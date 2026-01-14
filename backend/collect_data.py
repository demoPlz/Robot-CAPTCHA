"""Entry point for collecting robot manipulation data with crowd-sourced labeling. Adapted from lerobot's
control_robot.py.

Example usage:
    python backend/collect_data.py \
        --robot.type=trossen_ai_single_arm \
        --robot.max_relative_target=null \
        --control.type=record \
        --control.fps=30 \
        --control.single_task="Put the objects on the desk into the middle drawer" \
        --task-name=drawer \
        --control.repo_id=$USER/debug \
        --control.data_collection_policy_repo_id=$USER/debug_dcp \
        --control.tags='["tutorial"]' \
        --control.warmup_time_s=5 \
        --control.num_episodes=2 \
        --control.push_to_hub=false \
        --control.num_image_writer_processes=8 \
        --control.play_sound=false \
        --required-responses-per-critical-state=2 \
        --show-demo-videos

"""

import logging
from dataclasses import asdict
from pathlib import Path
from pprint import pformat
from threading import Thread
import time

import cv2  # for closing display windows
import numpy as np
from crowd_interface import *
from crowd_interface_config import CrowdInterfaceConfig
from flask_app import create_flask_app

# from safetensors.torch import load_file, save_file
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.policies.factory import make_policy
from lerobot.common.robot_devices.control_configs import (
    ControlPipelineConfig,
    RecordControlConfig,
)
from lerobot.common.robot_devices.control_utils import (
    init_keyboard_listener,
    record_episode_crowd,
    reset_environment_crowd,
    sanity_check_dataset_name,
    sanity_check_dataset_robot_compatibility,
)
from lerobot.common.robot_devices.robots.utils import Robot, make_robot_from_config
from lerobot.common.robot_devices.utils import safe_disconnect
from lerobot.common.utils.utils import has_method, init_logging, log_say
from lerobot.configs import parser
import socket
import subprocess
import time
from werkzeug.serving import make_server


def _stop_display_only(listener, display_cameras: bool):
    """Minimal UI teardown that does NOT touch the robot.

    - Listener is a daemon thread; it dies with the process.
    - Close any OpenCV windows if we showed cameras.

    """
    try:
        if display_cameras:
            cv2.destroyAllWindows()
    except Exception:
        pass


# Parse crowd interface config once at import time so @parser.wrap can run normally later
_CROWD_CONFIG = CrowdInterfaceConfig.from_cli_args()


@safe_disconnect
def record(robot: Robot, crowd_interface: CrowdInterface, cfg: RecordControlConfig) -> LeRobotDataset:
    if cfg.resume:
        dataset = LeRobotDataset(
            cfg.repo_id,
            root=cfg.root,
        )
        if len(robot.cameras) > 0:
            dataset.start_image_writer(
                num_processes=cfg.num_image_writer_processes,
                num_threads=cfg.num_image_writer_threads_per_camera * len(robot.cameras),
            )
        sanity_check_dataset_robot_compatibility(dataset, robot, cfg.fps, cfg.video)
    else:
        # Create empty dataset or load existing saved episodes
        sanity_check_dataset_name(cfg.repo_id, cfg.policy)
        dataset = LeRobotDataset.create(
            cfg.repo_id,
            cfg.fps,
            root=cfg.root,
            robot=robot,
            use_videos=cfg.video,
            image_writer_processes=cfg.num_image_writer_processes,
            image_writer_threads=cfg.num_image_writer_threads_per_camera * len(robot.cameras),
        )

    crowd_interface.init_dataset(cfg, robot)

    # If continuing from a previous dataset, drive robot to positions
    if _CROWD_CONFIG.continue_from_dataset:
        print(f"🔄 Continue mode active")
        
        # First, drive to home position (same as normal reset)
        print(f"🏠 Moving robot to home position first...")
        if not robot.is_connected:
            robot.connect()
        
        # Get the home position from robot
        initial_position = list(robot.follower_arms['main'].read("Present_Position"))
        print(f"   Robot starting from: {[f'{x:.3f}' for x in initial_position]}")
        
        # Now drive to the last critical state from the checkpoint
        print(f"📍 Now moving to last critical state from checkpoint...")
        continue_result = crowd_interface.continue_from_last_critical()
        
        if continue_result.get("status") == "error":
            print(f"❌ Failed to continue: {continue_result.get('message')}")
            print(f"   Will start from current robot position instead")
        else:
            target_positions = np.array(continue_result.get("joint_positions"))
            print(f"   Target: {[f'{x:.3f}' for x in target_positions]}")
            
            # Directly command robot to move to target position
            print(f"🤖 Driving robot...")
            robot.follower_arms['main'].write("Goal_Position", target_positions, duration=3.0)
            
            # Wait for robot to reach target position
            time.sleep(0.5)  # Initial delay for command to register
            
            # Monitor movement until robot reaches target
            max_wait_time = 10.0  # seconds
            start_time = time.time()
            while time.time() - start_time < max_wait_time:
                current_pos = np.array(list(robot.follower_arms['main'].read("Present_Position")))
                diff = current_pos - target_positions
                max_diff = np.max(np.abs(diff))
                
                if max_diff < 0.01:  # Close enough (1cm / 0.01 radians)
                    break
                    
                time.sleep(0.1)
            
            final_pos = np.array(list(robot.follower_arms['main'].read("Present_Position")))
            final_diff = final_pos - target_positions
            print(f"   Arrived at: {[f'{x:.3f}' for x in final_pos]}")
            print(f"   Error: {[f'{x:.3f}' for x in final_diff]} (max: {np.max(np.abs(final_diff)):.3f})")
            
            print(f"✅ Robot positioned at last critical state")
            print(f"   Please adjust the scene to match the checkpoint state")
            input("Press Enter when ready to continue recording...")

    # Load pretrained policy
    policy = make_policy(cfg.policy, ds_meta=dataset.meta) if cfg.policy is not None else None

    # Disable the leader arms since we use policy
    robot.leader_arms = []

    if not robot.is_connected:
        robot.connect()

    listener, events = init_keyboard_listener()

    # Pass events to crowd_interface for API control
    crowd_interface.set_events(events)

    # Skip safety stop in continue mode to avoid moving the robot from its positioned state
    if has_method(robot, "teleop_safety_stop") and not _CROWD_CONFIG.continue_from_dataset:
        robot.teleop_safety_stop()

    recorded_episodes = 0
    while True:
        if recorded_episodes >= cfg.num_episodes:
            break

        # Use crowd dataset's episode count (which may be > 0 if continuing from checkpoint)
        current_episode_index = crowd_interface.dataset_manager.dataset.meta.total_episodes
        log_say(f"Recording episode {current_episode_index}", cfg.play_sounds)
        # Ensure immediate-execution only fires for submissions belonging to the
        # *currently active* episode loop.
        crowd_interface.set_active_episode(current_episode_index)
        try:
            record_episode_crowd(
                robot=robot,
                dataset=dataset,
                events=events,
                episode_time_s=cfg.episode_time_s,
                display_cameras=cfg.display_cameras,
                policy=policy,
                fps=cfg.fps,
                single_task=cfg.single_task,
                crowd_interface=crowd_interface,
                episode_id=current_episode_index,
            )
        finally:
            # Leave no active episode once the loop exits (including early exit).
            crowd_interface.set_active_episode(None)

        if not events["stop_recording"] and ((recorded_episodes < cfg.num_episodes - 1) or events["rerecord_episode"]):
            log_say("Reset the environment", cfg.play_sounds)
            reset_environment_crowd(robot, events, cfg.reset_time_s, cfg.fps, crowd_interface)

        if events["rerecord_episode"]:
            log_say("Re-record episode", cfg.play_sounds)
            events["rerecord_episode"] = False
            events["exit_early"] = False
            dataset.clear_episode_buffer()
            continue

        dataset.save_episode()
        recorded_episodes += 1

        if events["stop_recording"]:
            break

    log_say("Stop recording from cameras", cfg.play_sounds, blocking=True)
    _stop_display_only(listener, cfg.display_cameras)
    
    # Auto-finalize async pool if in async mode
    if _CROWD_CONFIG.asynchronous_mode:
        log_say("Finalizing admin phase and preparing async pool", cfg.play_sounds)
        result = crowd_interface.state_manager.finalize_admin_phase()
        if result.get("status") == "success":
            states_count = result.get("states_in_pool", 0)
            log_say(f"Async pool ready: {states_count} states available for user labeling", cfg.play_sounds)
            print(f"📊 Async pool status: {states_count} states ready")
            print(f"   Users can now label via Netlify")
            
            # Wait for users to complete all labeling
            print(f"\n⏳ Waiting for users to complete async labeling...")
            print(f"   (Press Ctrl+C to stop waiting and exit)")
            
            import time
            check_interval = 5  # Check every 5 seconds
            last_status = None
            
            try:
                while True:
                    status = crowd_interface.state_manager.get_async_pool_status()
                    total = status.get("total_states", 0)
                    completed = status.get("states_completed", 0)
                    in_progress = status.get("states_in_progress", 0)
                    
                    # Only print when status changes to avoid spam
                    current_status = (completed, in_progress)
                    if current_status != last_status:
                        print(f"   📊 Progress: {completed}/{total} states completed, {in_progress} in progress")
                        last_status = current_status
                    
                    # Check if all states are fully labeled
                    if completed >= total and total > 0:
                        print(f"\n✅ All {total} states have been labeled by users!")
                        print(f"⏳ Waiting 10 seconds for dataset to save...")
                        time.sleep(10)
                        break
                    
                    time.sleep(check_interval)
            except KeyboardInterrupt:
                print(f"\n⚠️  User interrupted - exiting with partial labeling")
                print(f"   {completed}/{total} states completed")
        else:
            print(f"⚠️  Async finalization failed: {result.get('message')}")

    if cfg.push_to_hub:
        dataset.push_to_hub(tags=cfg.tags, private=cfg.private)
        crowd_interface.dataset.push_to_hub(tags=cfg.tags, private=cfg.private)

    log_say("Data collection session complete. Exiting", cfg.play_sounds)

    return dataset


@parser.wrap()
def control_robot(cfg: ControlPipelineConfig):
    # exact same pattern as lerobot/scripts/control_robot.py
    init_logging()
    logging.info(pformat(asdict(cfg)))

    # Disable Flask request logging to reduce terminal noise
    logging.getLogger("werkzeug").setLevel(logging.WARNING)

    # Log the prompt mode being used
    prompt_mode = "manual" if _CROWD_CONFIG.use_manual_prompt else "simple"
    logging.info(f"[Crowd] Prompt mode selected: {prompt_mode}")
    logging.info(f"[Crowd] Task name: {_CROWD_CONFIG.task_name}")

    # Use the crowd interface config to create CrowdInterface
    # Find an available port starting from 9000
    def find_free_port(start_port=9000, max_attempts=100):
        for port in range(start_port, start_port + max_attempts):
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.bind(('0.0.0.0', port))
                sock.close()
                return port
            except OSError:
                continue
        raise RuntimeError(f"No free port found in range {start_port}-{start_port + max_attempts}")
    
    port = find_free_port()
    print(f"\n🌐 Starting Flask server on port {port}")
    
    # Write port to config file for localhost frontend
    import json
    from datetime import datetime
    config_file = Path(__file__).parent.parent / "public" / "backend-config.json"
    config_file.parent.mkdir(parents=True, exist_ok=True)
    with open(config_file, "w") as f:
        json.dump({
            "backendUrl": f"http://127.0.0.1:{port}",
            "updatedAt": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "port": port
        }, f, indent=2)
    print(f"📝 Updated {config_file} with port {port}")
    
    crowd_interface = CrowdInterface(**_CROWD_CONFIG.to_crowd_interface_kwargs())
    crowd_interface.init_cameras()
    
    # Register cleanup handlers to ensure workers are killed on exit
    crowd_interface.register_cleanup_handlers()

    app = create_flask_app(crowd_interface)
    # Use a threaded WSGI server to handle multiple requests concurrently
    http_server = make_server("0.0.0.0", port, app, threaded=True)
    server_thread = Thread(target=http_server.serve_forever, name="flask-wsgi", daemon=True)
    server_thread.start()
    
    # Start cloudflared tunnel pointing to the Flask port
    print(f"🚇 Starting cloudflared tunnel for port {port}...")
    subprocess.Popen(
        ["pkill", "-f", "cloudflared"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    ).wait()
    time.sleep(0.5)
    
    tunnel_log = Path("/tmp/cloudflared.log")
    tunnel_log.unlink(missing_ok=True)
    
    tunnel_process = subprocess.Popen(
        ["cloudflared", "tunnel", "--url", f"http://localhost:{port}"],
        stdout=open(tunnel_log, "w"),
        stderr=subprocess.STDOUT
    )
    
    # Wait for tunnel URL to appear
    print("⏳ Waiting for tunnel URL...")
    tunnel_url = None
    for _ in range(30):  # Wait up to 15 seconds
        time.sleep(0.5)
        if tunnel_log.exists():
            content = tunnel_log.read_text()
            import re
            match = re.search(r'https://[a-z0-9-]+\.trycloudflare\.com', content)
            if match:
                tunnel_url = match.group(0)
                break
    
    if tunnel_url:
        print(f"✅ Tunnel active: {tunnel_url}")
        print(f"🚀 Auto-deploying frontend to Netlify...")
        
        # Temporarily update backend-config.json with tunnel URL for build
        with open(config_file, "w") as f:
            json.dump({
                "backendUrl": tunnel_url,
                "updatedAt": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
            }, f, indent=2)
        
        # Auto-deploy to Netlify
        deploy_result = subprocess.run(
            ["bash", "-c", "cd " + str(Path(__file__).parent.parent) + " && npm run build && npx netlify deploy --prod --dir=dist"],
            capture_output=True,
            text=True
        )
        
        # Restore localhost config after deployment
        with open(config_file, "w") as f:
            json.dump({
                "backendUrl": f"http://127.0.0.1:{port}",
                "updatedAt": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
                "port": port
            }, f, indent=2)
        
        if deploy_result.returncode == 0:
            print("✅ Frontend deployed to Netlify successfully")
            print(f"✅ Localhost config restored to port {port}")
        else:
            print("⚠️  Netlify deployment failed (you can deploy manually with: ./scripts/deploy_with_tunnel.sh)")
            if deploy_result.stderr:
                print(f"   Error: {deploy_result.stderr[:200]}")
    else:
        print("⚠️  Could not detect tunnel URL, check /tmp/cloudflared.log")
    print()

    robot = make_robot_from_config(cfg.robot)

    assert isinstance(cfg.control, RecordControlConfig), "This script is for data collection"

    record(robot, crowd_interface, cfg.control)

    if robot.is_connected:
        robot.disconnect()
    
    # Explicitly shutdown crowd_interface
    crowd_interface.shutdown()

    print("Data Collection Completed")


if __name__ == "__main__":
    control_robot()
