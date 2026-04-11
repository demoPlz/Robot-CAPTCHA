"""      
CrowdInterface - Main Backend Interface

Coordinates all subsystem managers to provide an API for crowd-sourced robot data collection.
"""

import atexit
import base64
import os
import signal
import sys
import tempfile
import time
from pathlib import Path
from threading import Lock
from typing import Optional

import numpy as np
import torch

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    print("⚠️  psutil not available - process cleanup will be limited")

from hardware_config import CAM_IDS, REAL_CALIB_PATHS, SIM_CALIB_PATHS
from interface_managers.action_selector_manager import ActionSelectorManager
from interface_managers.calibration_manager import CalibrationManager
from interface_managers.dataset_manager import DatasetManager
from interface_managers.demo_video_manager import DemoVideoManager
from interface_managers.drawer_position_manager import DrawerPositionManager
from interface_managers.mturk_manager import MTurkManager
from interface_managers.observation_stream_manager import ObservationStreamManager
from interface_managers.pose_estimation_manager import PoseEstimationManager
from interface_managers.sim_manager import SimManager
from interface_managers.state_manager import StateManager
from interface_managers.webcam_manager import WebcamManager


class CrowdInterface:
    """Main interface between frontend and backend for CAPTCHA-style crowd-sourced data collection for robot
    manipulation.

    Coordinates all subsystem managers:
    - State management (episodes, states, responses)
    - Camera/observation handling
    - Dataset operations
    - Calibration
    - Demo videos and prompts
    - Simulation rendering
    - Pose estimation

    """

    # =========================
    # Initialization
    # =========================

    def __init__(
        self,
        required_responses_per_state: int = 1,
        required_responses_per_critical_state: int = 3,
        required_approvals_per_critical_state: int = 3,
        num_expert_workers: int = 2,
        jitter_threshold: float = 0.01,
        autofill_critical_states: bool = False,
        num_autofill_actions: int | None = None,
        # Asynchronous mode
        asynchronous_mode: bool = False,
        async_admin_responses_per_state: int = 1,
        use_manual_prompt: bool = False,
        # --- saving critical-state cam_main frames ---
        save_maincam_sequence: bool = False,
        prompt_sequence_dir: str | None = None,
        prompt_sequence_clear: bool = False,
        # task text displayed to users and stored in dataset
        task_text: str | None = None,
        # used ONLY for prompt substitution and demo assets
        task_name: str | None = None,
        # --- demo video recording ---
        record_demo_videos: bool = False,
        demo_videos_dir: str | None = None,
        demo_videos_clear: bool = False,
        # --- read-only demo video display (independent of recording) ---
        show_demo_videos: bool = False,
        show_videos_dir: str | None = None,
        frontend_url: str | None = None,  # URL where frontend is hosted (e.g., Netlify)
        # --- tutorial state capture ---
        enable_tutorial_state_capture: bool = False,
        # --- sim ---
        use_sim: bool = True,
        use_gpu_physics: bool = False,
        max_animation_users: int = 2,
        usd_path: str | None = None,
        # --- objects ---
        objects: dict[str, str] | None = None,
        object_mesh_paths: dict[str, str] | None = None,
        mesh_scale: float = 1.0,
        joint_tracking: list | None = None,
        # --- pose estimation mode ---
        use_random_poses: bool = False,
        random_pose_bounds: dict | None = None,
        # --- action selection ---
        action_selector_mode: str = "random",
        action_selector_epsilon: float = 0.1,
        action_selector_model_path: str | None = None,
        # --- mturk integration ---
        use_mturk: bool = False,
        mturk_sandbox: bool = True,
        mturk_reward: float = 0.50,
        mturk_assignment_duration_seconds: int = 600,
        mturk_lifetime_seconds: int = 3600,
        mturk_auto_approval_delay_seconds: int = 60,
        mturk_assignment_coefficient: float = 1.0,
        mturk_title: str = "Control a robot arm to complete a manipulation task",
        mturk_description: str = "View a robot simulation and specify the next position for the robot to move to",
        mturk_keywords: str = "robot, manipulation, annotation, simulation",
        mturk_external_url: str | None = None,
        # --- mturk worker qualifications ---
        mturk_use_qualifications: bool = True,
        mturk_require_masters: bool = False,
        mturk_min_approval_rate: int = 95,
        mturk_min_approved_hits: int = 100,
        mturk_require_location: list[str] | None = None,
        # --- home position ---
        home_position_deg: list[float] | None = None,
        # --- container presets (sorting task) ---
        container_presets_deg: list[list[float]] | None = None,
    ):

        # --- Shutdown tracking ---
        self._shutdown_complete = False
        self._cleanup_registered = False

        # --- UI prompt mode (simple vs MANUAL) ---
        self.use_manual_prompt = bool(use_manual_prompt or int(os.getenv("USE_MANUAL_PROMPT", "0")))

        # --- Tutorial state capture ---
        self.enable_tutorial_state_capture = enable_tutorial_state_capture
        print(f"[CrowdInterface] Tutorial state capture enabled: {self.enable_tutorial_state_capture}")

        # --- Sim ---
        self.use_sim = use_sim
        self.use_gpu_physics = use_gpu_physics
        self.max_animation_users = max_animation_users
        self.usd_path = usd_path

        # --- MTurk integration ---
        self.use_mturk = use_mturk

        # --- Objects ---
        self.objects = objects
        self.object_mesh_paths = object_mesh_paths
        self.mesh_scale = mesh_scale
        
        # --- Pose estimation mode ---
        self.use_random_poses = use_random_poses
        self.random_pose_bounds = random_pose_bounds if random_pose_bounds is not None else {
            "x_min": -0.3, "x_max": 0.3,
            "y_min": -0.3, "y_max": 0.3,
            "z_min": 0.0, "z_max": 0.3
        }

        # -------- Observation disk cache (spills heavy per-state obs to disk) --------
        # Set CROWD_OBS_CACHE to override where temporary per-state observations are stored.
        self._obs_cache_root = Path(
            os.getenv("CROWD_OBS_CACHE", os.path.join(tempfile.gettempdir(), "crowd_obs_cache"))
        )
        try:
            self._obs_cache_root.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

        self.goal_lock = Lock()
        self._gripper_motion = 1  # Initialize gripper motion

        # Reset state management
        self.is_resetting = False
        self.reset_start_time = None
        self.reset_duration_s = 0

        # Control events for keyboard-like functionality
        self.events = None

        # N responses pattern
        self.required_responses_per_state = required_responses_per_state
        self.required_responses_per_critical_state = required_responses_per_critical_state
        self.required_approvals_per_critical_state = required_approvals_per_critical_state
        self.jitter_threshold = jitter_threshold
        self.autofill_critical_states = bool(autofill_critical_states)
        # If not specified, default to "complete on first submission"
        if num_autofill_actions is None:
            self.num_autofill_actions = self.required_responses_per_critical_state
        else:
            self.num_autofill_actions = int(num_autofill_actions)
        # Clamp to [1, required_responses_per_critical_state]
        self.num_autofill_actions = max(1, min(self.num_autofill_actions, self.required_responses_per_critical_state))
        
        # Asynchronous mode settings
        self.asynchronous_mode = asynchronous_mode
        self.async_admin_responses_per_state = async_admin_responses_per_state

        # IP address banning system
        self.banned_ips: set[str] = set()
        self.banned_ips_lock = Lock()

        # Episode-based state management (shared with StateManager via reference)
        self.pending_states_by_episode = {}  # episode_id -> {state_id -> {state: dict, responses_received: int}}
        self.completed_states_by_episode = (
            {}
        )  # episode_id -> {state_id -> {responses_received: int, completion_time: float}}
        self.completed_states_buffer_by_episode = (
            {}
        )  # episode_id -> {state_id -> completed_state_dict} - buffer for chronological add_frame
        self.served_states_by_episode = {}  # episode_id -> {session_id -> state_id}
        self.episodes_being_completed = set()  # Track episodes currently being processed for completion

        self.state_lock = Lock()  # Protects all episode-based state management

        # Task text used for UI prompt and dataset frames
        self.task_text = task_text
        # Task name used for prompt placeholder substitution and demo images (from --task-name)
        self.task_name = task_name
        # Gripper starts closed for insertion/switches tasks, open for everything else
        self.initial_gripper_open = (task_name not in ("insertion", "switches"))
        # Home position (degrees): defaults to standard if not provided
        self.home_position_deg = home_position_deg if home_position_deg is not None else [0, 60, 75, -60, 0, 0, 2]
        self.container_presets_deg = container_presets_deg if container_presets_deg is not None else []
        # Frontend URL for serving uncompressed videos from CDN
        self.frontend_url = frontend_url

        # Calibration manager
        repo_root = Path(__file__).resolve().parent.parent
        self.calibration = CalibrationManager(
            use_sim=self.use_sim,
            repo_root=repo_root,
            real_calib_paths=REAL_CALIB_PATHS,
            sim_calib_paths=SIM_CALIB_PATHS,
        )

        # Demo video manager
        self.video_manager = DemoVideoManager(
            task_name=task_name,
            record_demo_videos=record_demo_videos,
            demo_videos_dir=demo_videos_dir,
            demo_videos_clear=demo_videos_clear,
            show_demo_videos=show_demo_videos,
            show_videos_dir=show_videos_dir,
            save_maincam_sequence=save_maincam_sequence,
            prompt_sequence_dir=prompt_sequence_dir,
            prompt_sequence_clear=prompt_sequence_clear,
            repo_root=repo_root,
        )

        # Webcam manager
        self.webcam_manager = WebcamManager(
            cam_ids=CAM_IDS,
            undistort_maps=self.calibration.get_undistort_maps(),
            jpeg_quality=int(os.getenv("JPEG_QUALITY", "80")),
        )

        # Observation stream manager
        self.obs_stream = ObservationStreamManager(encoder_func=self.webcam_manager.encode_jpeg_base64)

        # Sim manager
        self.sim_manager = SimManager(
            use_sim=self.use_sim,
            use_gpu_physics=self.use_gpu_physics,
            task_name=task_name,
            usd_path=self.usd_path,
            obs_cache_root=self._obs_cache_root,
            state_lock=self.state_lock,
            pending_states_by_episode=self.pending_states_by_episode,
            completed_states_by_episode=self.completed_states_by_episode,
            webcam_manager=self.webcam_manager,
            calibration_manager=self.calibration,
            max_animation_users=self.max_animation_users,
            objects=objects,
            state_ready_callback=self._on_critical_state_ready,  # NEW: MTurk HIT creation
        )

        # Debounced episode finalization
        self.episode_finalize_grace_s = 2.0

        # Precompute immutable views and camera poses to avoid per-tick allocations

        self._exec_gate_by_session: dict[str, dict] = {}

        # Pose estimation manager
        self.pose_estimator = PoseEstimationManager(
            obs_cache_root=self._obs_cache_root,
            object_mesh_paths=object_mesh_paths,
            objects=objects,
            calibration_manager=self.calibration,
            state_lock=self.state_lock,
            pending_states_by_episode=self.pending_states_by_episode,
            use_random_poses=use_random_poses,
            random_pose_bounds=self.random_pose_bounds,
            mesh_scale=mesh_scale,
            task_name=task_name,
        )

        # Drawer position manager (only if joint_tracking is configured)
        if joint_tracking:
            self.drawer_position = DrawerPositionManager(
                calibration_manager=self.calibration,
                drawer_joint_name=joint_tracking[0] if joint_tracking else "Drawer_Joint",
                repo_root=repo_root,
            )
        else:
            self.drawer_position = None

        # --- Episode save behavior: datasets are always auto-saved after finalization ---
        # Manual save is only used for demo video recording workflow
        self._episodes_pending_save: set[str] = set()

        # Dataset manager
        self.dataset_manager = DatasetManager(
            required_responses_per_critical_state=self.required_responses_per_critical_state,
            obs_cache_root=self._obs_cache_root,
            asynchronous_mode=asynchronous_mode,
        )

        # Action selector manager
        self.action_selector = ActionSelectorManager(
            mode=action_selector_mode,
            epsilon=action_selector_epsilon,
            learned_model_path=action_selector_model_path,
            device="cpu",  # Can be made configurable if GPU is needed
        )

        # MTurk manager (optional)
        self.mturk_manager = None
        if use_mturk:
            try:
                # Auto-detect cloudflare tunnel URL if not explicitly provided
                effective_external_url = mturk_external_url
                if not effective_external_url:
                    effective_external_url = self._detect_cloudflare_tunnel_url()
                    if effective_external_url:
                        print(f"🔗 Auto-detected cloudflare tunnel URL: {effective_external_url}")
                    else:
                        print(f"⚠️  No cloudflare tunnel URL detected. MTurk HITs will fail to create.")
                        print(f"   Run './start_tunnel.sh' or set --mturk-external-url manually")
                
                self.mturk_manager = MTurkManager(
                    sandbox=mturk_sandbox,
                    reward=mturk_reward,
                    assignment_duration_seconds=mturk_assignment_duration_seconds,
                    lifetime_seconds=mturk_lifetime_seconds,
                    auto_approval_delay_seconds=mturk_auto_approval_delay_seconds,
                    assignment_coefficient=mturk_assignment_coefficient,
                    title=mturk_title,
                    description=mturk_description,
                    keywords=mturk_keywords,
                    external_url=effective_external_url,
                    num_expert_workers=num_expert_workers,
                    required_responses_per_critical_state=required_responses_per_critical_state,
                    use_qualifications=mturk_use_qualifications,
                    require_masters=mturk_require_masters,
                    min_approval_rate=mturk_min_approval_rate,
                    min_approved_hits=mturk_min_approved_hits,
                    require_location=mturk_require_location,
                    get_state_data_callback=self._get_state_data_for_mturk,
                )
                print(f"✅ MTurk integration enabled ({'sandbox' if mturk_sandbox else 'production'})")
            except Exception as e:
                print(f"⚠️  Failed to initialize MTurk manager: {e}")
                print(f"   Continuing without MTurk integration")
                self.mturk_manager = None

        # State manager (handles episode-based state lifecycle)
        self.state_manager = StateManager(
            required_responses_per_state=self.required_responses_per_state,
            required_responses_per_critical_state=self.required_responses_per_critical_state,
            required_approvals_per_critical_state=self.required_approvals_per_critical_state,
            autofill_critical_states=self.autofill_critical_states,
            num_autofill_actions=self.num_autofill_actions,
            asynchronous_mode=self.asynchronous_mode,
            async_admin_responses_per_state=self.async_admin_responses_per_state,
            use_manual_prompt=self.use_manual_prompt,
            use_sim=self.use_sim,
            task_text=self.task_text,
            jitter_threshold=self.jitter_threshold,
            obs_cache_root=self._obs_cache_root,
            state_lock=self.state_lock,
            pending_states_by_episode=self.pending_states_by_episode,
            completed_states_by_episode=self.completed_states_by_episode,
            completed_states_buffer_by_episode=self.completed_states_buffer_by_episode,
            episode_finalize_grace_s=self.episode_finalize_grace_s,
            episodes_pending_save=self._episodes_pending_save,
            obs_stream_manager=self.obs_stream,
            pose_estimation_manager=self.pose_estimator,
            drawer_position_manager=self.drawer_position,
            sim_manager=self.sim_manager,
            action_selector_manager=self.action_selector,
            dataset_manager=self.dataset_manager,
            persist_views_callback=self._persist_views_to_disk,
            persist_obs_callback=self._persist_obs_to_disk,
            snapshot_views_callback=self.snapshot_latest_views,
            save_episode_callback=self.dataset_manager.save_episode,
            home_position_deg=self.home_position_deg,
            container_presets_deg=self.container_presets_deg,
        )

    # =========================
    # Camera & Observation Management
    # =========================

    def init_cameras(self):
        """Open webcams and start background capture.

        Delegates to WebcamManager.

        """
        self.webcam_manager.init_cameras()

    def snapshot_latest_views(self) -> dict[str, str]:
        """Snapshot the latest **JPEG base64 strings** for each camera.

        Includes both webcam views and observation camera previews.

        """
        # Get webcam views from manager
        out = self.webcam_manager.snapshot_latest_views()

        # Include latest observation camera previews from manager
        out.update(self.obs_stream.get_latest_obs_jpeg())

        return out

    def state_to_json(self, state: dict) -> dict:
        """
        Build the JSON payload for the labeling frontend:
        - If the state contains 'view_paths', load the state-aligned JPEGs from disk (correct behavior).
        - Otherwise, fall back to the latest live previews.
        Also attach static camera models/poses.
        """
        if not state:
            return {}
        out = dict(state)  # shallow copy (we'll remove internal fields)

        # Remove tensor fields that frontend doesn't need
        out.pop("actions", None)
        
        # Remove any other non-JSON-serializable fields
        import torch
        import math
        def make_serializable(obj):
            """Recursively convert tensors and other non-serializable objects."""
            if isinstance(obj, torch.Tensor):
                # Convert tensor to list, then recursively process to handle NaN/Inf
                return make_serializable(obj.tolist())
            elif isinstance(obj, float):
                if math.isnan(obj) or math.isinf(obj):
                    return None  # Convert NaN/Inf to null in JSON
                return obj
            elif isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [make_serializable(item) for item in obj]
            return obj
        
        # Convert any remaining tensors (though actions should be removed already)
        out = make_serializable(out)

        # Remove internal/disk paths that shouldn't be exposed to client
        out.pop("obs_path", None)  # don't expose obs cache paths

        # Prefer state-aligned snapshots if available
        views = {}
        view_paths = out.pop("view_paths", None)  # don't expose file paths to the client
        views = self._load_views_from_disk(view_paths)

        out["views"] = views
        out["camera_poses"] = self.calibration.get_camera_poses()
        out["camera_models"] = self.calibration.get_camera_models()
        out["gripper_tip_calib"] = self.calibration.get_gripper_tip_calib()

        # --- Attach example video URL (direct file URL; byte-range capable) ---
        if self.video_manager.show_demo_videos:
            # Detect if request is from Netlify or localhost
            from flask import request
            origin = request.headers.get('Origin', '')
            referer = request.headers.get('Referer', '')
            
            # Check if request comes from Netlify frontend
            is_netlify_origin = (
                'robot-captcha.netlify.app' in origin or
                'robot-captcha.netlify.app' in referer
            )
            
            # Prefer a VLM-selected clip if available and present
            video_id = state.get("video_prompt")
            chosen_url = None
            if video_id is not None:
                p, _ = self.video_manager.find_show_video_by_id(video_id)
                if p:
                    # Serve from static demos directory for fast loading
                    video_filename = Path(p).name
                    # If from Netlify, use Netlify CDN (same origin, no CORS)
                    # If from localhost, use backend tunnel with high quality videos
                    if is_netlify_origin and self.frontend_url:
                        chosen_url = f"{self.frontend_url.rstrip('/')}/demos_hq/{video_filename}"
                    else:
                        # Serve high quality through backend tunnel (works for localhost and other origins)
                        chosen_url = f"/demos_hq/{video_filename}"

            # Fallback: latest available .webm
            if not chosen_url:
                lp, lid = self.video_manager.find_latest_show_video()
                if lp and lid:
                    video_filename = Path(lp).name
                    if is_netlify_origin and self.frontend_url:
                        chosen_url = f"{self.frontend_url.rstrip('/')}/demos_hq/{video_filename}"
                    else:
                        chosen_url = f"/demos_hq/{video_filename}"

            if chosen_url:
                out["example_video_url"] = chosen_url

        return out

    def encode_jpeg_base64(self, img_rgb: np.ndarray, quality: int | None = None) -> str:
        """Encode an RGB image to a base64 JPEG data URL.

        Delegates to WebcamManager.

        """
        return self.webcam_manager.encode_jpeg_base64(img_rgb, quality)

    # =========================
    # State Management (Delegated to StateManager)
    # =========================

    def add_state(
        self,
        joint_positions: dict,
        gripper_motion: int = None,
        obs_dict: dict[str, torch.Tensor] = None,
        episode_id: str = None,
        left_carriage_external_force: float | None = None,
    ):
        """Add a new state to the current episode.

        Delegates to StateManager.

        """
        return self.state_manager.add_state(
            joint_positions=joint_positions,
            gripper_motion=gripper_motion,
            obs_dict=obs_dict,
            episode_id=episode_id,
            left_carriage_external_force=left_carriage_external_force,
        )

    def set_last_state_to_critical(self):
        """Mark the last added state as critical.

        Delegates to StateManager.

        """
        return self.state_manager.set_last_state_to_critical()

    def get_latest_state(self, user_email: str = None, user_name: str = None) -> dict:
        """Get the latest pending critical state for labeling.

        Delegates to StateManager.
        
        Args:
            user_email: Optional user email for timing tracking
            user_name: Optional user name for test user detection

        """
        return self.state_manager.get_latest_state(user_email=user_email, user_name=user_name)

    def record_response(self, response_data: dict):
        """Record a user response for a state.

        Delegates to StateManager.

        """
        return self.state_manager.record_response(response_data)

    def get_pending_states_info(self) -> dict:
        """Get episode-based state information for monitoring.

        Delegates to StateManager.

        """
        return self.state_manager.get_pending_states_info()

    def set_active_episode(self, episode_id):
        """Mark which episode the robot loop is currently in.

        Delegates to StateManager.

        """
        return self.state_manager.set_active_episode(episode_id)

    def set_prompt_ready(
        self, state_info: dict, episode_id: int, state_id: int, text: str | None, video_id: int | None
    ) -> None:
        """Set prompt fields and mark state as ready.

        Delegates to StateManager.

        """
        return self.state_manager.set_prompt_ready(state_info, episode_id, state_id, text, video_id)

    def get_latest_goal(self) -> dict | None:
        """Get and clear the latest goal for robot execution.

        Delegates to StateManager.

        """
        return self.state_manager.get_latest_goal()

    def undo_to_previous_critical_state(self) -> dict | None:
        """Undo to the previous critical state.

        Removes all states after the previous critical state and returns
        the robot position to revert to.

        Delegates to StateManager.

        Returns:
            dict with 'joint_positions', 'gripper', 'episode_id', 'reverted_to_state_id'
            or None if undo is not possible

        """
        return self.state_manager.undo_to_previous_critical_state()
    
    def clear_episode(self, episode_id: int) -> None:
        """Clear episode for re-recording (clears both crowd state and dataset buffer).
        
        Args:
            episode_id: Episode to clear
        """
        # Clear crowd interface state
        self.state_manager.clear_episode(episode_id)
        
        # Clear dataset buffer if it exists and matches this episode
        if (self.dataset_manager.dataset.episode_buffer is not None and
            self.dataset_manager.dataset.episode_buffer.get("episode_index") == episode_id):
            self.dataset_manager.dataset.clear_episode_buffer()
            print(f"🗑️  Cleared dataset episode buffer for episode {episode_id}")

    # =========================
    # Reset State Management
    # =========================

    def start_reset(self, duration_s: float):
        """Start the reset countdown timer."""
        self.is_resetting = True
        self.reset_start_time = time.time()
        self.reset_duration_s = duration_s
        print(f"🔄 Starting reset countdown: {duration_s}s")

    def stop_reset(self):
        """Stop the reset countdown timer."""
        self.is_resetting = False
        self.reset_start_time = None
        self.reset_duration_s = 0

    def get_reset_countdown(self) -> float:
        """Get remaining reset time in seconds, or 0 if not resetting."""
        if not self.is_resetting or self.reset_start_time is None:
            return 0

        elapsed = time.time() - self.reset_start_time
        remaining = max(0, self.reset_duration_s - elapsed)

        # Auto-stop when countdown reaches 0
        if remaining <= 0 and self.is_resetting:
            self.stop_reset()

        return remaining

    def is_in_reset(self) -> bool:
        """Check if currently in reset state."""
        return self.is_resetting and self.get_reset_countdown() > 0

    # =========================
    # Dataset Management (Delegated to DatasetManager)
    # =========================

    def init_dataset(self, cfg, robot, phase1_resumed: bool = False):
        """Initialize dataset for data collection policy training.

        Delegates to DatasetManager.

        """
        # Initialize dataset (may set single_task from cfg, but we use config task_text for UI)
        dataset_task = self.dataset_manager.init_dataset(cfg, robot, phase1_resumed=phase1_resumed)
        
        # Use task_text from config if provided, otherwise fall back to dataset's single_task
        if not self.task_text:
            self.task_text = dataset_task
        
        # Update state manager's task_text
        self.state_manager.task_text = self.task_text
        
        # Update async logger's output directory to dataset root (if async mode and logger exists)
        if self.state_manager.async_user_logger is not None and self.dataset_manager.dataset is not None:
            from interface_managers.async_user_logger import AsyncUserLogger
            dataset_root = self.dataset_manager.dataset.root
            self.state_manager.async_user_logger = AsyncUserLogger(dataset_root)
            print(f"📊 Async user logger updated to use dataset root: {dataset_root}/async_user_submissions.jsonl")

    # =========================
    # Calibration Management (Delegated to CalibrationManager)
    # =========================

    def save_gripper_tip_calibration(self, calib: dict) -> str:
        """Save gripper tip calibration and return the written path.

        Delegates to CalibrationManager.

        """
        return self.calibration.save_gripper_tip_calibration(calib)

    # =========================
    # Observation Cache Management (disk persistence)
    # =========================

    def _episode_cache_dir(self, episode_id: str) -> Path:
        """Get or create cache directory for an episode."""
        d = self._obs_cache_root / str(episode_id)
        if not d.exists():
            try:
                d.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass
        return d

    def _persist_obs_to_disk(self, episode_id: str, state_id: int, obs: dict) -> str | None:
        """Writes the observations dict to a single file for the state and returns the path."""
        try:
            p = self._episode_cache_dir(episode_id) / f"{state_id}.pt"
            # Tensors/ndarrays/py objects handled by torch.save
            torch.save(obs, p)
            return str(p)
        except Exception as e:
            print(f"⚠️  failed to persist obs ep={episode_id} state={state_id}: {e}")
            return None

    def _persist_views_to_disk(self, episode_id: str, state_id: int, views_b64: dict[str, str]) -> dict[str, str]:
        """Persist base64 (data URL) JPEGs for each camera to disk.

        Returns a mapping: camera_name -> absolute file path.

        """
        if not views_b64:
            return {}
        out: dict[str, str] = {}
        try:
            d = self._episode_cache_dir(episode_id) / "views"
            d.mkdir(parents=True, exist_ok=True)
            for cam, data_url in views_b64.items():
                # Expect "data:image/jpeg;base64,....."
                if not isinstance(data_url, str):
                    continue
                idx = data_url.find("base64,")
                if idx == -1:
                    continue
                b64 = data_url[idx + len("base64,") :]
                try:
                    raw = base64.b64decode(b64)
                except Exception:
                    continue
                path = d / f"{state_id}_{cam}.jpg"
                with open(path, "wb") as f:
                    f.write(raw)
                out[cam] = str(path)
        except Exception as e:
            print(f"⚠️  failed to persist views ep={episode_id} state={state_id}: {e}")
        return out

    def _load_views_from_disk(self, view_paths: dict[str, str]) -> dict[str, str]:
        """Load per-camera JPEG files and return data URLs."""
        if not view_paths:
            return {}
        out: dict[str, str] = {}
        for cam, path in view_paths.items():
            try:
                with open(path, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode("ascii")
                out[cam] = f"data:image/jpeg;base64,{b64}"
            except Exception:
                # Missing/removed file -> skip this camera
                pass
        return out

    # =========================
    # Prompting and Demo Media Management
    # =========================

    def _prompts_root_dir(self) -> Path:
        """Root folder containing prompts/."""
        return (Path(__file__).resolve().parent / ".." / "data" / "prompts").resolve()

    def _task_dir(self, task_name: str | None = None) -> Path:
        tn = task_name or self.task_name
        return (self._prompts_root_dir() / tn).resolve()

    def _parse_description_bank_entries(self, file_path: str) -> list[dict]:
        """Read description bank from file.

        Each line is a text prompt.
        Line number corresponds to video number.
        Returns: [{"id": int, "text": "<line content>", "full": "<line content>"}]

        """
        entries = []
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line_content = line.strip()
                    if line_content:  # Skip empty lines
                        entries.append({"id": line_num, "text": line_content, "full": line_content})
        except FileNotFoundError:
            print(f"Description bank file not found: {file_path}")
        except Exception as e:
            print(f"Error reading description bank file {file_path}: {e}")

        return entries

    def get_description_bank(self) -> dict:
        """Return both the raw description-bank text and its parsed entries.

        Reads from prompts/{task-name}/descriptions.txt where each line is a text prompt. Line number corresponds to
        video number.

        """
        task_name = self.task_name  # task_name is an attribute, not a method
        if not task_name:
            print("Warning: No task name set, cannot load description bank")
            return {"raw_text": "", "entries": []}

        # Construct file path: prompts/{task-name}/descriptions.txt
        file_path = self._task_dir(task_name) / "descriptions.txt"

        # Read raw text for compatibility
        raw_text = ""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                raw_text = f.read()
        except FileNotFoundError:
            print(f"Description bank file not found: {file_path}")
        except Exception as e:
            print(f"Error reading description bank file {file_path}: {e}")

        return {"raw_text": raw_text, "entries": self._parse_description_bank_entries(str(file_path))}

    def get_acceptance_criteria(self, instruction_id: int) -> list[str]:
        """Get acceptance criteria for a specific instruction.

        Reads from prompts/{task-name}/acceptance_criteria.txt.
        Format: Blocks separated by "=== N ===" where N is the instruction number.
        Each line after the header (non-empty) is a criterion.

        Args:
            instruction_id: 1-based instruction number (matches line in descriptions.txt)

        Returns:
            List of criteria strings for that instruction, or empty list if not found.
        """
        task_name = self.task_name
        if not task_name:
            print("Warning: No task name set, cannot load acceptance criteria")
            return []

        file_path = self._task_dir(task_name) / "acceptance_criteria.txt"
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
        except FileNotFoundError:
            print(f"Acceptance criteria file not found: {file_path}")
            return []
        except Exception as e:
            print(f"Error reading acceptance criteria file {file_path}: {e}")
            return []

        # Parse blocks: === N === marks the start of instruction N's criteria
        import re
        pattern = rf"===\s*{instruction_id}\s*===\s*\n(.*?)(?====\s*\d+\s*===|$)"
        match = re.search(pattern, content, re.DOTALL)
        
        if not match:
            return []
        
        block = match.group(1)
        # Each non-empty line is a criterion
        criteria = [line.strip() for line in block.strip().split('\n') if line.strip()]
        return criteria

    # =========================
    # Simulation & Animation Management (Delegated to SimManager)
    # =========================

    def start_animation(
        self,
        session_id: str,
        goal_pose: dict = None,
        goal_joints: list = None,
        duration: float = 3.0,
        gripper_action: str = None,
        episode_id: str = None,
        state_id: int = None,
    ) -> dict:
        """Start animation for a user session.

        Delegates to SimManager.

        """
        return self.sim_manager.start_animation(
            session_id, goal_pose, goal_joints, duration, gripper_action, episode_id, state_id
        )

    def stop_animation(self, session_id: str) -> dict:
        """Stop animation for a user session.

        Delegates to SimManager.

        """
        return self.sim_manager.stop_animation(session_id)

    def get_animation_status(self) -> dict:
        """Get current animation status and availability.

        Delegates to SimManager.

        """
        return self.sim_manager.get_animation_status()

    def reinitialize_simulation(self) -> dict:
        """Manually reinitialize simulation if it failed during startup.

        Delegates to SimManager.

        """
        return self.sim_manager.reinitialize_simulation()

    def capture_animation_frame(self, session_id: str) -> dict:
        """Capture current animation frame for a user session.

        Delegates to SimManager.

        """
        return self.sim_manager.capture_animation_frame(session_id)

    def release_animation_session(self, session_id: str) -> dict:
        """Release animation slot for a disconnected session.

        Delegates to SimManager.

        """
        return self.sim_manager.release_animation_session(session_id)

    # =========================
    # MTurk Integration (Delegated to MTurkManager)
    # =========================

    def create_mturk_hit(self, episode_id: int, state_id: int, state_info: dict = None) -> str | None:
        """Create MTurk HIT for a critical state.

        Args:
            episode_id: Episode ID
            state_id: State ID
            state_info: State information dict (optional, will fetch if not provided)

        Returns:
            HIT ID if successful, None otherwise

        """
        if not self.mturk_manager:
            print(f"❌ Cannot create HIT: MTurk manager not initialized")
            return None

        # Get state data if not provided (avoid deadlock when called from callback)
        if state_info is None:
            with self.state_lock:
                if episode_id in self.pending_states_by_episode:
                    state_info = self.pending_states_by_episode[episode_id].get(state_id)

        if not state_info:
            print(f"⚠️  Cannot create HIT: state not found (episode={episode_id}, state={state_id})")
            return None

        # Create HIT
        return self.mturk_manager.create_hit_for_state(
            episode_id=episode_id,
            state_id=state_id,
            state_data=state_info,
        )

    def update_mturk_assignment_count(self, episode_id: int, state_id: int):
        """Notify MTurk manager of a new assignment submission.

        Args:
            episode_id: Episode ID
            state_id: State ID

        """
        if self.mturk_manager:
            self.mturk_manager.update_hit_assignment_count(episode_id, state_id)
    
    def _get_state_data_for_mturk(self, episode_id: int, state_id: int) -> Optional[dict]:
        """Callback for MTurk manager to fetch state data.
        
        Used when creating replacement HITs after timeout.
        
        Args:
            episode_id: Episode ID
            state_id: State ID
            
        Returns:
            State data dict or None if not found
        """
        with self.state_lock:
            if episode_id in self.pending_states_by_episode:
                return self.pending_states_by_episode[episode_id].get(state_id)
        return None
    
    def get_mturk_submitted_assignments(self, episode_id: int, state_id: int) -> list[dict]:
        """Get submitted MTurk assignments for a state (for monitoring).
        
        Args:
            episode_id: Episode ID
            state_id: State ID
            
        Returns:
            List of assignment info dicts
        """
        if not self.mturk_manager:
            return []
        return self.mturk_manager.get_submitted_assignments(episode_id, state_id)

    def get_mturk_hit_status(self, episode_id: int, state_id: int) -> dict | None:
        """Get MTurk HIT status for a state.

        Args:
            episode_id: Episode ID
            state_id: State ID

        Returns:
            HIT metadata dict or None

        """
        if not self.mturk_manager:
            return None
        return self.mturk_manager.get_hit_status(episode_id, state_id)

    def get_all_mturk_hits(self) -> dict:
        """Get status of all MTurk HITs.

        Returns:
            Dict of all HIT metadata

        """
        if not self.mturk_manager:
            return {}
        return self.mturk_manager.get_all_hits_status()

    def delete_mturk_hit(self, episode_id: int, state_id: int) -> bool:
        """Delete MTurk HIT.

        Args:
            episode_id: Episode ID
            state_id: State ID

        Returns:
            True if successful, False otherwise

        """
        if not self.mturk_manager:
            return False
        return self.mturk_manager.delete_hit(episode_id, state_id)

    def _on_critical_state_ready(self, episode_id: int, state_id: int, state_info: dict):
        """Callback when a critical state becomes fully ready for labeling.
        
        Auto-creates MTurk HIT if MTurk is enabled.
        
        Args:
            episode_id: Episode ID
            state_id: State ID
            state_info: State information dict
        """
        print(f"🎯 Critical state ready for labeling: episode={episode_id}, state={state_id}", flush=True)
        
        # Auto-create MTurk HIT if enabled
        if self.mturk_manager:
            print(f"🚀 Attempting to create MTurk HIT for episode={episode_id}, state={state_id}", flush=True)
            try:
                # Pass state_info directly to avoid deadlock (callback already holds state_lock)
                hit_id = self.create_mturk_hit(episode_id, state_id, state_info)
                if hit_id:
                    print(f"✅ Auto-created MTurk HIT: {hit_id}", flush=True)
                else:
                    print(f"⚠️  Failed to auto-create MTurk HIT for episode={episode_id}, state={state_id}", flush=True)
            except Exception as e:
                print(f"❌ Exception while creating MTurk HIT: {e}", flush=True)
                import traceback
                traceback.print_exc()
        else:
            print(f"⚠️  MTurk manager not initialized - skipping HIT creation", flush=True)

    # =========================
    # Utility Methods
    # =========================
    
    def _detect_cloudflare_tunnel_url(self) -> str | None:
        """Auto-detect cloudflare tunnel URL from log file.
        
        Returns:
            Tunnel URL if found, None otherwise
        """
        import re
        
        possible_paths = [
            Path("/tmp/cloudflared.log"),
            Path.home() / ".cloudflared" / "cloudflared.log",
            Path("cloudflared.log"),
        ]
        
        for log_path in possible_paths:
            if log_path.exists():
                try:
                    with open(log_path, 'r') as f:
                        content = f.read()
                        match = re.search(r'https://[a-z0-9-]+\.trycloudflare\.com', content)
                        if match:
                            return match.group(0)
                except Exception:
                    continue
        
        return None
    
    def _repo_root(self) -> Path:
        """Root of the repo (backend assumes this file lives under <repo>/scripts or similar)."""
        return (Path(__file__).resolve().parent / "..").resolve()

    def rel_path_from_repo(self, p: str | Path | None) -> str | None:
        if not p:
            return None
        try:
            rp = Path(p).resolve()
            return str(rp.relative_to(self._repo_root()))
        except Exception:
            # If not inside the repo root, return the basename as a safe hint.
            return os.path.basename(str(p))

    # =========================
    # Cleanup and Shutdown
    # =========================

    def _find_orphaned_workers(self) -> list[tuple[int, str]]:
        """Find orphaned worker processes (Isaac Sim, pose workers) that may have been left running.
        
        Returns:
            List of (pid, cmdline) tuples for orphaned workers
        """
        if not HAS_PSUTIL:
            print("⚠️  psutil not available - cannot scan for orphaned workers")
            return []
        
        orphaned = []
        current_pid = os.getpid()
        
        try:
            for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'ppid']):
                try:
                    cmdline = proc.info.get('cmdline', [])
                    if not cmdline:
                        continue
                    
                    cmdline_str = ' '.join(cmdline)
                    
                    # Look for Isaac Sim workers and related processes
                    is_isaac_worker = (
                        'isaac_sim_worker.py' in cmdline_str or 
                        'persistent_isaac_sim_worker.py' in cmdline_str or
                        ('python' in cmdline_str and 'isaac' in cmdline_str.lower())
                    )
                    
                    if is_isaac_worker:
                        # Add all Isaac Sim workers - we'll clean them up
                        orphaned.append((proc.info['pid'], cmdline_str))
                    
                    # Look for pose workers (any6d environment)
                    if 'pose_worker.py' in cmdline_str and 'any6d' in cmdline_str:
                        ppid = proc.info.get('ppid', 0)
                        if ppid == 1 or ppid == current_pid:
                            orphaned.append((proc.info['pid'], cmdline_str))
                            continue
                        
                        try:
                            psutil.Process(ppid)
                        except psutil.NoSuchProcess:
                            orphaned.append((proc.info['pid'], cmdline_str))
                
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    continue
        
        except Exception as e:
            print(f"⚠️  Error scanning for orphaned workers: {e}")
        
        return orphaned

    def _kill_orphaned_workers(self):
        """Find and kill any orphaned worker processes."""
        orphaned = self._find_orphaned_workers()
        
        if not orphaned:
            print("✓ No orphaned worker processes found")
            return
        
        print(f"🧹 Found {len(orphaned)} orphaned worker process(es)")
        for pid, cmdline in orphaned:
            # Truncate long command lines for display
            short_cmd = cmdline[:100] + '...' if len(cmdline) > 100 else cmdline
            print(f"   PID {pid}: {short_cmd}")
        
        # Kill them
        for pid, cmdline in orphaned:
            try:
                if HAS_PSUTIL:
                    proc = psutil.Process(pid)
                    proc.terminate()
                    try:
                        proc.wait(timeout=2.0)
                        print(f"✓ Terminated orphaned worker PID {pid}")
                    except psutil.TimeoutExpired:
                        proc.kill()
                        print(f"✓ Force killed orphaned worker PID {pid}")
                else:
                    os.kill(pid, signal.SIGTERM)
                    time.sleep(0.5)
                    try:
                        os.kill(pid, 0)  # Check if still alive
                        os.kill(pid, signal.SIGKILL)
                        print(f"✓ Force killed orphaned worker PID {pid}")
                    except ProcessLookupError:
                        print(f"✓ Terminated orphaned worker PID {pid}")
            except (ProcessLookupError, psutil.NoSuchProcess):
                print(f"✓ Worker PID {pid} already gone")
            except Exception as e:
                print(f"⚠️  Failed to kill worker PID {pid}: {e}")

    def shutdown(self):
        """Gracefully shutdown all managers and worker processes.
        
        This method:
        1. Stops all managed workers (pose estimation, webcam, observation stream)
        2. Shuts down Isaac Sim simulation
        3. Closes MTurk connection
        4. Scans for and kills any orphaned worker processes
        """
        if self._shutdown_complete:
            return
        
        print("\\n🛑 Starting CrowdInterface shutdown...")
        
        # Stop pose estimation workers
        if hasattr(self, 'pose_estimation_manager') and self.pose_estimation_manager:
            try:
                print("Stopping pose estimation workers...")
                self.pose_estimation_manager.stop()
            except Exception as e:
                print(f"⚠️  Error stopping pose estimation manager: {e}")
        
        # Stop webcam manager
        if hasattr(self, 'webcam_manager') and self.webcam_manager:
            try:
                print("Stopping webcam manager...")
                self.webcam_manager.stop()
            except Exception as e:
                print(f"⚠️  Error stopping webcam manager: {e}")
        
        # Stop observation stream manager
        if hasattr(self, 'obs_stream') and self.obs_stream:
            try:
                print("Stopping observation stream manager...")
                self.obs_stream.stop()
            except Exception as e:
                print(f"⚠️  Error stopping observation stream manager: {e}")
        
        # Shutdown simulation
        if hasattr(self, 'sim_manager') and self.sim_manager:
            try:
                print("Shutting down simulation...")
                self.sim_manager.shutdown()
            except Exception as e:
                print(f"⚠️  Error shutting down sim manager: {e}")
        
        # Shutdown MTurk connection
        if hasattr(self, 'mturk_manager') and self.mturk_manager:
            try:
                print("Shutting down MTurk connection...")
                self.mturk_manager.shutdown()
            except Exception as e:
                print(f"⚠️  Error shutting down MTurk manager: {e}")
        
        # Kill any orphaned workers that might have been missed
        print("Scanning for orphaned worker processes...")
        self._kill_orphaned_workers()
        
        self._shutdown_complete = True
        print("✅ CrowdInterface shutdown complete\\n")

    def register_cleanup_handlers(self):
        """Register cleanup handlers for graceful shutdown on exit/interrupt.
        
        This should be called once after CrowdInterface initialization to ensure
        workers are cleaned up even on unexpected exits.
        """
        if self._cleanup_registered:
            return
        
        def cleanup_handler():
            """Cleanup handler for atexit."""
            if not self._shutdown_complete:
                self.shutdown()
        
        def signal_handler(signum, frame):
            """Signal handler for SIGINT/SIGTERM."""
            print(f"\\n⚠️  Received signal {signum}, shutting down...")
            self.shutdown()
            sys.exit(0)
        
        # Register atexit handler
        atexit.register(cleanup_handler)
        
        # Register signal handlers
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        self._cleanup_registered = True
        print("✓ Cleanup handlers registered (atexit, SIGINT, SIGTERM)")

    def set_events(self, events):
        """Set the events object for keyboard-like control functionality."""
        self.events = events

    def continue_from_last_critical(self) -> dict:
        """Continue data collection from last critical state in saved dataset.
        
        This is called via the Continue button after dataset has been loaded.
        The old frames have already been copied during init_dataset.
        This just drives the robot to the last critical state position.
        
        Returns dict with status
        """
        try:
            from crowd_interface_config import CrowdInterfaceConfig
            cfg = CrowdInterfaceConfig()
            
            if not cfg.continue_from_dataset:
                return {"status": "error", "message": "No continue_from_dataset configured"}
            
            # Determine if continue_from is an absolute path or repo ID
            from pathlib import Path
            continue_path = Path(cfg.continue_from_dataset)
            root_for_load = None if continue_path.is_absolute() else (
                self.dataset_manager.dataset.root if self.dataset_manager.dataset else Path("data")
            )
            
            # Get last critical state from old dataset
            last_state = self.dataset_manager.get_last_critical_state_from_dataset(
                cfg.continue_from_dataset,
                root_for_load
            )
            
            if not last_state:
                return {"status": "error", "message": "No critical states found in dataset"}
            
            print(f"📍 Continue from: episode {last_state['episode_index']}, frame {last_state['frame_index']}")
            print(f"📍 Joint positions: {last_state['joint_positions']}")
            
            # Don't set latest_goal - we'll manually drive the robot in collect_data.py
            # and then just start recording from that position
            
            print(f"🤖 Will drive robot to last critical state position")
            print(f"▶️ Ready to continue data collection")
            
            return {
                "status": "success",
                "message": "Joint positions for last critical state",
                "joint_positions": last_state['joint_positions'],
            }
            
        except Exception as e:
            print(f"❌ Error in continue_from_last_critical: {e}")
            import traceback
            traceback.print_exc()
            return {"status": "error", "message": str(e)}

    def save_all_finalized_episodes(self):
        """Save all finalized episodes in order with consistent schema.
        
        This is called at the very end after all episodes are collected.
        It determines the global maximum action count, updates the schema once,
        then saves all episodes in order.
        """
        # Force all pending finalization timers to fire immediately
        # This prevents race conditions where timers fire during the batch save
        with self.state_manager.state_lock:
            pending_timers = list(self.state_manager._episode_finalize_timers.items())
            for episode_id, timer in pending_timers:
                timer.cancel()
                # Manually trigger finalization now (caller already holds state_lock)
                self.state_manager._finalize_episode_logic(episode_id)
        
            # In async Phase 2 mode, states stay in pending_states_by_episode and no
            # finalization timers are set. Directly finalize all episodes from buffer.
            if self.asynchronous_mode and self.state_manager.async_pool_finalized:
                if not hasattr(self.state_manager, '_finalized_episodes'):
                    self.state_manager._finalized_episodes = {}
                
                for episode_id, buffer in self.state_manager.completed_states_buffer_by_episode.items():
                    if episode_id not in self.state_manager._finalized_episodes:
                        episode_timing = self.state_manager._calculate_episode_timing(episode_id, buffer)
                        self.state_manager._finalized_episodes[episode_id] = {
                            'buffer': buffer,
                            'timing': episode_timing,
                        }
                        print(f"💾 Episode {episode_id} finalized for batch save ({len(buffer)} states)")
        
        if not hasattr(self.state_manager, '_finalized_episodes'):
            print(f"⚠️  No finalized episodes to save")
            return
        
        finalized = self.state_manager._finalized_episodes
        if not finalized:
            print(f"⚠️  No finalized episodes to save")
            return
        
        print(f"\n{'='*80}")
        print(f"💾 BATCH SAVE: Saving {len(finalized)} episodes with consistent schema")
        print(f"{'='*80}")
        
        # Find global maximum action count across ALL episodes
        global_max_actions = 0
        for episode_id in sorted(finalized.keys()):
            buffer = finalized[episode_id]['buffer']
            
            # Count max actions in this episode
            episode_max = 0
            for state in buffer.values():
                if state.get('critical') and state.get('execution_history'):
                    # Count ALL actions in execution_history (both approved and rejected)
                    total_count = len(state['execution_history'])
                    episode_max = max(episode_max, total_count)
            
            global_max_actions = max(global_max_actions, episode_max)
            print(f"   Episode {episode_id}: max {episode_max} actions")
        
        print(f"   📊 Global maximum: {global_max_actions} actions")
        
        # Update schema once with global max
        if global_max_actions > 0:
            self.dataset_manager._update_dataset_action_shape_dynamic(global_max_actions)
        
        # Set flag to skip dynamic shape updates during individual episode saves
        self.dataset_manager._batch_save_in_progress = True
        
        # Save all episodes in order
        for episode_id in sorted(finalized.keys()):
            buffer = finalized[episode_id]['buffer']
            timing = finalized[episode_id]['timing']
            
            print(f"\n💾 Saving episode {episode_id} with {len(buffer)} states...")
            self.state_manager._save_episode_callback(buffer, timing)
            
            # Clean up
            del self.state_manager.completed_states_buffer_by_episode[episode_id]
            print(f"✅ Episode {episode_id} saved successfully")
        
        # Clear batch save flag
        self.dataset_manager._batch_save_in_progress = False
        
        # Clean up finalized episodes buffer
        self.state_manager._finalized_episodes = {}
        
        print(f"\n{'='*80}")
        print(f"✅ BATCH SAVE COMPLETE: All episodes saved with consistent schema")
        print(f"{'='*80}\n")

    # =========================
    # IP Address Management
    # =========================

    def ban_ip(self, ip_address: str) -> dict:
        """Ban an IP address from submitting.
        
        Args:
            ip_address: IP address to ban
            
        Returns:
            Status dict
        """
        with self.banned_ips_lock:
            self.banned_ips.add(ip_address)
        print(f"🚫 Banned IP: {ip_address}")
        return {"status": "success", "message": f"Banned IP: {ip_address}"}

    def unban_ip(self, ip_address: str) -> dict:
        """Unban an IP address.
        
        Args:
            ip_address: IP address to unban
            
        Returns:
            Status dict
        """
        with self.banned_ips_lock:
            if ip_address in self.banned_ips:
                self.banned_ips.remove(ip_address)
                print(f"✅ Unbanned IP: {ip_address}")
                return {"status": "success", "message": f"Unbanned IP: {ip_address}"}
            else:
                return {"status": "error", "message": f"IP not banned: {ip_address}"}

    def is_ip_banned(self, ip_address: str) -> bool:
        """Check if an IP address is banned.
        
        Args:
            ip_address: IP address to check
            
        Returns:
            True if banned, False otherwise
        """
        with self.banned_ips_lock:
            return ip_address in self.banned_ips

    def get_banned_ips(self) -> list[str]:
        """Get list of all banned IP addresses.
        
        Returns:
            List of banned IPs
        """
        with self.banned_ips_lock:
            return sorted(list(self.banned_ips))

    def get_ip_submission_stats(self) -> dict:
        """Get submission statistics grouped by IP address.
        
        Returns:
            Dict mapping IP addresses to submission counts and user info
        """
        ip_stats = {}
        
        # Aggregate from async_user_logger if available
        if self.state_manager.async_user_logger:
            for email, stats in self.state_manager.async_user_logger.user_stats.items():
                # ip_addresses is stored as a set, so convert to list if it's a set
                ip_addresses = stats.get("ip_addresses", set())
                if isinstance(ip_addresses, set):
                    ip_addresses = list(ip_addresses)
                
                for ip in ip_addresses:
                    if ip not in ip_stats:
                        ip_stats[ip] = {
                            "total_submissions": 0,
                            "users": [],
                            "is_banned": self.is_ip_banned(ip)
                        }
                    ip_stats[ip]["total_submissions"] += stats["total_submissions"]
                    ip_stats[ip]["users"].append({
                        "email": email,
                        "name": stats["name"],
                        "submissions": stats["total_submissions"],
                        "approved": stats["approved_count"],
                        "rejected": stats["rejected_count"]
                    })
        
        return ip_stats

    def load_main_cam_from_obs(self, obs: dict) -> np.ndarray | None:
        """Extract 'observation.images.cam_main' as RGB uint8 HxWx3; returns None if missing."""
        if not isinstance(obs, dict):
            return None
        for k in ("observation.images.cam_main", "observation.images.main", "observation.cam_main"):
            if k in obs:
                return self.obs_stream._to_uint8_rgb(obs[k])
        return None

    # =========================
    # Phase 1/2 Checkpoint
    # =========================

    def save_phase1_checkpoint(self) -> Path:
        """Save Phase 1 checkpoint to dataset root directory.
        
        Returns:
            Path to saved checkpoint file
        """
        if self.dataset_manager.dataset is None:
            raise RuntimeError("Dataset not initialized - cannot save checkpoint")
        
        dataset = self.dataset_manager.dataset
        checkpoint_path = Path(dataset.root) / "phase1_checkpoint.json"
        
        # Gather dataset config needed to recreate in Phase 2
        dataset_config = {
            "repo_id": dataset.repo_id,
            "root": str(dataset.root),
            "fps": dataset.fps,
            "features": dataset.features,
            "robot_type": dataset.meta.robot_type,
        }
        
        result = self.state_manager.save_phase1_checkpoint(checkpoint_path, dataset_config=dataset_config)
        
        if result["status"] != "success":
            raise RuntimeError(f"Failed to save checkpoint: {result.get('message')}")
        
        return Path(result["path"])

    def load_phase1_checkpoint(self, checkpoint_path: Path) -> dict:
        """Load Phase 1 checkpoint and prepare for Phase 2.
        
        Args:
            checkpoint_path: Path to checkpoint JSON file
            
        Returns:
            dict with loaded config and dataset_config
        """
        return self.state_manager.load_phase1_checkpoint(checkpoint_path)

    def save_phase2_checkpoint(self, checkpoint_path: Path = None) -> dict:
        """Save Phase 2 async labeling progress checkpoint.
        
        Args:
            checkpoint_path: Path to save checkpoint. If None, saves to dataset root.
            
        Returns:
            dict with status and checkpoint path
        """
        if checkpoint_path is None:
            if self.dataset_manager.dataset is None:
                raise RuntimeError("Dataset not initialized - cannot save checkpoint")
            checkpoint_path = Path(self.dataset_manager.dataset.root) / "phase2_checkpoint.json"
        
        return self.state_manager.save_phase2_checkpoint(checkpoint_path)

    def load_phase2_checkpoint(self, checkpoint_path: Path) -> dict:
        """Load Phase 2 checkpoint and restore async labeling progress.
        
        Must be called after finalize_admin_phase().
        
        Args:
            checkpoint_path: Path to checkpoint JSON file
            
        Returns:
            dict with status, restored_states, restored_approved
        """
        return self.state_manager.load_phase2_checkpoint(checkpoint_path)
