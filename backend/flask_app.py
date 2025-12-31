"""Flask app to expose CrowdInterface to frontend."""

import json
import mimetypes
import os
import re
import traceback
from pathlib import Path

from crowd_interface import CrowdInterface
from flask import Flask, Response, jsonify, make_response, request, send_from_directory
from flask_cors import CORS


def create_flask_app(crowd_interface: CrowdInterface) -> Flask:
    """Create and configure Flask app with the crowd interface."""
    app = Flask(__name__)
    CORS(
        app,
        resources={r"/api/*": {"origins": "*"}},
        allow_headers=["Content-Type", "ngrok-skip-browser-warning", "X-Session-ID"],
        methods=["GET", "POST", "OPTIONS"],
        supports_credentials=False,
        expose_headers=["Content-Type"],
    )

    @app.after_request
    def add_cors_headers(response):
        """Ensure CORS headers are always present for Cloudflare Tunnel compatibility."""
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, ngrok-skip-browser-warning, X-Session-ID"
        return response

    @app.route("/api/get-state")
    def get_state():

        state = crowd_interface.get_latest_state()

        # Check if this is a status response (no real state)
        if isinstance(state, dict) and state.get("status"):
            # Return status response directly without processing through _state_to_json
            response = jsonify(state)
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
            return response

        # Process as a real state
        payload = crowd_interface.state_to_json(state)

        # Prefer text_prompt (manual or VLM), otherwise simple fallback
        text = payload.get("text_prompt")
        if isinstance(text, str) and text.strip():
            payload["prompt"] = text.strip()
        else:
            payload["prompt"] = f"{crowd_interface.task_text or 'crowdsourced_task'}"

        # Tell the frontend what to do with demo videos
        payload["demo_video"] = crowd_interface.video_manager.get_demo_video_config()

        response = jsonify(payload)
        # Prevent caching
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response

    @app.route("/api/test")
    def test():
        # Count total states across all episodes
        total_states = sum(len(states) for states in crowd_interface.pending_states_by_episode.values())
        return jsonify({"message": "Flask server is working", "states_count": total_states})

    @app.route("/api/cloudflared-url")
    def get_cloudflared_url():
        """Return the current cloudflared tunnel URL by reading from the log file.
        
        This endpoint allows the frontend to automatically discover the tunnel URL
        without manual configuration.
        """
        try:
            # Common locations for cloudflared log/config files
            home = Path.home()
            possible_paths = [
                Path("/tmp/cloudflared.log"),
                home / ".cloudflared" / "cloudflared.log",
                Path("cloudflared.log"),
            ]
            
            # Try to find and read the cloudflared URL from logs
            for log_path in possible_paths:
                if log_path.exists():
                    try:
                        with open(log_path, 'r') as f:
                            content = f.read()
                            # Look for trycloudflare.com URL in the logs
                            match = re.search(r'https://[a-z0-9-]+\.trycloudflare\.com', content)
                            if match:
                                url = match.group(0)
                                return jsonify({"url": url, "source": str(log_path)})
                    except Exception as e:
                        print(f"Error reading {log_path}: {e}")
                        continue
            
            # If no URL found in logs, return localhost as fallback
            return jsonify({
                "url": "http://127.0.0.1:9000",
                "source": "fallback",
                "message": "No cloudflared tunnel detected, using localhost"
            })
            
        except Exception as e:
            return jsonify({
                "url": "http://127.0.0.1:9000",
                "source": "error",
                "error": str(e)
            }), 500

    @app.route("/api/demo-video-config", methods=["GET"])
    def demo_video_config():
        """Lightweight config endpoint so the new frontend can fetch once on load.

        Mirrors the 'demo_video' object we also embed in /api/get-state.

        """
        try:
            return jsonify(crowd_interface.video_manager.get_demo_video_config())
        except Exception as e:
            return jsonify({"enabled": False, "error": str(e)}), 500

    @app.route("/api/submit-goal", methods=["POST"])
    def submit_goal():
        try:

            # Validate request data
            data = request.get_json(force=True, silent=True)
            if data is None:
                return jsonify({"status": "error", "message": "Invalid JSON data"}), 400

            # Check for required fields
            required_fields = ["state_id", "episode_id", "joint_positions", "gripper"]
            missing_fields = [field for field in required_fields if field not in data]
            if missing_fields:
                return jsonify({"status": "error", "message": f"Missing required fields: {missing_fields}"}), 400

            # Generate or retrieve session ID from request headers or IP
            session_id = request.headers.get("X-Session-ID", request.remote_addr or "unknown")

            # Record this as a response to the correct state for this session
            # The frontend now includes state_id in the request data
            crowd_interface.record_response(data)
            
            # Notify MTurk manager of assignment submission (if MTurk enabled)
            # Distinguish expert (localhost direct) vs MTurk (via tunnel) workers
            episode_id = data["episode_id"]
            state_id = data["state_id"]
            
            # Check for X-Forwarded-For header (set by cloudflared tunnel)
            # If present, request came through tunnel (MTurk worker)
            # If absent and localhost, it's direct localhost access (expert worker)
            forwarded_for = request.headers.get("X-Forwarded-For")
            cf_connecting_ip = request.headers.get("CF-Connecting-IP")
            remote_addr = request.remote_addr or ""
            
            # MTurk worker: has forwarded header OR non-localhost direct access
            is_mturk_worker = forwarded_for or cf_connecting_ip or remote_addr not in ["127.0.0.1", "::1", "localhost"]
            
            if is_mturk_worker:
                origin = forwarded_for or cf_connecting_ip or remote_addr
                print(f"🌐 MTurk worker submission: episode={episode_id}, state={state_id} (from {origin})")
                crowd_interface.update_mturk_assignment_count(episode_id, state_id)
            else:
                print(f"👤 Expert worker submission: episode={episode_id}, state={state_id}")
            
            return jsonify({"status": "ok"})

        except KeyError as e:
            print(f"❌ KeyError in submit_goal (missing data field): {e}")
            return jsonify({"status": "error", "message": f"Missing required field: {e}"}), 400
        except Exception as e:
            print(f"❌ Error in submit_goal endpoint: {e}")
            traceback.print_exc()
            return jsonify({"status": "error", "message": str(e)}), 500

    @app.route("/api/undo", methods=["POST"])
    def undo():
        """Undo to the previous critical state.

        Returns the robot position to revert to, or an error if undo is not possible. The robot control loop should
        consume this response and execute the movement.

        """
        try:
            result = crowd_interface.undo_to_previous_critical_state()

            if result is None:
                return jsonify({"status": "error", "message": "Cannot undo: need at least 2 critical states"}), 400

            return jsonify(
                {
                    "status": "ok",
                    "joint_positions": result["joint_positions"],
                    "gripper": result["gripper"],
                    "episode_id": result["episode_id"],
                    "reverted_to_state_id": result["reverted_to_state_id"],
                    "message": f"Reverted to state {result['reverted_to_state_id']}",
                }
            )

        except Exception as e:
            print(f"❌ Error in undo endpoint: {e}")
            traceback.print_exc()
            return jsonify({"status": "error", "message": str(e)}), 500

    @app.route("/api/pending-states-info")
    def pending_states_info():
        """Debug endpoint to see pending states information."""
        info = crowd_interface.get_pending_states_info()
        return jsonify(info)

    # --- PATCH: Endpoints for monitor modal -------------------------------------

    @app.route("/api/description-bank", methods=["GET"])
    def api_description_bank():
        """
        Return the description bank for the current task as both:
          - 'entries': [{id, text, full}]
          - 'raw_text': the unparsed text (for debugging or custom parsing)
        """
        try:
            bank = crowd_interface.get_description_bank()
            return jsonify(
                {
                    "ok": True,
                    "task_name": (crowd_interface.task_name or "default"),
                    "entries": bank["entries"],
                    "raw_text": bank["raw_text"],
                }
            )
        except Exception as e:
            print(f"❌ /api/description-bank error: {e}")
            return jsonify({"ok": False, "error": str(e)}), 500

    @app.route("/api/state-details", methods=["GET"])
    def api_state_details():
        """
        Query params: episode_id=<str|int>, state_id=<int>
        Returns:
          - maincam_data_url
          - is_critical
          - flex_text_prompt
          - flex_video_id
          - description_bank / description_bank_text
        """
        try:

            ep = request.args.get("episode_id", type=int)
            sid = request.args.get("state_id", type=int)
            if ep is None or sid is None:
                return jsonify({"ok": False, "error": "episode_id and state_id are required"}), 400

            # Defaults
            flex_text = ""
            flex_video_id = None
            is_imp = False
            obs_path = None

            with crowd_interface.state_lock:
                # Prefer pending
                p_ep = crowd_interface.pending_states_by_episode.get(ep, {})
                p_info = p_ep.get(sid)
                if p_info is not None:
                    is_imp = bool(p_info.get("critical", False))
                    obs_path = p_info.get("obs_path")
                    # Text: use new field name
                    flex_text = p_info.get("text_prompt") or ""
                    # Video id: use new field name
                    raw_vid = p_info.get("video_prompt")
                    try:
                        flex_video_id = int(raw_vid) if raw_vid is not None else None
                    except Exception:
                        flex_video_id = None
                else:
                    # Completed metadata
                    c_ep = crowd_interface.completed_states_by_episode.get(ep, {})
                    c_meta = c_ep.get(sid)
                    if c_meta is None:
                        return jsonify({"ok": False, "error": f"state {sid} not found in episode {ep}"}), 404
                    is_imp = bool(c_meta.get("critical", False))  # Use consistent field name
                    flex_text = c_meta.get("text_prompt") or ""
                    raw_vid = c_meta.get("video_prompt")
                    try:
                        flex_video_id = int(raw_vid) if raw_vid is not None else None
                    except Exception:
                        flex_video_id = None
                    man = crowd_interface.completed_states_buffer_by_episode.get(ep, {}).get(sid)
                    if isinstance(man, dict):
                        obs_path = man.get("obs_path")

            # Load maincam image (if possible)
            maincam_url = None
            if obs_path:
                obs = crowd_interface.dataset_manager.load_obs_from_disk(obs_path)
                img = crowd_interface.load_main_cam_from_obs(obs)
                if img is not None:
                    maincam_url = crowd_interface.encode_jpeg_base64(img)

            # Description bank
            bank = crowd_interface.get_description_bank()

            return jsonify(
                {
                    "ok": True,
                    "episode_id": ep,
                    "state_id": sid,
                    "critical": is_imp,
                    "text_prompt": flex_text,
                    "video_prompt": flex_video_id,
                    "maincam_data_url": maincam_url,
                    "description_bank": bank["entries"],
                    "description_bank_text": bank["raw_text"],
                }
            )
        except Exception as e:
            print(f"❌ /api/state-details error: {e}")
            return jsonify({"ok": False, "error": str(e)}), 500

    @app.route("/api/update-flex-selection", methods=["POST"])
    def api_update_flex_selection():
        """
        Body JSON:
        {
          "episode_id": <str or int>,
          "state_id": <int>,
          "video_prompt": <int>
          "text_prompt": <str>
        }
        """
        try:
            data = request.get_json(force=True, silent=True) or {}
            ep_raw = data.get("episode_id")
            if ep_raw is None:
                return jsonify({"ok": False, "error": "episode_id is required"}), 400
            ep = int(ep_raw)

            sid = data.get("state_id")
            if sid is None:
                return jsonify({"ok": False, "error": "state_id is required"}), 400
            sid = int(sid)

            vid = data.get("video_prompt")
            if vid is None:
                return jsonify({"ok": False, "error": "video_prompt is required"}), 400
            vid = int(vid)

            txt = (data.get("text_prompt") or "").strip()

            updated = False
            with crowd_interface.state_lock:
                # pending?
                p_ep = crowd_interface.pending_states_by_episode.get(ep, {})
                p_info = p_ep.get(sid)
                if p_info is not None:
                    crowd_interface.set_prompt_ready(p_info, ep, sid, txt if txt else None, vid)
                    updated = True
                else:
                    # completed metadata path
                    c_ep = crowd_interface.completed_states_by_episode.get(ep, {})
                    c_info = c_ep.get(sid)
                    if c_info is not None:
                        # metadata mirrors - use new field names
                        if txt:
                            c_info["text_prompt"] = txt
                        c_info["video_prompt"] = vid
                        c_info["prompt_ready"] = True
                        updated = True

            if not updated:
                return jsonify({"ok": False, "error": f"state {sid} not found in episode {ep}"}), 404

            return jsonify(
                {"ok": True, "episode_id": ep, "state_id": sid, "video_prompt": vid, "text_prompt": txt or None}
            )
        except Exception as e:
            print(f"❌ /api/update-flex-selection error: {e}")
            return jsonify({"ok": False, "error": str(e)}), 500

    @app.route("/api/monitor/latest-state", methods=["GET"])
    def monitor_latest_state():
        """Read-only monitoring endpoint for episode-based state monitoring.

        Avoid building a combined dict of all pending states on every call.

        """
        try:
            with crowd_interface.state_lock:
                current_episode = crowd_interface.state_manager.current_serving_episode

                total_pending = 0
                newest_state_id = None
                newest_state_data = None
                newest_episode_id = None

                for ep_id, ep_states in crowd_interface.pending_states_by_episode.items():
                    n = len(ep_states)
                    total_pending += n
                    if n == 0:
                        continue
                    # Max by key without materializing a merged dict
                    ep_max_id = max(ep_states.keys())
                    if newest_state_id is None or ep_max_id > newest_state_id:
                        newest_state_id = ep_max_id
                        newest_state_data = ep_states[ep_max_id]
                        newest_episode_id = ep_id

                if total_pending == 0 or newest_state_data is None:
                    return jsonify(
                        {
                            "status": "no_pending_states",
                            "message": "No pending states.",
                            "views": crowd_interface.snapshot_latest_views(),  # still show previews
                            "total_pending_states": 0,
                            "current_serving_episode": current_episode,
                            "is_resetting": crowd_interface.is_in_reset(),
                            "reset_countdown": crowd_interface.get_reset_countdown(),
                        }
                    )

            # Build response outside the lock
            # newest_state_data IS the state info directly (flattened structure)
            monitoring_data = {
                "status": "success",
                "state_id": newest_state_id,
                "episode_id": newest_episode_id,
                "current_serving_episode": current_episode,
                "responses_received": newest_state_data["responses_received"],
                "responses_required": (
                    crowd_interface.required_responses_per_critical_state
                    if newest_state_data.get("critical", False)
                    else crowd_interface.required_responses_per_state
                ),
                "critical": newest_state_data.get("critical", False),
                "views": crowd_interface.snapshot_latest_views(),  # lightweight snapshot (pre-encoded)
                "joint_positions": newest_state_data.get("joint_positions", {}),
                "gripper": newest_state_data.get("gripper", 0),
                "is_resetting": crowd_interface.is_in_reset(),
                "reset_countdown": crowd_interface.get_reset_countdown(),
                "total_pending_states": total_pending,
                "tutorial_state_capture_enabled": crowd_interface.enable_tutorial_state_capture,
            }

            response = jsonify(monitoring_data)
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
            return response

        except Exception as e:
            return jsonify({"status": "error", "message": f"Monitoring error: {str(e)}"}), 500

    @app.route("/api/control/next-episode", methods=["POST"])
    def next_episode():
        """Trigger next episode (equivalent to 'q' keyboard input)"""
        try:
            if crowd_interface.events is not None:
                print("API trigger: Exiting current loop...")
                crowd_interface.events["exit_early"] = True
                return jsonify({"status": "success", "message": "Next episode triggered"})
            else:
                return jsonify({"status": "error", "message": "Events not initialized"}), 400
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500

    @app.route("/api/control/rerecord", methods=["POST"])
    def rerecord_episode():
        """Trigger re-record episode (equivalent to 'r' keyboard input)"""
        try:
            if crowd_interface.events is not None:
                print("API trigger: Exiting loop and re-record the last episode...")
                crowd_interface.events["rerecord_episode"] = True
                crowd_interface.events["exit_early"] = True
                return jsonify({"status": "success", "message": "Re-record episode triggered"})
            else:
                return jsonify({"status": "error", "message": "Events not initialized"}), 400
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500

    @app.route("/api/control/pending-approval", methods=["GET"])
    def get_pending_approval():
        """Get the critical state awaiting approval."""
        try:
            pending = crowd_interface.state_manager.get_pending_approval_state()
            if pending is None:
                return jsonify({"status": "none"})

            # Load current state image
            current_image_url = None
            if pending["obs_path"]:
                obs = crowd_interface.dataset_manager.load_obs_from_disk(pending["obs_path"])
                img = crowd_interface.load_main_cam_from_obs(obs)
                if img is not None:
                    current_image_url = crowd_interface.encode_jpeg_base64(img)

            # Load previous critical state image
            previous_image_url = None
            if pending.get("previous_critical_obs_path"):
                obs = crowd_interface.dataset_manager.load_obs_from_disk(pending["previous_critical_obs_path"])
                img = crowd_interface.load_main_cam_from_obs(obs)
                if img is not None:
                    previous_image_url = crowd_interface.encode_jpeg_base64(img)

            return jsonify(
                {
                    "status": "pending",
                    "episode_id": pending["episode_id"],
                    "state_id": pending["state_id"],
                    "current_image_url": current_image_url,
                    "previous_image_url": previous_image_url,
                    "tutorial_state_capture_enabled": crowd_interface.enable_tutorial_state_capture,
                }
            )
        except Exception as e:
            print(f"❌ Error in pending-approval endpoint: {e}")
            import traceback

            traceback.print_exc()
            return jsonify({"status": "error", "message": str(e)}), 500

    @app.route("/api/control/approve-critical", methods=["POST"])
    def approve_critical():
        """Approve a pending critical state."""
        try:
            data = request.json
            episode_id = data.get("episode_id")
            state_id = data.get("state_id")

            if episode_id is None or state_id is None:
                return jsonify({"status": "error", "message": "Missing episode_id or state_id"}), 400

            success = crowd_interface.state_manager.approve_critical_state(episode_id, state_id)

            if not success:
                return jsonify({"status": "error", "message": "No matching pending approval"}), 400

            return jsonify({"status": "success"})
        except Exception as e:
            print(f"❌ Error in approve-critical endpoint: {e}")
            import traceback

            traceback.print_exc()
            return jsonify({"status": "error", "message": str(e)}), 500

    @app.route("/api/control/pending-pre-execution-approval", methods=["GET"])
    def get_pending_pre_execution_approval():
        """Get the action awaiting pre-execution approval"""
        try:
            pending = crowd_interface.state_manager.get_pending_pre_execution_approval()
            if pending is None:
                return jsonify({"status": "none"})

            # Load current state observation image - use main cam observation or real webcam_front
            current_image_url = None
            view_paths = pending.get("view_paths", {})
            
            # First priority: main cam observation from obs_path
            if pending.get("obs_path"):
                try:
                    obs = crowd_interface.dataset_manager.load_obs_from_disk(pending["obs_path"])
                    img = crowd_interface.load_main_cam_from_obs(obs)
                    if img is not None:
                        current_image_url = crowd_interface.encode_jpeg_base64(img)
                except Exception as e:
                    print(f"⚠️  failed to load obs from {pending.get('obs_path')}: {e}")
            
            # Second priority: real-life webcam front view (not simulated front)
            if not current_image_url and "webcam_front" in view_paths:
                try:
                    import base64
                    from pathlib import Path
                    front_path = view_paths["webcam_front"]
                    if Path(front_path).exists():
                        with open(front_path, 'rb') as f:
                            img_data = f.read()
                            current_image_url = f"data:image/jpeg;base64,{base64.b64encode(img_data).decode()}"
                except Exception as e:
                    print(f"⚠️  Failed to load webcam_front for current image: {e}")

            # Load view paths (static sim/webcam views)
            view_urls = {}
            for view_name, view_path in view_paths.items():
                try:
                    import base64
                    from pathlib import Path
                    if Path(view_path).exists():
                        with open(view_path, 'rb') as f:
                            img_data = f.read()
                            view_urls[view_name] = f"data:image/jpeg;base64,{base64.b64encode(img_data).decode()}"
                except Exception as e:
                    print(f"⚠️  Failed to load view {view_name}: {e}")

            # The action IS the joint positions (joint_0 through joint_5, left_carriage_joint)
            # Convert action list to joint positions for robot rendering
            action = pending["action"]  # This is already joint positions as a list
            
            return jsonify(
                {
                    "status": "pending",
                    "episode_id": pending["episode_id"],
                    "state_id": pending["state_id"],
                    "action": action,  # Joint positions: [joint_0, joint_1, joint_2, joint_3, joint_4, joint_5, left_carriage_joint]
                    "current_image_url": current_image_url,
                    "view_urls": view_urls,  # Static views for rendering
                    "camera_poses": crowd_interface.calibration.get_camera_poses(),
                    "camera_models": crowd_interface.calibration.get_camera_models(),
                }
            )
        except Exception as e:
            print(f"❌ Error in pending-pre-execution-approval endpoint: {e}")
            import traceback

            traceback.print_exc()
            return jsonify({"status": "error", "message": str(e)}), 500

    @app.route("/api/control/approve-pre-execution", methods=["POST"])
    def approve_pre_execution():
        """Approve a pending pre-execution action"""
        try:
            data = request.json
            episode_id = data.get("episode_id")
            state_id = data.get("state_id")

            if episode_id is None or state_id is None:
                return jsonify({"status": "error", "message": "Missing episode_id or state_id"}), 400

            success = crowd_interface.state_manager.approve_pre_execution(episode_id, state_id)

            if not success:
                return jsonify({"status": "error", "message": "No matching pending pre-execution approval"}), 400

            return jsonify({"status": "success"})
        except Exception as e:
            print(f"❌ Error in approve-pre-execution endpoint: {e}")
            import traceback

            traceback.print_exc()
            return jsonify({"status": "error", "message": str(e)}), 500

    @app.route("/api/control/reject-pre-execution", methods=["POST"])
    def reject_pre_execution():
        """Reject a pending pre-execution action (triggers resampling)"""
        try:
            data = request.json
            episode_id = data.get("episode_id")
            state_id = data.get("state_id")

            if episode_id is None or state_id is None:
                return jsonify({"status": "error", "message": "Missing episode_id or state_id"}), 400

            success = crowd_interface.state_manager.reject_pre_execution(episode_id, state_id)

            if not success:
                return jsonify({"status": "error", "message": "No matching pending pre-execution approval"}), 400

            return jsonify({"status": "success"})
        except Exception as e:
            print(f"❌ Error in reject-pre-execution endpoint: {e}")
            import traceback

            traceback.print_exc()
            return jsonify({"status": "error", "message": str(e)}), 500

    @app.route("/api/control/reject-critical", methods=["POST"])
    def reject_critical():
        """Reject a pending critical state (triggers undo)"""
        try:
            data = request.json
            episode_id = data.get("episode_id")
            state_id = data.get("state_id")

            if episode_id is None or state_id is None:
                return jsonify({"status": "error", "message": "Missing episode_id or state_id"}), 400

            success = crowd_interface.state_manager.reject_critical_state(episode_id, state_id)

            if not success:
                return jsonify({"status": "error", "message": "No matching pending approval"}), 400

            return jsonify({"status": "success"})
        except Exception as e:
            print(f"❌ Error in reject-critical endpoint: {e}")
            import traceback

            traceback.print_exc()
            return jsonify({"status": "error", "message": str(e)}), 500

    @app.route("/api/control/discard-jitter-states", methods=["POST"])
    def discard_jitter_states():
        """Discard jitter states after the last approved critical state"""
        try:
            data = request.json
            episode_id = data.get("episode_id")

            if episode_id is None:
                return jsonify({"status": "error", "message": "Missing episode_id"}), 400

            success = crowd_interface.state_manager.discard_jitter_states(episode_id)

            if not success:
                return jsonify({"status": "error", "message": "No approved state found or no states to discard"}), 400

            return jsonify({"status": "success"})
        except Exception as e:
            print(f"❌ Error in discard-jitter-states endpoint: {e}")
            import traceback

            traceback.print_exc()
            return jsonify({"status": "error", "message": str(e)}), 500

    @app.route("/api/control/pending-undo-classification", methods=["GET"])
    def get_pending_undo_classification():
        """Get the state awaiting undo classification (new state vs old state)"""
        try:
            pending = crowd_interface.state_manager.get_pending_undo_classification()
            if pending is None:
                return jsonify({"status": "none"})

            # Load previous state image (target before undo)
            previous_image_url = None
            if pending.get("previous_obs_path"):
                obs = crowd_interface.dataset_manager.load_obs_from_disk(pending["previous_obs_path"])
                img = crowd_interface.load_main_cam_from_obs(obs)
                if img is not None:
                    previous_image_url = crowd_interface.encode_jpeg_base64(img)

            # Load arrived state image (after undo motion)
            arrived_image_url = None
            if pending.get("arrived_obs_path"):
                obs = crowd_interface.dataset_manager.load_obs_from_disk(pending["arrived_obs_path"])
                img = crowd_interface.load_main_cam_from_obs(obs)
                if img is not None:
                    arrived_image_url = crowd_interface.encode_jpeg_base64(img)

            return jsonify(
                {
                    "status": "pending",
                    "episode_id": pending["episode_id"],
                    "state_id": pending["state_id"],
                    "previous_image_url": previous_image_url,
                    "arrived_image_url": arrived_image_url,
                    "num_remaining_actions": pending.get("num_remaining_actions", 0),
                }
            )
        except Exception as e:
            print(f"❌ Error in pending-undo-classification endpoint: {e}")
            import traceback

            traceback.print_exc()
            return jsonify({"status": "error", "message": str(e)}), 500

    @app.route("/api/control/classify-undo-new-state", methods=["POST"])
    def classify_undo_new_state():
        """Classify post-undo arrival as a new state (requires new action submissions)"""
        try:
            data = request.json
            episode_id = data.get("episode_id")
            state_id = data.get("state_id")

            if episode_id is None or state_id is None:
                return jsonify({"status": "error", "message": "Missing episode_id or state_id"}), 400

            success = crowd_interface.state_manager.classify_undo_as_new_state(episode_id, state_id)

            if not success:
                return jsonify({"status": "error", "message": "No matching pending classification"}), 400

            return jsonify({"status": "success"})
        except Exception as e:
            print(f"❌ Error in classify-undo-new-state endpoint: {e}")
            import traceback

            traceback.print_exc()
            return jsonify({"status": "error", "message": str(e)}), 500

    @app.route("/api/control/classify-undo-old-state", methods=["POST"])
    def classify_undo_old_state():
        """Classify post-undo arrival as old state (resample from existing actions)"""
        try:
            data = request.json
            episode_id = data.get("episode_id")
            state_id = data.get("state_id")

            if episode_id is None or state_id is None:
                return jsonify({"status": "error", "message": "Missing episode_id or state_id"}), 400

            success = crowd_interface.state_manager.classify_undo_as_old_state(episode_id, state_id)

            if not success:
                return jsonify({"status": "error", "message": "No matching pending classification"}), 400

            return jsonify({"status": "success"})
        except Exception as e:
            print(f"❌ Error in classify-undo-old-state endpoint: {e}")
            import traceback

            traceback.print_exc()
            return jsonify({"status": "error", "message": str(e)}), 500

    @app.route("/api/control/stop", methods=["POST"])
    def stop_recording():
        """Trigger stop recording (equivalent to 'x' keyboard input)"""
        try:
            if crowd_interface.events is not None:
                print("API trigger: Stopping data recording...")
                crowd_interface.events["stop_recording"] = True
                crowd_interface.events["exit_early"] = True
                return jsonify({"status": "success", "message": "Stop recording triggered"})
            else:
                return jsonify({"status": "error", "message": "Events not initialized"}), 400
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500

    @app.route("/api/control/start-episode", methods=["POST"])
    def start_episode():
        """Skip remaining reset time and start the next episode immediately."""
        try:
            if crowd_interface.is_in_reset():
                crowd_interface.stop_reset()
                return jsonify({"status": "success", "message": "Reset skipped, starting episode"})
            else:
                return jsonify({"status": "error", "message": "Not currently in reset state"})
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500

    @app.route("/api/control/set-critical", methods=["POST"])
    def set_last_state_critical():
        """Manually mark the last state as critical."""
        try:
            crowd_interface.set_last_state_to_critical()
            return jsonify({"status": "success", "message": "Last state marked as critical"})
        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({"status": "error", "message": str(e)}), 500

    @app.route("/api/control/mark-state-as-end", methods=["POST"])
    def mark_state_as_end():
        """Mark a specific critical state with 'End.' prompt, auto-filling it with current position.

        Expected JSON: {"episode_id": str, "state_id": int}

        """
        try:
            data = request.get_json()
            if not data:
                return jsonify({"status": "error", "message": "No JSON data provided"}), 400

            episode_id = data.get("episode_id")
            state_id = data.get("state_id")

            if episode_id is None or state_id is None:
                return jsonify({"status": "error", "message": "episode_id and state_id are required"}), 400

            # Convert state_id to int
            try:
                state_id = int(state_id)
            except (ValueError, TypeError):
                return jsonify({"status": "error", "message": "state_id must be an integer"}), 400

            # Find the state
            state_info = None
            with crowd_interface.state_manager.state_lock:
                # Check pending states first
                pending_states = crowd_interface.state_manager.pending_states_by_episode.get(episode_id, {})
                if state_id in pending_states:
                    state_info = pending_states[state_id]
                else:
                    # Check completed states
                    completed_states = crowd_interface.state_manager.completed_states_by_episode.get(episode_id, {})
                    if state_id in completed_states:
                        return jsonify({"status": "error", "message": "State is already completed"}), 400
                    else:
                        return jsonify({"status": "error", "message": "State not found"}), 404

                # Verify it's a critical state
                if not state_info.get("critical", False):
                    return jsonify({"status": "error", "message": "Only critical states can be marked as end"}), 400

            # Apply "End." auto-fill using the existing logic
            # This will auto-fill with current position and move state to completed
            crowd_interface.state_manager.set_prompt_ready(
                state_info, episode_id, state_id, "End.", None  # This triggers auto-fill in set_prompt_ready
            )

            return jsonify(
                {
                    "status": "success",
                    "message": f"Marked state {state_id} in episode {episode_id} as end",
                    "episode_id": episode_id,
                    "state_id": state_id,
                }
            )

        except Exception as e:
            print(f"❌ Error in mark-state-as-end: {e}")
            import traceback

            traceback.print_exc()
            return jsonify({"status": "error", "message": str(e)}), 500

    @app.route("/api/control/save-tutorial-state", methods=["POST"])
    def save_tutorial_state():
        """Save a critical state as a tutorial state for worker qualification testing.
        
        Only available when tutorial state capture is enabled in config.

        Expected JSON: {"episode_id": str, "state_id": int, "state_name": str}

        Saves:
        - All camera images (real and simulated)
        - Robot state (joint positions, gripper)
        - Object positions
        - Task information
        - Metadata

        """
        try:
            # Check if tutorial state capture is enabled
            if not crowd_interface.enable_tutorial_state_capture:
                return jsonify({"status": "error", "error": "Tutorial state capture is not enabled"}), 400
            
            data = request.get_json()
            if not data:
                return jsonify({"status": "error", "error": "No JSON data provided"}), 400

            episode_id = data.get("episode_id")
            state_id = data.get("state_id")
            state_name = data.get("state_name")
            
            print(f"[Tutorial Save] Received: episode_id={episode_id} (type={type(episode_id)}), state_id={state_id} (type={type(state_id)}), state_name={state_name}")

            if episode_id is None or state_id is None or not state_name:
                return jsonify({"status": "error", "error": "episode_id, state_id, and state_name are required"}), 400

            # Convert to int (episode_id might be string from JSON)
            try:
                episode_id = int(episode_id)
                state_id = int(state_id)
            except (ValueError, TypeError):
                return jsonify({"status": "error", "error": "episode_id and state_id must be integers"}), 400

            # Validate state name
            import re
            if not re.match(r'^[a-zA-Z0-9_-]+$', state_name):
                return jsonify({"status": "error", "error": "state_name must contain only letters, numbers, underscores, and hyphens"}), 400

            # Get the state from pending_states_by_episode (currently being served)
            state_info = None
            obs_path = None
            
            print(f"[Tutorial Save] Looking for episode_id={episode_id}, state_id={state_id}")
            
            with crowd_interface.state_manager.state_lock:
                print(f"[Tutorial Save] pending_states_by_episode keys: {list(crowd_interface.state_manager.pending_states_by_episode.keys())}")
                if episode_id in crowd_interface.state_manager.pending_states_by_episode:
                    episode_states = crowd_interface.state_manager.pending_states_by_episode[episode_id]
                    print(f"[Tutorial Save] Episode {episode_id} state keys: {list(episode_states.keys())}")
                    state_info = episode_states.get(state_id)
                    if state_info:
                        print(f"[Tutorial Save] Found state: {state_info.keys()}")
                else:
                    print(f"[Tutorial Save] Episode {episode_id} not found in pending_states_by_episode")
            
            if not state_info:
                return jsonify({"status": "error", "error": f"State {state_id} in episode {episode_id} is not currently being served"}), 404
            
            obs_path = state_info.get("obs_path")
            if not obs_path:
                return jsonify({"status": "error", "error": "No observation data available for this state"}), 404
            
            # Get view paths (rendered camera views for serving)
            view_paths = state_info.get("view_paths", {})
            if not view_paths:
                return jsonify({"status": "error", "error": "No view images available for this state"}), 404

            # Create tutorial states directory
            from pathlib import Path
            import json
            import base64
            
            tutorial_dir = Path("data/tutorial_states") / state_name
            tutorial_dir.mkdir(parents=True, exist_ok=True)

            # Load observation from disk
            obs = crowd_interface.dataset_manager.load_obs_from_disk(obs_path)
            if not obs:
                return jsonify({"status": "error", "error": "Failed to load observation data"}), 500

            # Extract robot state from observation
            joint_positions = obs.get("observation.state", [])
            
            # Save state metadata
            state_metadata = {
                "episode_id": str(episode_id),
                "state_id": state_id,
                "state_name": state_name,
                "task": crowd_interface.task_text,
                "robot_state": {
                    "joint_positions": joint_positions.tolist() if hasattr(joint_positions, 'tolist') else list(joint_positions),
                },
                "saved_at": __import__('time').time(),
            }

            with open(tutorial_dir / "state.json", 'w') as f:
                json.dump(state_metadata, f, indent=2)

            # Copy the 6 rendered view images (real-life + simulated cameras)
            import shutil
            for view_name, view_path in view_paths.items():
                src_path = Path(view_path)
                if src_path.exists():
                    # Keep the view name as-is (e.g., webcam_front, sim_front, etc.)
                    dst_path = tutorial_dir / f"{view_name}.jpg"
                    shutil.copy2(src_path, dst_path)
                    print(f"  Copied {view_name}: {src_path} -> {dst_path}")
                else:
                    print(f"  Warning: View {view_name} not found at {view_path}")
            
            print(f"✅ Tutorial state saved: {state_name} (episode {episode_id}, state {state_id})")
            print(f"   Saved {len(view_paths)} view images to {tutorial_dir}")
            return jsonify({"status": "success", "message": f"Tutorial state '{state_name}' saved successfully"})

        except Exception as e:
            print(f"❌ Error saving tutorial state: {e}")
            import traceback
            traceback.print_exc()
            return jsonify({"status": "error", "error": str(e)}), 500

    @app.route("/api/control/fast-forward", methods=["POST"])
    def fast_forward():
            print(f"❌ Error saving tutorial state: {e}")
            import traceback
            traceback.print_exc()
            return jsonify({"status": "error", "error": str(e)}), 500

    @app.route("/api/save-calibration", methods=["POST"])
    def save_calibration():
        """Save manual calibration to ../calib/manual_calibration_{camera}.json Also updates the in-memory camera
        models/poses so the user immediately sees results.

        Expected JSON:
        {
          "camera": "front",
          "intrinsics": {"width": W, "height": H, "Knew": [[fx,0,cx],[0,fy,cy],[0,0,1]]},
          "extrinsics": {"T_three": [[...4x4...]]}
        }

        """
        # Allow multiplexing to gripper tip handler
        data = request.get_json(force=True, silent=True) or {}
        typ = (data.get("type") or "").strip().lower()
        if typ == "gripper_tips":
            calib = data.get("gripper_tip_calib") or {}
            # minimal validation
            if not isinstance(calib, dict) or "left" not in calib or "right" not in calib:
                return jsonify({"error": "gripper_tip_calib must include 'left' and 'right' {x,y,z}"}), 400
            try:
                out_path = crowd_interface.save_gripper_tip_calibration(calib)
                return jsonify({"status": "ok", "path": out_path})
            except (ValueError, IOError) as e:
                return jsonify({"error": str(e)}), 400
        cam = data.get("camera")
        intr = data.get("intrinsics") or {}
        extr = data.get("extrinsics") or {}
        if not cam:
            return jsonify({"error": "missing 'camera'"}), 400
        if "Knew" not in intr or "width" not in intr or "height" not in intr:
            return jsonify({"error": "intrinsics must include width, height, Knew"}), 400
        if "T_three" not in extr:
            return jsonify({"error": "extrinsics must include T_three (4x4)"}), 400

        # Resolve ../data/calib path relative to this file
        base_dir = Path(__file__).resolve().parent
        calib_dir = (base_dir / ".." / "data" / "calib").resolve()
        calib_dir.mkdir(parents=True, exist_ok=True)
        out_path = calib_dir / f"manual_calibration_{cam}.json"

        # Write JSON file
        to_write = {
            "camera": cam,
            "intrinsics": {
                "width": int(intr["width"]),
                "height": int(intr["height"]),
                "Knew": intr["Knew"],
            },
            "extrinsics": {"T_three": extr["T_three"]},
        }
        try:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(to_write, f, indent=2)
        except Exception as e:
            return jsonify({"error": f"failed to write calibration: {e}"}), 500

        # Update in-memory models so the next /api/get-state reflects it immediately
        try:
            # intrinsics
            crowd_interface._camera_models[cam] = {
                "model": "pinhole",
                "rectified": crowd_interface._camera_models.get(cam, {}).get("rectified", False),
                "width": int(intr["width"]),
                "height": int(intr["height"]),
                "Knew": intr["Knew"],
            }
            # extrinsics (pose)
            crowd_interface._camera_poses[f"{cam}_pose"] = extr["T_three"]
        except Exception:
            # Non-fatal; file already saved
            pass

        return jsonify({"status": "ok", "path": str(out_path)})

    @app.route("/api/save-gripper-tips", methods=["POST"])
    def save_gripper_tips():
        try:
            data = request.get_json(force=True, silent=True) or {}
            calib = data.get("gripper_tip_calib") or {}
            if not isinstance(calib, dict) or "left" not in calib or "right" not in calib:
                return jsonify({"error": "gripper_tip_calib must include 'left' and 'right' {x,y,z}"}), 400
            out_path = crowd_interface.save_gripper_tip_calibration(calib)
            return jsonify({"status": "ok", "path": out_path})
        except (ValueError, IOError) as e:
            return jsonify({"error": str(e)}), 400
        except Exception as e:
            return jsonify({"error": f"unexpected error: {e}"}), 500

    @app.route("/api/demo-videos/<filename>")
    def serve_demo_video(filename):
        """Serve demo video files for the frontend example video feature."""
        if not crowd_interface._demo_videos_dir:
            return jsonify({"error": "Demo videos directory not configured"}), 404

        try:
            # Sanitize filename to prevent directory traversal
            filename = os.path.basename(filename)
            file_path = crowd_interface._demo_videos_dir / filename

            if not file_path.exists():
                return jsonify({"error": "Video file not found"}), 404

            # Determine MIME type
            mime_type = mimetypes.guess_type(str(file_path))[0] or "video/webm"

            # Create response with proper headers for video streaming
            response = make_response()
            response.headers["Content-Type"] = mime_type
            response.headers["Accept-Ranges"] = "bytes"
            response.headers["Cache-Control"] = "public, max-age=3600"  # Cache for 1 hour

            # Read and return the file
            with open(file_path, "rb") as f:
                response.data = f.read()

            return response

        except Exception as e:
            print(f"❌ Error serving demo video {filename}: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route("/api/show-videos/<video_id>")
    def serve_show_video(video_id):
        """Serve read-only example videos by numeric id from prompts/{task-name}/videos (or custom dir).

        This endpoint is independent of the recording feature and supports HTTP Range.

        """

        if not crowd_interface.video_manager.show_demo_videos or not crowd_interface.video_manager._show_videos_dir:
            return jsonify({"error": "Show demo videos is not enabled"}), 404

        # Sanitize; we only accept digits for ids.
        vid = "".join(c for c in str(video_id) if c.isdigit())
        if not vid:
            return jsonify({"error": "Invalid video id"}), 400

        file_path, mime = crowd_interface.video_manager.find_show_video_by_id(vid)
        if not file_path:
            return jsonify({"error": "Video file not found"}), 404

        try:
            file_size = os.path.getsize(file_path)
            range_header = request.headers.get("Range", None)

            if range_header:
                # Format: "bytes=start-end"
                m = re.match(r"bytes=(\d+)-(\d*)", range_header)
                if m:
                    start = int(m.group(1))
                    end = int(m.group(2)) if m.group(2) else file_size - 1
                    end = min(end, file_size - 1)
                    if start > end or start >= file_size:
                        # RFC 7233
                        resp = Response(status=416)
                        resp.headers["Content-Range"] = f"bytes */{file_size}"
                        return resp

                    length = end - start + 1
                    with open(file_path, "rb") as f:
                        f.seek(start)
                        data = f.read(length)

                    rv = Response(data, 206, mimetype=mime, direct_passthrough=True)
                    rv.headers["Content-Range"] = f"bytes {start}-{end}/{file_size}"
                    rv.headers["Accept-Ranges"] = "bytes"
                    rv.headers["Content-Length"] = str(length)
                    rv.headers["Cache-Control"] = "public, max-age=3600"
                    return rv

            # No Range: return full file
            with open(file_path, "rb") as f:
                data = f.read()
            rv = make_response(data)
            rv.headers["Content-Type"] = mime
            rv.headers["Content-Length"] = str(file_size)
            rv.headers["Accept-Ranges"] = "bytes"
            rv.headers["Cache-Control"] = "public, max-age=3600"
            return rv

        except Exception as e:
            print(f"❌ Error serving show video {video_id}: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route("/api/show-videos/latest.webm")
    def serve_latest_show_video():
        """Serve the most recent .webm in the show_videos_dir with full HTTP Range support.

        Content-Type: video/webm
        Accept-Ranges: bytes

        """

        if not crowd_interface.video_manager.show_demo_videos or not crowd_interface.video_manager._show_videos_dir:
            return jsonify({"error": "Show demo videos is not enabled"}), 404

        # Resolve the latest numeric .webm (e.g., 1.webm, 2.webm, ...)
        latest_path, latest_id = crowd_interface.video_manager.find_latest_show_video()
        if not latest_path or not latest_path.exists():
            return jsonify({"error": "No video file found"}), 404

        try:
            file_path = latest_path
            mime = "video/webm"  # force WebM for the player

            file_size = os.path.getsize(file_path)
            range_header = request.headers.get("Range", None)

            if range_header:
                # Format: "bytes=start-end"
                m = re.match(r"bytes=(\d+)-(\d*)", range_header)
                if m:
                    start = int(m.group(1))
                    end = int(m.group(2)) if m.group(2) else file_size - 1
                    end = min(end, file_size - 1)
                    if start > end or start >= file_size:
                        resp = Response(status=416)
                        resp.headers["Content-Range"] = f"bytes */{file_size}"
                        return resp

                    length = end - start + 1
                    with open(file_path, "rb") as f:
                        f.seek(start)
                        data = f.read(length)

                    rv = Response(data, 206, mimetype=mime, direct_passthrough=True)
                    rv.headers["Content-Range"] = f"bytes {start}-{end}/{file_size}"
                    rv.headers["Accept-Ranges"] = "bytes"
                    rv.headers["Content-Length"] = str(length)
                    rv.headers["Cache-Control"] = "public, max-age=3600"
                    return rv

            # No Range header → return the whole file
            with open(file_path, "rb") as f:
                data = f.read()
            rv = make_response(data)
            rv.headers["Content-Type"] = mime
            rv.headers["Content-Length"] = str(file_size)
            rv.headers["Accept-Ranges"] = "bytes"
            rv.headers["Cache-Control"] = "public, max-age=3600"
            return rv

        except Exception as e:
            print(f"❌ Error serving latest show video: {e}")
            return jsonify({"error": str(e)}), 500

    # Streaming recording endpoints for canvas-based recording
    # Simple single recording session (no multi-user support needed)
    current_recording = None  # {recording_id, task_name, ext, chunks: [(seq, bytes)], started_at, metadata}

    @app.route("/api/record/start", methods=["POST"])
    def record_start():
        nonlocal current_recording

        if not crowd_interface.video_manager.record_demo_videos:
            return jsonify({"error": "Demo video recording is not enabled"}), 400

        try:
            data = request.get_json() or {}
            recording_id = data.get("recording_id")
            task_name = data.get("task_name") or crowd_interface.task_name or "default"
            ext = "webm"  # VP9-only

            if not recording_id:
                return jsonify({"error": "missing recording_id"}), 400

            # Initialize single recording session
            current_recording = {
                "recording_id": recording_id,
                "task_name": task_name,
                "ext": ext,
                "chunks": [],
                "started_at": data.get("started_at"),
                "metadata": data,
            }

            return jsonify({"ok": True})

        except Exception as e:
            print(f"❌ Error starting recording: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route("/api/record/chunk", methods=["POST"])
    def record_chunk():
        nonlocal current_recording

        if not crowd_interface.video_manager.record_demo_videos:
            return jsonify({"error": "Demo video recording is not enabled"}), 400

        try:
            recording_id = request.args.get("rid")
            seq = request.args.get("seq", "0")

            if not current_recording:
                return jsonify({"error": "no active recording"}), 404

            if recording_id != current_recording["recording_id"]:
                return jsonify({"error": "mismatched recording_id"}), 400

            # Get the raw bytes from the request
            chunk_data = request.get_data()
            if not chunk_data:
                return jsonify({"error": "no data"}), 400

            # Store chunk in memory (ordered by sequence)
            current_recording["chunks"].append((int(seq), chunk_data))

            return jsonify({"ok": True})

        except Exception as e:
            print(f"❌ Error storing chunk: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route("/api/record/stop", methods=["POST"])
    def record_stop():
        nonlocal current_recording

        if not crowd_interface.video_manager.record_demo_videos:
            return jsonify({"error": "Demo video recording is not enabled"}), 400

        try:
            data = request.get_json() or {}
            recording_id = data.get("recording_id")

            if not current_recording:
                return jsonify({"error": "no active recording"}), 404

            if recording_id != current_recording["recording_id"]:
                return jsonify({"error": "mismatched recording_id"}), 400

            # Sort chunks by sequence number
            chunks = sorted(current_recording["chunks"], key=lambda x: x[0])

            if not chunks:
                current_recording = None
                return jsonify({"error": "no chunks received"}), 400

            # Combine all chunks into a single video file
            try:
                # Get next filename using the counter system
                ext = current_recording["ext"]
                filename, index = crowd_interface.video_manager.next_video_filename(ext)
                file_path = crowd_interface.video_manager._demo_videos_dir / filename

                # Write all chunks to the file
                with open(file_path, "wb") as f:
                    for seq, chunk_data in chunks:
                        f.write(chunk_data)

                # Clean up current recording
                current_recording = None

                # No cloud upload - storing locally only
                public_url = None

                return jsonify(
                    {
                        "ok": True,
                        "filename": filename,
                        "path": str(file_path),
                        "save_dir_rel": crowd_interface.rel_path_from_repo(file_path.parent),
                        "public_url": public_url,
                        "index": index,
                    }
                )

            except Exception as e:
                print(f"❌ Error finalizing recording: {e}")
                current_recording = None
                return jsonify({"error": f"failed to save: {e}"}), 500

        except Exception as e:
            print(f"❌ Error stopping recording: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route("/api/record/save", methods=["POST"])
    def record_save():
        nonlocal current_recording
        """Manual save endpoint for demo video recordings."""

        if not crowd_interface.video_manager.record_demo_videos:
            return jsonify({"error": "Demo video recording is not enabled"}), 400

        try:
            data = request.get_json() or {}
            recording_id = data.get("recording_id")

            if not current_recording:
                return jsonify({"error": "No active recording session found"}), 404

            if recording_id != current_recording["recording_id"]:
                return jsonify({"error": "Recording ID mismatch"}), 400

            # Sort chunks by sequence number
            chunks = sorted(current_recording["chunks"], key=lambda x: x[0])

            if not chunks:
                current_recording = None
                return jsonify({"error": "No recording data to save"}), 400

            # Get next filename using the counter system
            ext = current_recording["ext"]
            filename, index = crowd_interface.video_manager.next_video_filename(ext)
            file_path = crowd_interface.video_manager._demo_videos_dir / filename

            # Write all chunks to the file
            with open(file_path, "wb") as f:
                for seq, chunk_data in chunks:
                    f.write(chunk_data)

            # Clean up current recording
            current_recording = None

            # No cloud upload - storing locally only
            public_url = None

            return jsonify(
                {
                    "ok": True,
                    "status": "success",
                    "message": "Recording saved successfully",
                    "filename": filename,
                    "path": str(file_path),
                    "save_dir_rel": crowd_interface.rel_path_from_repo(file_path.parent),
                    "public_url": public_url,
                    "index": index,
                }
            )

        except Exception as e:
            print(f"❌ Error saving recording: {e}")
            return jsonify({"error": str(e)}), 500

    # ============================================================================
    # Animation API Endpoints
    # ============================================================================

    def get_session_id():
        """Extract session ID from request headers."""
        return request.headers.get("X-Session-ID", "anonymous")

    @app.route("/api/animation/status", methods=["GET"])
    def animation_status():
        """Get animation slot availability and current status."""
        try:
            status = crowd_interface.get_animation_status()
            return jsonify(status)
        except Exception as e:
            print(f"❌ Error getting animation status: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route("/api/animation/start", methods=["POST"])
    def start_animation():
        """Start animation for the current user session."""
        try:
            session_id = get_session_id()
            data = request.get_json() or {}

            # Extract goal pose and animation parameters
            goal_pose = data.get("goal_pose")
            goal_joints = data.get("goal_joints")
            duration = data.get("duration", 3.0)
            gripper_action = data.get("gripper_action")  # NEW: extract gripper action

            # Validate input
            if not goal_pose and not goal_joints:
                return jsonify({"error": "Must provide either goal_pose or goal_joints"}), 400

            result = crowd_interface.start_animation(
                session_id=session_id,
                goal_pose=goal_pose,
                goal_joints=goal_joints,
                duration=duration,
                gripper_action=gripper_action,
            )

            if result.get("status") == "error":
                return jsonify(result), 400

            return jsonify(result)

        except Exception as e:
            print(f"❌ Error starting animation: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route("/api/animation/stop", methods=["POST"])
    def stop_animation():
        """Stop animation for the current user session."""
        try:
            session_id = get_session_id()

            result = crowd_interface.stop_animation(session_id)

            if result.get("status") == "error":
                return jsonify(result), 400

            return jsonify(result)

        except Exception as e:
            print(f"❌ Error stopping animation: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route("/api/animation/frame", methods=["GET"])
    def capture_animation_frame():
        """Capture current animation frame for the user session."""
        try:
            session_id = get_session_id()

            result = crowd_interface.capture_animation_frame(session_id)

            if result.get("status") == "error":
                return jsonify(result), 400

            return jsonify(result)

        except Exception as e:
            print(f"❌ Error capturing animation frame: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route("/api/animation/release", methods=["POST"])
    def release_animation_session():
        """Release animation slot for disconnected session."""
        try:
            session_id = get_session_id()

            result = crowd_interface.release_animation_session(session_id)

            return jsonify(result)

        except Exception as e:
            print(f"❌ Error releasing animation session: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route("/api/simulation/init", methods=["POST"])
    def init_simulation():
        """Initialize simulation mode by triggering initial state capture."""
        try:
            if hasattr(crowd_interface, "isaac_manager") and crowd_interface.isaac_manager:
                # Create a basic config to trigger simulation initialization
                basic_config = {
                    "usd_path": f"public/assets/usd/{crowd_interface.task_name}_flattened_tray.usd",
                    "robot_joints": [0.0] * 7,  # Default joint positions
                    "object_poses": {
                        "Cube_Blue": {"pos": [0.2, 0.0, 0.1], "rot": [0, 0, 0, 1]},
                        "Cube_Red": {"pos": [0.2, 0.2, 0.1], "rot": [0, 0, 0, 1]},
                        "Tennis": {"pos": [0.2, -0.2, 0.1], "rot": [0, 0, 0, 1]},
                    },
                }
                # Use the actual method that exists
                result = crowd_interface.isaac_manager.capture_initial_state(basic_config)
                if isinstance(result, dict) and "status" in result:
                    return jsonify(result)
                else:
                    # capture_initial_state returns file paths, not status
                    return jsonify({"status": "success", "message": "Simulation initialized", "files": result})
            else:
                return jsonify({"status": "error", "message": "Isaac manager not available"}), 400
        except Exception as e:
            print(f"❌ Error initializing simulation: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route("/api/simulation/status", methods=["GET"])
    def simulation_status():
        """Get simulation status."""
        try:
            if hasattr(crowd_interface, "isaac_manager") and crowd_interface.isaac_manager:
                status = {
                    "simulation_initialized": crowd_interface.isaac_manager.simulation_initialized,
                    "worker_ready": crowd_interface.isaac_manager.worker_ready,
                    "animation_initialized": crowd_interface.isaac_manager.animation_initialized,
                }
                return jsonify(status)
            else:
                return jsonify({"status": "error", "message": "Isaac manager not available"}), 400
        except Exception as e:
            print(f"❌ Error getting simulation status: {e}")
            return jsonify({"error": str(e)}), 500

    # =========================
    # MTurk Endpoints
    # =========================

    @app.route("/api/mturk/instructions", methods=["GET"])
    def mturk_instructions():
        """Get MTurk general instructions from file.
        
        Returns:
            JSON array of instruction lines (format: "Label: instruction text")
        """
        try:
            project_root = Path(__file__).parent.parent
            instructions_path = project_root / "data" / "prompts" / "mturk_instructions.txt"
            
            if not instructions_path.exists():
                # Return default instructions
                return jsonify([
                    "Goal: Help control a robot arm to complete a manipulation task",
                    "Task: You'll see a robot simulation. Your job is to specify the next position for the robot to move to",
                    "Controls: Use the interface controls to adjust the robot's position and gripper",
                    "Verification: Use the \"Simulate\" button to preview your action before submitting",
                    "Submit: Once you're satisfied with your choice, click \"Confirm\" to submit",
                    "Important: After submitting, you'll see a completion message. Click \"Submit HIT\" to finalize your work"
                ])
            
            with open(instructions_path, 'r') as f:
                # Read entire file and split only on <br> tags
                content = f.read()
                # Split by <br> and strip only leading/trailing whitespace from each section
                sections = [section.strip() for section in content.split('<br>') if section.strip()]
            
            return jsonify(sections)
            
        except Exception as e:
            print(f"Error loading MTurk instructions: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route("/api/mturk/task-instructions", methods=["GET"])
    def mturk_task_instructions():
        """Get task-specific MTurk instructions from file.
        
        Returns:
            JSON object with 'title' and 'sections' array
            Format: {"title": "Task Instructions", "sections": ["section1", "section2", ...]}
        """
        try:
            task_name = crowd_interface.task_name if crowd_interface else "drawer"
            project_root = Path(__file__).parent.parent
            instructions_path = project_root / "data" / "prompts" / f"{task_name}_instructions.txt"
            
            if not instructions_path.exists():
                # Return empty if no task-specific instructions
                return jsonify({"title": None, "sections": []})
            
            with open(instructions_path, 'r') as f:
                content = f.read()
            
            # First line is the title if it starts with TITLE:
            lines = content.split('\n', 1)
            title = None
            sections_content = content
            
            if lines[0].startswith('TITLE:'):
                title = lines[0][6:].strip()  # Remove 'TITLE:' prefix
                sections_content = lines[1] if len(lines) > 1 else ''
            
            # Split remaining content by <br> tags
            sections = [s.strip() for s in sections_content.split('<br>') if s.strip()]
            
            return jsonify({"title": title, "sections": sections})
            
        except Exception as e:
            print(f"Error loading task-specific MTurk instructions: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route("/api/mturk/create-hit", methods=["POST"])
    def mturk_create_hit():
        """Create MTurk HIT for a critical state.
        
        Request body:
            {
                "episode_id": int,
                "state_id": int
            }
        """
        try:
            data = request.get_json()
            if not data or "episode_id" not in data or "state_id" not in data:
                return jsonify({"status": "error", "message": "Missing episode_id or state_id"}), 400

            episode_id = data["episode_id"]
            state_id = data["state_id"]

            hit_id = crowd_interface.create_mturk_hit(episode_id, state_id)

            if hit_id:
                return jsonify({"status": "success", "hit_id": hit_id})
            else:
                return jsonify({"status": "error", "message": "Failed to create HIT"}), 500

        except Exception as e:
            print(f"❌ Error creating MTurk HIT: {e}")
            traceback.print_exc()
            return jsonify({"status": "error", "message": str(e)}), 500

    @app.route("/api/mturk/hit-status", methods=["GET"])
    def mturk_hit_status():
        """Get MTurk HIT status for a specific state.
        
        Query params:
            episode_id: int
            state_id: int
        """
        try:
            episode_id = request.args.get("episode_id", type=int)
            state_id = request.args.get("state_id", type=int)

            if episode_id is None or state_id is None:
                return jsonify({"status": "error", "message": "Missing episode_id or state_id"}), 400

            hit_status = crowd_interface.get_mturk_hit_status(episode_id, state_id)

            if hit_status:
                return jsonify({"status": "success", "hit": hit_status})
            else:
                return jsonify({"status": "not_found", "message": "No HIT found for this state"}), 404

        except Exception as e:
            print(f"❌ Error getting MTurk HIT status: {e}")
            return jsonify({"status": "error", "message": str(e)}), 500

    @app.route("/api/mturk/all-hits", methods=["GET"])
    def mturk_all_hits():
        """Get status of all MTurk HITs."""
        try:
            hits = crowd_interface.get_all_mturk_hits()
            return jsonify({"status": "success", "hits": hits})

        except Exception as e:
            print(f"❌ Error getting all MTurk HITs: {e}")
            return jsonify({"status": "error", "message": str(e)}), 500

    @app.route("/api/mturk/delete-hit", methods=["POST"])
    def mturk_delete_hit():
        """Delete MTurk HIT.
        
        Request body:
            {
                "episode_id": int,
                "state_id": int
            }
        """
        try:
            data = request.get_json()
            if not data or "episode_id" not in data or "state_id" not in data:
                return jsonify({"status": "error", "message": "Missing episode_id or state_id"}), 400

            episode_id = data["episode_id"]
            state_id = data["state_id"]

            success = crowd_interface.delete_mturk_hit(episode_id, state_id)

            if success:
                return jsonify({"status": "success", "message": "HIT deleted"})
            else:
                return jsonify({"status": "error", "message": "Failed to delete HIT"}), 500

        except Exception as e:
            print(f"❌ Error deleting MTurk HIT: {e}")
            return jsonify({"status": "error", "message": str(e)}), 500

    # Serve static files from src/ directory (for MTurk workers)
    # This MUST be last to avoid catching API routes
    @app.route("/<path:filepath>")
    def serve_static(filepath):
        """Serve static files from dist/ (built) or src/ directory."""
        # Get the project root (one level up from backend/)
        project_root = Path(__file__).parent.parent
        dist_dir = project_root / "dist"
        src_dir = project_root / "src"
        public_dir = project_root / "public"
        
        # Try to serve from dist/ (Vite build output) first, then fall back to src/
        try:
            # Check public directory first for any file
            public_file = public_dir / filepath
            if public_file.exists() and public_file.is_file():
                return send_from_directory(public_dir, filepath)
            
            # For HTML files, try dist first
            if filepath.startswith("pages/") or filepath.endswith(".html"):
                # HTML files are in dist/src/pages/ after build
                if not filepath.startswith("pages/"):
                    filepath = f"pages/{filepath}"
                
                dist_file = dist_dir / "src" / filepath
                if dist_file.exists():
                    return send_from_directory(dist_dir / "src", filepath)
                else:
                    return send_from_directory(src_dir, filepath)
            
            # For assets, check dist/assets
            elif filepath.startswith("assets/"):
                dist_file = dist_dir / filepath
                if dist_file.exists():
                    return send_from_directory(dist_dir, filepath)
            
            # For CSS/JS, check dist then src
            elif filepath.startswith("css/") or filepath.startswith("js/"):
                dist_file = dist_dir / filepath
                if dist_file.exists():
                    return send_from_directory(dist_dir, filepath)
                
                src_file = src_dir / filepath
                if src_file.exists():
                    return send_from_directory(src_dir, filepath)
            
            return "Not Found", 404
        except Exception as e:
            print(f"❌ Error serving {filepath}: {e}")
            import traceback
            traceback.print_exc()
            return "Not Found", 404

    return app
