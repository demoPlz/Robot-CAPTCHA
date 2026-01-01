"""State Manager Module.

Manages episode-based state lifecycle for the crowd interface. Handles state creation, labeling, auto-labeling, and
episode finalization.

"""

import queue
import random
import time
from threading import Lock, Thread, Timer

import torch

# Joint names constant (shared with crowd_interface)
JOINT_NAMES = ["joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "left_carriage_joint"]


class StateManager:
    """Manages episode-based state lifecycle for crowd interface.

    Responsibilities:
    - State creation and management (add_state, set_last_state_to_critical)
    - Response recording and completion tracking
    - Auto-labeling of non-critical states
    - Episode finalization and dataset saving
    - State monitoring and info retrieval

    Attributes:
        pending_states_by_episode: episode_id -> {state_id -> state_info}
        completed_states_by_episode: episode_id -> {state_id -> state_info}
        completed_states_buffer_by_episode: episode_id -> {state_id -> state_info} (for chronological dataset writes)
        current_serving_episode: Episode ID currently being served
        episodes_completed: Set of fully completed episode IDs
        next_state_id: Auto-incrementing state ID counter
        state_lock: Lock protecting all state data structures

    """

    def __init__(
        self,
        required_responses_per_state: int,
        required_responses_per_critical_state: int,
        required_approvals_per_critical_state: int,
        autofill_critical_states: bool,
        num_autofill_actions: int,
        use_manual_prompt: bool,
        use_sim: bool,
        task_text: str | None,
        jitter_threshold: float,
        obs_cache_root,
        state_lock: Lock,
        pending_states_by_episode: dict,
        completed_states_by_episode: dict,
        completed_states_buffer_by_episode: dict,
        episode_finalize_grace_s: float,
        episodes_pending_save: set,
        # Managers for external dependencies
        obs_stream_manager,
        pose_estimation_manager,
        drawer_position_manager,
        sim_manager,
        action_selector_manager,
        dataset_manager,
        # Callbacks for external operations
        persist_views_callback,
        persist_obs_callback,
        snapshot_views_callback,
        save_episode_callback,
        state_ready_callback=None,  # NEW: Called when critical state becomes ready for labeling
    ):
        """Initialize state manager.

        Args:
            required_responses_per_state: Number of responses needed for non-critical states
            required_responses_per_critical_state: Number of responses needed for critical states
            autofill_critical_states: Whether to autofill critical states
            num_autofill_actions: Number of actions to autofill - 1
            use_manual_prompt: Whether manual prompting is enabled
            use_sim: Whether sim is enabled
            task_text: Task text for state info
            jitter_threshold: L2 distance threshold for automatic jitter detection (radians)
            obs_cache_root: Root directory for observation cache
            state_lock: Lock protecting shared state data structures
            pending_states_by_episode: Shared dict of pending states
            completed_states_by_episode: Shared dict of completed states
            completed_states_buffer_by_episode: Shared dict for chronological dataset writes
            episode_finalize_grace_s: Grace period before finalizing empty episodes
            episodes_pending_save: Shared set of episodes pending save
            obs_stream_manager: ObservationStreamManager instance
            pose_estimation_manager: PoseEstimationManager instance
            drawer_position_manager: DrawerPositionManager instance
            sim_manager: SimManager instance
            action_selector_manager: ActionSelectorManager instance for action selection
            dataset_manager: DatasetManager instance for logging
            persist_views_callback: Callback to persist views to disk
            persist_obs_callback: Callback to persist observations to disk
            snapshot_views_callback: Callback to snapshot current views
            save_episode_callback: Callback to save episode to dataset

        """
        self.required_responses_per_state = required_responses_per_state
        self.required_responses_per_critical_state = required_responses_per_critical_state
        self.jitter_threshold = jitter_threshold
        self.required_approvals_per_critical_state = required_approvals_per_critical_state
        self.autofill_critical_states = autofill_critical_states
        self.num_autofill_actions = num_autofill_actions
        self.use_manual_prompt = use_manual_prompt
        self.use_sim = use_sim
        self.task_text = task_text
        self._obs_cache_root = obs_cache_root

        # Shared state data structures (references)
        self.state_lock = state_lock
        self.pending_states_by_episode = pending_states_by_episode
        self.completed_states_by_episode = completed_states_by_episode
        self.completed_states_buffer_by_episode = completed_states_buffer_by_episode
        self._episodes_pending_save = episodes_pending_save

        # Episode state
        self.current_serving_episode = None
        self.episodes_completed = set()
        self.next_state_id = 0

        # Episode finalization
        self.episode_finalize_grace_s = episode_finalize_grace_s
        self._episode_finalize_timers: dict[str, Timer] = {}

        # Auto-labeling
        self.auto_label_queue = queue.Queue()
        self.auto_label_worker_thread = None
        
        # Pre-approval threads tracking
        self._pre_approval_threads: dict[tuple[int, int], Thread] = {}  # (episode_id, state_id) -> thread

        # Manager dependencies
        self.obs_stream = obs_stream_manager
        self.pose_estimator = pose_estimation_manager
        self.drawer_position = drawer_position_manager
        self.sim_manager = sim_manager
        self.action_selector = action_selector_manager
        self.dataset_manager = dataset_manager

        # Callbacks for external operations
        self._persist_views_callback = persist_views_callback
        self._persist_obs_callback = persist_obs_callback
        self._snapshot_views_callback = snapshot_views_callback
        self._save_episode_callback = save_episode_callback
        self._state_ready_callback = state_ready_callback  # NEW: MTurk HIT creation callback

        # Goal management
        self.latest_goal = None

        # Active episode tracking
        self._active_episode_id = None

        # Critical state approval tracking (post-execution)
        self.pending_approval_state = None  # {episode_id, state_id, approved: None/True/False}
        self.approval_lock = Lock()

        # Pre-execution approval tracking (before robot moves)
        self.pending_pre_execution_approval = None  # {episode_id, state_id, action, obs_path, view_paths, approved: None/True/False}
        self.pre_execution_approval_lock = Lock()

        # Undo state tracking
        self.pending_undo_classification = (
            None  # {episode_id, state_id, is_new_state: None/True/False, already_executed_actions: []}
        )
        self.undo_lock = Lock()
        self.undo_motion_start_state_id = None  # Track when undo motion begins to delete intermediate states

        # Start auto-labeling worker
        self._start_auto_label_worker()

    # =========================
    # State Management (Public API - called externally)
    # =========================

    def add_state(
        self,
        joint_positions: dict,
        gripper_motion: int = None,
        obs_dict: dict[str, torch.Tensor] = None,
        episode_id: str = None,
        left_carriage_external_force: float | None = None,
    ):
        """Called by lerobot code to add states to backend."""
        joint_positions_float = {k: float(v) for k, v in joint_positions.items()}

        state_id = self.next_state_id
        self.next_state_id += 1

        # Persist views to disk to avoid storing in memory
        view_paths = self._persist_views_callback(episode_id, state_id, self._snapshot_views_callback())  # legacy

        obs_dict_deep_copy = {}
        for key, value in obs_dict.items():
            obs_dict_deep_copy[key] = value.clone().detach()
        obs_path = self._persist_obs_callback(episode_id, state_id, obs_dict_deep_copy)
        if obs_path is None:
            print(f"⚠️  WARNING: obs_path is None for episode={episode_id} state={state_id}")
        else:
            print(f"✓ Persisted obs to: {obs_path}")
        del obs_dict_deep_copy

        # Push obs to monitoring frontend
        self.obs_stream.push_obs_view("obs_main", obs_dict.get("observation.images.cam_main"))
        self.obs_stream.push_obs_view("obs_wrist", obs_dict.get("observation.images.cam_wrist"))

        state_info = {
            # Identity
            "state_id": state_id,
            "episode_id": episode_id,
            # Robot state
            "joint_positions": joint_positions_float,
            "gripper": gripper_motion,
            "controls": ["x", "y", "z", "roll", "pitch", "yaw", "gripper"],  # legacy, will remove
            "left_carriage_external_force": left_carriage_external_force,
            # Observations
            "obs_path": obs_path,
            # Views
            "view_paths": view_paths,
            # Labels
            "actions": [],
            "responses_received": 0,
            # Critical state fields
            "critical": False,
            "prompt_ready": False if self.use_manual_prompt else True,
            "text_prompt": None,  # replaces flex_text_prompt
            "video_prompt": None,  # replaces flex_video_id
            # Task
            "task_text": self.task_text,
            # Sim
            "sim_ready": False if self.use_sim else True,
            # Approval (for critical states only)
            "approval_status": None,  # None=pending/non-critical, "approved", "rejected"
            # Execution history: list of {"action": tensor, "propensity": float, "approval": 1/-1/None}
            "execution_history": [],
            # Pre-approval tracking
            "num_pre_approvals_completed": 0,
            "pre_approval_loop_complete": False,  # Flag to track if pre-approval loop is done
            "final_executed_action": None,
            # Drawer joint positions will be computed when we call set_last_state_to_critical (like object_poses)
            # "drawer_joint_positions"
            # Poses of each object in self.object will be computed when we call set_last_state_to_critical
            # "object_poses"
            # No other fields; segmentation, and all others, no longer supported
        }

        with self.state_lock:
            # Initialize episode containers if needed
            if episode_id not in self.pending_states_by_episode:
                self.pending_states_by_episode[episode_id] = {}
                self.completed_states_by_episode[episode_id] = {}

            # Add state to pending states
            self.pending_states_by_episode[episode_id][state_id] = state_info

            self.current_serving_episode = episode_id

    def set_last_state_to_critical(self):
        # ---- Phase 1: figure out which state to mark, under lock ----
        with self.state_lock:
            if not self.pending_states_by_episode:
                return

            latest_episode_id = max(self.pending_states_by_episode.keys())
            episode_states = self.pending_states_by_episode[latest_episode_id]
            if not episode_states:
                return

            latest_state_id = max(episode_states.keys())
            info = episode_states[latest_state_id]

            if info["critical"]:
                # Already set
                return

            # ---- Phase 0.5: Auto-detect jitter states ----
            # Skip jitter detection if we're in an undo motion (robot returning to previous state)
            # because we EXPECT the robot to be at the same position as a previous critical state
            in_undo_motion = False
            with self.undo_lock:
                if self.pending_undo_classification is not None and self.pending_undo_classification.get("awaiting_robot_arrival", False):
                    in_undo_motion = True
            
            if not in_undo_motion:
                # If the state we want to mark critical is too similar to the previous critical state,
                # it's jitter. Delete it AND all intermediate states in one batch.
                all_states = {
                    **self.pending_states_by_episode.get(latest_episode_id, {}),
                    **self.completed_states_by_episode.get(latest_episode_id, {}),
                }
                
                # Find previous critical state
                previous_critical_states = [
                    (sid, sinfo)
                    for sid, sinfo in sorted(all_states.items())
                    if sid < latest_state_id and sinfo.get("critical", False)
                ]
                
                if previous_critical_states:
                    prev_state_id, prev_state_info = previous_critical_states[-1]
                    
                    # Check if the state we want to mark critical is jitter
                    is_jitter = self._is_jitter_state(
                        info["joint_positions"],
                        prev_state_info["joint_positions"],
                        threshold=self.jitter_threshold
                    )
                    
                    if is_jitter:
                        # Delete this state AND all states between prev_state_id and latest_state_id
                        states_to_delete = [
                            sid for sid in self.pending_states_by_episode[latest_episode_id].keys()
                            if prev_state_id < sid <= latest_state_id
                        ]
                        
                        print(f"🗑️  Auto-detected jitter: state {latest_state_id} too similar to critical state {prev_state_id}")
                        print(f"🗑️  Batch-deleting {len(states_to_delete)} states ({min(states_to_delete)}-{max(states_to_delete)})")
                        
                        for sid in states_to_delete:
                            sinfo = self.pending_states_by_episode[latest_episode_id][sid]
                            self._delete_obs_from_disk(sinfo.get("obs_path"))
                            del self.pending_states_by_episode[latest_episode_id][sid]
                        
                        print(f"✅ Jitter states removed (serving logic unaffected)")
                        # The previous critical state continues to be served
                        return

            info["critical"] = True
            self.demote_earlier_unanswered_criticals(latest_state_id, latest_episode_id)
            self.auto_label_previous_states(latest_state_id)

        # ---- Phase 1.4: Check if this is post-undo arrival, trigger classification if needed ----
        should_trigger_undo_classification = False
        undo_arrived_obs_path = None
        undo_classification_result = None
        with self.undo_lock:
            if self.pending_undo_classification is not None and self.pending_undo_classification.get(
                "awaiting_robot_arrival", False
            ):
                should_trigger_undo_classification = True
                # Get the obs_path from the state we just marked critical
                with self.state_lock:
                    ep = self.pending_states_by_episode.get(latest_episode_id)
                    if ep and latest_state_id in ep:
                        undo_arrived_obs_path = ep[latest_state_id].get("obs_path")

        if should_trigger_undo_classification and undo_arrived_obs_path:
            # Robot has arrived at undo target and we marked it critical
            # Now trigger classification modal (blocking)
            undo_classification_result = self.trigger_undo_classification_after_arrival(undo_arrived_obs_path)
            # If we resampled (old state), skip the approval phase since action is already set
            # If we're collecting new actions (new state), continue with normal approval flow
            if undo_classification_result and not undo_classification_result.get("classified_as_new_state", True):
                # Old state - action was resampled and latest_goal is set, skip to pose estimation
                print(f"⏩ Skipping approval phase - action already resampled from undo")
                # Skip approval, jump directly to Phase 2
                approved = True
            else:
                # New state - proceed with normal approval flow
                approved = None  # Will be set in approval phase below
        else:
            approved = None  # Will be set in approval phase below

        # ---- Phase 1.5: Wait for administrator approval ----
        if approved is None:  # Only do approval if not already handled by undo classification
            # CRITICAL: Mark state as "pending" BEFORE setting up modal
            # This protects it from demotion by future states arriving during modal setup
            with self.state_lock:
                if latest_episode_id in self.pending_states_by_episode:
                    if latest_state_id in self.pending_states_by_episode[latest_episode_id]:
                        self.pending_states_by_episode[latest_episode_id][latest_state_id]["approval_status"] = "pending"
            
            # Set pending approval and wait for response
            with self.approval_lock:
                self.pending_approval_state = {
                    "episode_id": latest_episode_id,
                    "state_id": latest_state_id,
                    "approved": None,  # None=pending, True=approved, False=rejected
                }

            print(f"⏸️  Waiting for administrator approval for state {latest_state_id}...")

            # Poll for approval decision (blocking)
            import time

            while True:
                time.sleep(0.1)
                with self.approval_lock:
                    if self.pending_approval_state is None:
                        # State was demoted, exit
                        print(f"⚠️  State {latest_state_id} was demoted, canceling approval")
                        return
                    if self.pending_approval_state["approved"] is not None:
                        approved = self.pending_approval_state["approved"]
                        self.pending_approval_state = None
                        break

        if not approved:
            # Administrator rejected - perform undo
            print(f"❌ Administrator rejected state {latest_state_id}, performing undo")
            self.undo_to_previous_critical_state()
            return

        print(f"✅ Administrator approved state {latest_state_id}, proceeding...")
        
        # Record final_executed_action for the PREVIOUS critical state (the one we moved FROM)
        # The action that got us to latest_state_id belongs to the previous critical state
        with self.state_lock:
            ep = self.completed_states_buffer_by_episode.get(latest_episode_id, {})
            if ep:
                # Find previous critical state
                critical_states = [sid for sid, sinfo in ep.items() if sinfo.get("critical", False)]
                critical_states.sort()
                
                if len(critical_states) > 0:
                    # Previous critical is the one before this arrival
                    prev_critical_state_id = critical_states[-1]
                    prev_state_info = ep[prev_critical_state_id]
                    
                    # Find the executed action (the one that was approved post-execution)
                    exec_history = prev_state_info.get("execution_history", [])
                    for entry in exec_history:
                        if entry.get("executed", False) and entry.get("post_execution_approved", False):
                            executed_action = entry["action"]
                            prev_state_info["final_executed_action"] = executed_action.tolist() if hasattr(executed_action, "tolist") else list(executed_action)
                            print(f"✅ Recorded final_executed_action for PREVIOUS state {prev_critical_state_id}")
                            break

        # ---- Phase 2: enqueue pose jobs and BLOCK until all are reported ----
        # Re-fetch info from state_lock after approval wait
        with self.state_lock:
            ep = self.pending_states_by_episode.get(latest_episode_id)
            if not ep or latest_state_id not in ep:
                print(f"⚠️  State {latest_state_id} was removed during approval")
                return
            info = ep[latest_state_id]

        poses_ready = self.pose_estimator.enqueue_pose_jobs_for_state(
            latest_episode_id, latest_state_id, info, wait=True, timeout_s=None
        )

        # ---- Phase 2.5: Estimate drawer position if tracking enabled ----
        drawer_positions_ready = False
        if self.drawer_position and self.drawer_position.enabled:
            try:
                # Load the observation
                obs_dict = torch.load(info["obs_path"], map_location="cpu")
                drawer_joint_positions = self.drawer_position.get_joint_position_from_obs(obs_dict)

                if drawer_joint_positions:
                    # Update the state info with drawer position
                    with self.state_lock:
                        ep = self.pending_states_by_episode.get(latest_episode_id)
                        if ep and latest_state_id in ep:
                            ep[latest_state_id]["drawer_joint_positions"] = drawer_joint_positions
                            drawer_positions_ready = True
                            print(
                                f"🗄️  Drawer position captured for critical state (ep={latest_episode_id}, state={latest_state_id})"
                            )
                else:
                    print(f"⚠️  Drawer position estimation failed for ep={latest_episode_id}, state={latest_state_id}")
            except Exception as e:
                print(f"⚠️  Error estimating drawer position: {e}")

        # ---- Phase 3: only then consider sim ----
        with self.state_lock:
            # Re-lookup the state in case the dict changed
            ep = self.pending_states_by_episode.get(latest_episode_id)
            if not ep or latest_state_id not in ep:
                return
            info = ep[latest_state_id]

            if self.use_sim and poses_ready:
                info["sim_ready"] = False  # Mark as not ready initially
                self.sim_manager.enqueue_sim_capture(latest_episode_id, latest_state_id, info)
            else:
                # Not using sim, or poses not ready within timeout
                info["sim_ready"] = not self.use_sim
                if self.use_sim and not poses_ready:
                    print(
                        f"⏭️  Skipping/deferring sim capture: poses not ready for ep={latest_episode_id}, state={latest_state_id}"
                    )

    def get_latest_state(self) -> dict:
        """Get a pending state from current serving episode. 
        
        Only serves states that have been approved by the monitor admin.
        This ensures jitter states don't affect what users see.
        """

        with self.state_lock:
            episode_id = self.current_serving_episode
            
            if episode_id not in self.pending_states_by_episode:
                # No pending critical states left
                return {"status": "no_pending_states", "blocked_critical_states": False}
            
            # Find the latest APPROVED critical state
            pending_states = self.pending_states_by_episode[episode_id]
            approved_critical_states = [
                (state_id, state_info)
                for state_id, state_info in pending_states.items()
                if state_info.get("critical", False) and state_info.get("approval_status") == "approved"
            ]
            
            if not approved_critical_states:
                # No approved states yet - check if there are pending approval states
                pending_approval_states = [
                    state_id for state_id, state_info in pending_states.items()
                    if state_info.get("critical", False) and state_info.get("approval_status") == "pending"
                ]
                
                if pending_approval_states:
                    # There are states awaiting approval
                    return {
                        "status": "no_ready_states",
                        "blocked_critical_states": True,
                    }
                else:
                    # No critical states at all
                    return {"status": "no_pending_states", "blocked_critical_states": False}
            
            # Get the latest approved state (highest state_id)
            latest_approved_state_id, state_info = max(approved_critical_states, key=lambda x: x[0])
            
            # Check if state is ready (prompt_ready and sim_ready)
            if state_info["critical"] and (not state_info["prompt_ready"] or not state_info["sim_ready"]):
                # State is approved but not ready yet
                return {
                    "status": "no_ready_states",
                    "blocked_critical_states": True,
                }

            # Return the latest approved state for labeling
            return state_info.copy()

    def record_response(self, response_data: dict):
        """Record a response for a specific state.

        Handles all the side-effects.

        """
        should_run_pre_approval = False
        state_info_copy = None
        
        with self.state_lock:
            state_id = response_data["state_id"]
            episode_id = response_data["episode_id"]

            if (
                episode_id not in self.pending_states_by_episode
                or state_id not in self.pending_states_by_episode[episode_id]
            ):
                # State already fully labeled
                return

            state_info = self.pending_states_by_episode[episode_id][state_id]

            required_responses = (
                self.required_responses_per_critical_state
                if state_info["critical"]
                else self.required_responses_per_state
            )

            joint_positions = response_data["joint_positions"]
            gripper_action = response_data["gripper"]

            state_info["responses_received"] += 1

            goal_positions = []
            for joint_name in JOINT_NAMES:
                joint_value = joint_positions[joint_name]
                goal_positions.append(float(joint_value[0]))

            goal_positions[-1] = 0.044 if gripper_action > 0 else 0.0
            goal_positions = torch.tensor(goal_positions, dtype=torch.float32)
            state_info["actions"].append(goal_positions)
            
            # Track actual number of unique worker submissions (not including autofill)
            if "actual_num_submissions" not in state_info:
                state_info["actual_num_submissions"] = 0
            state_info["actual_num_submissions"] += 1

            # Autofill
            if state_info["critical"] and self.autofill_critical_states:
                remaining = state_info["responses_received"]
                clones_to_add = min(self.num_autofill_actions - 1, remaining)
                for _ in range(clones_to_add):
                    state_info["actions"].append(goal_positions.clone())
                state_info["responses_received"] += clones_to_add

            # Handle completion
            if state_info["responses_received"] >= required_responses:
                # Build action tensor for all submissions
                all_actions = torch.cat(state_info["actions"][:required_responses], dim=0)

                if required_responses < self.required_responses_per_critical_state:
                    # Pad unimportant states's action tensor
                    missing_responses = self.required_responses_per_critical_state - required_responses
                    action_dim = len(JOINT_NAMES)
                    padding_size = missing_responses * action_dim
                    padding = torch.full((padding_size,), float("nan"), dtype=torch.float32)
                    all_actions = torch.cat([all_actions, padding], dim=0)

                state_info["action_to_save"] = all_actions

                # Save to completed states buffer (for forming training set)
                if episode_id not in self.completed_states_buffer_by_episode:
                    self.completed_states_buffer_by_episode[episode_id] = {}
                self.completed_states_buffer_by_episode[episode_id][state_id] = state_info

                # Save to completed states (for monitoring)
                if episode_id not in self.completed_states_by_episode:
                    self.completed_states_by_episode[episode_id] = {}
                self.completed_states_by_episode[episode_id][state_id] = state_info

                # Remove from pending
                del self.pending_states_by_episode[episode_id][state_id]
                
                # Flag for pre-approval loop if critical
                if state_info["critical"]:
                    should_run_pre_approval = True
                    state_info_copy = state_info.copy()

        # Run pre-approval loop in background thread (only for completed critical states)
        if should_run_pre_approval and state_info_copy:
            thread = Thread(
                target=self._run_pre_approval_loop_wrapper,
                args=(state_info_copy, episode_id, state_id),
                daemon=True,
            )
            self._pre_approval_threads[(episode_id, state_id)] = thread
            thread.start()
            
        with self.state_lock:
                # Handle episode completion
                if episode_id in self.pending_states_by_episode and not self.pending_states_by_episode[episode_id]:
                    self._schedule_episode_finalize_after_grace(episode_id)

    def get_pending_states_info(self) -> dict:
        """Get episode-based state information for monitoring."""
        with self.state_lock:
            episodes_info = {}
            total_pending = 0

            # Include episodes that have either pending states OR completed states (so completed states remain visible)
            all_episode_ids = set(self.pending_states_by_episode.keys()) | set(self.completed_states_by_episode.keys())

            # Process each episode
            for episode_id in sorted(all_episode_ids):
                episode_states = {}

                # Add pending states from this episode
                if episode_id in self.pending_states_by_episode:
                    for state_id, info in self.pending_states_by_episode[episode_id].items():
                        required_responses = (
                            self.required_responses_per_critical_state
                            if info.get("critical", False)
                            else self.required_responses_per_state
                        )
                        _txt = info.get("text_prompt")  # Updated field name
                        has_flex_text = bool(str(_txt or "").strip())
                        _vid = info.get("video_prompt")  # Updated field name
                        has_flex_video = _vid is not None

                        episode_states[state_id] = {
                            "responses_received": info["responses_received"],
                            "responses_needed": required_responses - info["responses_received"],
                            "critical": bool(info.get("critical", False)),
                            "has_flex_text": has_flex_text,
                            "has_flex_video": has_flex_video,
                            # Legacy aliases to avoid breaking older monitor UI
                            "has_vlm_text": has_flex_text,
                            "has_video_id": has_flex_video,
                        }
                        total_pending += 1

                # Add completed states from this episode
                if episode_id in self.completed_states_by_episode:
                    for state_id, info in self.completed_states_by_episode[episode_id].items():
                        _txt = info.get("text_prompt")  # Updated field name
                        has_flex_text = bool(str(_txt or "").strip())
                        _vid = info.get("video_prompt")  # Updated field name
                        has_flex_video = _vid is not None

                        episode_states[state_id] = {
                            "responses_received": info["responses_received"],
                            "responses_needed": 0,  # Completed
                            "critical": bool(info.get("critical", False)),
                            "has_flex_text": has_flex_text,
                            "has_flex_video": has_flex_video,
                            "has_vlm_text": has_flex_text,  # legacy
                            "has_video_id": has_flex_video,  # legacy
                        }

                episodes_info[episode_id] = {
                    "states": episode_states,
                    "pending_count": len(self.pending_states_by_episode.get(episode_id, {})),
                    "completed_count": len(self.completed_states_by_episode.get(episode_id, {})),
                    "is_current_serving": episode_id == self.current_serving_episode,
                    "is_completed": episode_id in self.episodes_completed,
                    "pending_save": episode_id in self._episodes_pending_save,
                }

            return {
                "total_pending": total_pending,
                "current_serving_episode": self.current_serving_episode,
                "required_responses_per_state": self.required_responses_per_state,
                "required_responses_per_critical_state": self.required_responses_per_critical_state,
                "episodes": episodes_info,
            }

    def get_latest_goal(self) -> dict | None:
        """Get next approved action to execute.

        Phase 2: Execution loop - select first non-executed pre-approved action from execution_history.

        Returns:
            dict with 'action' (list of floats) and 'is_undo' (bool), or None if no goal available

        """
        # PRIORITY: Check if there's an undo goal (from undo_to_previous_critical_state)
        if self.latest_goal is not None:
            goal_tensor = self.latest_goal
            self.latest_goal = None  # Clear after consuming
            
            # Convert to list
            action_list = goal_tensor.tolist() if hasattr(goal_tensor, "tolist") else list(goal_tensor)
            print(f"↩️  Executing undo motion to previous critical state")
            return action_list
        
        # Phase 2: Execution loop - find first non-executed pre-approved action
        with self.state_lock:
            # Find the latest completed critical state
            latest_episode_id = max(self.completed_states_by_episode.keys()) if self.completed_states_by_episode else None
            if latest_episode_id is None:
                return None
                
            ep = self.completed_states_by_episode.get(latest_episode_id, {})
            if not ep:
                return None
                
            # Find latest critical state
            critical_states = [sid for sid, sinfo in ep.items() if sinfo.get("critical", False)]
            if not critical_states:
                return None
                
            latest_critical_state_id = max(critical_states)
            state_info = ep[latest_critical_state_id]
            
            # MUST check if pre-approval loop is complete FIRST, before looking at actions
            if not state_info.get("pre_approval_loop_complete", False):
                # Pre-approval loop still running - don't execute anything yet
                return None
            
            exec_history = state_info.get("execution_history", [])
            
            # Find first non-executed approved action
            for entry in exec_history:
                if entry.get("approval") == 1 and not entry.get("executed", False):
                    # Mark as executed
                    entry["executed"] = True
                    selected_action = entry["action"]
                    
                    # Convert to list if tensor
                    action_list = selected_action.tolist() if hasattr(selected_action, "tolist") else list(selected_action)
                    
                    print(f"✅ Executing pre-approved action for state {latest_critical_state_id}")
                    return action_list
            
            # No approved actions available - return None silently
            return None

    def get_pending_undo_classification(self) -> dict | None:
        """Get the state awaiting undo classification from administrator."""
        with self.undo_lock:
            if self.pending_undo_classification is None or self.pending_undo_classification["is_new_state"] is not None:
                return None

            # Don't show modal until robot has arrived
            if self.pending_undo_classification.get("awaiting_robot_arrival", True):
                return None

            episode_id = self.pending_undo_classification["episode_id"]
            state_id = self.pending_undo_classification["state_id"]
            previous_obs_path = self.pending_undo_classification.get("previous_obs_path")
            arrived_obs_path = self.pending_undo_classification.get("arrived_obs_path")
            already_executed = self.pending_undo_classification.get("already_executed_actions", [])

            return {
                "episode_id": episode_id,
                "state_id": state_id,
                "previous_obs_path": previous_obs_path,  # Previous critical state (target)
                "arrived_obs_path": arrived_obs_path,  # State after undo motion
                "num_remaining_actions": len(already_executed),
            }

    def classify_undo_as_new_state(self, episode_id: int, state_id: int) -> bool:
        """Classify post-undo state as a new state (requires new action submissions)."""
        with self.undo_lock:
            if self.pending_undo_classification is None:
                return False
            if (
                self.pending_undo_classification["episode_id"] != episode_id
                or self.pending_undo_classification["state_id"] != state_id
            ):
                return False

            self.pending_undo_classification["is_new_state"] = True
            return True

    def classify_undo_as_old_state(self, episode_id: int, state_id: int) -> bool:
        """Classify post-undo state as old state (resample from existing actions)."""
        with self.undo_lock:
            if self.pending_undo_classification is None:
                return False
            if (
                self.pending_undo_classification["episode_id"] != episode_id
                or self.pending_undo_classification["state_id"] != state_id
            ):
                return False

            self.pending_undo_classification["is_new_state"] = False
            return True

    def get_pending_approval_state(self) -> dict | None:
        """Get the state awaiting approval from administrator."""
        with self.approval_lock:
            if self.pending_approval_state is None or self.pending_approval_state["approved"] is not None:
                return None

            episode_id = self.pending_approval_state["episode_id"]
            state_id = self.pending_approval_state["state_id"]

            # Get state info
            with self.state_lock:
                if episode_id not in self.pending_states_by_episode:
                    return None
                if state_id not in self.pending_states_by_episode[episode_id]:
                    return None

                current_state = self.pending_states_by_episode[episode_id][state_id]

                # Find previous critical state for comparison
                previous_critical_obs_path = None
                all_states = {
                    **self.pending_states_by_episode.get(episode_id, {}),
                    **self.completed_states_by_episode.get(episode_id, {}),
                }

                previous_critical_states = [
                    (sid, sinfo)
                    for sid, sinfo in sorted(all_states.items())
                    if sid < state_id and sinfo.get("critical", False)
                ]

                if previous_critical_states:
                    _, prev_state = previous_critical_states[-1]
                    previous_critical_obs_path = prev_state.get("obs_path")

                return {
                    "episode_id": episode_id,
                    "state_id": state_id,
                    "obs_path": current_state.get("obs_path"),
                    "previous_critical_obs_path": previous_critical_obs_path,
                }

    def approve_critical_state(self, episode_id: int, state_id: int) -> bool:
        """Approve a pending critical state (post-execution approval)."""
        with self.approval_lock:
            if self.pending_approval_state is None:
                return False
            if (
                self.pending_approval_state["episode_id"] != episode_id
                or self.pending_approval_state["state_id"] != state_id
            ):
                return False

            self.pending_approval_state["approved"] = True

            # Mark the state as approved in pending_states_by_episode (where it actually is)
            with self.state_lock:
                if episode_id in self.pending_states_by_episode:
                    if state_id in self.pending_states_by_episode[episode_id]:
                        state_info = self.pending_states_by_episode[episode_id][state_id]
                        state_info["approval_status"] = "approved"
                        
                # Also mark the executed action as post-execution approved in completed states
                ep_completed = self.completed_states_buffer_by_episode.get(episode_id, {})
                if ep_completed:
                    # Find previous critical state (the one that has the executed action)
                    critical_states = [sid for sid, sinfo in ep_completed.items() if sinfo.get("critical", False)]
                    critical_states.sort()
                    
                    if len(critical_states) > 0:
                        prev_critical_state_id = critical_states[-1]
                        prev_state_info = ep_completed[prev_critical_state_id]
                        
                        # Mark the executed action as post-execution approved
                        exec_history = prev_state_info.get("execution_history", [])
                        for entry in exec_history:
                            if entry.get("executed", False):
                                entry["post_execution_approved"] = True
                                print(f"✅ Marked executed action as post-execution approved for state {prev_critical_state_id}")
                                break

            return True

    # =========================
    # Pre-Execution Approval (New)
    # =========================

    def get_pending_pre_execution_approval(self) -> dict | None:
        """Get the action awaiting pre-execution approval from administrator."""
        with self.pre_execution_approval_lock:
            if self.pending_pre_execution_approval is None:
                return None
            if self.pending_pre_execution_approval["approved"] is not None:
                print(f"   get_pending_pre_execution_approval: approved={self.pending_pre_execution_approval['approved']}, returning None")
                return None

            episode_id = self.pending_pre_execution_approval["episode_id"]
            state_id = self.pending_pre_execution_approval["state_id"]
            action = self.pending_pre_execution_approval["action"]
            obs_path = self.pending_pre_execution_approval["obs_path"]
            view_paths = self.pending_pre_execution_approval["view_paths"]

            return {
                "episode_id": episode_id,
                "state_id": state_id,
                "action": action,  # Joint positions as list
                "obs_path": obs_path,
                "view_paths": view_paths,
            }

    def approve_pre_execution(self, episode_id: int, state_id: int) -> bool:
        """Approve a pending pre-execution action."""
        with self.pre_execution_approval_lock:
            if self.pending_pre_execution_approval is None:
                return False
            if (
                self.pending_pre_execution_approval["episode_id"] != episode_id
                or self.pending_pre_execution_approval["state_id"] != state_id
            ):
                return False

            self.pending_pre_execution_approval["approved"] = True
            return True

    def reject_pre_execution(self, episode_id: int, state_id: int) -> bool:
        """Reject a pending pre-execution action (will trigger resampling)."""
        with self.pre_execution_approval_lock:
            if self.pending_pre_execution_approval is None:
                return False
            if (
                self.pending_pre_execution_approval["episode_id"] != episode_id
                or self.pending_pre_execution_approval["state_id"] != state_id
            ):
                return False

            self.pending_pre_execution_approval["approved"] = False
            return True

    def reject_critical_state(self, episode_id: int, state_id: int) -> bool:
        """Reject a pending critical state."""
        with self.approval_lock:
            if self.pending_approval_state is None:
                return False
            if (
                self.pending_approval_state["episode_id"] != episode_id
                or self.pending_approval_state["state_id"] != state_id
            ):
                return False

            self.pending_approval_state["approved"] = False

            # Mark the state as rejected in pending_states_by_episode (where it actually is)
            with self.state_lock:
                if episode_id in self.pending_states_by_episode:
                    if state_id in self.pending_states_by_episode[episode_id]:
                        state_info = self.pending_states_by_episode[episode_id][state_id]
                        state_info["approval_status"] = "rejected"

            return True

    def discard_jitter_states(self, episode_id: int) -> bool:
        """Find last approved critical state and discard all states after it.
        
        This is used when newer states have been created due to motor jitter after
        an approved state. Discarding allows the approved state to be served.
        
        Args:
            episode_id: Episode to clean up
            
        Returns:
            True if states were discarded, False if no approved state found or no states to discard
        """
        with self.state_lock:
            # Find all critical states with approval_status = "approved"
            all_states = {
                **self.pending_states_by_episode.get(episode_id, {}),
                **self.completed_states_by_episode.get(episode_id, {}),
            }
            
            approved_critical_states = [
                state_id for state_id, state_info in all_states.items()
                if state_info.get("critical") and state_info.get("approval_status") == "approved"
            ]
            
            if not approved_critical_states:
                print(f"⚠️  No approved critical state found in episode {episode_id}")
                return False
            
            # Get the last (highest state_id) approved critical state
            last_approved_state_id = max(approved_critical_states)
            print(f"📍 Last approved critical state: {last_approved_state_id}")
            
            # Collect all states after it
            states_to_delete = []
            if episode_id in self.pending_states_by_episode:
                for other_state_id in self.pending_states_by_episode[episode_id].keys():
                    if other_state_id > last_approved_state_id:
                        states_to_delete.append(other_state_id)
            
            if episode_id in self.completed_states_by_episode:
                for other_state_id in self.completed_states_by_episode[episode_id].keys():
                    if other_state_id > last_approved_state_id:
                        states_to_delete.append(other_state_id)
            
            if not states_to_delete:
                print(f"⚠️  No states to discard after state {last_approved_state_id}")
                return False
            
            # Remove duplicates and sort
            states_to_delete = sorted(set(states_to_delete))
            
            # Delete them
            deleted_count = 0
            for delete_state_id in states_to_delete:
                if episode_id in self.pending_states_by_episode:
                    if delete_state_id in self.pending_states_by_episode[episode_id]:
                        state_info = self.pending_states_by_episode[episode_id][delete_state_id]
                        self._delete_obs_from_disk(state_info.get("obs_path"))
                        del self.pending_states_by_episode[episode_id][delete_state_id]
                        deleted_count += 1
                
                if episode_id in self.completed_states_by_episode:
                    if delete_state_id in self.completed_states_by_episode[episode_id]:
                        state_info = self.completed_states_by_episode[episode_id][delete_state_id]
                        self._delete_obs_from_disk(state_info.get("obs_path"))
                        del self.completed_states_by_episode[episode_id][delete_state_id]
                        deleted_count += 1
                
                if episode_id in self.completed_states_buffer_by_episode:
                    if delete_state_id in self.completed_states_buffer_by_episode[episode_id]:
                        del self.completed_states_buffer_by_episode[episode_id][delete_state_id]
            
            # Clear pending approval if it was for one of the deleted states
            with self.approval_lock:
                if (self.pending_approval_state and 
                    self.pending_approval_state["episode_id"] == episode_id and
                    self.pending_approval_state["state_id"] in states_to_delete):
                    print(f"🗑️  Clearing pending approval for deleted state {self.pending_approval_state['state_id']}")
                    self.pending_approval_state = None
            
            # Reset next_state_id to point after the last approved state
            # This ensures get_latest_state() will serve the approved state
            self.next_state_id = last_approved_state_id + 1
            print(f"🔄 Reset next_state_id to {self.next_state_id}")
            
            print(f"🗑️  Discarded {deleted_count} jitter states after approved state {last_approved_state_id}")
            return deleted_count > 0

    def undo_to_previous_critical_state(self) -> dict | None:
        """Undo to the previous critical state by discarding all states since then.

        Returns the joint positions and gripper state of the previous critical state,
        or None if there is no previous critical state to undo to.

        This function:
        1. Finds the current latest critical state
        2. Finds the previous critical state (if any)
        3. Deletes all states after the previous critical state
        4. Returns the robot position to execute (previous critical state's position)

        """
        with self.state_lock:
            if not self.pending_states_by_episode and not self.completed_states_by_episode:
                print("⚠️  No states to undo")
                return None

            # Get the latest episode
            all_episode_ids = set(self.pending_states_by_episode.keys()) | set(self.completed_states_by_episode.keys())
            if not all_episode_ids:
                print("⚠️  No episodes found")
                return None

            latest_episode_id = max(all_episode_ids)

            # Combine pending and completed states for this episode
            episode_states = {
                **self.pending_states_by_episode.get(latest_episode_id, {}),
                **self.completed_states_by_episode.get(latest_episode_id, {}),
            }

            if not episode_states:
                print("⚠️  No states in latest episode")
                return None

            # Find all critical states in chronological order
            critical_states = [
                (state_id, state_info)
                for state_id, state_info in sorted(episode_states.items())
                if state_info.get("critical", False)
            ]

            if len(critical_states) < 2:
                print("⚠️  Need at least 2 critical states to undo (found {})".format(len(critical_states)))
                return None

            # Get the second-to-last critical state (the one to revert to)
            previous_critical_state_id, previous_critical_state_info = critical_states[-2]
            current_critical_state_id = critical_states[-1][0]

            print(f"🔙 Undoing: reverting from state {current_critical_state_id} to state {previous_critical_state_id}")

            # Delete all states AFTER the previous critical state (not including it)
            # This deletes both the current critical state and all intermediate non-critical states
            states_to_delete = [state_id for state_id in episode_states.keys() if state_id > previous_critical_state_id]

            deleted_count = 0
            for state_id in states_to_delete:
                # Remove from pending states
                if latest_episode_id in self.pending_states_by_episode:
                    if state_id in self.pending_states_by_episode[latest_episode_id]:
                        state_info = self.pending_states_by_episode[latest_episode_id][state_id]
                        # Clean up observation cache
                        self._delete_obs_from_disk(state_info.get("obs_path"))
                        del self.pending_states_by_episode[latest_episode_id][state_id]
                        deleted_count += 1

                # Remove from completed states
                if latest_episode_id in self.completed_states_by_episode:
                    if state_id in self.completed_states_by_episode[latest_episode_id]:
                        state_info = self.completed_states_by_episode[latest_episode_id][state_id]
                        # Clean up observation cache
                        self._delete_obs_from_disk(state_info.get("obs_path"))
                        del self.completed_states_by_episode[latest_episode_id][state_id]
                        deleted_count += 1

                # Remove from completed buffer (used for dataset writes)
                if latest_episode_id in self.completed_states_buffer_by_episode:
                    if state_id in self.completed_states_buffer_by_episode[latest_episode_id]:
                        del self.completed_states_buffer_by_episode[latest_episode_id][state_id]

            print(f"🗑️  Deleted {deleted_count} states after previous critical state {previous_critical_state_id}")

            # Mark the state_id where undo motion begins - all states >= this will be deleted later
            self.undo_motion_start_state_id = previous_critical_state_id + 1

            # Reset next_state_id to point after the previous critical state
            # This ensures get_latest_state() will serve the previous critical state during undo motion
            self.next_state_id = previous_critical_state_id + 1
            print(f"🔄 Reset next_state_id to {self.next_state_id} (undo to previous critical state)")

            # Return the robot position to execute (revert to previous critical state)
            joint_positions = previous_critical_state_info["joint_positions"]
            gripper_action = previous_critical_state_info.get("gripper", 0)

            print(
                f"↩️  Returning to state {previous_critical_state_id}: joints={joint_positions}, gripper={gripper_action}"
            )

            # Convert joint_positions dict to tensor in correct order (matching normal action format)
            goal_positions = []
            for joint_name in JOINT_NAMES:
                joint_value = joint_positions[joint_name]
                goal_positions.append(float(joint_value))

            # Set gripper position based on gripper action
            goal_positions[-1] = 0.044 if gripper_action > 0 else 0.0
            goal_tensor = torch.tensor(goal_positions, dtype=torch.float32)

            # Set as latest_goal for robot to consume
            self.latest_goal = goal_tensor

        # ---- Set up pending undo classification (will be triggered after robot arrives) ----
        # Store the previous critical state info for later classification
        # Build list of all actions that have been attempted (approved, rejected, or executed)
        attempted_actions = []
        execution_history = previous_critical_state_info.get("execution_history", [])
        for execution in execution_history:
            attempted_actions.append(execution["action"])
        
        with self.undo_lock:
            self.pending_undo_classification = {
                "episode_id": latest_episode_id,
                "state_id": previous_critical_state_id,
                "is_new_state": None,  # None=pending, True=new state, False=old state
                "already_executed_actions": attempted_actions,  # All actions that were attempted (approved or rejected)
                "previous_obs_path": previous_critical_state_info.get("obs_path"),  # For side-by-side comparison
                "awaiting_robot_arrival": True,  # Flag to indicate robot hasn't arrived yet
            }

        print(f"↩️  Robot will move to previous state {previous_critical_state_id}...")
        print(f"⏳ Waiting for robot to arrive before classification...")

        return {
            "episode_id": latest_episode_id,
            "reverted_to_state_id": previous_critical_state_id,
            "awaiting_classification": True,
        }

    def trigger_undo_classification_after_arrival(self, arrived_obs_path: str):
        """Called after robot arrives at the undo target state.

        This triggers the classification modal with side-by-side comparison. Blocks until administrator makes a
        decision.

        """
        with self.undo_lock:
            if self.pending_undo_classification is None:
                print("⚠️  No pending undo classification")
                return

            if not self.pending_undo_classification.get("awaiting_robot_arrival"):
                print("⚠️  Undo classification not awaiting arrival")
                return

            # Update with arrived observation for side-by-side comparison
            self.pending_undo_classification["arrived_obs_path"] = arrived_obs_path
            self.pending_undo_classification["awaiting_robot_arrival"] = False

            latest_episode_id = self.pending_undo_classification["episode_id"]
            previous_critical_state_id = self.pending_undo_classification["state_id"]

        print(f"📸 Robot arrived at previous state {previous_critical_state_id}")
        print(f"⏸️  Waiting for administrator classification: new state or old state?")

        # Poll for administrator decision (blocking)
        import time

        while True:
            time.sleep(0.1)
            with self.undo_lock:
                if self.pending_undo_classification is None:
                    print(f"⚠️  Undo classification was cancelled")
                    return None
                if self.pending_undo_classification["is_new_state"] is not None:
                    is_new_state = self.pending_undo_classification["is_new_state"]
                    already_executed = self.pending_undo_classification["already_executed_actions"]
                    self.pending_undo_classification = None
                    break

        # Delete all states created during undo motion (from undo_motion_start_state_id onwards)
        # This includes all states created while robot was moving back to previous critical state
        self._delete_states_from_id_onwards(latest_episode_id, self.undo_motion_start_state_id)
        self.undo_motion_start_state_id = None

        if is_new_state:
            # Treat as new state - the previous critical state remains completed
            # and we'll create a new state when add_state is called
            print(f"✅ Administrator classified as NEW STATE - will collect new actions")

            return {
                "classified_as_new_state": True,
            }
        else:
            # Treat as old state - resample from existing actions
            # We're acting as if the robot is back at the previous critical state
            print(f"✅ Administrator classified as OLD STATE - will resample from existing actions")

            with self.state_lock:
                # Re-fetch the previous critical state
                ep = self.completed_states_by_episode.get(latest_episode_id)
                if not ep or previous_critical_state_id not in ep:
                    print(f"⚠️  Previous critical state no longer exists")
                    return None

                state_info = ep[previous_critical_state_id]

                # Get all submitted actions for this state
                all_actions = state_info.get("actions", [])
                required_responses = self.required_responses_per_critical_state
                available_actions = all_actions[:required_responses]

                if not available_actions:
                    print(f"⚠️  No actions available to resample from")
                    return None

                # Filter out already executed actions (keep duplicates for now)
                remaining_actions_with_dupes = [
                    action
                    for action in available_actions
                    if not any(torch.equal(action, executed) for executed in already_executed)
                ]

                if not remaining_actions_with_dupes:
                    print(f"⚠️  All actions have been executed, cannot resample")
                    print(f"    Consider classifying as new state instead")
                    return None

                # Deduplicate remaining actions for selection
                remaining_unique = []
                for action in remaining_actions_with_dupes:
                    if not any(torch.equal(action, unique_a) for unique_a in remaining_unique):
                        remaining_unique.append(action)

                print(
                    f"🎲 Resampling from {len(remaining_unique)} unique actions (out of {len(remaining_actions_with_dupes)} remaining submissions)"
                )

                # Use action selector to pick from deduplicated remaining actions
                selected_action, base_propensity, selection_metadata = self.action_selector.select_action(
                    remaining_unique, state_info
                )

                # Compute propensity as submission frequency among actual worker submissions
                # This is the correct importance weight for learning (not counting autofilled clones)
                actual_total_submissions = state_info.get("actual_num_submissions", len(available_actions))
                count_selected = sum(1 for a in available_actions if torch.equal(a, selected_action))
                conditional_propensity = count_selected / actual_total_submissions

                # Update state info with new selection
                # Move selected action to front
                for idx, action in enumerate(state_info["actions"][:required_responses]):
                    if torch.equal(action, selected_action):
                        state_info["actions"][0], state_info["actions"][idx] = (
                            state_info["actions"][idx],
                            state_info["actions"][0],
                        )
                        break

                # Set as latest goal
                self.latest_goal = state_info["actions"][0]

                # Update metadata for this resampled selection
                state_info["action_selection_metadata"] = {
                    **selection_metadata,
                    "resampled": True,
                    "num_remaining_actions": len(remaining_actions),
                    "num_already_executed": len(already_executed),
                    "conditional_propensity": conditional_propensity,
                }
                state_info["action_propensity"] = conditional_propensity

                # Track this as executed
                if "executed_actions" not in state_info:
                    state_info["executed_actions"] = []
                state_info["executed_actions"].append(selected_action)
                
                # Record resampled execution attempt in history
                execution_index = len(state_info.get("execution_history", []))
                state_info["execution_history"].append({
                    "action": selected_action.clone(),
                    "propensity": conditional_propensity,
                    "selector_metadata": {**selection_metadata, "resampled": True},
                    "approval": None  # Will be set when approved/rejected
                })
                
                # Log this resampled execution attempt immediately
                self.dataset_manager.log_execution_attempt(
                    episode_index=latest_episode_id,
                    state_id=previous_critical_state_id,
                    execution_index=execution_index,
                    action=selected_action,
                    propensity=conditional_propensity,
                    selector_metadata={**selection_metadata, "resampled": True},
                )

                print(f"🎯 Resampled action using {selection_metadata['selector_used']} selector")
                print(f"   Propensity (submission frequency): {conditional_propensity:.4f}")
                actual_total = state_info.get("actual_num_submissions", len(available_actions))
                print(f"   (Selector propensity from {len(remaining_unique)} unique: {base_propensity:.4f}, appears {count_selected}/{actual_total} times in actual submissions)")

            return {
                "classified_as_new_state": False,
                "resampled": True,
            }

    def _get_propensities_for_actions(self, actions: list, state_info: dict) -> list[float]:
        """Compute propensities for a list of actions without sampling.

        This is used for computing conditional propensities when resampling.

        """
        if self.action_selector.mode == "random":
            # For random selector, all actions have equal propensity
            n = len(actions)
            return [1.0 / n] * n

        elif self.action_selector.mode == "learned":
            # For learned selector, compute softmax probabilities
            import torch.nn.functional as F

            with torch.no_grad():
                state_features = torch.zeros(1).to(self.action_selector.device)
                action_tensor = torch.stack(actions).to(self.action_selector.device)
                logits = self.action_selector.learned_selector.model(state_features, action_tensor)
                probs = F.softmax(logits, dim=0)
                return probs.cpu().tolist()

        elif self.action_selector.mode == "epsilon_greedy":
            # For epsilon-greedy, combine random and learned propensities
            import torch.nn.functional as F

            epsilon = self.action_selector.epsilon
            n = len(actions)

            with torch.no_grad():
                state_features = torch.zeros(1).to(self.action_selector.device)
                action_tensor = torch.stack(actions).to(self.action_selector.device)
                logits = self.action_selector.learned_selector.model(state_features, action_tensor)
                learned_probs = F.softmax(logits, dim=0)

            # Combine: epsilon * (1/n) + (1-epsilon) * learned_prob
            combined_probs = [
                epsilon * (1.0 / n) + (1 - epsilon) * learned_prob for learned_prob in learned_probs.cpu().tolist()
            ]
            return combined_probs

        else:
            # Fallback to uniform
            n = len(actions)
            return [1.0 / n] * n

    def _delete_states_from_id_onwards(self, episode_id: int, from_state_id: int):
        """Delete all states (pending, completed, buffer) from from_state_id onwards.

        Args:
            episode_id: Episode to delete states from
            from_state_id: Delete all states with state_id >= this value

        """
        if from_state_id is None:
            return

        with self.state_lock:
            states_to_delete = []

            # Collect states to delete from pending
            if episode_id in self.pending_states_by_episode:
                for state_id in self.pending_states_by_episode[episode_id].keys():
                    if state_id >= from_state_id:
                        states_to_delete.append(state_id)

            # Collect states to delete from completed
            if episode_id in self.completed_states_by_episode:
                for state_id in self.completed_states_by_episode[episode_id].keys():
                    if state_id >= from_state_id:
                        states_to_delete.append(state_id)

            # Remove duplicates and sort
            states_to_delete = sorted(set(states_to_delete))

            deleted_count = 0
            for state_id in states_to_delete:
                # Delete from pending
                if episode_id in self.pending_states_by_episode:
                    if state_id in self.pending_states_by_episode[episode_id]:
                        state_info = self.pending_states_by_episode[episode_id][state_id]
                        self._delete_obs_from_disk(state_info.get("obs_path"))
                        del self.pending_states_by_episode[episode_id][state_id]
                        deleted_count += 1

                # Delete from completed
                if episode_id in self.completed_states_by_episode:
                    if state_id in self.completed_states_by_episode[episode_id]:
                        state_info = self.completed_states_by_episode[episode_id][state_id]
                        self._delete_obs_from_disk(state_info.get("obs_path"))
                        del self.completed_states_by_episode[episode_id][state_id]
                        deleted_count += 1

                # Delete from buffer
                if episode_id in self.completed_states_buffer_by_episode:
                    if state_id in self.completed_states_buffer_by_episode[episode_id]:
                        del self.completed_states_buffer_by_episode[episode_id][state_id]

            if deleted_count > 0:
                print(f"🗑️  Deleted {deleted_count} states from state_id {from_state_id} onwards")
                # Update next_state_id to point to from_state_id (the first deleted state)
                # This ensures get_latest_state() will serve the state before from_state_id
                if self.next_state_id > from_state_id:
                    self.next_state_id = from_state_id
                    print(f"🔄 Reset next_state_id to {self.next_state_id} (states deleted from {from_state_id} onwards)")

    def _is_jitter_state(self, joint_positions_1: dict, joint_positions_2: dict, threshold: float = 0.01) -> bool:
        """Check if two joint position states are too similar (likely jitter).
        
        Args:
            joint_positions_1: First joint positions dict
            joint_positions_2: Second joint positions dict
            threshold: Maximum L2 distance to consider states as jitter (radians for joints)
            
        Returns:
            True if states are too similar (jitter), False otherwise
        """
        try:
            # First check if gripper moved significantly - if so, NOT jitter
            gripper_joint = JOINT_NAMES[-1]  # left_carriage_joint
            if gripper_joint in joint_positions_1 and gripper_joint in joint_positions_2:
                gripper_diff = abs(float(joint_positions_1[gripper_joint]) - float(joint_positions_2[gripper_joint]))
                if gripper_diff > 0.01:
                    # Gripper moved significantly, this is intentional motion, not jitter
                    return False
            
            # Compare arm joint positions (excluding gripper)
            total_diff_sq = 0.0
            num_joints = 0
            
            joints_to_check = JOINT_NAMES[:-1]  # Exclude gripper
            
            for joint_name in joints_to_check:
                if joint_name in joint_positions_1 and joint_name in joint_positions_2:
                    val1 = float(joint_positions_1[joint_name])
                    val2 = float(joint_positions_2[joint_name])
                    diff = val1 - val2
                    total_diff_sq += diff * diff
                    num_joints += 1
            
            if num_joints == 0:
                return False  # No joints to compare
            
            # Calculate L2 distance for arm joints
            import math
            l2_distance = math.sqrt(total_diff_sq)
            
            is_similar = l2_distance < threshold
            if is_similar:
                print(f"  Arm joint L2 distance: {l2_distance:.6f} < threshold {threshold} → JITTER")
            
            return is_similar
            
        except Exception as e:
            print(f"⚠️  Error comparing joint positions: {e}")
            return False  # On error, don't treat as jitter

    def _delete_obs_from_disk(self, obs_path: str | None):
        """Delete observation file from disk cache."""
        if not obs_path:
            return
        try:
            import os

            if os.path.exists(obs_path):
                os.remove(obs_path)
                print(f"🗑️  Deleted obs cache: {obs_path}")
        except Exception as e:
            print(f"⚠️  Failed to delete obs cache {obs_path}: {e}")

    def set_active_episode(self, episode_id):
        """Mark which episode the outer robot loop is currently in (or None)."""
        with self.state_lock:
            self._active_episode_id = episode_id

    def set_prompt_ready(
        self, state_info: dict, episode_id: int, state_id: int, text: str | None, video_id: int | None
    ) -> None:
        """Set text/video prompt fields and mark as ready."""
        state_info["text_prompt"] = text  # Updated field name
        state_info["video_prompt"] = video_id  # Updated field name
        state_info["prompt_ready"] = True

        # Check if this is a critical state with "end." text - auto-fill with current position
        if text and text.strip().lower() == "end.":
            self._auto_fill_end_state_locked(state_info, episode_id, state_id)

    def _run_pre_approval_loop_wrapper(self, state_info: dict, episode_id: int, state_id: int) -> None:
        """Wrapper for pre-approval loop that handles cleanup."""
        try:
            self._run_pre_approval_loop(state_info, episode_id, state_id)
        finally:
            # Clean up thread tracking
            key = (episode_id, state_id)
            if key in self._pre_approval_threads:
                del self._pre_approval_threads[key]

    def _run_pre_approval_loop(self, state_info: dict, episode_id: int, state_id: int) -> None:
        """Phase 1: Sample actions one-by-one for pre-approval until stopping condition met.
        
        Stopping condition: (num_pre_approvals_completed >= required_approvals_per_critical_state) AND (at least 1 approved)
        
        Args:
            state_info: State dict with actions list
            episode_id: Episode ID
            state_id: State ID
        """
        required_responses = self.required_responses_per_critical_state
        available_actions = state_info.get("actions", [])[:required_responses]
        
        if not available_actions:
            print(f"⚠️  No actions available for pre-approval loop (state {state_id})")
            return
            
        reviewed_actions = []
        num_approved = 0
        
        while True:
            # Check stopping condition
            num_reviewed = len(reviewed_actions)
            if num_reviewed >= self.required_approvals_per_critical_state and num_approved >= 1:
                print(f"✅ Pre-approval loop complete: {num_reviewed} reviewed, {num_approved} approved")
                break
                
            # Check if we have more actions to sample
            remaining_actions_with_dupes = [a for a in available_actions if not any(torch.equal(a, r["action"]) for r in reviewed_actions)]
            if not remaining_actions_with_dupes:
                print(f"⚠️  No more actions to review (reviewed {num_reviewed}, approved {num_approved})")
                break
            
            # Deduplicate remaining actions for selection
            remaining_unique = []
            for action in remaining_actions_with_dupes:
                if not any(torch.equal(action, unique_a) for unique_a in remaining_unique):
                    remaining_unique.append(action)
                
            # Sample next action from deduplicated remaining actions
            selected_action, selector_propensity, selection_metadata = self.action_selector.select_action(
                remaining_unique, state_info
            )
            
            # Compute true propensity as submission frequency
            # Propensity = (number of times this action appears) / (total number of actual submissions)
            # This is the correct importance weight for learning from crowd data
            actual_total_submissions = state_info.get("actual_num_submissions", len(available_actions))
            count_selected = sum(1 for a in available_actions if torch.equal(a, selected_action))
            true_propensity = count_selected / actual_total_submissions
            
            # Set up pre-execution approval modal (blocking)
            with self.pre_execution_approval_lock:
                self.pending_pre_execution_approval = {
                    "episode_id": episode_id,
                    "state_id": state_id,
                    "action": selected_action.tolist(),
                    "propensity": true_propensity,
                    "selector_metadata": selection_metadata,
                    "obs_path": state_info.get("obs_path"),
                    "view_paths": state_info.get("view_paths"),
                    "approved": None,
                }
                
            print(f"⏸️  Waiting for pre-approval decision (state {state_id}, action {num_reviewed + 1})")
            
            # Block until admin makes decision
            approved = None
            while approved is None:
                time.sleep(0.1)
                with self.pre_execution_approval_lock:
                    if self.pending_pre_execution_approval is None:
                        print(f"⚠️  Pre-approval was cleared, exiting loop")
                        return
                    if self.pending_pre_execution_approval["approved"] is not None:
                        approved = self.pending_pre_execution_approval["approved"]
                        self.pending_pre_execution_approval = None
                        break
                        
            # Record decision
            approval_value = 1 if approved else -1
            reviewed_actions.append({
                "action": selected_action,
                "propensity": true_propensity,  # Use propensity based on original submission counts
                "selector_metadata": selection_metadata,
                "approval": approval_value,
            })
            
            if approved:
                num_approved += 1
                print(f"✅ Action {num_reviewed + 1} approved ({num_approved} total approved)")
            else:
                print(f"❌ Action {num_reviewed + 1} rejected")
                
        # Store all reviewed actions in execution_history and mark loop as complete
        with self.state_lock:
            # Update in both completed_states_by_episode and completed_states_buffer_by_episode
            for ep_dict in [self.completed_states_by_episode, self.completed_states_buffer_by_episode]:
                if episode_id in ep_dict and state_id in ep_dict[episode_id]:
                    ep_dict[episode_id][state_id]["execution_history"] = reviewed_actions
                    ep_dict[episode_id][state_id]["num_pre_approvals_completed"] = len(reviewed_actions)
                    ep_dict[episode_id][state_id]["pre_approval_loop_complete"] = True
        
        print(f"📊 Pre-approval complete: {len(reviewed_actions)} actions reviewed, {num_approved} approved")

    # =========================
    # Internal Helper Methods
    # =========================

    def demote_earlier_unanswered_criticals(self, current_state_id, episode_id):
        """Demote critical states before state_id in episode with episode_id to non-critical.
        
        Only demotes states that:
        - Are earlier than current_state_id
        - Are marked critical
        - Have no actions yet
        - Have NOT passed post-execution approval (not approved/rejected yet)
        """
        for state_id in self.pending_states_by_episode[episode_id].keys():
            state_info = self.pending_states_by_episode[episode_id][state_id]
            
            # Don't demote if state has entered approval phase (pending, approved, or rejected)
            if state_info.get("approval_status") in ["pending", "approved", "rejected"]:
                continue  # In approval pipeline - protected from demotion
            
            # Don't demote if state is currently pending post-execution approval
            is_pending_approval = False
            with self.approval_lock:
                if (
                    self.pending_approval_state
                    and self.pending_approval_state["episode_id"] == episode_id
                    and self.pending_approval_state["state_id"] == state_id
                ):
                    is_pending_approval = True
            
            if is_pending_approval:
                continue  # Currently awaiting approval - protected from demotion
            
            # Only demote if: earlier, critical, no actions yet, and hasn't been approved
            if (
                state_id < current_state_id
                and state_info["critical"]
                and not state_info["actions"]
            ):
                self.pending_states_by_episode[episode_id][state_id]["critical"] = False
                print(f"⬇️  Demoted unanswered critical state {state_id} (new critical: {current_state_id})")

    def auto_label_previous_states(self, critical_state_id):
        self.auto_label_queue.put_nowait(critical_state_id)

    def _start_auto_label_worker(self):
        self.auto_label_worker_thread = Thread(target=self._auto_label_worker, daemon=True)
        self.auto_label_worker_thread.start()

    def _auto_label_worker(self):
        for critical_state_id in iter(self.auto_label_queue.get, None):
            self._auto_label(critical_state_id)

    def _auto_label(self, critical_state_id):
        """
        Given critical_state_id, auto-labels noncritical states in the same episode before the critical state with:"
        1. The executed action of the previous important state
        2. If no previous important state exists, the joint positions of the first state in the episode
        """
        with self.state_lock:
            episode_id = max(self.pending_states_by_episode.keys())

            episode_states = {
                **self.pending_states_by_episode[episode_id],
                **self.completed_states_by_episode[episode_id],
            }

            template_action = None

            previous_critical_id_in_episode = []
            for state_id in episode_states.keys():
                if (
                    episode_states[state_id]["critical"]
                    and state_id < critical_state_id
                    and len(episode_states[state_id]["actions"]) > 0
                ):
                    previous_critical_id_in_episode.append(state_id)

            if previous_critical_id_in_episode:  # Previous critical states exist
                latest_critical_state = episode_states[max(previous_critical_id_in_episode)]
                template_action = latest_critical_state["actions"][0]
            else:  # This is the first critical state in the episode
                first_state_id = min(episode_states.keys())
                first_state = episode_states[first_state_id]
                # Direct access to joint_positions and gripper in flattened structure
                joint_positions = first_state["joint_positions"]
                gripper_action = first_state["gripper"]
                goal_positions = []
                for joint_name in JOINT_NAMES:
                    joint_value = joint_positions[joint_name]
                    goal_positions.append(float(joint_value))

                goal_positions[-1] = 0.044 if gripper_action > 0 else 0.0
                template_action = torch.tensor(goal_positions, dtype=torch.float32)

            states_to_label = []
            for state_id, state_info in episode_states.items():
                if (
                    state_id < critical_state_id
                    and not state_info["critical"]
                    and state_id not in self.completed_states_by_episode[episode_id]
                ):
                    states_to_label.append(state_id)

            for state_id in states_to_label:
                state_info = episode_states[state_id]

                while state_info["responses_received"] < self.required_responses_per_state:
                    state_info["actions"].append(template_action.clone())
                    state_info["responses_received"] += 1

                all_actions = torch.cat(state_info["actions"][: self.required_responses_per_state], dim=0)

                # Pad with inf values to match critical state shape
                missing_responses = self.required_responses_per_critical_state - self.required_responses_per_state
                action_dim = len(JOINT_NAMES)
                padding_size = missing_responses * action_dim
                padding = torch.full((padding_size,), float("nan"), dtype=torch.float32)
                all_actions = torch.cat([all_actions, padding], dim=0)

                state_info["action_to_save"] = all_actions

                # Save to completed_states buffer (for forming training set)
                if episode_id not in self.completed_states_buffer_by_episode:
                    self.completed_states_buffer_by_episode[episode_id] = {}
                self.completed_states_buffer_by_episode[episode_id][state_id] = state_info

                # Save to completed states (for monitoring)
                if episode_id not in self.completed_states_by_episode:
                    self.completed_states_by_episode[episode_id] = {}
                self.completed_states_by_episode[episode_id][state_id] = state_info

                del self.pending_states_by_episode[episode_id][state_id]

    def _schedule_episode_finalize_after_grace(self, episode_id: int):
        delay = self.episode_finalize_grace_s
        timer = Timer(delay, self._finalize_episode_if_still_empty, args=(episode_id,))
        timer.daemon = True
        self._episode_finalize_timers[episode_id] = timer
        timer.start()

    def _finalize_episode_if_still_empty(self, episode_id: int):
        """Timer callback."""
        with self.state_lock:
            self._episode_finalize_timers.pop(episode_id, None)

            if self.pending_states_by_episode.get(episode_id):
                # New states has become pending in the episode
                return

            # Check if there's a pre-execution approval pending for this episode
            with self.pre_execution_approval_lock:
                if (self.pending_pre_execution_approval and 
                    self.pending_pre_execution_approval.get("episode_id") == episode_id):
                    # Don't finalize - waiting for pre-execution approval
                    return

            self.episodes_completed.add(episode_id)  # for monitoring

            buffer = self.completed_states_buffer_by_episode[episode_id]
            self._save_episode_callback(buffer)

            del self.completed_states_buffer_by_episode[episode_id]

    def _auto_fill_end_state_locked(self, state_info: dict, episode_id: int, state_id: int) -> None:
        """Auto-fill an critical state labeled as "end." with multiple copies of its current position.

        MUST be called with self.state_lock already held.

        """
        # Direct access to joint positions and gripper in flattened structure
        joint_positions = state_info.get("joint_positions", {})
        gripper_action = state_info.get("gripper", 0)

        # Convert joint positions to action tensor (same as autolabel logic)
        goal_positions = []
        for joint_name in JOINT_NAMES:
            v = joint_positions.get(joint_name, 0.0)
            v = float(v[0]) if isinstance(v, (list, tuple)) and len(v) > 0 else float(v)
            goal_positions.append(v)
        # Set gripper position based on gripper action
        goal_positions[-1] = 0.044 if gripper_action > 0 else 0.0

        position_action = torch.tensor(goal_positions, dtype=torch.float32)

        state_info["actions"] = [position_action for _ in range(self.required_responses_per_critical_state)]
        all_actions = torch.cat(state_info["actions"][: self.required_responses_per_critical_state], dim=0)

        state_info["action_to_save"] = all_actions

        # Mark as approved since we're auto-filling with "End."
        state_info["approval_status"] = "approved"

        self.completed_states_buffer_by_episode[episode_id][state_id] = state_info
        self.completed_states_by_episode[episode_id][state_id] = state_info

        del self.pending_states_by_episode[episode_id][state_id]

        # Clear pending approval if this state was awaiting approval
        with self.approval_lock:
            if (
                self.pending_approval_state
                and self.pending_approval_state["episode_id"] == episode_id
                and self.pending_approval_state["state_id"] == state_id
            ):
                print(f"✅ Auto-approved state {state_id} (marked as end)")
                self.pending_approval_state = None

        if not self.pending_states_by_episode[episode_id]:
            self._schedule_episode_finalize_after_grace(episode_id)
