"""State Manager Module.

Manages episode-based state lifecycle for the crowd interface. Handles state creation, labeling, auto-labeling, and
episode finalization.

"""

import json
import queue
import random
import time
from pathlib import Path
from threading import Lock, Thread, Timer

import torch

from interface_managers.flush_manager import FlushManager
from interface_managers.async_user_logger import AsyncUserLogger

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
        asynchronous_mode: bool,
        async_admin_responses_per_state: int,
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
            asynchronous_mode: Whether async mode is enabled (admin collects, then users label)
            async_admin_responses_per_state: Responses needed from admin before robot executes in async mode
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
        self.asynchronous_mode = asynchronous_mode
        self.async_admin_responses_per_state = async_admin_responses_per_state
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
        self.episodes_marked_as_end = set()  # Episodes with "end" state - block new states
        self.next_state_id = 0
        
        # Episode timing tracking
        self.episode_start_times = {}  # episode_id -> Unix timestamp
        self.episode_start_times_iso = {}  # episode_id -> ISO format
        
        # Asynchronous mode state pool
        # When async mode finalized, states moved here for weighted dynamic sampling
        self.async_state_pool = {}  # (episode_id, state_id) -> state_info
        self.async_user_submissions = {}  # user_email -> set of (episode_id, state_id) tuples already submitted
        self.async_user_names = {}  # user_email -> user_name (lowercase) for test user detection
        self.async_user_current_state = {}  # user_email -> (episode_id, state_id) - state user is currently working on (prevents refresh shopping)
        self.async_pool_finalized = False  # True when admin phase complete and pool is ready
        
        # Special test user tracking
        # test_rejected: track rejections per state to enforce 10-rejection limit
        self.test_rejected_state_rejections = {}  # (episode_id, state_id) -> rejection_count
        
        # Warning suppression
        self._no_user_email_warning_shown = False

        # Episode finalization
        self.episode_finalize_grace_s = episode_finalize_grace_s
        self._episode_finalize_timers: dict[str, Timer] = {}

        # Auto-labeling
        self.auto_label_queue = queue.Queue()
        self.auto_label_worker_thread = None
        
        # Pre-approval worker thread and queue
        self.pre_approval_queue = queue.Queue()  # Queue of (state_info, episode_id, state_id)
        self.pre_approval_worker_thread = None
        
        # Flush manager (handles incremental dataset saves)
        self.flush_manager = None  # Initialized after callbacks are set

        # Manager dependencies
        self.obs_stream = obs_stream_manager
        self.pose_estimator = pose_estimation_manager
        self.drawer_position = drawer_position_manager
        self.sim_manager = sim_manager
        self.action_selector = action_selector_manager
        self.dataset_manager = dataset_manager
        
        # Async mode user logger (only active in async mode)
        self.async_user_logger = None
        if self.asynchronous_mode:
            # Initialize logger with dataset directory (if available) or obs cache root
            if self.dataset_manager is not None and hasattr(self.dataset_manager, 'dataset') and self.dataset_manager.dataset is not None:
                log_dir = self.dataset_manager.dataset.root
                print(f"✅ Async user logger using dataset root: {log_dir}")
            else:
                # In async mode, dataset_manager may be None - use obs_cache_root instead
                log_dir = self._obs_cache_root
                print(f"✅ Async user logger using obs_cache_root: {log_dir}")
            self.async_user_logger = AsyncUserLogger(log_dir)
            print(f"📊 Async logs will be written to: {log_dir}/async_user_submissions.jsonl")

        # Callbacks for external operations
        self._persist_views_callback = persist_views_callback
        self._persist_obs_callback = persist_obs_callback
        self._snapshot_views_callback = snapshot_views_callback
        self._save_episode_callback = save_episode_callback
        self._state_ready_callback = state_ready_callback  # NEW: MTurk HIT creation callback

        # Goal management
        self.latest_goal = None
        self.latest_goal_is_undo = False  # Track if latest_goal is from undo

        # Active episode tracking
        self._active_episode_id = None

        # Critical state approval tracking (post-execution)
        self.pending_approval_state = None  # {episode_id, state_id, approved: None/True/False}
        self.approval_lock = Lock()

        # Pre-execution approval tracking (before robot moves) - FIFO queue
        self.pre_execution_approval_queue = []  # Queue of approval requests
        self.pending_pre_execution_approval = None  # Currently active approval (being shown to admin)
        self.pre_execution_approval_lock = Lock()
        self._pre_execution_approval_sequence = 0  # Monotonic counter to detect stale requests

        # Undo state tracking
        self.pending_undo_classification = (
            None  # {episode_id, state_id, is_new_state: None/True/False, already_executed_actions: []}
        )
        self.undo_lock = Lock()
        self.undo_motion_start_state_id = None  # Track when undo motion begins to delete intermediate states

        # Start auto-labeling worker
        self._start_auto_label_worker()
        
        # Start pre-approval worker
        self._start_pre_approval_worker()
        
        # Initialize flush manager
        self.flush_manager = FlushManager(
            state_lock=self.state_lock,
            completed_states_buffer_by_episode=self.completed_states_buffer_by_episode,
            pending_states_by_episode=self.pending_states_by_episode,
            episodes_pending_save=self._episodes_pending_save,
            save_episode_callback=self._save_episode_callback,
            required_responses_per_critical_state=self.required_responses_per_critical_state,
            calculate_episode_timing_callback=self._calculate_episode_timing,
        )

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
        # Check if this episode has been marked as "end" - reject new states
        if episode_id in self.episodes_marked_as_end:
            print(f"🚫 Rejecting new state for episode {episode_id} (already marked as end)")
            return
        
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
            # Timing tracking
            "state_created_at": time.time(),
            "state_created_at_iso": None,  # Will be set below
            "state_completed_at": None,
            "state_completed_at_iso": None,
            "user_timings": {},  # email -> {"served_at": timestamp, "served_at_iso": str, "submitted_at": timestamp, "submitted_at_iso": str, "duration_seconds": float}
        }
        
        # Set ISO timestamp
        import datetime
        state_info["state_created_at_iso"] = datetime.datetime.now().isoformat()

        with self.state_lock:
            # Initialize episode containers if needed
            if episode_id not in self.pending_states_by_episode:
                self.pending_states_by_episode[episode_id] = {}
                self.completed_states_by_episode[episode_id] = {}
                
                # Track episode start time (first state)
                self.episode_start_times[episode_id] = time.time()
                self.episode_start_times_iso[episode_id] = datetime.datetime.now().isoformat()
                print(f"📅 Episode {episode_id} started at {self.episode_start_times_iso[episode_id]}")

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
                
                # Find previous APPROVED critical state (not just any critical state)
                # We only want to compare against approved states to avoid comparing against
                # intermediate jitter states that were marked critical but never approved
                previous_approved_critical_states = [
                    (sid, sinfo)
                    for sid, sinfo in sorted(all_states.items())
                    if sid < latest_state_id 
                    and sinfo.get("critical", False)
                    and sinfo.get("approval_status") == "approved"
                ]
                
                if previous_approved_critical_states:
                    prev_state_id, prev_state_info = previous_approved_critical_states[-1]
                    
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
            # CRITICAL: Set approval_status ATOMICALLY with critical flag to protect from races
            # This must happen BEFORE calling demote_earlier_unanswered_criticals so the
            # protection logic at line 1725 works correctly
            info["approval_status"] = "pending"
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
                # New state - the arrival state shouldn't need approval since it's just the undo target
                # The NEXT state collected after this will need approval
                print(f"⏩ Classified as new state - auto-approving undo arrival state, next state will need approval")
                approved = True
                # Clear the pending approval status since we're auto-approving
                with self.approval_lock:
                    self.pending_approval_state = None
        else:
            approved = None  # Will be set in approval phase below

        # ---- Phase 1.5: Wait for administrator approval ----
        if approved is None:  # Only do approval if not already handled by undo classification
            # approval_status was already set to "pending" atomically with critical=True above
            # to protect from races in demote_earlier_unanswered_criticals
            
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
                    # Check if approval was set FIRST (before checking demotion)
                    # This prevents race where approval is set, but undo clears pending_approval_state
                    # before we wake up from sleep
                    if self.pending_approval_state and self.pending_approval_state["approved"] is not None:
                        approved = self.pending_approval_state["approved"]
                        self.pending_approval_state = None
                        break
                    
                    if self.pending_approval_state is None:
                        # State was demoted, exit
                        print(f"⚠️  State {latest_state_id} was demoted, canceling approval")
                        return

        if not approved:
            # Administrator rejected - perform undo
            print(f"❌ Administrator rejected state {latest_state_id}, performing undo")
            self.undo_to_previous_critical_state()
            return

        print(f"✅ Administrator approved state {latest_state_id}, proceeding...")
        
        # Mark state as approved so it can be served to users
        with self.state_lock:
            ep = self.pending_states_by_episode.get(latest_episode_id)
            if ep and latest_state_id in ep:
                ep[latest_state_id]["approval_status"] = "approved"
        
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
                    
                    # Mark the first currently_executing action as executed
                    # (this is the action that got us to the current state)
                    exec_history = prev_state_info.get("execution_history", [])
                    for entry in exec_history:
                        if entry.get("currently_executing", False):
                            entry["executed"] = True
                            entry["currently_executing"] = False  # Clear the flag
                            print(f"✅ Marked action as executed in previous state {prev_critical_state_id}'s execution_history")
                            break
                    
                    # Find the executed action for final_executed_action
                    # In async mode: first entry (admin's action) is the one executed
                    # In sync mode: look for post_execution_approved flag
                    if self.asynchronous_mode:
                        # Async mode: admin's action (first entry) was executed
                        if exec_history and len(exec_history) > 0:
                            executed_action = exec_history[0]["action"]
                            prev_state_info["final_executed_action"] = executed_action.tolist() if hasattr(executed_action, "tolist") else list(executed_action)
                            print(f"✅ Recorded final_executed_action for PREVIOUS state {prev_critical_state_id} (admin action)")
                    else:
                        # Sync mode: find post-execution approved action
                        for entry in exec_history:
                            if entry.get("executed", False) and entry.get("post_execution_approved", False):
                                executed_action = entry["action"]
                                prev_state_info["final_executed_action"] = executed_action.tolist() if hasattr(executed_action, "tolist") else list(executed_action)
                                print(f"✅ Recorded final_executed_action for PREVIOUS state {prev_critical_state_id}")
                                break
            
            # Also mark in completed_states_by_episode (same state object reference, but just to be safe)
            ep2 = self.completed_states_by_episode.get(latest_episode_id, {})
            if ep2 and len(critical_states) > 0:
                prev_critical_state_id = critical_states[-1]
                if prev_critical_state_id in ep2:
                    prev_state_info2 = ep2[prev_critical_state_id]
                    exec_history2 = prev_state_info2.get("execution_history", [])
                    for entry in exec_history2:
                        if entry.get("currently_executing", False):
                            entry["executed"] = True
                            entry["currently_executing"] = False
                            break

        # ---- Phase 2: enqueue pose jobs and BLOCK until all are reported ----
        # Re-fetch info from state_lock after approval wait
        with self.state_lock:
            ep = self.pending_states_by_episode.get(latest_episode_id)
            if not ep or latest_state_id not in ep:
                print(f"⚠️  State {latest_state_id} was removed during approval")
                return
            info = ep[latest_state_id]

        # Skip pose estimation if flag is set (poses already copied from last critical state)
        if info.get("skip_pose_estimation", False):
            print(f"⏭️  Skipping pose estimation (reusing poses from previous critical state)")
            poses_ready = True
        else:
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

    def get_latest_state(self, user_email: str = None) -> dict:
        """Get a pending state from current serving episode. 
        
        Only serves states that have been approved by the monitor admin.
        This ensures jitter states don't affect what users see.
        
        In async mode, serves random states from the finalized pool.
        
        Args:
            user_email: Email of user requesting state (for timing tracking)
        """
        
        # ASYNC MODE: Check if pool is finalized and if user is admin
        if self.asynchronous_mode:
            # Detect admin by checking if request comes from localhost (no X-Forwarded-For)
            from flask import request as flask_request
            is_admin = not flask_request.headers.get('X-Forwarded-For') if flask_request else False
            
            if not self.async_pool_finalized:
                if not is_admin:
                    # Non-admin users cannot access during admin phase
                    return {
                        "status": "admin_phase",
                        "message": "Data collection in progress. Please wait for labeling phase to begin."
                    }
            else:
                # Pool is finalized - serve from async pool
                # Check if user already has an assigned state (prevent refresh shopping)
                state_info = None
                
                with self.state_lock:
                    if user_email and user_email in self.async_user_current_state:
                        current_state_key = self.async_user_current_state[user_email]
                        if current_state_key in self.async_state_pool:
                            candidate_state = self.async_state_pool[current_state_key]
                            # Check if state still needs labels (might have filled while user was working)
                            num_approved = sum(1 for entry in candidate_state.get("execution_history", []) 
                                              if entry.get("approval") == 1)
                            if num_approved < self.required_responses_per_critical_state:
                                state_info = candidate_state
                            else:
                                # State is full - clear assignment so user gets a new state
                                del self.async_user_current_state[user_email]
                        else:
                            # State no longer in pool - clear stale assignment
                            del self.async_user_current_state[user_email]
                
                # If no existing assignment, get a new state
                if state_info is None:
                    state_info = self.get_async_pooled_state(user_email)
                
                if state_info:
                    # Track timing for async served state
                    # ALWAYS update served_at on every fetch (including refreshes)
                    # This starts the timer from when user realizes they can't skip and must work on this state
                    if user_email:
                        import datetime
                        if "user_timings" not in state_info:
                            state_info["user_timings"] = {}
                        
                        now = time.time()
                        now_iso = datetime.datetime.now().isoformat()
                        state_info["user_timings"][user_email] = {
                            "served_at": now,
                            "served_at_iso": now_iso,
                            "submitted_at": None,
                            "submitted_at_iso": None,
                            "duration_seconds": None,
                        }
                    
                    return state_info.copy()
                else:
                    # User has labeled all available states
                    return {"status": "no_pending_states", "message": "You have labeled all available states"}
        
        # SYNC MODE or ASYNC ADMIN PHASE: Serve from pending_states (existing logic)
        with self.state_lock:
            episode_id = self.current_serving_episode
            
            if episode_id not in self.pending_states_by_episode:
                # No pending critical states left
                return {"status": "no_pending_states", "blocked_critical_states": False}
            
            # Find the latest APPROVED critical state
            pending_states = self.pending_states_by_episode[episode_id]
            
            # In async mode during admin phase, exclude admin-complete states (already labeled by admin)
            if self.asynchronous_mode and not self.async_pool_finalized:
                from flask import request as flask_request
                is_admin = not flask_request.headers.get('X-Forwarded-For') if flask_request else False
                
                if is_admin:
                    # Admin should skip admin-complete states and move to next unlabeled state
                    approved_critical_states = [
                        (state_id, state_info)
                        for state_id, state_info in pending_states.items()
                        if state_info.get("critical", False) 
                        and state_info.get("approval_status") == "approved"
                        and not state_info.get("admin_complete", False)  # Skip states already labeled by admin
                    ]
                else:
                    # Non-admin (shouldn't happen in admin phase, but fallback)
                    approved_critical_states = [
                        (state_id, state_info)
                        for state_id, state_info in pending_states.items()
                        if state_info.get("critical", False) and state_info.get("approval_status") == "approved"
                    ]
            else:
                # Sync mode or async user phase - show all approved states
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
            
            # Track timing for this user - ALWAYS update to current time
            # This ensures duration measures from most recent fetch, not first encounter
            if user_email:
                import datetime
                if "user_timings" not in state_info:
                    state_info["user_timings"] = {}
                
                now = time.time()
                now_iso = datetime.datetime.now().isoformat()
                state_info["user_timings"][user_email] = {
                    "served_at": now,
                    "served_at_iso": now_iso,
                    "submitted_at": None,
                    "submitted_at_iso": None,
                    "duration_seconds": None,
                }
            else:
                # Only show warning once per session (to avoid spam from monitor.html polling)
                if not self._no_user_email_warning_shown:
                    print(f"⚠️  No user_email provided for get_latest_state() - timing not tracked (suppressing further warnings)")
                    self._no_user_email_warning_shown = True

            # Return the latest approved state for labeling
            return state_info.copy()

    def record_response(self, response_data: dict):
        """Record a response for a specific state.

        Handles all the side-effects.

        """
        should_run_pre_approval = False
        state_info_copy = None
        should_run_pre_approval_now = False
        single_action_state_copy = None
        
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

            # Determine required responses based on mode and user type
            # Track if this is from admin (will be set by caller in flask_app.py)
            is_admin_submission = response_data.get("is_admin", False)
            
            # Check if this is a gripper-only action (no position change)
            joint_positions = response_data["joint_positions"]
            gripper_action = response_data["gripper"]
            
            # Define home position: [0, 60°, 75°, -60°, 0°, 0°, 2°] in radians
            import math
            HOME_POSITION_DEG = [0, 60, 75, -60, 0, 0, 2]
            HOME_POSITION_RAD = [deg * math.pi / 180.0 for deg in HOME_POSITION_DEG]
            
            # Detect gripper-only: compare with previous state's joint positions
            is_gripper_only = False
            is_home_position = False
            is_manual_autofill = False
            
            if self.asynchronous_mode and is_admin_submission and state_info["critical"]:
                # Check if admin manually requested autofill
                is_manual_autofill = response_data.get("autofill_requested", False)
                if is_manual_autofill:
                    print(f"⚡ Manual autofill requested for state {state_id} - auto-filling all slots")
                
                # Get previous joint positions from state_info
                current_joint_positions = state_info.get("joint_positions", {})
                # Check if all joint positions are the same (only gripper changed)
                position_changed = False
                for joint_name in JOINT_NAMES[:-1]:  # Exclude gripper (last joint)
                    current_val = current_joint_positions.get(joint_name, 0.0)
                    submitted_val = joint_positions.get(joint_name, [0.0])[0] if isinstance(joint_positions.get(joint_name), list) else joint_positions.get(joint_name, 0.0)
                    if abs(float(submitted_val) - float(current_val)) > 0.001:  # 0.001 radian threshold
                        position_changed = True
                        break
                
                is_gripper_only = not position_changed
                if is_gripper_only:
                    print(f"🤏 Gripper-only action detected for state {state_id} - auto-filling all slots")
                
                # Check if submitted action matches home position exactly
                home_match = True
                for i, joint_name in enumerate(JOINT_NAMES[:-1]):  # Exclude gripper
                    submitted_val = joint_positions.get(joint_name, [0.0])[0] if isinstance(joint_positions.get(joint_name), list) else joint_positions.get(joint_name, 0.0)
                    if abs(float(submitted_val) - HOME_POSITION_RAD[i]) > 0.001:  # 0.001 radian threshold
                        home_match = False
                        break
                
                is_home_position = home_match
                if is_home_position:
                    print(f"🏠 Home position action detected for state {state_id} - auto-filling all slots")
            
            if self.asynchronous_mode and state_info["critical"] and is_admin_submission:
                if is_gripper_only or is_home_position or is_manual_autofill:
                    # Gripper-only, home position, or manual autofill: instantly fill all slots
                    required_responses = self.required_responses_per_critical_state
                else:
                    # Normal async mode admin: only need async_admin_responses_per_state responses before executing
                    required_responses = self.async_admin_responses_per_state
            else:
                # Sync mode OR async mode user responses: use full requirement
                required_responses = (
                    self.required_responses_per_critical_state
                    if state_info["critical"]
                    else self.required_responses_per_state
                )

            state_info["responses_received"] += 1

            goal_positions = []
            for joint_name in JOINT_NAMES:
                joint_value = joint_positions[joint_name]
                goal_positions.append(float(joint_value[0]))

            goal_positions[-1] = 0.044 if gripper_action > 0 else 0.0
            goal_positions = torch.tensor(goal_positions, dtype=torch.float32)
            state_info["actions"].append(goal_positions)
            
            # Track user identity for this submission
            user_name = response_data.get("user_name")
            user_email = response_data.get("user_email")
            used_animation = response_data.get("used_animation", False)  # Track if user clicked animation
            ip_address = response_data.get("ip_address")  # Track IP address
            
            if "user_submissions" not in state_info:
                state_info["user_submissions"] = []
            state_info["user_submissions"].append({
                "name": user_name,
                "email": user_email,
                "action_index": len(state_info["actions"]) - 1,  # Index of the action just appended
                "used_animation": used_animation,  # NEW: Track animation usage
                "ip_address": ip_address,  # NEW: Track IP address
            })
            
            # Track submission timing for this user
            import datetime
            now = time.time()
            now_iso = datetime.datetime.now().isoformat()
            
            if user_email:
                # Ensure user_timings dict exists
                if "user_timings" not in state_info:
                    state_info["user_timings"] = {}
                
                # If user wasn't tracked before (missed get_latest_state call), initialize now
                if user_email not in state_info["user_timings"]:
                    state_info["user_timings"][user_email] = {
                        "served_at": None,  # Unknown when they were served this state
                        "served_at_iso": None,
                        "submitted_at": now,
                        "submitted_at_iso": now_iso,
                        "duration_seconds": None,  # Can't calculate without start time
                    }
                    print(f"⚠️  User {user_name} ({user_email}) timing started at submission (no get_latest_state call)")
                else:
                    # User was tracked - update submission time and calculate duration
                    timing = state_info["user_timings"][user_email]
                    timing["submitted_at"] = now
                    timing["submitted_at_iso"] = now_iso
                    if timing["served_at"]:
                        timing["duration_seconds"] = now - timing["served_at"]
                
                # Track submission in async mode (mark as submitted so they can't submit again)
                if self.asynchronous_mode and self.async_pool_finalized:
                    state_key = (episode_id, state_id)
                    if user_email in self.async_user_submissions:
                        self.async_user_submissions[user_email].add(state_key)
                    # Clear current state assignment - user can now get a new state
                    if user_email in self.async_user_current_state:
                        del self.async_user_current_state[user_email]
            
            # Track actual number of unique worker submissions (not including autofill)
            if "actual_num_submissions" not in state_info:
                state_info["actual_num_submissions"] = 0
            state_info["actual_num_submissions"] += 1
            
            # In async mode: trigger immediate pre-approval for each user submission
            should_run_pre_approval_now = False
            single_action_state_copy = None
            if state_info.get("critical") and self.asynchronous_mode and self.async_pool_finalized and not is_admin_submission:
                should_run_pre_approval_now = True
                # Create a copy with only the last action for immediate review
                single_action_state_copy = state_info.copy()
                single_action_state_copy["actions"] = [state_info["actions"][-1].clone()]  # Deep copy the tensor
                last_submission = state_info["user_submissions"][-1].copy()
                last_submission["action_index"] = 0
                single_action_state_copy["user_submissions"] = [last_submission]

            # Autofill
            if state_info["critical"] and self.autofill_critical_states:
                remaining = state_info["responses_received"]
                clones_to_add = min(self.num_autofill_actions - 1, remaining)
                for _ in range(clones_to_add):
                    state_info["actions"].append(goal_positions.clone())
                    # Add execution history entry for monitor progress bar
                    if "execution_history" not in state_info:
                        state_info["execution_history"] = []
                    state_info["execution_history"].append({
                        "action": goal_positions.clone(),
                        "propensity": 1.0,
                        "approval": 1,  # Auto-approved for display
                        "executed": False,
                        "submitted_by": [{"email": "Auto-filled", "user_id": "autofill"}]
                    })
                state_info["responses_received"] += clones_to_add
            
            # Additional autofill for gripper-only or home position actions in async mode
            if (is_gripper_only or is_home_position or is_manual_autofill) and self.asynchronous_mode and is_admin_submission:
                # Fill remaining slots with the same action
                clones_needed = required_responses - state_info["responses_received"]
                for _ in range(clones_needed):
                    state_info["actions"].append(goal_positions.clone())
                    # Add execution history entry for monitor progress bar
                    if "execution_history" not in state_info:
                        state_info["execution_history"] = []
                    state_info["execution_history"].append({
                        "action": goal_positions.clone(),
                        "propensity": 1.0,
                        "approval": 1,  # Auto-approved for display
                        "executed": False,
                        "submitted_by": [{"email": "Auto-filled", "user_id": "autofill"}]
                    })
                state_info["responses_received"] += clones_needed
                if is_manual_autofill:
                    action_type = "manual autofill"
                elif is_gripper_only:
                    action_type = "gripper-only"
                else:
                    action_type = "home position"
                print(f"   Auto-filled {clones_needed} more slots for {action_type} action")
                
                # Mark so it won't be added to async pool
                if is_gripper_only:
                    state_info["gripper_only_autofilled"] = True
                if is_home_position:
                    state_info["home_position_autofilled"] = True
                if is_manual_autofill:
                    state_info["manual_autofilled"] = True

            # Handle completion
            should_check_finalization = False
            if state_info["responses_received"] >= required_responses:
                # Record state completion time (when all actions received)
                import datetime
                state_info["state_completed_at"] = time.time()
                state_info["state_completed_at_iso"] = datetime.datetime.now().isoformat()
                
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

                # Determine if pre-approval should run for critical states
                should_run_pre_approval = False
                state_info_copy = None
                if state_info["critical"]:
                    if self.asynchronous_mode and is_admin_submission:
                        # Auto-approve admin submission in async mode - skip pre-approval
                        print(f"✅ Auto-approving admin submission for state {state_id} (async mode)")
                        should_run_pre_approval = False
                        # Mark as pre-approved and ready for post-approval (critical state approval)
                        # Initialize execution_history if not already present (from autofill)
                        if "execution_history" not in state_info:
                            state_info["execution_history"] = []
                        # Add the admin's action as first entry (or insert at beginning if autofills exist)
                        admin_entry = {
                            "action": state_info["actions"][0],
                            "propensity": 1.0,
                            "approval": 1,  # Auto-approved
                            "executed": False,
                            "submitted_by": state_info.get("user_submissions", [])[:1]  # First submission (admin)
                        }
                        # Insert admin entry at the beginning (before any autofills)
                        state_info["execution_history"].insert(0, admin_entry)
                        state_info["pre_approval_loop_complete"] = True
                    elif self.asynchronous_mode and self.async_pool_finalized and not is_admin_submission:
                        # Handled by immediate per-action pre-approval above
                        should_run_pre_approval = False
                    else:
                        # Sync mode or async admin phase: normal pre-approval
                        should_run_pre_approval = True
                        state_info_copy = state_info.copy()

                # In async mode: check if we have enough APPROVED actions
                if self.asynchronous_mode and state_info.get("critical"):
                    # Immediate per-action review handles approval
                    # Check if enough APPROVED after each review completes
                    num_approved = sum(1 for entry in state_info.get("execution_history", []) 
                                      if entry.get("approval") == 1)
                    fully_complete = num_approved >= self.required_responses_per_critical_state
                else:
                    # Sync mode or non-critical: complete when we have enough total responses
                    fully_complete = state_info["responses_received"] >= self.required_responses_per_critical_state
                
                if fully_complete:
                    # Fully labeled - save to completed states buffer
                    if episode_id not in self.completed_states_buffer_by_episode:
                        self.completed_states_buffer_by_episode[episode_id] = {}
                    self.completed_states_buffer_by_episode[episode_id][state_id] = state_info

                    # Save to completed states (for monitoring)
                    if episode_id not in self.completed_states_by_episode:
                        self.completed_states_by_episode[episode_id] = {}
                    self.completed_states_by_episode[episode_id][state_id] = state_info
                    
                    # Set final_executed_action to admin's action (first in execution_history)
                    if state_info.get("final_executed_action") is None:
                        if state_info.get("execution_history") and len(state_info["execution_history"]) > 0:
                            admin_action = state_info["execution_history"][0]["action"]
                            state_info["final_executed_action"] = admin_action.tolist() if hasattr(admin_action, "tolist") else list(admin_action)
                    
                    # Queue pre-approval request (single worker will process it)
                    if state_info.get("critical") and should_run_pre_approval and state_info_copy:
                        self.pre_approval_queue.put((state_info_copy, episode_id, state_id))

                    # Remove from pending (this triggers episode save when all pending are done)
                    del self.pending_states_by_episode[episode_id][state_id]
                    
                    # DON'T schedule finalization here - wait until after immediate pre-approval threads start
                    # to avoid race condition where finalization fires before threads add to queue
                    should_check_finalization = True
                else:
                    # Admin-complete in async mode - mark as ready for async serving but keep in pending
                    state_info["admin_complete"] = True
                    state_info["awaiting_user_labels"] = True
                    
                    # Add to BOTH completed_states and completed_states_buffer
                    if episode_id not in self.completed_states_by_episode:
                        self.completed_states_by_episode[episode_id] = {}
                    self.completed_states_by_episode[episode_id][state_id] = state_info
                    
                    if episode_id not in self.completed_states_buffer_by_episode:
                        self.completed_states_buffer_by_episode[episode_id] = {}
                    self.completed_states_buffer_by_episode[episode_id][state_id] = state_info
                    
                    # Set final_executed_action to admin's action ONCE (first in execution_history)
                    # Only set if not already set (avoid redundant setting on every user submission)
                    if state_info.get("final_executed_action") is None:
                        if state_info.get("execution_history") and len(state_info["execution_history"]) > 0:
                            admin_action = state_info["execution_history"][0]["action"]
                            state_info["final_executed_action"] = admin_action.tolist() if hasattr(admin_action, "tolist") else list(admin_action)
                    
                    # DON'T delete from pending - keep it there for user labeling
                    # DON'T set latest_goal - robot doesn't move in async user labeling phase

        # Check finalization
        if should_check_finalization:
            with self.state_lock:
                remaining_pending = len(self.pending_states_by_episode.get(episode_id, {}))
        
        # Queue pre-approval request (worker thread will process it)
        if should_run_pre_approval_now and single_action_state_copy:
            self.pre_approval_queue.put((single_action_state_copy, episode_id, state_id))
        
        # Check if episode is now empty and handle finalization
        if should_check_finalization:
            with self.state_lock:
                if episode_id in self.pending_states_by_episode and not self.pending_states_by_episode[episode_id]:
                    # Check if there are any pre-approvals queued or active for this episode
                    with self.pre_execution_approval_lock:
                        has_pending = (
                            self.pending_pre_execution_approval is not None and
                            self.pending_pre_execution_approval.get("episode_id") == episode_id
                        )
                        has_queued = any(
                            req.get("episode_id") == episode_id 
                            for req in self.pre_execution_approval_queue
                        )
                    
                    if not has_pending and not has_queued:
                        print(f"📦 Episode {episode_id} complete - scheduling finalization (grace: {self.episode_finalize_grace_s}s)")
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
                        is_critical = info.get("critical", False)
                        required_responses = (
                            self.required_responses_per_critical_state
                            if is_critical
                            else self.required_responses_per_state
                        )
                        _txt = info.get("text_prompt")  # Updated field name
                        has_flex_text = bool(str(_txt or "").strip())
                        _vid = info.get("video_prompt")  # Updated field name
                        has_flex_video = _vid is not None
                        
                        # For critical states, count only approved submissions
                        # For non-critical, use total responses_received
                        if is_critical:
                            num_approved = sum(1 for entry in info.get("execution_history", []) 
                                             if entry.get("approval") == 1)
                            responses_received = num_approved
                        else:
                            responses_received = info["responses_received"]

                        episode_states[state_id] = {
                            "responses_received": responses_received,
                            "responses_needed": required_responses - responses_received,
                            "critical": bool(is_critical),
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
                        is_critical = info.get("critical", False)
                        _txt = info.get("text_prompt")  # Updated field name
                        has_flex_text = bool(str(_txt or "").strip())
                        _vid = info.get("video_prompt")  # Updated field name
                        has_flex_video = _vid is not None
                        
                        # For critical states, count only approved submissions
                        # For non-critical, use total responses_received
                        if is_critical:
                            num_approved = sum(1 for entry in info.get("execution_history", []) 
                                             if entry.get("approval") == 1)
                            responses_received = num_approved
                        else:
                            responses_received = info["responses_received"]

                        episode_states[state_id] = {
                            "responses_received": responses_received,
                            "responses_needed": 0,  # Completed
                            "critical": bool(is_critical),
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
        # PRIORITY: Check if there's a queued goal (from undo or admin immediate execution)
        if self.latest_goal is not None:
            goal_tensor = self.latest_goal
            is_undo = self.latest_goal_is_undo
            self.latest_goal = None  # Clear after consuming
            self.latest_goal_is_undo = False  # Reset flag
            
            # Convert to list
            action_list = goal_tensor.tolist() if hasattr(goal_tensor, "tolist") else list(goal_tensor)
            if is_undo:
                print(f"↩️  Executing undo motion to previous critical state")
            else:
                print(f"🤖 Executing admin action (immediate execution mode)")
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
            
            # Find first non-executed approved action that's not currently being executed
            # Mark as "currently_executing" to prevent returning it again until robot arrives
            for entry in exec_history:
                if entry.get("approval") == 1 and not entry.get("executed", False) and not entry.get("currently_executing", False):
                    # Mark as currently executing (will be marked as executed when robot arrives)
                    entry["currently_executing"] = True
                    selected_action = entry["action"]
                    
                    # Convert to list if tensor
                    action_list = selected_action.tolist() if hasattr(selected_action, "tolist") else list(selected_action)
                    
                    print(f"✅ Returning pre-approved action for state {latest_critical_state_id} (will mark executed when robot arrives)")
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

    def approve_critical_state(self, episode_id: int, state_id: int, skip_pose_estimation: bool = False) -> bool:
        """Approve a pending critical state (post-execution approval).
        
        Args:
            episode_id: Episode ID
            state_id: State ID  
            skip_pose_estimation: If True, reuse object poses from last critical state instead of running pose estimation
            
        Returns:
            bool: True if approval was successful
        """
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
                        
                        # Handle pose estimation skip
                        if skip_pose_estimation:
                            # Find last critical state and copy its object poses
                            # Check pending, completed_buffer, and completed states (for async mode)
                            critical_states = []
                            
                            # Check pending_states_by_episode (async mode - approved but not submitted yet)
                            ep_pending = self.pending_states_by_episode.get(episode_id, {})
                            for sid, sinfo in ep_pending.items():
                                if sinfo.get("critical", False) and sid < state_id:
                                    critical_states.append((sid, sinfo))
                            
                            # Check completed_states_buffer_by_episode (buffered states)
                            ep_buffer = self.completed_states_buffer_by_episode.get(episode_id, {})
                            for sid, sinfo in ep_buffer.items():
                                if sinfo.get("critical", False):
                                    critical_states.append((sid, sinfo))
                            
                            # Check completed_states_by_episode (fully completed states)
                            ep_completed = self.completed_states_by_episode.get(episode_id, {})
                            for sid, sinfo in ep_completed.items():
                                if sinfo.get("critical", False):
                                    critical_states.append((sid, sinfo))
                            
                            if critical_states:
                                # Sort by state_id and get the most recent
                                critical_states.sort(key=lambda x: x[0])
                                last_critical_state_id, last_critical_info = critical_states[-1]
                                
                                # Copy object poses from last critical state
                                if "object_poses" in last_critical_info:
                                    state_info["object_poses"] = last_critical_info["object_poses"].copy()
                                    state_info["skip_pose_estimation"] = True
                                    print(f"✅ Reusing object poses from state {last_critical_state_id} (skipped pose estimation)")
                                else:
                                    print(f"⚠️  Last critical state {last_critical_state_id} has no object_poses - will run pose estimation")
                            else:
                                print(f"⚠️  No previous critical state found - will run pose estimation")
                        
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
                # Approval decision already made - return None and clear stale data
                return None

            # Return a copy with sequence number for frontend deduplication
            return {
                "episode_id": self.pending_pre_execution_approval["episode_id"],
                "state_id": self.pending_pre_execution_approval["state_id"],
                "action": self.pending_pre_execution_approval["action"],
                "obs_path": self.pending_pre_execution_approval["obs_path"],
                "view_paths": self.pending_pre_execution_approval["view_paths"],
                "sequence": self.pending_pre_execution_approval["sequence"],
                "text_prompt": self.pending_pre_execution_approval.get("text_prompt"),
                "video_prompt": self.pending_pre_execution_approval.get("video_prompt"),
                "submitted_by": self.pending_pre_execution_approval.get("submitted_by", []),
                "original_joint_positions": self.pending_pre_execution_approval.get("original_joint_positions", []),
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

            # Atomically mark as approved - backend loop will detect and clear
            self.pending_pre_execution_approval["approved"] = True
            pass  # Approved
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

            # Atomically mark as rejected - backend loop will detect and clear
            self.pending_pre_execution_approval["approved"] = False
            pass  # Rejected
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
            self.latest_goal_is_undo = True  # Mark as undo action

        # ---- Set up pending undo classification (will be triggered after robot arrives) ----
        # Store the previous critical state info for later classification
        # Build list of all actions that have been ACTUALLY EXECUTED (not just pre-approved)
        actually_executed_actions = []
        execution_history = previous_critical_state_info.get("execution_history", [])
        for execution in execution_history:
            if execution.get("executed", False):
                actually_executed_actions.append(execution["action"])
        
        with self.undo_lock:
            self.pending_undo_classification = {
                "episode_id": latest_episode_id,
                "state_id": previous_critical_state_id,
                "is_new_state": None,  # None=pending, True=new state, False=old state
                "already_executed_actions": actually_executed_actions,  # Only actions that were actually executed (not just pre-approved)
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
                
                # Debug logging
                print(f"🔍 DEBUG: Total submitted actions: {len(all_actions)}")
                print(f"🔍 DEBUG: Available actions (first {required_responses}): {len(available_actions)}")
                print(f"🔍 DEBUG: Already executed actions: {len(already_executed)}")
                exec_history = state_info.get("execution_history", [])
                print(f"🔍 DEBUG: Execution history entries: {len(exec_history)}")
                for i, entry in enumerate(exec_history):
                    print(f"   Entry {i}: executed={entry.get('executed', False)}, approval={entry.get('approval', 'N/A')}")

                if not available_actions:
                    print(f"⚠️  No actions available to resample from")
                    return None

                # Filter out already executed actions (keep duplicates for now)
                remaining_actions_with_dupes = [
                    action
                    for action in available_actions
                    if not any(torch.equal(action, executed) for executed in already_executed)
                ]
                
                print(f"🔍 DEBUG: Remaining actions after filtering: {len(remaining_actions_with_dupes)}")

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
                    "num_remaining_actions": len(remaining_unique),
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
    
    def clear_episode(self, episode_id: int) -> None:
        """Clear all crowd interface state for an episode (for re-recording).
        
        This removes:
        - All pending/completed/buffered states
        - Observation cache files
        - Async pool entries (if any)
        - Pending approvals
        - Episode tracking metadata
        - Running threads/timers
        
        Args:
            episode_id: Episode to clear
        """
        import shutil
        
        with self.state_lock:
            # 1. Delete states from all dictionaries
            if episode_id in self.pending_states_by_episode:
                for state_info in self.pending_states_by_episode[episode_id].values():
                    self._delete_obs_from_disk(state_info.get("obs_path"))
                del self.pending_states_by_episode[episode_id]
                print(f"🗑️  Cleared pending states for episode {episode_id}")
            
            if episode_id in self.completed_states_by_episode:
                for state_info in self.completed_states_by_episode[episode_id].values():
                    self._delete_obs_from_disk(state_info.get("obs_path"))
                del self.completed_states_by_episode[episode_id]
                print(f"🗑️  Cleared completed states for episode {episode_id}")
            
            if episode_id in self.completed_states_buffer_by_episode:
                del self.completed_states_buffer_by_episode[episode_id]
                print(f"🗑️  Cleared completed states buffer for episode {episode_id}")
            
            # 2. Clear async pool entries for this episode
            if self.asynchronous_mode:
                keys_to_remove = [key for key in self.async_state_pool.keys() if key[0] == episode_id]
                for key in keys_to_remove:
                    del self.async_state_pool[key]
                if keys_to_remove:
                    print(f"🗑️  Removed {len(keys_to_remove)} states from async pool")
                
                # Remove from all users' submissions
                for user_email in self.async_user_submissions:
                    before_count = len(self.async_user_submissions[user_email])
                    self.async_user_submissions[user_email] = {
                        k for k in self.async_user_submissions[user_email] if k[0] != episode_id
                    }
                    after_count = len(self.async_user_submissions[user_email])
                    if before_count != after_count:
                        print(f"🗑️  Cleared {before_count - after_count} submissions for {user_email}")
            
            # 3. Clear episode tracking
            self.episodes_completed.discard(episode_id)
            self.episodes_marked_as_end.discard(episode_id)
            self.episode_start_times.pop(episode_id, None)
            self.episode_start_times_iso.pop(episode_id, None)
            
            # 4. Reset current serving episode if it's this one
            if self.current_serving_episode == episode_id:
                self.current_serving_episode = None
            
            # 5. Reset next_state_id to 0 for fresh start
            self.next_state_id = 0
            print(f"🔄 Reset next_state_id to 0")
        
        # 6. Clear pending approvals (separate lock)
        with self.approval_lock:
            if (self.pending_approval_state and 
                self.pending_approval_state["episode_id"] == episode_id):
                self.pending_approval_state = None
                print(f"🗑️  Cleared pending approval for episode {episode_id}")
        
        # 7. Clear pre-execution approvals
        with self.pre_execution_approval_lock:
            if (self.pending_pre_execution_approval and
                self.pending_pre_execution_approval.get("episode_id") == episode_id):
                self.pending_pre_execution_approval = None
                print(f"🗑️  Cleared pending pre-execution approval")
            
            # Remove from queue
            queue_before = len(self.pre_execution_approval_queue)
            self.pre_execution_approval_queue = [
                req for req in self.pre_execution_approval_queue
                if req.get("episode_id") != episode_id
            ]
            queue_after = len(self.pre_execution_approval_queue)
            if queue_before != queue_after:
                print(f"🗑️  Removed {queue_before - queue_after} items from pre-execution approval queue")
        
        # 8. Cancel episode finalization timer
        timer = self._episode_finalize_timers.pop(episode_id, None)
        if timer:
            timer.cancel()
            print(f"⏹️  Cancelled finalization timer for episode {episode_id}")
        
        # 9. Clear pre-approval queue for this episode
        # (Worker thread will skip items when it sees state is gone)
        
        # 10. Delete entire observation cache directory for episode
        obs_dir = self._obs_cache_root / str(episode_id)
        if obs_dir.exists():
            shutil.rmtree(obs_dir)
            print(f"🗑️  Deleted observation cache directory: {obs_dir}")
        
        print(f"🧹 Cleared all crowd interface state for episode {episode_id}")

    def set_prompt_ready(
        self, state_info: dict, episode_id: int, state_id: int, text: str | None, video_id: int | None
    ) -> None:
        """Set text/video prompt fields and mark as ready."""
        state_info["text_prompt"] = text  # Updated field name
        state_info["video_prompt"] = video_id  # Updated field name
        state_info["prompt_ready"] = True

        # Check if this is a critical state with "end" text - auto-fill with current position
        # Accept "End", "End.", "end", "end." etc.
        if text and text.strip().rstrip(".").lower() == "end":
            with self.state_lock:
                self._auto_fill_end_state_locked(state_info, episode_id, state_id)

    def _run_pre_approval_loop(self, state_info: dict, episode_id: int, state_id: int) -> None:
        """Phase 1: Sample actions one-by-one for pre-approval until stopping condition met.
        
        Stopping condition: (num_pre_approvals_completed >= required_approvals_per_critical_state) AND (at least 1 approved)
        
        Args:
            state_info: State dict with actions list (may be single_action_state_copy with only 1 action for immediate mode)
            episode_id: Episode ID
            state_id: State ID
        """
        required_responses = self.required_responses_per_critical_state
        # Use actions directly from passed state_info (may be filtered to single action in immediate mode)
        available_actions = state_info.get("actions", [])
        
        if not available_actions:
            print(f"⚠️  No actions available for pre-approval loop (state {state_id})")
            return
        
        # Check existing execution_history (e.g., admin auto-approved actions)
        with self.state_lock:
            existing_history = []
            for ep_dict in [self.completed_states_by_episode, self.completed_states_buffer_by_episode]:
                if episode_id in ep_dict and state_id in ep_dict[episode_id]:
                    existing_history = ep_dict[episode_id][state_id].get("execution_history", [])
                    break
        
        num_existing_approved = sum(1 for e in existing_history if e.get("approval") == 1)
            
        reviewed_actions = []
        num_approved = 0
        
        while True:
            total_approved = num_existing_approved + num_approved
            num_reviewed = len(reviewed_actions)
            
            if total_approved >= self.required_responses_per_critical_state:
                print(f"✅ Pre-approval loop complete: {num_reviewed} new reviewed, {num_approved} new approved, {total_approved} total approved")
                break
                
            # Check if we have more actions to sample
            # Filter out: 1) actions reviewed in this loop, 2) actions already in execution_history
            remaining_actions_with_dupes = [
                a for a in available_actions 
                if not any(torch.equal(a, r["action"]) for r in reviewed_actions)
                and not any(torch.equal(a, e["action"]) for e in existing_history)
            ]
            if not remaining_actions_with_dupes:
                break  # No more actions to review
            
            # Deduplicate remaining actions for selection
            remaining_unique = []
            for action in remaining_actions_with_dupes:
                if not any(torch.equal(action, unique_a) for unique_a in remaining_unique):
                    remaining_unique.append(action)
                
            # Select action: use submission order if responses == approvals, else use selector
            if self.required_responses_per_critical_state == self.required_approvals_per_critical_state:
                # When all submissions need approval, present in submission order
                selected_action = remaining_unique[0]
                # Still compute selector propensity for metadata
                _, selector_propensity, selection_metadata = self.action_selector.select_action(
                    remaining_unique, state_info
                )
                selection_metadata["selector_used"] = "submission_order"
            else:
                # When sampling subset for approval, use action selector
                selected_action, selector_propensity, selection_metadata = self.action_selector.select_action(
                    remaining_unique, state_info
                )
            
            # Compute true propensity as submission frequency
            # Propensity = (number of times this action appears) / (total number of actual submissions)
            # This is the correct importance weight for learning from crowd data
            actual_total_submissions = state_info.get("actual_num_submissions", len(available_actions))
            count_selected = sum(1 for a in available_actions if torch.equal(a, selected_action))
            true_propensity = count_selected / actual_total_submissions
            
            # Find which user(s) submitted this action
            user_submissions = state_info.get("user_submissions", [])
            action_users = []
            for i, action in enumerate(available_actions):
                if torch.equal(action, selected_action):
                    # Find user who submitted this action
                    for user_sub in user_submissions:
                        if user_sub["action_index"] == i:
                            action_users.append({
                                "name": user_sub["name"],
                                "email": user_sub["email"],
                                "used_animation": user_sub.get("used_animation", False),
                                "ip_address": user_sub.get("ip_address")
                            })
                            break
            
            # Get current robot joint positions
            original_joint_positions_list = list(state_info.get("joint_positions", {}).values())
            
            # Check for auto-approve/reject users by name (test_approved, test_rejected)
            user_names = [u.get("name", "").lower() for u in action_users]
            is_test_approved = "test_approved" in user_names
            is_test_rejected = "test_rejected" in user_names
            
            # Store user name mapping for test users
            if is_test_approved or is_test_rejected:
                user_name = next((u.get("name", "").lower() for u in action_users), "")
                user_email = next((u.get("email", "") for u in action_users), "")
                if user_name and user_email:
                    with self.state_lock:
                        self.async_user_names[user_email] = user_name
            
            # Auto-approve/reject test users (bypass monitor like localhost does)
            if is_test_approved or is_test_rejected:
                approved = is_test_approved
                print(f"{'✅' if approved else '❌'} Auto-{'approved' if approved else 'rejected'} test user (state {state_id})")
            else:
                # Normal approval flow: Set up pre-execution approval modal (blocking)
                with self.pre_execution_approval_lock:
                    self._pre_execution_approval_sequence += 1
                    approval_request = {
                        "episode_id": episode_id,
                        "state_id": state_id,
                        "action": selected_action.tolist(),
                        "propensity": true_propensity,
                        "selector_metadata": selection_metadata,
                        "obs_path": state_info.get("obs_path"),
                        "view_paths": state_info.get("view_paths"),
                        "approved": None,
                        "sequence": self._pre_execution_approval_sequence,
                        "submitted_by": action_users,  # List of users who submitted this action
                        "text_prompt": state_info.get("text_prompt"),
                        "video_prompt": state_info.get("video_prompt"),
                        "original_joint_positions": original_joint_positions_list,  # Starting position
                    }
                    
                    my_sequence = approval_request["sequence"]
                    
                    # If no approval is currently active, show this one immediately
                    if self.pending_pre_execution_approval is None:
                        self.pending_pre_execution_approval = approval_request
                    else:
                        # Queue it - it will be shown after current one is done
                        self.pre_execution_approval_queue.append(approval_request)
                
                # Block until THIS specific request is approved/rejected
                approved = None
                while approved is None:
                    time.sleep(0.1)
                    with self.pre_execution_approval_lock:
                        # Check if this is the active request and has been decided
                        if (self.pending_pre_execution_approval and 
                            self.pending_pre_execution_approval["sequence"] == my_sequence and
                            self.pending_pre_execution_approval["approved"] is not None):
                            approved = self.pending_pre_execution_approval["approved"]
                            break
                        
            # Record decision
            approval_value = 1 if approved else -1
            reviewed_actions.append({
                "action": selected_action,
                "propensity": true_propensity,  # Use propensity based on original submission counts
                "selector_metadata": selection_metadata,
                "approval": approval_value,
                "submitted_by": action_users,  # Track who submitted this action
            })
            
            if approved:
                num_approved += 1
            else:
                # Handle rejection: give users another chance by re-inserting state in their queue
                state_key = (episode_id, state_id)
                for user in action_users:
                    user_email = user.get("email")
                    if not user_email:
                        continue
                    
                    # Skip localhost/expert users
                    if "127.0.0.1" in user_email or "localhost" in user_email.lower():
                        continue
                    
                    # Track rejections for test_rejected user (check by name)
                    user_name = next((u.get("name", "").lower() for u in action_users), "")
                    if user_name:
                        with self.state_lock:
                            self.async_user_names[user_email] = user_name
                    if user_name == "test_rejected":
                        with self.state_lock:
                            if state_key not in self.test_rejected_state_rejections:
                                self.test_rejected_state_rejections[state_key] = 0
                            self.test_rejected_state_rejections[state_key] += 1
                            rejection_count = self.test_rejected_state_rejections[state_key]
                            print(f"📊 test_rejected has {rejection_count} rejection(s) for state {state_id}")
                    
                    # Re-queue this state for the user (with weighted sampling, just remove from submitted set)
                    # Exception: test_rejected with limit rejections should NOT be re-queued
                    with self.state_lock:
                        if user_email in self.async_user_submissions:
                            # Check if test_rejected has reached rejection limit for this state (check by name)
                            user_name = next((u.get("name", "").lower() for u in action_users), "")
                            if user_name == "test_rejected":
                                rejection_count = self.test_rejected_state_rejections.get(state_key, 0)
                                if rejection_count >= self.required_responses_per_critical_state:
                                    print(f"🚫 test_rejected reached {self.required_responses_per_critical_state} rejections for state {state_id} - not re-queuing")
                                    continue  # Don't remove from submitted set - they can't resubmit
                            
                            # Remove from submitted set so they can try again
                            # With weighted dynamic sampling, they'll naturally get this state
                            # with appropriate priority based on its completion status
                            self.async_user_submissions[user_email].discard(state_key)
            
            # Log async user submissions (only for user submissions, not admin/autofill)
            if self.async_user_logger and self.asynchronous_mode and self.async_pool_finalized:
                # Get queue length for logging
                queue_length = len(self.pre_execution_approval_queue)
                
                for user in action_users:
                    user_email = user.get("email")
                    user_name = user.get("name", "Unknown")
                    
                    if not user_email:
                        continue
                    
                    # Filter out localhost expert submissions
                    if "127.0.0.1" in user_email or "localhost" in user_email.lower():
                        continue
                    
                    # Get user's timing info for this state
                    duration_seconds = 0.0
                    used_animation = user.get("used_animation", False)
                    ip_address = user.get("ip_address")
                    
                    if user_email in state_info.get("user_timings", {}):
                        timing = state_info["user_timings"][user_email]
                        duration_seconds = timing.get("duration_seconds", 0.0)
                    
                    # Calculate user's current approval rate
                    user_approved = 0
                    user_rejected = 0
                    
                    # Count all previously reviewed submissions for this user
                    with self.state_lock:
                        for ep_id in self.completed_states_buffer_by_episode:
                            for s_id in self.completed_states_buffer_by_episode[ep_id]:
                                s_info = self.completed_states_buffer_by_episode[ep_id][s_id]
                                exec_history = s_info.get("execution_history", [])
                                for exec_entry in exec_history:
                                    for submitted_user in exec_entry.get("submitted_by", []):
                                        if submitted_user.get("email") == user_email:
                                            if exec_entry.get("approval") == 1:
                                                user_approved += 1
                                            elif exec_entry.get("approval") == -1:
                                                user_rejected += 1
                    
                    # Include current submission
                    if approved:
                        user_approved += 1
                    else:
                        user_rejected += 1
                    
                    user_total = user_approved + user_rejected
                    user_approval_rate = user_approved / user_total if user_total > 0 else None
                    
                    # Log this submission
                    self.async_user_logger.log_submission(
                        user_email=user_email,
                        user_name=user_name,
                        episode_id=episode_id,
                        state_id=state_id,
                        duration_seconds=duration_seconds,
                        used_animation=used_animation,
                        approval_status=approval_value,
                        current_approval_rate=user_approval_rate,
                        current_approval_count=user_approved,
                        current_total_count=user_total,
                        ip_address=ip_address,
                        queue_length=queue_length,
                    )
            
            # After recording decision, clear current and activate next from queue (only for normal flow)
            if not (is_test_approved or is_test_rejected):
                with self.pre_execution_approval_lock:
                    if self.pre_execution_approval_queue:
                        self.pending_pre_execution_approval = self.pre_execution_approval_queue.pop(0)
                    else:
                        self.pending_pre_execution_approval = None
                
        # Store all reviewed actions in execution_history and mark loop as complete
        with self.state_lock:
            # Find the state in either completed_states_buffer (preferred) or completed_states
            state_ref = None
            if episode_id in self.completed_states_buffer_by_episode and state_id in self.completed_states_buffer_by_episode[episode_id]:
                state_ref = self.completed_states_buffer_by_episode[episode_id][state_id]
            elif episode_id in self.completed_states_by_episode and state_id in self.completed_states_by_episode[episode_id]:
                state_ref = self.completed_states_by_episode[episode_id][state_id]
            
            if state_ref is not None:
                # APPEND new reviews to existing execution_history (don't overwrite admin auto-approvals)
                existing_history = state_ref.get("execution_history", [])
                state_ref["execution_history"] = existing_history + reviewed_actions
                state_ref["num_pre_approvals_completed"] = len(existing_history) + len(reviewed_actions)
                state_ref["pre_approval_loop_complete"] = True
                
                # Also update in the other dict if it exists there (keep them in sync)
                if episode_id in self.completed_states_by_episode and state_id in self.completed_states_by_episode[episode_id]:
                    self.completed_states_by_episode[episode_id][state_id] = state_ref
                if episode_id in self.completed_states_buffer_by_episode and state_id in self.completed_states_buffer_by_episode[episode_id]:
                    self.completed_states_buffer_by_episode[episode_id][state_id] = state_ref
            
            # Check if state is now fully complete (enough approved actions) and remove from pending
            if episode_id in self.pending_states_by_episode and state_id in self.pending_states_by_episode[episode_id]:
                state_info_ref = self.pending_states_by_episode[episode_id][state_id]
                total_approved = sum(1 for entry in state_info_ref.get("execution_history", []) 
                                    if entry.get("approval") == 1)
                
                if total_approved >= self.required_responses_per_critical_state:
                    print(f"✅ State {state_id} now has {total_approved} approved actions - removing from pending")
                    del self.pending_states_by_episode[episode_id][state_id]
                    
                    # Check if episode is now complete
                    if not self.pending_states_by_episode.get(episode_id):
                        print(f"📦 Episode {episode_id} has no more pending states after async labeling")

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
    
    def _start_pre_approval_worker(self):
        """Start the single worker thread that processes pre-approval requests."""
        self.pre_approval_worker_thread = Thread(target=self._pre_approval_worker, daemon=True)
        self.pre_approval_worker_thread.start()
    
    def _pre_approval_worker(self):
        """Single worker thread that processes pre-approval requests sequentially."""
        while True:
            try:
                # Block until a request is available
                state_info, episode_id, state_id = self.pre_approval_queue.get()
                
                # Process the pre-approval loop
                self._run_pre_approval_loop(state_info, episode_id, state_id)
                
                # Check if episode should finalize after this approval completes
                with self.state_lock:
                    if episode_id not in self.pending_states_by_episode or not self.pending_states_by_episode[episode_id]:
                        # Check if there are any pre-approvals queued or active for this episode
                        with self.pre_execution_approval_lock:
                            has_pending = (
                                self.pending_pre_execution_approval is not None and
                                self.pending_pre_execution_approval.get("episode_id") == episode_id
                            )
                            has_queued = any(
                                req.get("episode_id") == episode_id 
                                for req in self.pre_execution_approval_queue
                            )
                        
                        if not has_pending and not has_queued:
                            print(f"🏁 Pre-approval complete for episode {episode_id} - scheduling finalization")
                            self._schedule_episode_finalize_after_grace(episode_id)
                
                self.pre_approval_queue.task_done()
            except Exception as e:
                print(f"❌ Error in pre-approval worker: {e}")
                import traceback
                traceback.print_exc()

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
        """Schedule episode finalization after grace period.
        
        Cancels any existing timer for this episode before scheduling a new one.
        """
        # Cancel existing timer if present
        if episode_id in self._episode_finalize_timers:
            old_timer = self._episode_finalize_timers[episode_id]
            if old_timer.is_alive():
                print(f"⏱️  Cancelling existing finalization timer for episode {episode_id}")
                old_timer.cancel()
        
        delay = self.episode_finalize_grace_s
        timer = Timer(delay, self._finalize_episode_if_still_empty, args=(episode_id,))
        timer.daemon = True
        self._episode_finalize_timers[episode_id] = timer
        timer.start()

    def _finalize_episode_if_still_empty(self, episode_id: int):
        """Timer callback."""
        print(f"⏰ Finalization timer fired for episode {episode_id}")
        with self.state_lock:
            self._finalize_episode_logic(episode_id)
    
    def _finalize_episode_logic(self, episode_id: int):
        """Finalize episode logic (caller must hold state_lock)."""
        self._episode_finalize_timers.pop(episode_id, None)

        if self.pending_states_by_episode.get(episode_id):
            # New states has become pending in the episode
            print(f"⚠️  Episode {episode_id} has new pending states - skipping finalization")
            return

        # Check if there's a pre-execution approval pending or queued for this episode
        with self.pre_execution_approval_lock:
            # Check active approval
            if (self.pending_pre_execution_approval and 
                self.pending_pre_execution_approval.get("episode_id") == episode_id):
                self._schedule_episode_finalize_after_grace(episode_id)
                return
            
            # Check queued approvals
            queued_for_episode = [
                req for req in self.pre_execution_approval_queue 
                if req.get("episode_id") == episode_id
            ]
            if queued_for_episode:
                self._schedule_episode_finalize_after_grace(episode_id)
                return
        
        # All checks passed - finalize episode
        print(f"💾 Episode {episode_id} finalized (buffered for batch save)")
        self.episodes_completed.add(episode_id)  # for monitoring

        buffer = self.completed_states_buffer_by_episode[episode_id]
        
        # Calculate episode timing
        episode_timing = self._calculate_episode_timing(episode_id, buffer)
        
        # Store for batch save at the end (don't save immediately)
        if not hasattr(self, '_finalized_episodes'):
            self._finalized_episodes = {}
        self._finalized_episodes[episode_id] = {
            'buffer': buffer,
            'timing': episode_timing
        }
        
        print(f"✅ Episode {episode_id} ready for batch save ({len(self._finalized_episodes)} episodes buffered)")
        
        # Don't delete buffer yet - we'll need it for batch save
        # del self.completed_states_buffer_by_episode[episode_id]

    def _calculate_episode_timing(self, episode_id: int, buffer: list) -> dict:
        """Calculate comprehensive timing statistics for an episode.
        
        Returns dict with:
        - episode_start_time, episode_end_time (Unix timestamps)
        - episode_start_time_iso, episode_end_time_iso (ISO strings)
        - total_episode_duration_seconds
        - per_user_stats: {email: {total_time, avg_time, num_submissions, min_time, max_time}}
        - per_state_stats: {state_id: {avg_time, num_users, state_duration}}
        - overall_avg_submission_time, overall_avg_state_duration
        """
        import datetime
        
        episode_end_time = time.time()
        episode_end_time_iso = datetime.datetime.now().isoformat()
        episode_start_time = self.episode_start_times.get(episode_id, episode_end_time)
        episode_start_time_iso = self.episode_start_times_iso.get(episode_id, episode_end_time_iso)
        
        total_episode_duration = episode_end_time - episode_start_time
        
        # Calculate per-user and per-state statistics from completed states
        per_user_times = {}  # email -> [duration1, duration2, ...]
        per_user_counts = {}  # email -> num_submissions (even without duration)
        per_state_times = {}  # state_id -> {"durations": [d1, d2, ...], "state_duration": s, "num_users": N}
        
        # Buffer is a dict {state_id -> state_info}, iterate over values
        for state_info in buffer.values():
            state_id = state_info.get("state_id")
            user_timings = state_info.get("user_timings", {})
            
            # Per-user timing - track both durations and counts
            for email, timing_info in user_timings.items():
                # Count submissions (even without duration)
                if email not in per_user_counts:
                    per_user_counts[email] = 0
                per_user_counts[email] += 1
                
                # Track duration if available
                duration = timing_info.get("duration_seconds")
                if duration is not None:
                    if email not in per_user_times:
                        per_user_times[email] = []
                    per_user_times[email].append(duration)
            
            # Per-state timing
            if state_id is not None:
                if state_id not in per_state_times:
                    per_state_times[state_id] = {"durations": [], "state_duration": None, "num_users": 0}
                
                # Count users for this state
                per_state_times[state_id]["num_users"] = len(user_timings)
                
                # Collect all user durations for this state
                for timing_info in user_timings.values():
                    duration = timing_info.get("duration_seconds")
                    if duration is not None:
                        per_state_times[state_id]["durations"].append(duration)
                
                # State completion duration (from creation to all responses received)
                state_created = state_info.get("state_created_at")
                state_completed = state_info.get("state_completed_at")
                if state_created and state_completed:
                    per_state_times[state_id]["state_duration"] = state_completed - state_created
        
        # Aggregate per-user stats
        per_user_stats = {}
        
        # Combine users from both timing data and counts
        all_users = set(per_user_times.keys()) | set(per_user_counts.keys())
        
        for email in all_users:
            durations = per_user_times.get(email, [])
            num_submissions = per_user_counts.get(email, 0)
            
            per_user_stats[email] = {
                "total_time": sum(durations) if durations else 0,
                "avg_time": sum(durations) / len(durations) if durations else 0,
                "num_submissions": num_submissions,
                "min_time": min(durations) if durations else 0,
                "max_time": max(durations) if durations else 0,
            }
        
        # Aggregate per-state stats (only include critical states with actual timing data)
        per_state_stats = {}
        for state_id, state_data in per_state_times.items():
            durations = state_data["durations"]
            state_duration = state_data["state_duration"]
            num_users = state_data["num_users"]
            
            # Only include states that have timing data (user submissions or state completion)
            if num_users > 0 or state_duration is not None:
                per_state_stats[state_id] = {
                    "avg_time": sum(durations) / len(durations) if durations else 0,
                    "num_users": num_users,
                    "state_duration": state_duration,
                }
        
        # Overall averages
        all_submission_times = [d for durations in per_user_times.values() for d in durations]
        all_state_durations = [s["state_duration"] for s in per_state_times.values() if s["state_duration"] is not None]
        
        return {
            "episode_start_time": episode_start_time,
            "episode_start_time_iso": episode_start_time_iso,
            "episode_end_time": episode_end_time,
            "episode_end_time_iso": episode_end_time_iso,
            "total_episode_duration_seconds": total_episode_duration,
            "per_user_stats": per_user_stats,
            "per_state_stats": per_state_stats,
            "overall_avg_submission_time": sum(all_submission_times) / len(all_submission_times) if all_submission_times else 0,
            "overall_avg_state_duration": sum(all_state_durations) / len(all_state_durations) if all_state_durations else 0,
        }

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
        
        # Create execution_history with all actions auto-approved
        state_info["execution_history"] = [
            {
                "action": position_action.clone(),
                "propensity": 1.0,
                "approval": 1,  # Auto-approved
                "submitted_by": [],  # Auto-filled, no user
            }
            for _ in range(self.required_responses_per_critical_state)
        ]
        state_info["num_pre_approvals_completed"] = self.required_responses_per_critical_state
        state_info["pre_approval_loop_complete"] = True

        # Mark as approved since we're auto-filling with "End."
        state_info["approval_status"] = "approved"
        
        # Mark this episode as ended - block any new states
        self.episodes_marked_as_end.add(episode_id)
        print(f"🏁 Episode {episode_id} marked as END - no more states will be accepted")

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
                # Set approved to True so the waiting thread sees it as approval, not demotion
                self.pending_approval_state["approved"] = True

        if not self.pending_states_by_episode[episode_id]:
            self._schedule_episode_finalize_after_grace(episode_id)

    # =========================
    # Manual Flush (Save without completing trajectory)
    # =========================

    def flush_episode_now(self, episode_id: int) -> dict:
        """Request flush of collected frames for an episode to the dataset.
        
        Delegates to FlushManager for non-blocking, thread-safe operation.
        
        Args:
            episode_id: The episode to flush
            
        Returns:
            dict with "status" and "message" keys
        """
        return self.flush_manager.flush_episode_now(episode_id)
    
    def get_flush_status(self, episode_id: int = None) -> dict:
        """Get status of flush operations.
        
        Delegates to FlushManager.
        
        Args:
            episode_id: If provided, get status for specific episode. Otherwise get all.
            
        Returns:
            dict with flush status information
        """
        return self.flush_manager.get_flush_status(episode_id)


    # =========================
    # Asynchronous Mode Management
    # =========================

    def finalize_admin_phase(self, robot=None) -> dict:
        """Finalize admin data collection phase and prepare for async labeling.
        
        Marks all admin-complete states as ready for async serving. States remain in 
        pending_states_by_episode until they receive all required user labels.
        
        Args:
            robot: Optional Robot instance to send to docking position
        
        Returns:
            dict with status and counts
        """
        if not self.asynchronous_mode:
            return {"status": "error", "message": "Not in asynchronous mode"}
        
        if self.async_pool_finalized:
            return {"status": "error", "message": "Admin phase already finalized"}
        
        # Move REAL robot to home then dock position at the start of async labeling
        import math
        import time
        HOME_POSITION_DEG = [0, 60, 75, -60, 0, 0, 2]
        HOME_POSITION_RAD = [deg * math.pi / 180.0 for deg in HOME_POSITION_DEG]
        DOCK_POSITION_RAD = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # Sleep pose - all joints at 0
        
        if robot is not None:
            print("🏠 Moving REAL robot to docking position for async labeling phase...")
            try:
                if not robot.is_connected:
                    robot.connect()
                
                # Stage 1: Move to home position
                print("   Stage 1: Moving to home position...")
                robot.follower_arms['main'].write("Goal_Position", HOME_POSITION_RAD, duration=5.0)
                time.sleep(5.5)  # Wait for movement to complete
                print("   ✓ Reached home position")
                
                # Stage 2: Move to docked/sleep position (all joints at 0)
                print("   Stage 2: Moving to docked position (sleep pose)...")
                robot.follower_arms['main'].write("Goal_Position", DOCK_POSITION_RAD, duration=3.0)
                time.sleep(3.5)  # Wait for movement to complete
                print("✓ Robot docked and ready for async labeling")
            except Exception as e:
                print(f"⚠️  Failed to dock robot: {e}")
        
        with self.state_lock:
            # Count admin-complete states that are ready for async serving
            # These are states that got admin response but need more user labels
            states_ready = 0
            states_skipped = 0
            
            for episode_id, states in self.pending_states_by_episode.items():
                for state_id, state_info in states.items():
                    if state_info.get("critical", False) and state_info.get("admin_complete", False):
                        # Check if this was a gripper-only, home position, or manually autofilled state (already fully labeled)
                        is_gripper_only = state_info.get("gripper_only_autofilled", False)
                        is_home_position = state_info.get("home_position_autofilled", False)
                        is_manual_autofill = state_info.get("manual_autofilled", False)
                        
                        if is_gripper_only or is_home_position or is_manual_autofill:
                            # These are already fully complete and will be in completed_states
                            states_skipped += 1
                        else:
                            # Mark as available for async serving
                            state_info["async_pool_ready"] = True
                            
                            # Create pool key for tracking user assignments
                            pool_key = (episode_id, state_id)
                            self.async_state_pool[pool_key] = state_info  # Reference, not copy
                            states_ready += 1
            
            self.async_pool_finalized = True
            
            print(f"{'='*80}")
            print(f"🔄 Async pool ready: {states_ready} states available for user labeling")
            if states_skipped > 0:
                print(f"   Skipped {states_skipped} auto-filled states (gripper-only, home position, or manual autofill)")
            print(f"{'='*80}")
            
            return {
                "status": "success",
                "states_in_pool": states_ready,
                "states_skipped": states_skipped,
                "message": f"Admin phase complete. {states_ready} states ready for labeling"
            }
    
    def get_async_pooled_state(self, user_email: str) -> dict | None:
        """Get next state using weighted dynamic sampling.
        
        States that need fewer labels get higher weights (quadratically), ensuring
        balanced progress across all states and preventing stragglers.
        
        Weight formula: weight = (max_need - need + 1)²
        where need = required_responses - approved_count
        
        Args:
            user_email: Email of user requesting state
            
        Returns:
            State info dict or None if no states available
        """
        if not self.asynchronous_mode or not self.async_pool_finalized:
            return None
        
        if not user_email:
            return None
        
        with self.state_lock:
            # Initialize user's submission tracking on first access
            if user_email not in self.async_user_submissions:
                self.async_user_submissions[user_email] = set()
                pass  # New user starting async labeling
            
            user_submitted = self.async_user_submissions[user_email]
            
            # Build list of available states (unsubmitted by user, not full)
            available_states = []
            needs = []
            
            for state_key, state_info in self.async_state_pool.items():
                # Get user's name from stored mapping (for test user detection)
                user_name = self.async_user_names.get(user_email, "").lower()
                
                # Skip if user already submitted to this state (except test_approved)
                if state_key in user_submitted and user_name != "test_approved":
                    continue
                
                # Special handling for test_rejected: check if they've reached rejection limit
                if user_name == "test_rejected":
                    rejection_count = self.test_rejected_state_rejections.get(state_key, 0)
                    if rejection_count >= self.required_responses_per_critical_state:
                        # Skip this state - test_rejected has been rejected required_responses times already
                        continue
                
                # Count APPROVED submissions for critical states
                num_approved = sum(1 for entry in state_info.get("execution_history", []) 
                                  if entry.get("approval") == 1)
                need = self.required_responses_per_critical_state - num_approved
                
                # Skip if state is already full
                if need <= 0:
                    continue
                
                available_states.append((state_key, state_info))
                needs.append(need)
            
            # No available states
            if not available_states:
                return None
            
            # Calculate weights: states closer to completion get higher weight
            # weight = (max_need - need + 1)²
            max_need = max(needs)
            weights = [(max_need - need + 1) ** 2 for need in needs]
            
            # Weighted random selection
            import random
            selected_idx = random.choices(range(len(available_states)), weights=weights, k=1)[0]
            state_key, state_info = available_states[selected_idx]
            
            episode_id, state_id = state_key
            selected_need = needs[selected_idx]
            selected_weight = weights[selected_idx]
            
            # Assign this state to the user (prevents refresh shopping)
            self.async_user_current_state[user_email] = state_key
            
            return state_info

    
    def get_async_pool_status(self) -> dict:
        """Get current status of async pool."""
        with self.state_lock:
            total_states = len(self.async_state_pool)
            
            # Calculate completion status for each state
            states_needing_labels = 0
            states_completed = 0
            total_approved = 0
            total_submissions = 0
            total_needed = total_states * self.required_responses_per_critical_state
            
            for pool_key, state_info in self.async_state_pool.items():
                # Count APPROVED submissions for critical states
                exec_history = state_info.get("execution_history", [])
                num_approved = sum(1 for entry in exec_history if entry.get("approval") == 1)
                total_approved += num_approved
                total_submissions += len(exec_history)
                if num_approved < self.required_responses_per_critical_state:
                    states_needing_labels += 1
                else:
                    states_completed += 1
            
            # User statistics
            total_users = len(self.async_user_submissions)
            users_maxed_out = sum(
                1 for seen_set in self.async_user_submissions.values()
                if len(seen_set) >= total_states
            )
            
            # Pre-approval status
            with self.pre_execution_approval_lock:
                queue_length = len(self.pre_execution_approval_queue)
                has_active_approval = self.pending_pre_execution_approval is not None
            
            # Check pre-approval queue size
            approval_queue_size = self.pre_approval_queue.qsize()
            
            return {
                "async_mode": self.asynchronous_mode,
                "pool_finalized": self.async_pool_finalized,
                "total_states": total_states,
                "states_needing_labels": states_needing_labels,
                "states_completed": states_completed,
                "total_approved": total_approved,
                "total_submissions": total_submissions,
                "total_needed": total_needed,
                "required_per_state": self.required_responses_per_critical_state,
                "total_users": total_users,
                "users_maxed_out": users_maxed_out,
                "pending_approvals_queue": queue_length,
                "active_approval": has_active_approval,
                "approval_queue_size": approval_queue_size,
            }

    def reset_async_pool(self):
        """Reset async pool (for testing or restarting admin phase)."""
        with self.state_lock:
            self.async_state_pool.clear()
            self.async_user_submissions.clear()
            self.async_pool_finalized = False
            print("🔄 Async pool reset")
    
    def generate_async_user_summary(self):
        """Generate final summary of async user performance.
        
        Call this after all async data collection is complete to generate
        comprehensive statistics about user submissions, approval rates,
        animation usage, and timing.
        
        Returns:
            dict: Summary statistics
        """
        if not self.async_user_logger:
            print("⚠️  Async user logger not initialized (not in async mode)")
            return {"error": "Not in async mode"}
        
        print("📊 Generating final async user summary...")
        self.async_user_logger.generate_final_summary()
        
        return {
            "status": "success",
            "summary_file": str(self.async_user_logger.summary_log_path),
            "submission_log": str(self.async_user_logger.submission_log_path),
        }
    
    def get_user_approval_count(self, user_email: str) -> dict:
        """Get approval count for a specific user.
        
        Args:
            user_email: User email address
            
        Returns:
            dict: {"approved": int, "rejected": int, "disapproved": int, "total": int, "rate": float}
        """
        if not user_email:
            return {"approved": 0, "rejected": 0, "disapproved": 0, "total": 0, "rate": 0.0}
        
        user_approved = 0
        user_rejected = 0
        
        # Count all submissions across all completed states
        # Use completed_states_by_episode (permanent storage) to include finalized episodes
        # completed_states_buffer_by_episode is deleted when episode is finalized
        with self.state_lock:
            for ep_id in self.completed_states_by_episode:
                for s_id in self.completed_states_by_episode[ep_id]:
                    s_info = self.completed_states_by_episode[ep_id][s_id]
                    exec_history = s_info.get("execution_history", [])
                    for exec_entry in exec_history:
                        for submitted_user in exec_entry.get("submitted_by", []):
                            if submitted_user.get("email") == user_email:
                                if exec_entry.get("approval") == 1:
                                    user_approved += 1
                                elif exec_entry.get("approval") == -1:
                                    user_rejected += 1
        
        user_total = user_approved + user_rejected
        approval_rate = user_approved / user_total if user_total > 0 else 0.0
        
        return {
            "approved": user_approved,
            "rejected": user_rejected,
            "disapproved": user_rejected,  # Alias for rejected (user-facing term)
            "total": user_total,
            "rate": approval_rate
        }

    # =========================
    # Phase 1/2 Checkpoint (for separating admin collection from async user labeling)
    # =========================

    def _serialize_state_for_checkpoint(self, state_info: dict, checkpoint_dir: Path = None) -> dict:
        """Convert state_info to JSON-serializable format (tensors -> lists, views -> base64, copy obs)."""
        import base64
        import shutil
        
        state_id = state_info.get("state_id", "?")
        ep_id = state_info.get("episode_id", "?")
        
        out = {}
        for k, v in state_info.items():
            if isinstance(v, torch.Tensor):
                out[k] = v.tolist()
            elif k == "actions" and isinstance(v, list):
                out[k] = [a.tolist() if isinstance(a, torch.Tensor) else a for a in v]
            elif k == "execution_history" and isinstance(v, list):
                out[k] = []
                for entry in v:
                    new_entry = dict(entry)
                    if "action" in new_entry and isinstance(new_entry["action"], torch.Tensor):
                        new_entry["action"] = new_entry["action"].tolist()
                    out[k].append(new_entry)
            elif k == "view_paths" and isinstance(v, dict):
                # Embed view images as base64 to survive tmp cleanup
                out[k] = v  # Keep paths for reference
                views_data = {}
                for cam, path in v.items():
                    try:
                        with open(path, "rb") as f:
                            views_data[cam] = base64.b64encode(f.read()).decode("ascii")
                    except Exception:
                        pass  # File missing - skip
                out["views_data"] = views_data
            elif k == "obs_path" and v and checkpoint_dir:
                # Copy obs file to checkpoint directory
                src_path = Path(v)
                if src_path.exists():
                    obs_ep_id = state_info.get("episode_id", "unknown")
                    obs_state_id = state_info.get("state_id", "unknown")
                    obs_subdir = checkpoint_dir / "obs_cache"
                    obs_subdir.mkdir(parents=True, exist_ok=True)
                    dst_path = obs_subdir / f"ep{obs_ep_id}_state{obs_state_id}.pt"
                    
                    try:
                        src_size = src_path.stat().st_size
                        
                        # Copy with explicit binary read/write
                        with open(src_path, "rb") as sf:
                            data = sf.read()
                        with open(dst_path, "wb") as df:
                            df.write(data)
                            df.flush()
                            import os
                            os.fsync(df.fileno())
                        
                        # Verify destination exists and has same size
                        if dst_path.exists():
                            dst_size = dst_path.stat().st_size
                            if dst_size == src_size:
                                out[k] = str(dst_path)
                            else:
                                print(f"⚠️  Size mismatch copying obs! src={src_size}, dst={dst_size}")
                                out[k] = v
                        else:
                            print(f"⚠️  Destination file doesn't exist after copy!")
                            out[k] = v
                    except Exception as e:
                        print(f"⚠️  Failed to copy obs {src_path} -> {dst_path}: {e}")
                        out[k] = v
                else:
                    print(f"⚠️  Source obs file not found: {src_path}")
                    out[k] = v
            else:
                out[k] = v
        return out

    def _deserialize_state_from_checkpoint(self, state_data: dict, checkpoint_dir: Path = None) -> dict:
        """Convert checkpoint state data back to runtime format (lists -> tensors, restore views).
        
        Args:
            state_data: Serialized state data from checkpoint
            checkpoint_dir: Directory containing the checkpoint (used to find obs_cache)
        """
        import base64
        out = {}
        for k, v in state_data.items():
            if k == "actions" and isinstance(v, list):
                out[k] = [torch.tensor(a, dtype=torch.float32) if isinstance(a, list) else a for a in v]
            elif k == "execution_history" and isinstance(v, list):
                out[k] = []
                for entry in v:
                    new_entry = dict(entry)
                    if "action" in new_entry and isinstance(new_entry["action"], list):
                        new_entry["action"] = torch.tensor(new_entry["action"], dtype=torch.float32)
                    out[k].append(new_entry)
            elif k == "views_data":
                # Skip - will be processed together with view_paths
                pass
            elif k == "obs_path" and checkpoint_dir is not None:
                # Update obs_path to point to checkpoint's obs_cache directory
                original_path = Path(v)
                new_path = checkpoint_dir / "obs_cache" / original_path.name
                out[k] = str(new_path)
            else:
                out[k] = v
        
        # Restore view files from embedded base64 data
        views_data = state_data.get("views_data", {})
        view_paths = state_data.get("view_paths", {})
        if views_data and view_paths:
            new_view_paths = {}
            for cam, b64_data in views_data.items():
                original_path = view_paths.get(cam)
                if original_path:
                    try:
                        # Ensure directory exists
                        path = Path(original_path)
                        path.parent.mkdir(parents=True, exist_ok=True)
                        # Write file
                        with open(path, "wb") as f:
                            f.write(base64.b64decode(b64_data))
                        new_view_paths[cam] = str(path)
                    except Exception as e:
                        print(f"⚠️  Failed to restore view {cam}: {e}")
            out["view_paths"] = new_view_paths
        
        return out

    def save_phase1_checkpoint(self, checkpoint_path: Path, dataset_config: dict = None) -> dict:
        """Save all state data needed for Phase 2 (async user labeling).
        
        Called at end of Phase 1 (admin collection) before finalize_admin_phase().
        
        Args:
            checkpoint_path: Path to save checkpoint JSON file
            dataset_config: Dataset configuration to recreate dataset in Phase 2
            
        Returns:
            dict with status and checkpoint path
        """
        checkpoint_path = Path(checkpoint_path)
        checkpoint_dir = checkpoint_path.parent  # Dataset directory
        
        # Track obs copy results
        obs_copy_stats = {"success": 0, "failed": 0, "skipped": 0}
        
        def serialize_with_tracking(s, cdir):
            result = self._serialize_state_for_checkpoint(s, cdir)
            obs_path = s.get("obs_path")
            if obs_path and cdir:
                result_path = result.get("obs_path", "")
                if result_path and str(cdir) in result_path:
                    # Check if destination file actually exists
                    if Path(result_path).exists():
                        obs_copy_stats["success"] += 1
                    else:
                        obs_copy_stats["failed"] += 1
                        print(f"⚠️  obs copy reported success but file missing: {result_path}")
                else:
                    obs_copy_stats["skipped"] += 1
            return result
        
        with self.state_lock:
            checkpoint = {
                "version": 1,
                "saved_at": time.time(),
                "saved_at_iso": __import__("datetime").datetime.now().isoformat(),
                # Dataset config for recreation in Phase 2
                "dataset_config": dataset_config,
                # State data (pass checkpoint_dir to copy obs files)
                "pending_states_by_episode": {
                    ep: {str(sid): serialize_with_tracking(s, checkpoint_dir) for sid, s in states.items()}
                    for ep, states in self.pending_states_by_episode.items()
                },
                "completed_states_by_episode": {
                    ep: {str(sid): serialize_with_tracking(s, checkpoint_dir) for sid, s in states.items()}
                    for ep, states in self.completed_states_by_episode.items()
                },
                "completed_states_buffer_by_episode": {
                    ep: {str(sid): serialize_with_tracking(s, checkpoint_dir) for sid, s in states.items()}
                    for ep, states in self.completed_states_buffer_by_episode.items()
                },
                # Episode metadata
                "episode_start_times": dict(self.episode_start_times),
                "episode_start_times_iso": dict(self.episode_start_times_iso),
                "episodes_marked_as_end": list(self.episodes_marked_as_end),
                "next_state_id": self.next_state_id,
                # Config values needed for Phase 2
                "config": {
                    "required_responses_per_state": self.required_responses_per_state,
                    "required_responses_per_critical_state": self.required_responses_per_critical_state,
                    "required_approvals_per_critical_state": self.required_approvals_per_critical_state,
                    "asynchronous_mode": self.asynchronous_mode,
                    "async_admin_responses_per_state": self.async_admin_responses_per_state,
                    "task_text": self.task_text,
                },
            }
        
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        with open(checkpoint_path, "w") as f:
            json.dump(checkpoint, f, indent=2)
        
        # Print copy stats
        print(f"   📊 Obs copy stats: {obs_copy_stats['success']} success, {obs_copy_stats['failed']} failed, {obs_copy_stats['skipped']} skipped")
        
        # Count unique states that should have obs files
        # States may appear in multiple dicts (e.g. completed + buffer), so deduplicate by (ep, sid)
        unique_state_keys = set()
        total_states_raw = 0
        for states_dict in [self.pending_states_by_episode, self.completed_states_by_episode, self.completed_states_buffer_by_episode]:
            for ep, states in states_dict.items():
                total_states_raw += len(states)
                for sid in states:
                    unique_state_keys.add((ep, sid))
        total_states = len(unique_state_keys)
        
        # Verify obs files were copied
        obs_cache_dir = checkpoint_dir / "obs_cache"
        if obs_cache_dir.exists():
            obs_files = list(obs_cache_dir.glob("*.pt"))
            print(f"✅ Phase 1 checkpoint saved to: {checkpoint_path}")
            print(f"   📁 obs_cache contains {len(obs_files)} files (expected ~{total_states})")
            
            if len(obs_files) == 0:
                print(f"\n" + "="*60)
                print(f"❌ CRITICAL ERROR: No obs files were copied to checkpoint!")
                print(f"   This checkpoint will NOT work for Phase 2!")
                print(f"   Source obs files may not exist in /tmp/crowd_obs_cache/")
                print(f"="*60 + "\n")
                return {"status": "error", "path": str(checkpoint_path), "message": "No obs files copied - checkpoint unusable"}
            
            if obs_copy_stats['failed'] > 0:
                print(f"\n" + "="*60)
                print(f"⚠️  WARNING: {obs_copy_stats['failed']} obs file copies failed!")
                print(f"   Some states may be missing observation data.")
                print(f"="*60 + "\n")
        else:
            print(f"✅ Phase 1 checkpoint saved to: {checkpoint_path}")
            print(f"   ⚠️  WARNING: obs_cache directory does not exist!")
            return {"status": "error", "path": str(checkpoint_path), "message": "obs_cache directory not created"}
        
        return {"status": "success", "path": str(checkpoint_path)}

    def load_phase1_checkpoint(self, checkpoint_path: Path) -> dict:
        """Load Phase 1 checkpoint and restore state for Phase 2.
        
        Args:
            checkpoint_path: Path to checkpoint JSON file
            
        Returns:
            dict with status and loaded config
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            return {"status": "error", "message": f"Checkpoint not found: {checkpoint_path}"}
        
        with open(checkpoint_path, "r") as f:
            checkpoint = json.load(f)
        
        version = checkpoint.get("version", 0)
        if version != 1:
            return {"status": "error", "message": f"Unsupported checkpoint version: {version}"}
        
        # Get checkpoint directory for resolving obs_cache paths
        checkpoint_dir = checkpoint_path.parent
        
        with self.state_lock:
            # Restore state data
            self.pending_states_by_episode.clear()
            for ep, states in checkpoint["pending_states_by_episode"].items():
                # Convert episode_id to int if numeric
                ep_key = int(ep) if ep.isdigit() else ep
                self.pending_states_by_episode[ep_key] = {
                    int(sid): self._deserialize_state_from_checkpoint(s, checkpoint_dir) for sid, s in states.items()
                }
            
            self.completed_states_by_episode.clear()
            for ep, states in checkpoint["completed_states_by_episode"].items():
                ep_key = int(ep) if ep.isdigit() else ep
                self.completed_states_by_episode[ep_key] = {
                    int(sid): self._deserialize_state_from_checkpoint(s, checkpoint_dir) for sid, s in states.items()
                }
            
            self.completed_states_buffer_by_episode.clear()
            for ep, states in checkpoint["completed_states_buffer_by_episode"].items():
                ep_key = int(ep) if ep.isdigit() else ep
                self.completed_states_buffer_by_episode[ep_key] = {
                    int(sid): self._deserialize_state_from_checkpoint(s, checkpoint_dir) for sid, s in states.items()
                }
            
            # CRITICAL: Restore shared object references between the three dicts.
            # During normal Phase 1 operation, pending/completed/buffer all point to
            # the SAME Python dict for a given (episode, state) key. Deserialization
            # creates independent copies, which breaks code that mutates one dict
            # (e.g. appending to execution_history) and expects the change to be
            # visible through the other dicts.  Without this, the pre-approval
            # worker writes to an orphaned object and labels are silently lost.
            shared_refs_count = 0
            for ep in self.pending_states_by_episode:
                for sid in self.pending_states_by_episode[ep]:
                    canonical = self.pending_states_by_episode[ep][sid]
                    if ep in self.completed_states_by_episode and sid in self.completed_states_by_episode[ep]:
                        self.completed_states_by_episode[ep][sid] = canonical
                        shared_refs_count += 1
                    if ep in self.completed_states_buffer_by_episode and sid in self.completed_states_buffer_by_episode[ep]:
                        self.completed_states_buffer_by_episode[ep][sid] = canonical
            if shared_refs_count > 0:
                print(f"   🔗 Restored {shared_refs_count} shared object references across state dicts")
            
            # Restore episode metadata
            self.episode_start_times = {
                (int(k) if k.isdigit() else k): v 
                for k, v in checkpoint["episode_start_times"].items()
            }
            self.episode_start_times_iso = {
                (int(k) if k.isdigit() else k): v 
                for k, v in checkpoint["episode_start_times_iso"].items()
            }
            self.episodes_marked_as_end = set(
                int(e) if isinstance(e, str) and e.isdigit() else e 
                for e in checkpoint["episodes_marked_as_end"]
            )
            self.next_state_id = checkpoint["next_state_id"]
        
        # Validate obs files exist
        obs_cache_dir = checkpoint_dir / "obs_cache"
        missing_obs = 0
        total_obs_checks = 0
        for states_dict in [self.pending_states_by_episode, self.completed_states_by_episode, self.completed_states_buffer_by_episode]:
            for ep, states in states_dict.items():
                for sid, s in states.items():
                    obs_path = s.get("obs_path")
                    if obs_path:
                        total_obs_checks += 1
                        if not Path(obs_path).exists():
                            missing_obs += 1
        
        print(f"✅ Phase 1 checkpoint loaded from: {checkpoint_path}")
        print(f"   Episodes: {len(self.pending_states_by_episode)}")
        total_states = sum(len(s) for s in self.pending_states_by_episode.values())
        print(f"   Pending states: {total_states}")
        print(f"   📁 Obs file validation: {total_obs_checks - missing_obs}/{total_obs_checks} files exist")
        
        if missing_obs > 0:
            print(f"\n" + "="*60)
            print(f"❌ CRITICAL ERROR: {missing_obs} obs files are MISSING!")
            print(f"   Expected location: {obs_cache_dir}")
            print(f"   This checkpoint cannot be used for Phase 2.")
            print(f"   You need to re-run Phase 1 to create a valid checkpoint.")
            print(f"="*60 + "\n")
            return {"status": "error", "message": f"{missing_obs} obs files missing - checkpoint unusable"}
        
        return {
            "status": "success",
            "config": checkpoint.get("config", {}),
            "dataset_config": checkpoint.get("dataset_config", {}),
            "saved_at_iso": checkpoint.get("saved_at_iso"),
        }

