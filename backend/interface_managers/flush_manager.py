"""Flush Manager Module.

Manages incremental dataset flushing for long trajectories. Handles background saving of completed
states to the dataset without blocking data collection.

"""

import queue
import time
from threading import Lock, Thread


class FlushManager:
    """Manages incremental flushing of episode data to dataset.

    Responsibilities:
    - Queue-based flush requests (non-blocking)
    - Background worker thread for dataset writes
    - Progress tracking and status queries
    - Duplicate request prevention

    Attributes:
        flush_queue: Thread-safe queue for flush requests
        flush_worker_thread: Persistent background worker
        flush_in_progress: Dict tracking active flushes
        flush_lock: Lock protecting flush_in_progress

    """

    def __init__(
        self,
        state_lock: Lock,
        completed_states_buffer_by_episode: dict,
        pending_states_by_episode: dict,
        episodes_pending_save: set,
        save_episode_callback,
        required_responses_per_critical_state: int,
        calculate_episode_timing_callback,
    ):
        """Initialize flush manager.

        Args:
            state_lock: Shared lock protecting state data structures
            completed_states_buffer_by_episode: Reference to episode buffers (completed states)
            pending_states_by_episode: Reference to pending states (for finding last approved critical)
            episodes_pending_save: Reference to pending save tracking set
            save_episode_callback: Callback to save episode to dataset
            required_responses_per_critical_state: Number of responses needed for critical states
            calculate_episode_timing_callback: Callback to calculate timing statistics

        """
        self.state_lock = state_lock
        self.completed_states_buffer_by_episode = completed_states_buffer_by_episode
        self.pending_states_by_episode = pending_states_by_episode
        self.episodes_pending_save = episodes_pending_save
        self.save_episode_callback = save_episode_callback
        self.required_responses_per_critical_state = required_responses_per_critical_state
        self.calculate_episode_timing_callback = calculate_episode_timing_callback

        # Flush worker infrastructure
        self.flush_queue = queue.Queue()
        self.flush_worker_thread = None
        self.flush_in_progress = {}  # episode_id -> {"status": "in_progress", "num_states": N, "start_time": T}
        self.flush_lock = Lock()  # Protects flush_in_progress dict

        # Start worker
        self._start_flush_worker()

    def _start_flush_worker(self):
        """Start persistent flush worker thread that processes flush requests from queue."""

        def _flush_worker_loop():
            print("🧵 Flush worker thread started")
            while True:
                try:
                    # Block until flush request arrives
                    episode_id = self.flush_queue.get()

                    if episode_id is None:  # Poison pill for shutdown
                        print("🧵 Flush worker thread shutting down")
                        break

                    # Mark as in progress
                    with self.flush_lock:
                        if episode_id in self.flush_in_progress:
                            print(f"⚠️  Flush already in progress for episode {episode_id}, skipping duplicate")
                            self.flush_queue.task_done()
                            continue
                        self.flush_in_progress[episode_id] = {
                            "status": "in_progress",
                            "start_time": time.time(),
                        }

                    # Acquire lock briefly to copy buffer
                    buffer_copy = None
                    num_states = 0
                    try:
                        with self.state_lock:
                            # Find last approved critical state across both pending and completed
                            last_approved_critical_id = self._find_last_approved_critical_state(episode_id)
                            
                            if last_approved_critical_id is None:
                                print(f"⚠️  Episode {episode_id} has no approved critical states to flush")
                                with self.flush_lock:
                                    del self.flush_in_progress[episode_id]
                                self.flush_queue.task_done()
                                continue
                            
                            print(f"📍 Last approved critical state: {last_approved_critical_id}")
                            
                            # Collect all states up to and including last approved critical
                            buffer_copy = self._build_flush_buffer(episode_id, last_approved_critical_id)
                            
                            if not buffer_copy:
                                print(f"⚠️  No states to flush for episode {episode_id}")
                                with self.flush_lock:
                                    del self.flush_in_progress[episode_id]
                                self.flush_queue.task_done()
                                continue

                            num_states = len(buffer_copy)

                            # Update progress tracking
                            with self.flush_lock:
                                self.flush_in_progress[episode_id]["num_states"] = num_states

                        # Release lock before slow I/O operation
                        print(f"💾 Flushing {num_states} states for episode {episode_id}...")

                        # Calculate timing statistics for the flush
                        episode_timing = self.calculate_episode_timing_callback(episode_id, buffer_copy)

                        # Call save callback with timing stats
                        self.save_episode_callback(buffer_copy, episode_timing)

                        print(f"✅ Successfully flushed episode {episode_id} ({num_states} states)")

                        # NOTE: Don't remove states from buffers after flush!
                        # Flush is a checkpoint save - states remain active until episode completes
                        # or user explicitly stops. This allows continued labeling after flush.

                    except Exception as e:
                        print(f"❌ Error flushing episode {episode_id}: {e}")
                        import traceback

                        traceback.print_exc()
                        # Don't remove from buffer on error so user can retry
                        with self.state_lock:
                            self.episodes_pending_save.discard(episode_id)

                    finally:
                        # Clear in-progress status
                        with self.flush_lock:
                            if episode_id in self.flush_in_progress:
                                del self.flush_in_progress[episode_id]
                        self.flush_queue.task_done()

                except Exception as e:
                    print(f"❌ Unexpected error in flush worker: {e}")
                    import traceback

                    traceback.print_exc()
                    # Continue processing other requests

        self.flush_worker_thread = Thread(target=_flush_worker_loop, daemon=True, name="FlushWorker")
        self.flush_worker_thread.start()

    def _find_last_approved_critical_state(self, episode_id: int) -> int | None:
        """Find the last approved critical state in an episode.
        
        Searches both pending and completed states.
        MUST be called with state_lock held.
        
        Args:
            episode_id: Episode to search
            
        Returns:
            state_id of last approved critical state, or None if none found
        """
        all_states = {}
        
        # Merge pending and completed states
        if episode_id in self.pending_states_by_episode:
            all_states.update(self.pending_states_by_episode[episode_id])
        if episode_id in self.completed_states_buffer_by_episode:
            all_states.update(self.completed_states_buffer_by_episode[episode_id])
        
        # Find all approved critical states
        approved_critical = [
            state_id
            for state_id, state_info in all_states.items()
            if state_info.get("critical", False) and state_info.get("approval_status") == "approved"
        ]
        
        if not approved_critical:
            return None
        
        # Return the highest (last) state_id
        return max(approved_critical)
    
    def _build_flush_buffer(self, episode_id: int, last_approved_critical_id: int) -> dict:
        """Build a buffer of states to flush, up to and including last approved critical state.
        
        MUST be called with state_lock held.
        
        For the last approved critical state:
        - If it has action_to_save (fully labeled), use it
        - If not (partially labeled or unlabeled), create filler action_to_save with NaNs
        
        Args:
            episode_id: Episode to build buffer for
            last_approved_critical_id: State ID of last approved critical state
            
        Returns:
            Dictionary of state_id -> state_info for states to save
        """
        import torch
        
        buffer = {}
        
        # Collect all states from completed buffer with state_id <= last_approved_critical_id
        # Make COPIES to avoid modifying shared state objects
        completed = self.completed_states_buffer_by_episode.get(episode_id, {})
        for state_id, state_info in completed.items():
            if state_id <= last_approved_critical_id:
                buffer[state_id] = state_info.copy()
        
        # Check if the last approved critical state itself needs to be added
        if last_approved_critical_id not in buffer:
            # It's still in pending states - need to prepare it for saving
            pending = self.pending_states_by_episode.get(episode_id, {})
            if last_approved_critical_id in pending:
                # Create a COPY so we don't modify the original pending state
                state_info_copy = pending[last_approved_critical_id].copy()
                
                # Check if it has action_to_save
                if "action_to_save" not in state_info_copy:
                    # Need to create filler action_to_save with all NaNs
                    print(f"⚠️  State {last_approved_critical_id} not fully labeled, using NaN fillers")
                    
                    # Determine action dimensionality (7 joints by default)
                    action_dim = 7  # [joint_0, joint_1, joint_2, joint_3, joint_4, joint_5, left_carriage_joint]
                    total_action_size = self.required_responses_per_critical_state * action_dim
                    
                    # Create all-NaN action tensor
                    state_info_copy["action_to_save"] = torch.full(
                        (total_action_size,), float("nan"), dtype=torch.float32
                    )
                
                buffer[last_approved_critical_id] = state_info_copy
        
        return buffer

        # Start the worker thread
        self.flush_worker_thread = Thread(target=_flush_worker_loop, daemon=True, name="FlushWorker")
        self.flush_worker_thread.start()

    def flush_episode_now(self, episode_id: int) -> dict:
        """Request flush of collected frames for an episode to the dataset.

        This is non-blocking and thread-safe. The request is queued and processed
        by a persistent worker thread, avoiding deadlocks and race conditions.

        Args:
            episode_id: The episode to flush

        Returns:
            dict with "status" and "message" keys

        """
        # Check if flush already in progress (without blocking on state_lock)
        with self.flush_lock:
            if episode_id in self.flush_in_progress:
                progress = self.flush_in_progress[episode_id]
                elapsed = time.time() - progress.get("start_time", 0)
                return {
                    "status": "already_in_progress",
                    "message": f"Flush already in progress for episode {episode_id} (running for {elapsed:.1f}s)",
                    "episode_id": episode_id,
                    "num_states": progress.get("num_states", 0),
                }

        # Quick validation that episode exists (without deep inspection)
        episode_exists = False
        try:
            with self.state_lock:
                episode_exists = episode_id in self.completed_states_buffer_by_episode
        except Exception as e:
            # If we can't even check, still queue it - worker will handle gracefully
            print(f"⚠️  Could not verify episode {episode_id} exists: {e}")
            episode_exists = True  # Optimistic - let worker handle

        if not episode_exists:
            return {"status": "error", "message": f"Episode {episode_id} not found"}

        # Queue the flush request (non-blocking)
        try:
            self.flush_queue.put_nowait(episode_id)
            print(f"📥 Queued flush request for episode {episode_id}")
            return {
                "status": "queued",
                "message": f"Flush request queued for episode {episode_id}",
                "episode_id": episode_id,
            }
        except queue.Full:
            return {"status": "error", "message": "Flush queue is full, try again later"}

    def get_flush_status(self, episode_id: int = None) -> dict:
        """Get status of flush operations.

        Args:
            episode_id: If provided, get status for specific episode. Otherwise get all.

        Returns:
            dict with flush status information

        """
        with self.flush_lock:
            if episode_id is not None:
                if episode_id in self.flush_in_progress:
                    progress = self.flush_in_progress[episode_id]
                    return {
                        "episode_id": episode_id,
                        "status": "in_progress",
                        "num_states": progress.get("num_states", 0),
                        "elapsed": time.time() - progress.get("start_time", 0),
                    }
                else:
                    return {"episode_id": episode_id, "status": "idle"}
            else:
                # Return all in-progress flushes
                return {
                    "in_progress": [
                        {
                            "episode_id": ep_id,
                            "num_states": info.get("num_states", 0),
                            "elapsed": time.time() - info.get("start_time", 0),
                        }
                        for ep_id, info in self.flush_in_progress.items()
                    ],
                    "queue_size": self.flush_queue.qsize(),
                }

    def shutdown(self):
        """Gracefully shutdown the flush worker thread."""
        if self.flush_worker_thread and self.flush_worker_thread.is_alive():
            print("🛑 Shutting down flush worker...")
            self.flush_queue.put(None)  # Poison pill
            self.flush_worker_thread.join(timeout=5.0)
            if self.flush_worker_thread.is_alive():
                print("⚠️  Flush worker did not shutdown in time")
            else:
                print("✅ Flush worker shutdown complete")
