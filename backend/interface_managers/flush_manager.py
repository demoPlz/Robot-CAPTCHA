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
        episodes_pending_save: set,
        save_episode_callback,
    ):
        """Initialize flush manager.

        Args:
            state_lock: Shared lock protecting state data structures
            completed_states_buffer_by_episode: Reference to episode buffers
            episodes_pending_save: Reference to pending save tracking set
            save_episode_callback: Callback to save episode to dataset

        """
        self.state_lock = state_lock
        self.completed_states_buffer_by_episode = completed_states_buffer_by_episode
        self.episodes_pending_save = episodes_pending_save
        self.save_episode_callback = save_episode_callback

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
                            buffer = self.completed_states_buffer_by_episode.get(episode_id, {})
                            if not buffer:
                                print(f"⚠️  Episode {episode_id} has no completed states to flush")
                                with self.flush_lock:
                                    del self.flush_in_progress[episode_id]
                                self.flush_queue.task_done()
                                continue

                            # Create deep copy of buffer
                            buffer_copy = dict(buffer)
                            num_states = len(buffer_copy)

                            # Update progress tracking
                            with self.flush_lock:
                                self.flush_in_progress[episode_id]["num_states"] = num_states
                    except Exception as e:
                        print(f"❌ Error copying buffer for episode {episode_id}: {e}")
                        with self.flush_lock:
                            del self.flush_in_progress[episode_id]
                        self.flush_queue.task_done()
                        continue

                    # Perform actual save WITHOUT holding state_lock
                    print(f"💾 Flushing {num_states} states from episode {episode_id} to dataset...")
                    try:
                        self.save_episode_callback(buffer_copy)
                        print(f"✅ Successfully flushed episode {episode_id} ({num_states} states)")

                        # After successful save, remove saved states from buffer
                        with self.state_lock:
                            current_buffer = self.completed_states_buffer_by_episode.get(episode_id, {})
                            # Remove only the states we just saved
                            for state_id in buffer_copy.keys():
                                if state_id in current_buffer:
                                    del current_buffer[state_id]

                            # Remove from pending save set
                            self.episodes_pending_save.discard(episode_id)

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
