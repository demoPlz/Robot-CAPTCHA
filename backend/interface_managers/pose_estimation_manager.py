"""Pose Estimation Manager Module.

Handles 6D pose estimation for objects using any6d workers. Manages cross-environment communication via disk-based job
queues.

"""

import json
import os
import subprocess
import time
import uuid
from pathlib import Path
from threading import Lock, Thread

import numpy as np
from scipy.spatial.transform import Rotation as R


class PoseEstimationManager:
    """Manages 6D pose estimation for objects using any6d workers.

    Responsibilities:
    - Spawn and manage any6d worker processes (one per object)
    - Disk-based job queue management (inbox/outbox)
    - Background results watcher thread
    - Intrinsics selection and job creation
    - Results integration into state info

    Attributes:
        pose_jobs_root: Root directory for job queues
        pose_inbox: Directory where jobs are enqueued
        pose_outbox: Directory where results are written
        pose_tmp: Temporary directory for atomic writes

    """

    def __init__(
        self,
        obs_cache_root: Path,
        object_mesh_paths: dict[str, str] | None,
        objects: dict[str, str] | None,
        calibration_manager,
        state_lock: Lock,
        pending_states_by_episode: dict,
        pose_camera_name: str = "realsense",  # RealSense D455 is used for pose estimation
        use_random_poses: bool = False,
        random_pose_bounds: dict | None = None,
    ):
        """Initialize pose estimation manager.

        Args:
            obs_cache_root: Root directory for observation cache (parent of pose_jobs)
            object_mesh_paths: Dict of object name -> mesh file path
            objects: Dict of object name -> language prompt
            calibration_manager: CalibrationManager instance for intrinsics and extrinsics access
            state_lock: Lock protecting pending_states_by_episode
            pending_states_by_episode: Reference to episode state dict for result integration
            pose_camera_name: Name of camera used for pose estimation (default: "realsense" for D455)
                             Used for transforming poses from camera frame to world frame.
                             Requires extrinsics calibration file: data/calib/extrinsics_realsense_d455.npz
            use_random_poses: If True, skip real pose estimation and use random fixed poses
            random_pose_bounds: Dict with x_min, x_max, y_min, y_max, z_min, z_max for random pose bounds

        """
        self.object_mesh_paths = object_mesh_paths
        self.objects = objects
        self.calibration = calibration_manager
        self.state_lock = state_lock
        self.pending_states_by_episode = pending_states_by_episode
        self.pose_camera_name = pose_camera_name  # Store camera name for coordinate transformation
        self.use_random_poses = use_random_poses
        self.random_pose_bounds = random_pose_bounds if random_pose_bounds is not None else {
            "x_min": -0.3, "x_max": 0.3,
            "y_min": -0.3, "y_max": 0.3,
            "z_min": 0.0, "z_max": 0.3
        }

        # Disk-backed job queue shared with any6d env workers
        self.pose_jobs_root = (obs_cache_root / "pose_jobs").resolve()
        self.pose_inbox = self.pose_jobs_root / "inbox"
        self.pose_outbox = self.pose_jobs_root / "outbox"
        self.pose_tmp = self.pose_jobs_root / "tmp"

        # Create directories
        for d in (self.pose_inbox, self.pose_outbox, self.pose_tmp):
            try:
                d.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass

        # Clean up stale jobs from previous runs
        self._cleanup_job_queues()

        # Worker process management
        self._pose_worker_procs: dict[str, subprocess.Popen] = {}
        self._pose_results_thread: Thread | None = None

        # === Last known good poses for fallback on estimation failure ===
        # Maps object_name -> {"pos": [x,y,z], "rot": [x,y,z,w]} or None
        self.last_known_poses: dict[str, dict | None] = {}
        
        # === Random pose generation for each object (when use_random_poses=True) ===
        # Store one random pose per object that will be reused for all states
        self._random_fixed_poses: dict[str, dict] = {}

        # Skip worker initialization if using random poses
        if self.use_random_poses:
            print("🎲 Random pose mode enabled - skipping pose estimation workers")
            self._generate_random_fixed_poses()
        else:
            # Start workers and results watcher
            self._start_pose_workers()
            self._start_pose_results_watcher()

    # =========================
    # Random Pose Generation
    # =========================
    
    def _generate_random_fixed_poses(self):
        """Generate one random pose per object that will be reused for all states.
        
        Poses are generated within the bounds specified in random_pose_bounds.
        """
        if not self.object_mesh_paths:
            return
        
        # Only generate poses for objects that are tracked
        objects_to_track = [obj for obj in self.object_mesh_paths.keys() 
                           if not self.objects or obj in self.objects]
        
        print(f"🎲 Generating random fixed poses for {len(objects_to_track)} objects...")
        
        for obj in objects_to_track:
            # Generate random position within bounds
            x = np.random.uniform(self.random_pose_bounds["x_min"], 
                                 self.random_pose_bounds["x_max"])
            y = np.random.uniform(self.random_pose_bounds["y_min"], 
                                 self.random_pose_bounds["y_max"])
            z = np.random.uniform(self.random_pose_bounds["z_min"], 
                                 self.random_pose_bounds["z_max"])
            
            # Generate random orientation as quaternion
            # Use uniform sampling on unit sphere for rotation
            quat = R.random().as_quat()  # Returns [x, y, z, w]
            
            self._random_fixed_poses[obj] = {
                "pos": [float(x), float(y), float(z)],
                "rot": quat.tolist()
            }
            
            print(f"   {obj}: pos=[{x:.3f}, {y:.3f}, {z:.3f}], "
                  f"rot=[{quat[0]:.3f}, {quat[1]:.3f}, {quat[2]:.3f}, {quat[3]:.3f}]")
        
        print("✅ Random fixed poses generated")
    
    def _apply_random_poses_to_state(self, episode_id: int, state_id: int):
        """Apply the pre-generated random poses to a state.
        
        Args:
            episode_id: Episode ID
            state_id: State ID
        """
        with self.state_lock:
            ep = self.pending_states_by_episode.get(episode_id)
            if not ep or state_id not in ep:
                print(f"⚠️  State disappeared (ep={episode_id}, state={state_id})")
                return
            
            state_info = ep[state_id]
            
            # Initialize object_poses dict if not present
            if "object_poses" not in state_info:
                state_info["object_poses"] = {}
            
            # Copy the random poses to this state
            for obj, pose in self._random_fixed_poses.items():
                state_info["object_poses"][obj] = pose.copy()
            
            print(f"🎲 Applied random fixed poses to state (ep={episode_id}, state={state_id})")

    # =========================
    # Pose Transformation Utilities
    # =========================

    def _matrix_to_pos_quat(self, T: list[list[float]]) -> dict:
        """Convert 4x4 transformation matrix to position and quaternion.

        Args:
            T: 4x4 transformation matrix (list of lists)

        Returns:
            Dict with "pos" (list of 3 floats [x, y, z]) and "rot" (list of 4 floats [x, y, z, w] quaternion)

        """
        T_np = np.array(T)
        position = T_np[:3, 3].tolist()
        rotation_matrix = T_np[:3, :3]
        quat_xyzw = R.from_matrix(rotation_matrix).as_quat()  # Returns [x, y, z, w]
        return {"pos": position, "rot": quat_xyzw.tolist()}

    def _transform_camera_to_world(self, pose_cam_T_obj: list[list[float]], camera_name: str) -> dict:
        """Transform object pose from camera frame to world frame and convert to pos+quat.

        Args:
            pose_cam_T_obj: 4x4 matrix representing object pose in camera frame
            camera_name: Name of camera (e.g. "front", "left", "right", "top")

        Returns:
            Dict with "pos" and "rot" in world frame, or None if camera calibration not available

        """
        # Get camera pose in world frame from calibration manager
        camera_poses = self.calibration.get_camera_poses()
        camera_pose_key = f"{camera_name}_pose"

        if camera_pose_key not in camera_poses:
            print(f"⚠️  No calibration found for camera '{camera_name}', cannot transform to world frame")
            print(f"   Available camera pose keys: {list(camera_poses.keys())}")
            print(f"   Looking for key: '{camera_pose_key}'")
            return None

        world_T_cam = np.array(camera_poses[camera_pose_key])  # 4x4 matrix: world <- camera
        cam_T_obj = np.array(pose_cam_T_obj)  # 4x4 matrix: camera <- object

        # Compute world <- object = (world <- camera) @ (camera <- object)
        world_T_obj = world_T_cam @ cam_T_obj

        # Convert to position + quaternion
        return self._matrix_to_pos_quat(world_T_obj.tolist())

    # =========================
    # Initialization and Cleanup
    # =========================

    def _cleanup_job_queues(self):
        """Remove stale job files from inbox and outbox directories.

        Called on initialization to prevent workers from processing jobs from previous runs.

        """
        try:
            # Clean inbox (pending jobs)
            for f in self.pose_inbox.glob("*.json"):
                try:
                    f.unlink()
                except Exception:
                    pass

            # Clean outbox (completed results)
            for f in self.pose_outbox.glob("*.json"):
                try:
                    f.unlink()
                except Exception:
                    pass

            # Clean tmp (partially written jobs)
            for f in self.pose_tmp.glob("*.json"):
                try:
                    f.unlink()
                except Exception:
                    pass

            print("🧹 Cleaned up stale pose jobs from previous runs")
        except Exception as e:
            print(f"⚠️  Failed to cleanup job queues: {e}")

    def _start_pose_workers(self):
        """Spawn ONE persistent worker per object (they run continuously and process jobs sequentially). Worker script
        path can be overridden via $POSE_WORKER_SCRIPT.

        Set SKIP_POSE_WORKERS=1 to disable auto-spawning (useful for manual debugging).

        """
        if os.getenv("SKIP_POSE_WORKERS", "0") == "1":
            print("🐛 SKIP_POSE_WORKERS=1: Not spawning pose workers (attach manually)")
            return

        if not self.object_mesh_paths:
            print("⚠️  No object_mesh_paths provided; pose workers not started.")
            return

        worker_script = os.getenv(
            "POSE_WORKER_SCRIPT", str((Path(__file__).resolve().parent.parent / "any6d" / "pose_worker.py").resolve())
        )
        pose_env = os.getenv("POSE_ENV", "any6d")

        # Build CUDA library paths for any6d
        conda_prefix = Path.home() / "miniconda3" / "envs" / pose_env
        cuda_lib_path = f"{conda_prefix}/lib:{conda_prefix}/targets/x86_64-linux/lib"
        worker_env = os.environ.copy()
        worker_env["LD_LIBRARY_PATH"] = cuda_lib_path

        # Spawn ONE persistent worker per object (parallel processing)
        print("🔄 Starting pose estimation workers (one per object)...")

        # Track ready status for each worker (True=ready, False=pending, None=failed)   
        # Only spawn workers for objects in self.objects (filter by presence in objects dict)
        workers_status = {obj: False for obj in self.object_mesh_paths.keys() if not self.objects or obj in self.objects}
        status_lock = Lock()

        for obj, mesh_path in self.object_mesh_paths.items():
            # Skip objects not in self.objects
            if self.objects and obj not in self.objects:
                continue
            
            lang_prompt = (self.objects or {}).get(obj, obj)

            cmd = [
                "conda",
                "run",
                "--no-capture-output",
                "-n",
                pose_env,
                "python",
                worker_script,
                "--jobs-dir",
                str(self.pose_jobs_root),
                "--object",
                obj,
                "--mesh",
                str(mesh_path),
                "--prompt",
                str(lang_prompt),
            ]

            try:
                proc = subprocess.Popen(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=worker_env
                )
                self._pose_worker_procs[obj] = proc
                print(f"✓ Pose worker for '{obj}' started (PID {proc.pid})")

                # Start thread to print worker output and detect ready/failure signals
                def _print_worker_output(proc, obj_name):
                    try:
                        for line in iter(proc.stdout.readline, ""):
                            if line:
                                print(f"[{obj_name}] {line.rstrip()}")
                                # Detect ready signal: "✅ worker ready (watching ...)"
                                if "✅" in line and "worker ready" in line:
                                    with status_lock:
                                        workers_status[obj_name] = True
                                # Detect initialization failures: "✖ mesh load failed" or "✖ engines init failed"
                                elif "✖" in line and ("mesh load failed" in line or "engines init failed" in line):
                                    with status_lock:
                                        workers_status[obj_name] = None  # None indicates failure
                    except Exception:
                        pass
                    finally:
                        proc.stdout.close()

                Thread(target=_print_worker_output, args=(proc, obj), daemon=True).start()

            except Exception as e:
                print(f"⚠️  Failed to start pose worker for '{obj}': {e}")
                with status_lock:
                    workers_status[obj] = None  # Mark as failed

        # Wait for all workers to be ready or fail (with timeout)
        print("⏳ Waiting for pose workers to initialize...")
        timeout_s = float(os.getenv("POSE_WORKER_INIT_TIMEOUT", "30.0"))
        deadline = time.time() + timeout_s

        while time.time() < deadline:
            with status_lock:
                # Check if any workers failed (None status)
                failed = [obj for obj, status in workers_status.items() if status is None]
                if failed:
                    print(f"❌ Worker initialization FAILED for: {failed}")
                    print(f"❌ Check worker logs above for details (mesh load or engine init errors)")
                    # Continue anyway - maybe other workers can still work
                    return

                # Check if all workers are ready (True status)
                if all(status is True for status in workers_status.values()):
                    print("✅ All pose workers ready!")
                    return
            time.sleep(0.1)

        # Timeout - report which workers are still pending
        with status_lock:
            pending = [obj for obj, status in workers_status.items() if status is False]
            failed = [obj for obj, status in workers_status.items() if status is None]

        if failed:
            print(f"❌ Workers failed during initialization: {failed}")
        if pending:
            print(f"⚠️  Timeout waiting for pose workers: {pending} not ready after {timeout_s}s")
        if pending or failed:
            print(f"⚠️  Continuing anyway, but pose estimation may fail for affected objects...")

    def _start_pose_results_watcher(self):
        self._pose_results_thread = Thread(target=self._pose_results_watcher, daemon=True)
        self._pose_results_thread.start()

    def _pose_results_watcher(self):
        """Poll pose_jobs/outbox for result JSONs and fold them into state_info['object_poses']."""
        print("📬 Results watcher thread started")
        while True:
            try:
                for p in self.pose_outbox.glob("*.json"):
                    try:
                        with open(p, "r", encoding="utf-8") as f:
                            result = json.load(f)
                        os.remove(p)

                        ep_id = result.get("episode_id")
                        st_id = result.get("state_id")
                        obj = result.get("object")
                        pose_cam_T_obj = result.get("pose_cam_T_obj")  # 4x4 matrix in camera frame, may be None

                        # Transform pose from camera frame to world frame and convert to pos+quat
                        if pose_cam_T_obj is not None:
                            try:
                                # Extract camera frame position for debugging
                                cam_T_obj = np.array(pose_cam_T_obj)
                                pos_cam = cam_T_obj[:3, 3]

                                pose_world = self._transform_camera_to_world(pose_cam_T_obj, self.pose_camera_name)
                                if pose_world is None:
                                    print(
                                        f"⚠️  Could not transform pose to world frame for {obj} "
                                        f"(camera={self.pose_camera_name}), skipping"
                                    )
                                    continue

                                # Debug output: show before/after transformation
                                print(f"🔄 [{obj}] Pose transform:")
                                print(f"   Camera frame: X={pos_cam[0]:+.3f}, Y={pos_cam[1]:+.3f}, Z={pos_cam[2]:+.3f}")
                                print(
                                    f"   World frame:  X={pose_world['pos'][0]:+.3f}, Y={pose_world['pos'][1]:+.3f}, Z={pose_world['pos'][2]:+.3f}"
                                )
                            except Exception as e:
                                print(f"⚠️  Failed to transform pose for {obj}: {e}")
                                import traceback

                                traceback.print_exc()
                                continue
                        else:
                            pose_world = None

                        # RETRY LOGIC: Decide if we should retry (outside lock to avoid deadlock during file I/O)
                        should_retry = False
                        retry_job_data = None
                        obs_path_for_retry = None
                        
                        if pose_world is None:
                            retry_count = result.get("retry_count", 0)
                            max_retries = 5
                            
                            if retry_count < max_retries:
                                # Need to get obs_path from state before releasing lock
                                with self.state_lock:
                                    ep = self.pending_states_by_episode.get(ep_id)
                                    if ep and st_id in ep:
                                        obs_path_for_retry = ep[st_id].get("obs_path")
                                
                                if obs_path_for_retry:
                                    should_retry = True
                                    print(f"🔄 Pose estimation failed for {obj}, retrying ({retry_count + 1}/{max_retries})...")
                                    
                                    job_id = f"{ep_id}_{st_id}_{obj}_{uuid.uuid4().hex[:8]}"
                                    retry_job_data = {
                                        "job_id": job_id,
                                        "episode_id": int(ep_id),
                                        "state_id": int(st_id),
                                        "object": obj,
                                        "obs_path": obs_path_for_retry,
                                        "K": self._intrinsics_for_pose(),
                                        "prompt": (self.objects or {}).get(obj, obj),
                                        "est_refine_iter": int(os.getenv("POSE_EST_ITERS", "20")),
                                        "track_refine_iter": int(os.getenv("POSE_TRACK_ITERS", "8")),
                                        "retry_count": retry_count + 1,
                                    }
                        
                        # Write retry job file OUTSIDE the lock
                        if should_retry and retry_job_data:
                            tmp = self.pose_tmp / f"{retry_job_data['job_id']}.json"
                            dst = self.pose_inbox / f"{retry_job_data['job_id']}.json"
                            try:
                                with open(tmp, "w", encoding="utf-8") as f:
                                    json.dump(retry_job_data, f)
                                os.replace(tmp, dst)
                                print(f"   ✅ Retry job enqueued: {dst.name}")
                                # Skip setting object_poses - wait for retry result
                                continue
                            except Exception as e:
                                print(f"⚠️  Failed to enqueue retry job: {e}")
                                # Fall through to set fallback pose
                        
                        # Set pose in state (success, fallback, or None)
                        with self.state_lock:
                            ep = self.pending_states_by_episode.get(ep_id)
                            if not ep or st_id not in ep:
                                continue
                            st = ep[st_id]
                            if "object_poses" not in st:
                                st["object_poses"] = {}

                            if pose_world is None:
                                retry_count = result.get("retry_count", 0)
                                max_retries = 5
                                
                                # Max retries exceeded or retry enqueue failed - use fallback
                                if retry_count >= max_retries:
                                    print(f"⚠️  Max retries ({max_retries}) exceeded for {obj}")
                                
                                fallback_pose = self.last_known_poses.get(obj)
                                if fallback_pose is not None:
                                    print(f"⚠️  Pose estimation failed for {obj}, using last known pose:")
                                    print(
                                        f"   Fallback: X={fallback_pose['pos'][0]:+.3f}, Y={fallback_pose['pos'][1]:+.3f}, Z={fallback_pose['pos'][2]:+.3f}"
                                    )
                                    st["object_poses"][obj] = fallback_pose
                                else:
                                    print(f"❌ Pose estimation failed for {obj} and no previous pose available")
                                    st["object_poses"][obj] = None
                            else:
                                # SUCCESS: Store the new pose and update last known pose
                                st["object_poses"][obj] = pose_world
                                self.last_known_poses[obj] = pose_world
                                print(f"✅ Updated last known pose for {obj}")
                    except Exception as e:
                        print(f"⚠️  Failed to process pose result {p.name}: {e}")
                        try:
                            os.remove(p)
                        except Exception:
                            pass
            except Exception:
                # Keep the watcher alive
                time.sleep(0.2)
            time.sleep(0.1)

    def _intrinsics_for_pose(self) -> list[list[float]]:
        """Returns 3x3 K to send to pose workers.

        Returns a Python list-of-lists (JSON-serializable).

        """
        realsense_calib = self.calibration.repo_root / "data" / "calib" / "intrinsics_realsense_d455.npz"
        if realsense_calib.exists():
            data = np.load(realsense_calib, allow_pickle=True)
            K = np.asarray(data["Knew"], dtype=np.float64)  # Use Knew (same as K for RealSense)
            return K.tolist()
        else:
            print("⚠️  RealSense D455 intrinsics not found")
            exit(1)

    def enqueue_pose_jobs_for_state(
        self,
        episode_id: str,
        state_id: int,
        state_info: dict,
        wait: bool = True,
        timeout_s: float | None = None,
    ) -> bool:
        """Enqueue one pose-estimation job per object into pose_jobs/inbox, then (optionally) block until results for
        *all* objects are folded into pending_states_by_episode[episode_id][state_id]['object_poses'] by the results
        watcher.

        Returns:
            True  -> all objects reported (success or failure) within timeout
            False -> state disappeared or timed out before all objects reported

        """
        # If using random poses, apply them immediately and return
        if self.use_random_poses:
            self._apply_random_poses_to_state(episode_id, state_id)
            return True
        
        if not self.object_mesh_paths:
            # Nothing to do; treat as ready.
            return True

        # Only process objects that are in self.objects (if objects dict is provided)
        expected_objs = [obj for obj in self.object_mesh_paths.keys() if not self.objects or obj in self.objects]
        
        if not expected_objs:
            # Nothing to do; treat as ready.
            return True

        # ---------- Enqueue jobs (do not mark object_poses yet) ----------
        print(f"📬 Enqueueing pose jobs for episode={episode_id} state={state_id}")
        for obj, mesh_path in self.object_mesh_paths.items():
            # Skip objects not in self.objects
            if self.objects and obj not in self.objects:
                continue
                
            job_id = f"{episode_id}_{state_id}_{obj}_{uuid.uuid4().hex[:8]}"
            job = {
                "job_id": job_id,
                "episode_id": int(episode_id),
                "state_id": int(state_id),
                "object": obj,
                "obs_path": state_info.get("obs_path"),
                "K": self._intrinsics_for_pose(),  # 3x3 list
                "prompt": (self.objects or {}).get(obj, obj),  # language prompt
                # Optional knobs:
                "est_refine_iter": int(os.getenv("POSE_EST_ITERS", "20")),
                "track_refine_iter": int(os.getenv("POSE_TRACK_ITERS", "8")),
            }
            print(f"   📝 Creating job {job_id}")
            print(f"      obj={obj}, obs_path={job['obs_path']}")
            tmp = self.pose_tmp / f"{job_id}.json"
            dst = self.pose_inbox / f"{job_id}.json"
            try:
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(job, f)
                os.replace(tmp, dst)  # atomic move
                print(f"   ✅ Job written to inbox: {dst.name}")
            except Exception as e:
                print(f"⚠️  Failed to enqueue pose job {job_id}: {e}")

        if not wait:
            return True

        # ---------- Wait for watcher to fold ALL results into state ----------
        # NOTE: Do NOT hold self.state_lock while sleeping; watcher needs it.
        try:
            timeout = float(timeout_s if timeout_s is not None else os.getenv("POSE_WAIT_TIMEOUT_S", "20.0"))
        except Exception:
            timeout = 20.0
        deadline = time.time() + max(0.0, timeout)

        # We consider a job "done" when the watcher has inserted a key for that object,
        # regardless of success (pose may be None on failure). Presence == finished.
        while True:
            with self.state_lock:
                ep = self.pending_states_by_episode.get(episode_id)
                if not ep or state_id not in ep:
                    print(f"⚠️  State disappeared (ep={episode_id}, state={state_id})")
                    return False
                st = ep[state_id]
                poses = st.get("object_poses", {})
                done = all(obj in poses for obj in expected_objs)

            if done:
                return True

            # if time.time() > deadline:
            #     with self.state_lock:
            #         poses_now = list(self.pending_states_by_episode.get(episode_id, {}).get(state_id, {}).get("object_poses", {}).keys())
            #     print(f"⚠️  Timed out waiting for poses (ep={episode_id}, state={state_id}). "
            #         f"Have={poses_now}, expected={expected_objs}")
            #     return False

            time.sleep(0.02)

    def stop(self):
        """Stop all pose worker processes.
        
        Attempts graceful termination first (SIGTERM), then force kills (SIGKILL) if needed.
        """
        if not self._pose_worker_procs:
            return
        
        print(f"🛑 Stopping {len(self._pose_worker_procs)} pose worker(s)...")
        
        for obj, proc in self._pose_worker_procs.items():
            try:
                # Check if already dead
                if proc.poll() is not None:
                    print(f"✓ Pose worker for '{obj}' already terminated")
                    continue
                
                # Try graceful termination
                proc.terminate()
                try:
                    proc.wait(timeout=2.0)
                    print(f"✓ Pose worker for '{obj}' stopped gracefully")
                except subprocess.TimeoutExpired:
                    # Force kill if termination times out
                    print(f"⚠️  Pose worker for '{obj}' didn't respond to SIGTERM, force killing...")
                    proc.kill()
                    proc.wait(timeout=1.0)
                    print(f"✓ Pose worker for '{obj}' force killed")
            except Exception as e:
                print(f"⚠️  Error stopping pose worker for '{obj}': {e}")
                # Try to force kill anyway
                try:
                    proc.kill()
                except Exception:
                    pass
        
        self._pose_worker_procs.clear()
        print("✅ All pose workers stopped")
