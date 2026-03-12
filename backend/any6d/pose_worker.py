#!/usr/bin/env python3
"""pose_worker.py.

Runs inside the any6d conda environment.
- Watches <jobs_dir>/inbox for JSON jobs
- Loads RGB/Depth tensors from 'obs_path'
- Runs estimate_pose_from_tensors(...) using pose_fn
- Writes results to <jobs_dir>/outbox as JSON (+ optional visualization PNG)

Usage:
  conda run -n any6d python pose_worker.py \
    --jobs-dir /tmp/crowd_obs_cache/pose_jobs \
    --object Cube_Blue \
    --mesh assets/meshes/cube_blue.obj \
    --prompt "blue cube"

Debug mode:
  Set POSE_WORKER_DEBUG=1 to enable debugpy - worker will WAIT for debugger to attach

"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import trimesh

# Parent death detection
try:
    import ctypes
    import signal as sig
    # PR_SET_PDEATHSIG = 1 on Linux - send signal when parent dies
    libc = ctypes.CDLL("libc.so.6")
    PR_SET_PDEATHSIG = 1
    HAS_PDEATHSIG = True
except Exception:
    HAS_PDEATHSIG = False

# ========== DEBUGPY SUPPORT ==========
# Check if debug mode is enabled via environment variable
if os.getenv("POSE_WORKER_DEBUG", "0") == "1":
    try:
        import debugpy

        # Use a different port for each worker to avoid conflicts
        # Parse --objects argument to get first object name for port calculation
        objects_arg = sys.argv[sys.argv.index("--objects") + 1] if "--objects" in sys.argv else "unknown"
        first_object = objects_arg.split(',')[0].strip()
        # Hash object name to get a consistent port offset (0-9)
        port_offset = hash(first_object) % 10
        debug_port = 5678 + port_offset

        # Check if already listening (in case of restart)
        if not debugpy.is_client_connected():
            try:
                debugpy.listen(("localhost", debug_port))
                print(f"\n{'='*60}", flush=True)
                print(f"🐛 [{first_object}] DEBUGPY READY", flush=True)
                print(f"🐛 Port: {debug_port}", flush=True)
                print(f"🐛 Objects: {objects_arg}", flush=True)
                print(f"🐛 Attach config: 'Attach to Pose Worker ({first_object})'", flush=True)
                print(f"🐛 WAITING FOR DEBUGGER TO ATTACH...", flush=True)
                print(f"{'='*60}\n", flush=True)

                # BLOCK until debugger attaches
                debugpy.wait_for_client()
                print(f"✅ [{first_object}] Debugger attached! Continuing...\n", flush=True)

                # Optional: break at the start
                # debugpy.breakpoint()
            except Exception as listen_err:
                print(f"⚠️  Failed to listen on port {debug_port}: {listen_err}", flush=True)
                print(f"⚠️  Continuing without debugger...", flush=True)
    except ImportError:
        print("⚠️  debugpy not installed in any6d environment.", flush=True)
        print("⚠️  Install with: conda activate any6d && pip install debugpy", flush=True)
    except Exception as e:
        print(f"⚠️  Debugpy setup failed: {e}", flush=True)
        import traceback

        traceback.print_exc()
# =====================================

# Ensure pose_fn is importable (assumes same repo)
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.append(str(THIS_DIR))

from estimate_pose import (
    create_engines,
    estimate_pose_from_tensors,
    estimate_pose_repeated,
    estimate_pose_multi_frame,
    reset_tracking,
    visualize_estimation,
    PoseOutput,
)


def _load_obs(obs_path: str) -> dict:
    try:
        return torch.load(obs_path, map_location="cpu")
    except Exception as e:
        raise RuntimeError(f"failed to torch.load obs_path={obs_path}: {e}")


def _extract_rgb_depth(obs: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Try common keys. Returns (rgb_t, depth_t).

    - RGB expected shape (480,640,3) uint8 or float
    - Depth expected shape (480,640) float meters

    """
    # RGB
    rgb = obs.get("observation.images.cam_main")
    if rgb is None:
        raise KeyError("RGB not found in obs dict (expected observation.images.cam_main, etc.)")
    if isinstance(rgb, np.ndarray):
        rgb = torch.from_numpy(rgb)
    assert isinstance(rgb, torch.Tensor), "RGB must be torch.Tensor or numpy.ndarray"
    if rgb.ndim == 3 and rgb.shape[0] == 3 and rgb.shape[-1] != 3:
        rgb = rgb.permute(1, 2, 0).contiguous()  # CHW->HWC

    # DEPTH
    depth = obs.get("depth")
    if depth is None:
        raise KeyError("depth not found in obs dict (expected 'depth' key or similar)")
    if isinstance(depth, np.ndarray):
        depth = torch.from_numpy(depth)
    assert isinstance(depth, torch.Tensor), "Depth must be torch.Tensor or numpy.ndarray"
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth.squeeze(-1)
    return rgb, depth


def _load_observations(obs_paths: list[str], K: np.ndarray) -> list[dict]:
    """Load multiple observation files and prepare them for multi-frame estimation.
    
    Returns list of dicts with 'rgb', 'depth', 'K' keys.
    """
    observations = []
    for obs_path in obs_paths:
        try:
            obs = _load_obs(obs_path)
            rgb_t, depth_t = _extract_rgb_depth(obs)
            observations.append({
                'rgb': rgb_t,
                'depth': depth_t,
                'K': torch.from_numpy(K.copy()),
            })
        except Exception as e:
            print(f"⚠️  Failed to load observation {obs_path}: {e}", flush=True)
            # Skip this observation and continue with others
            continue
    return observations


def _as_K_array(K_like) -> np.ndarray:
    K = np.asarray(K_like, dtype=np.float32)
    if K.shape != (3, 3):
        raise ValueError(f"K must be 3x3, got {K.shape}")
    return K


def claim_job(inbox: Path, tmpdir: Path, object_names: list[str]) -> Path | None:
    """Atomically move one job for any object in 'object_names' from inbox -> tmpdir and return its path.
    
    Args:
        inbox: Directory containing job files
        tmpdir: Directory to move claimed jobs to
        object_names: List of object names this worker handles (jobs for any of these will be claimed)
    """
    for p in inbox.glob("*.json"):
        try:
            with open(p, "r", encoding="utf-8") as f:
                hdr = json.load(f)
            # Accept job if it's for any of our objects
            if hdr.get("object") not in object_names:
                continue
        except Exception:
            # unreadable; try to move it aside to avoid hot-looping
            try:
                bad = tmpdir / f"bad_{p.name}"
                os.replace(p, bad)
            except Exception:
                pass
            continue

        # Try to claim (atomic move)
        dest = tmpdir / p.name
        try:
            os.replace(p, dest)
            return dest
        except FileNotFoundError:
            # Raced; continue
            continue
        except Exception:
            continue
    return None


def write_json_atomic(obj: dict, outdir: Path, name: str):
    tmp = outdir / (name + ".tmp")
    dst = outdir / name
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f)
    os.replace(tmp, dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs-dir", required=True, type=Path)
    ap.add_argument("--objects", required=True, help="Comma-separated list of object names this worker handles")
    ap.add_argument("--mesh", required=True, type=str)
    ap.add_argument("--mesh-scale", type=float, default=1.0, help="Scale factor to apply to mesh (e.g., 0.001 for mm to m)")
    ap.add_argument("--est-refine-iter", type=int, default=None)
    ap.add_argument("--track-refine-iter", type=int, default=None)
    ap.add_argument("--save-viz", action="store_true", help="Save RGB overlay PNG per job")
    ap.add_argument("--multi-frame", action="store_true", help="Enable multi-frame averaging with outlier rejection")
    ap.add_argument("--num-frames", type=int, default=5, help="Number of frames to average (multi-frame mode)")
    ap.add_argument("--repeated-inference", action="store_true", default=False, help="Run inference multiple times on same frame for outlier rejection")
    ap.add_argument("--num-inferences", type=int, default=5, help="Number of inferences to run (repeated mode)")
    ap.add_argument("--rgb-noise-std", type=float, default=1.0, help="RGB noise std dev for repeated inference (default: 1.0)")
    ap.add_argument("--depth-noise-std", type=float, default=0.001, help="Depth noise std dev for repeated inference (default: 0.001)")
    ap.add_argument("--inlier-threshold", type=float, default=0.05, help="Distance threshold for outlier rejection (meters)")
    ap.add_argument("--min-inliers", type=int, default=2, help="Minimum inliers required for valid result")
    args = ap.parse_args()
    
    # Parse comma-separated object names
    object_names = [obj.strip() for obj in args.objects.split(',')]
    
    # Create worker ID for logging (show first object, or "obj1+2more" for multiple)
    if len(object_names) == 1:
        worker_id = object_names[0]
    else:
        worker_id = f"{object_names[0]}+{len(object_names)-1}more"

    # Set up signal handler for graceful shutdown
    def signal_handler(signum, frame):
        print(f"\n[{worker_id}] 🛑 Received signal {signum}, shutting down...", flush=True)
        sys.exit(0)
    
    import signal
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    inbox = (args.jobs_dir / "inbox").resolve()
    outbox = (args.jobs_dir / "outbox").resolve()
    tmpdir = (args.jobs_dir / "tmp").resolve()
    inbox.mkdir(parents=True, exist_ok=True)
    outbox.mkdir(parents=True, exist_ok=True)
    tmpdir.mkdir(parents=True, exist_ok=True)

    # Load mesh & create engines (expensive) once
    print(f"[{worker_id}] 🔄 Initializing models...", flush=True)
    print(f"[{worker_id}]    Handling objects: {', '.join(object_names)}", flush=True)
    try:
        mesh = trimesh.load(args.mesh)
        if mesh.is_empty:
            raise RuntimeError("mesh is empty")
        if args.mesh_scale != 1.0:
            mesh.apply_scale(args.mesh_scale)
            print(f"[{worker_id}] Applied mesh scale factor {args.mesh_scale}", flush=True)
        _ = mesh.vertex_normals
    except Exception as e:
        print(f"[{worker_id}] ✖ mesh load failed: {e}", flush=True)
        sys.exit(1)

    try:
        engines = create_engines(mesh, debug=0)
    except Exception as e:
        print(f"[{worker_id}] ✖ engines init failed: {e}", flush=True)
        sys.exit(2)

    print(f"[{worker_id}] ✅ worker ready (watching {inbox})", flush=True)

    # Set up parent death detection (Linux only)
    if HAS_PDEATHSIG:
        try:
            # When parent dies, this process receives SIGTERM
            libc.prctl(PR_SET_PDEATHSIG, sig.SIGTERM)
            print(f"[{worker_id}] ✓ Parent death detection enabled (will exit if parent dies)", flush=True)
        except Exception as e:
            print(f"[{worker_id}] ⚠️  Failed to set parent death signal: {e}", flush=True)

    while True:
        job_path = claim_job(inbox, tmpdir, object_names)
        if not job_path:
            time.sleep(0.1)
            continue

        try:
            with open(job_path, "r", encoding="utf-8") as f:
                job = json.load(f)
        except Exception as e:
            print(f"[{worker_id}] ⚠️ bad job file {job_path.name}: {e}", flush=True)
            try:
                job_path.unlink()
            except Exception:
                pass
            continue

        job_id = job.get("job_id", "unknown")
        episode_id = int(job.get("episode_id"))
        state_id = int(job.get("state_id"))
        object_name = job.get("object")  # Get object name from job
        obs_path = job.get("obs_path")  # Single observation (legacy)
        obs_paths = job.get("obs_paths")  # Multiple observations (multi-frame)
        prompt = job.get("prompt", object_name)  # Use job's prompt (for color-specific masks)
        K = _as_K_array(job.get("K"))
        est_iters = int(job.get("est_refine_iter") or args.est_refine_iter or 20)
        track_iters = int(job.get("track_refine_iter") or args.track_refine_iter or 8)

        # Determine mode priority: multi-frame > repeated > single
        # Default is repeated inference for single frames
        use_multi_frame = args.multi_frame or obs_paths is not None
        use_repeated = args.repeated_inference if not use_multi_frame else False
        if not use_multi_frame and "repeated_inference" in job:
            use_repeated = bool(job.get("repeated_inference"))
        
        if use_multi_frame and obs_paths:
            print(f"[{object_name}] 🔍 Processing MULTI-FRAME job {job_id} (ep={episode_id}, state={state_id})", flush=True)
            print(f"[{object_name}]    {len(obs_paths)} observations, prompt: {prompt}", flush=True)
        elif use_repeated:
            print(f"[{object_name}] 🔍 Processing REPEATED-INFERENCE job {job_id} (ep={episode_id}, state={state_id})", flush=True)
            print(f"[{object_name}]    {args.num_inferences} inferences with noise, prompt: {prompt}", flush=True)
        else:
            print(f"[{object_name}] 🔍 Processing SINGLE-FRAME job {job_id} (ep={episode_id}, state={state_id})", flush=True)
            print(f"[{object_name}]    obs_path: {obs_path}, prompt: {prompt}", flush=True)

        result = {
            "job_id": job_id,
            "episode_id": episode_id,
            "state_id": state_id,
            "object": object_name,
            "success": False,
            "retry_count": job.get("retry_count", 0),  # Preserve retry count for retry logic
        }

        try:
            if use_multi_frame and obs_paths:
                print(f"[{object_name}] 📂 Loading {len(obs_paths)} observations...", flush=True)
                observations = _load_observations(obs_paths, K)
                if not observations:
                    raise RuntimeError("Failed to load any observations from provided paths")
                print(f"[{object_name}] ✓ Loaded {len(observations)} observations successfully", flush=True)
            else:
                print(f"[{object_name}] 📂 Loading observation from {obs_path}...", flush=True)
                obs = _load_obs(obs_path)
                print(f"[{object_name}] 🖼️  Extracting RGB and depth...", flush=True)
                rgb_t, depth_t = _extract_rgb_depth(obs)
                print(
                    f"[{object_name}] ✓ Observation loaded successfully (RGB shape={rgb_t.shape}, depth shape={depth_t.shape})",
                    flush=True,
                )
        except Exception as e:
            print(f"[{object_name}] ❌ Failed to load observation(s): {e}", flush=True)
            result["error"] = f"obs load/extract failed: {e}"
            write_json_atomic(result, outbox, f"{job_id}.json")
            try:
                job_path.unlink()
            except Exception:
                pass
            continue

        # Run pose (single, repeated, or multi-frame mode)
        try:
            # Reset tracking state before each job to avoid accumulation
            reset_tracking(engines)
            
            if use_multi_frame and obs_paths:
                # Multi-frame mode with outlier rejection
                print(f"[{object_name}] 🎯 Running MULTI-FRAME pose estimation ({len(observations)} frames)...", flush=True)
                out = estimate_pose_multi_frame(
                    mesh=mesh,
                    observations=observations,
                    language_prompt=prompt,
                    num_frames=args.num_frames,
                    inlier_threshold=args.inlier_threshold,
                    min_inliers=args.min_inliers,
                    est_refine_iter=est_iters,
                    track_refine_iter=track_iters,
                    debug=0,
                    engines=engines,
                )
                
                if out.success:
                    result["success"] = True
                    result["pose_cam_T_obj"] = out.pose_cam_T_obj.detach().cpu().numpy().tolist()
                    result["num_frames_processed"] = out.extras.get("num_frames_processed", 0)
                    result["successful_frames"] = out.extras.get("successful_frames", 0)
                    result["failed_frames"] = out.extras.get("failed_frames", 0)
                    result["num_inliers"] = out.extras.get("num_inliers", 0)
                    result["num_outliers"] = out.extras.get("num_outliers", 0)
                    
                    print(f"[{object_name}] ✅ Multi-frame pose estimation SUCCESS!", flush=True)
                    print(f"[{object_name}]    Processed: {result['num_frames_processed']} frames", flush=True)
                    print(f"[{object_name}]    Success: {result['successful_frames']}, Failed: {result['failed_frames']}", flush=True)
                    print(f"[{object_name}]    Inliers: {result['num_inliers']}, Outliers: {result['num_outliers']}", flush=True)
                    print(f"[{object_name}]    Averaged pose: {result['pose_cam_T_obj']}", flush=True)
                else:
                    result["error"] = out.extras.get("error", "multi-frame pose failed")
                    if "successful_frames" in out.extras:
                        result["successful_frames"] = out.extras["successful_frames"]
                    if "failed_frames" in out.extras:
                        result["failed_frames"] = out.extras["failed_frames"]
                    
                    error_msg = result["error"]
                    if "successful_frames" in result:
                        error_msg += f", successful={result['successful_frames']}"
                    if "failed_frames" in result:
                        error_msg += f", failed={result['failed_frames']}"
                    
                    print(f"[{object_name}] ❌ Multi-frame pose estimation FAILED: {error_msg}", flush=True)
                
                # Use last observation for visualization
                rgb_t = observations[-1]['rgb']
                depth_t = observations[-1]['depth']
                
            elif use_repeated:
                # Repeated inference mode with noise perturbations
                print(f"[{object_name}] 🎯 Running REPEATED-INFERENCE pose estimation ({args.num_inferences} inferences)...", flush=True)
                out = estimate_pose_repeated(
                    mesh=mesh,
                    rgb_t=rgb_t,
                    depth_t=depth_t,
                    K=K,
                    language_prompt=prompt,
                    num_inferences=args.num_inferences,
                    rgb_noise_std=args.rgb_noise_std,
                    depth_noise_std=args.depth_noise_std,
                    inlier_threshold=args.inlier_threshold,
                    min_inliers=args.min_inliers,
                    est_refine_iter=est_iters,
                    track_refine_iter=track_iters,
                    debug=0,
                    engines=engines,
                )
                
                if out.success:
                    result["success"] = True
                    result["pose_cam_T_obj"] = out.pose_cam_T_obj.detach().cpu().numpy().tolist()
                    result["num_inferences"] = out.extras.get("num_inferences", 0)
                    result["successful_inferences"] = out.extras.get("successful_inferences", 0)
                    result["failed_inferences"] = out.extras.get("failed_inferences", 0)
                    result["num_inliers"] = out.extras.get("num_inliers", 0)
                    result["num_outliers"] = out.extras.get("num_outliers", 0)
                    
                    print(f"[{object_name}] ✅ Repeated-inference pose estimation SUCCESS!", flush=True)
                    print(f"[{object_name}]    Inferences: {result['num_inferences']}", flush=True)
                    print(f"[{object_name}]    Success: {result['successful_inferences']}, Failed: {result['failed_inferences']}", flush=True)
                    print(f"[{object_name}]    Inliers: {result['num_inliers']}, Outliers: {result['num_outliers']}", flush=True)
                    print(f"[{object_name}]    Averaged pose: {result['pose_cam_T_obj']}", flush=True)
                else:
                    result["error"] = out.extras.get("error", "repeated inference failed")
                    if "successful_inferences" in out.extras:
                        result["successful_inferences"] = out.extras["successful_inferences"]
                    if "failed_inferences" in out.extras:
                        result["failed_inferences"] = out.extras["failed_inferences"]
                    
                    error_msg = result["error"]
                    if "successful_inferences" in result:
                        error_msg += f", successful={result['successful_inferences']}"
                    if "failed_inferences" in result:
                        error_msg += f", failed={result['failed_inferences']}"
                    
                    print(f"[{object_name}] ❌ Repeated-inference pose estimation FAILED: {error_msg}", flush=True)
                
            else:
                # Single-frame mode (original behavior)
                reset_tracking(engines)
                
                print(f"[{object_name}] 🎯 Running pose estimation...", flush=True)
                out = estimate_pose_from_tensors(
                    mesh=mesh,
                    rgb_t=rgb_t,
                    depth_t=depth_t,
                    K=K,
                    language_prompt=prompt,
                    est_refine_iter=est_iters,
                    track_refine_iter=track_iters,
                    debug=0,
                    engines=engines,
                )
                
                if out.success:
                    result["success"] = True
                    result["pose_cam_T_obj"] = out.pose_cam_T_obj.detach().cpu().numpy().tolist()
                    result["mask_area"] = int(out.extras.get("mask_area", 0))
                    if "score" in out.extras:
                        result["score"] = float(out.extras["score"])
                    print(f"[{object_name}] ✅ Pose estimation SUCCESS! mask_area={result['mask_area']}", flush=True)
                    if "score" in result:
                        print(f"[{object_name}]    confidence score: {result['score']:.3f}", flush=True)
                    print(f"[{object_name}]    pose_cam_T_obj: {result['pose_cam_T_obj']}", flush=True)
                else:
                    # Extract all available error information
                    result["error"] = out.extras.get("error", "pose failed")
                    if "reason" in out.extras:
                        result["reason"] = out.extras["reason"]
                    if "mask_area" in out.extras:
                        result["mask_area"] = int(out.extras["mask_area"])
                    if "score" in out.extras:
                        result["score"] = float(out.extras["score"])

                    # Print detailed error info
                    error_parts = [f"error={result['error']}"]
                    if "reason" in result:
                        error_parts.append(f"reason={result['reason']}")
                    if "mask_area" in result:
                        error_parts.append(f"mask_area={result['mask_area']}")
                    if "score" in result:
                        error_parts.append(f"score={result['score']:.3f}")

                    error_msg = ", ".join(error_parts)
                    print(f"[{object_name}] ❌ Pose estimation FAILED: {error_msg}", flush=True)
                    print(f"[{object_name}]    extras: {out.extras}", flush=True)
                    
        except Exception as e:
            result["error"] = f"pose exception: {e}"
            print(f"[{object_name}] ❌ Pose estimation EXCEPTION: {e}", flush=True)
            import traceback

            traceback.print_exc()
            
            # Create a dummy failed output for visualization to work
            out = PoseOutput(
                success=False,
                pose_cam_T_obj=None,
                mask=None,
                bbox_obj_frame=None,
                to_origin=None,
                extras={"error": str(e)},
            )

        # TEMPORARY: Always save visualization (both success and failure cases)
        try:
            viz = visualize_estimation(
                rgb_t=rgb_t,
                depth_t=depth_t,
                K=K,
                pose_out=out,
                axis_scale=0.05,
                depth_min=100.0,  # Depth is in millimeters, so 100mm = 0.1m
                depth_max=1000.0,  # 1000mm = 1.0m
                overlay_mask=True,
            )

            # Save all visualizations to outbox with clear naming
            status_str = "success" if result["success"] else "failed"
            png_path = outbox / f"viz_{job_id}_{status_str}.png"
            import cv2  # local import to avoid import cycles

            cv2.imwrite(str(png_path), viz["rgb_with_pose_bgr"])
            result["pose_viz_path"] = str(png_path)

            # Also save depth and mask visualizations
            if "depth_viz_bgr" in viz:
                depth_path = outbox / f"viz_{job_id}_{status_str}_depth.png"
                cv2.imwrite(str(depth_path), viz["depth_viz_bgr"])
            if "mask_gray" in viz:
                mask_path = outbox / f"viz_{job_id}_{status_str}_mask.png"
                cv2.imwrite(str(mask_path), viz["mask_gray"])

            print(f"[{object_name}] 💾 Saved visualization to {png_path}", flush=True)
        except Exception as e:
            # viz is best-effort, but log errors for debugging
            print(f"[{object_name}] ⚠️ Visualization failed: {e}", flush=True)

        # Emit result JSON
        try:
            write_json_atomic(result, outbox, f"{job_id}.json")
            print(
                f"[{object_name}] 📤 Result written to outbox: {job_id}.json (success={result['success']})", flush=True
            )
        except Exception as e:
            print(f"[{object_name}] ⚠️ failed to write result for {job_id}: {e}", flush=True)

        # Remove the claimed job file
        try:
            job_path.unlink()
            print(f"[{object_name}] 🗑️  Job file removed", flush=True)
        except Exception:
            pass
        
        # === VRAM cleanup: critical for long-running workers ===
        # Explicitly delete large tensors to free GPU memory
        try:
            # Delete observation tensors
            if 'rgb_t' in locals():
                del rgb_t
            if 'depth_t' in locals():
                del depth_t
            if 'obs' in locals():
                del obs
            if 'observations' in locals():
                del observations
            
            # Delete inference outputs
            if 'out' in locals():
                del out
            
            # Delete visualization tensors
            if 'viz' in locals():
                del viz
            
            # Force CUDA cache cleanup every job to prevent accumulation
            if torch.cuda.is_available():
                # Clear any cached gradients (shouldn't exist in inference mode, but be safe)
                torch.cuda.empty_cache()
                # Synchronize to ensure all GPU operations are complete
                torch.cuda.synchronize()
                
                # Optional: More aggressive cleanup every N jobs
                # Uncomment if you still see accumulation:
                # import gc
                # gc.collect()
                # torch.cuda.reset_peak_memory_stats()
                
        except Exception as e:
            print(f"[{object_name}] ⚠️ Cleanup warning: {e}", flush=True)


if __name__ == "__main__":
    main()
