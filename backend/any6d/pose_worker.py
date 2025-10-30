#!/usr/bin/env python3
"""
pose_worker.py
Runs inside the Any6D310 conda environment.
- Watches <jobs_dir>/inbox for JSON jobs
- Loads RGB/Depth tensors from 'obs_path'
- Runs estimate_pose_from_tensors(...) using pose_fn
- Writes results to <jobs_dir>/outbox as JSON (+ optional visualization PNG)

Usage:
  conda run -n Any6D310 python pose_worker.py \
    --jobs-dir /tmp/crowd_obs_cache/pose_jobs \
    --object Cube_Blue \
    --mesh assets/meshes/cube_blue.obj \
    --prompt "blue cube"
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

# Ensure pose_fn is importable (assumes same repo)
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.append(str(THIS_DIR))

from estimate_pose import (
    create_engines,
    estimate_pose_from_tensors,
    reset_tracking,
    visualize_estimation,
)

def _load_obs(obs_path: str) -> dict:
    try:
        return torch.load(obs_path, map_location="cpu")
    except Exception as e:
        raise RuntimeError(f"failed to torch.load obs_path={obs_path}: {e}")

def _extract_rgb_depth(obs: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Try common keys. Returns (rgb_t, depth_t).
    - RGB expected shape (480,640,3) uint8 or float
    - Depth expected shape (480,640) float meters
    """
    # RGB
    rgb = (
        obs.get("observation.images.cam_main") or
        obs.get("observation.images.main") or
        obs.get("observation.cam_main") or
        obs.get("rgb") or
        obs.get("image")
    )
    if rgb is None:
        raise KeyError("RGB not found in obs dict (expected observation.images.cam_main, etc.)")
    if isinstance(rgb, np.ndarray):
        rgb = torch.from_numpy(rgb)
    assert isinstance(rgb, torch.Tensor), "RGB must be torch.Tensor or numpy.ndarray"
    if rgb.ndim == 3 and rgb.shape[0] == 3 and rgb.shape[-1] != 3:
        rgb = rgb.permute(1, 2, 0).contiguous()  # CHW->HWC

    # DEPTH
    depth = (
        obs.get("depth") or
        obs.get("observation.depth") or
        obs.get("observation.depth_main") or
        obs.get("observation.depth.cam_main")
    )
    if depth is None:
        raise KeyError("depth not found in obs dict (expected 'depth' key or similar)")
    if isinstance(depth, np.ndarray):
        depth = torch.from_numpy(depth)
    assert isinstance(depth, torch.Tensor), "Depth must be torch.Tensor or numpy.ndarray"
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth.squeeze(-1)
    return rgb, depth

def _as_K_array(K_like) -> np.ndarray:
    K = np.asarray(K_like, dtype=np.float32)
    if K.shape != (3, 3):
        raise ValueError(f"K must be 3x3, got {K.shape}")
    return K

def claim_job(inbox: Path, tmpdir: Path, object_name: str) -> Path | None:
    """
    Atomically move one job for 'object_name' from inbox -> tmpdir and return its path.
    """
    for p in inbox.glob("*.json"):
        try:
            with open(p, "r", encoding="utf-8") as f:
                hdr = json.load(f)
            if hdr.get("object") != object_name:
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
    ap.add_argument("--object", required=True, help="Object name this worker owns")
    ap.add_argument("--mesh", required=True, type=str)
    ap.add_argument("--prompt", default=None, type=str)
    ap.add_argument("--est-refine-iter", type=int, default=None)
    ap.add_argument("--track-refine-iter", type=int, default=None)
    ap.add_argument("--save-viz", action="store_true", help="Save RGB overlay PNG per job")
    args = ap.parse_args()

    inbox = (args.jobs_dir / "inbox").resolve()
    outbox = (args.jobs_dir / "outbox").resolve()
    tmpdir = (args.jobs_dir / "tmp").resolve()
    inbox.mkdir(parents=True, exist_ok=True)
    outbox.mkdir(parents=True, exist_ok=True)
    tmpdir.mkdir(parents=True, exist_ok=True)

    # Load mesh & create engines (expensive) once
    try:
        mesh = trimesh.load(args.mesh)
        if mesh.is_empty:
            raise RuntimeError("mesh is empty")
        _ = mesh.vertex_normals
    except Exception as e:
        print(f"[{args.object}] ✖ mesh load failed: {e}", flush=True)
        sys.exit(1)

    try:
        engines = create_engines(mesh, debug=0)
    except Exception as e:
        print(f"[{args.object}] ✖ engines init failed: {e}", flush=True)
        sys.exit(2)

    print(f"[{args.object}] ✅ worker ready (watching {inbox})", flush=True)

    while True:
        job_path = claim_job(inbox, tmpdir, args.object)
        if not job_path:
            time.sleep(0.1)
            continue

        try:
            with open(job_path, "r", encoding="utf-8") as f:
                job = json.load(f)
        except Exception as e:
            print(f"[{args.object}] ⚠️ bad job file {job_path.name}: {e}", flush=True)
            try:
                job_path.unlink()
            except Exception:
                pass
            continue

        job_id = job.get("job_id", "unknown")
        episode_id = str(job.get("episode_id"))
        state_id = int(job.get("state_id"))
        obs_path = job.get("obs_path")
        prompt = job.get("prompt") or args.prompt or args.object
        K = _as_K_array(job.get("K"))
        est_iters = int(job.get("est_refine_iter") or args.est_refine_iter or 20)
        track_iters = int(job.get("track_refine_iter") or args.track_refine_iter or 8)

        result = {
            "job_id": job_id,
            "episode_id": episode_id,
            "state_id": state_id,
            "object": args.object,
            "success": False,
        }

        try:
            obs = _load_obs(obs_path)
            rgb_t, depth_t = _extract_rgb_depth(obs)
        except Exception as e:
            result["error"] = f"obs load/extract failed: {e}"
            write_json_atomic(result, outbox, f"{job_id}.json")
            try:
                job_path.unlink()
            except Exception:
                pass
            continue

        # Run pose (stateless per call)
        try:
            # Make each call independent
            reset_tracking(engines)

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
            else:
                result["error"] = out.extras.get("error", "pose failed")
        except Exception as e:
            result["error"] = f"pose exception: {e}"

        # Optional visualization
        try:
            if result["success"] and (args.save_viz or bool(os.getenv("POSE_SAVE_VIZ", "0") == "1")):
                viz = visualize_estimation(
                    rgb_t=rgb_t, depth_t=depth_t, K=K, pose_out=out,
                    axis_scale=0.05, depth_min=0.1, depth_max=1.0, overlay_mask=True
                )
                png_path = outbox / f"viz_{job_id}.png"
                import cv2  # local import to avoid import cycles
                cv2.imwrite(str(png_path), viz["rgb_with_pose_bgr"])
                result["pose_viz_path"] = str(png_path)
        except Exception:
            # viz is best-effort
            pass

        # Emit result JSON
        try:
            write_json_atomic(result, outbox, f"{job_id}.json")
        except Exception as e:
            print(f"[{args.object}] ⚠️ failed to write result for {job_id}: {e}", flush=True)

        # Remove the claimed job file
        try:
            job_path.unlink()
        except Exception:
            pass


if __name__ == "__main__":
    main()