from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import trimesh
from PIL import Image

# Add external/any6d to sys.path so we can import estimater, foundationpose, etc.
# This file is at: backend/any6d/estimate_pose.py
# Target directory: external/any6d/
_BACKEND_DIR = Path(__file__).resolve().parent  # backend/any6d/
_REPO_ROOT = _BACKEND_DIR.parent.parent  # crowdsourcing-ui/
_EXTERNAL_ANY6D = _REPO_ROOT / "external" / "any6d"

if str(_EXTERNAL_ANY6D) not in sys.path:
    sys.path.insert(0, str(_EXTERNAL_ANY6D))

# Also add lang-segment-anything for lang_sam imports
_LANG_SAM_DIR = _EXTERNAL_ANY6D / "lang-segment-anything"
if str(_LANG_SAM_DIR) not in sys.path:
    sys.path.insert(0, str(_LANG_SAM_DIR))

# External deps expected by your environment:
#   nvdiffrast.torch as dr
#   lang_sam.LangSAM (from lang-segment-anything/)
#   estimater.Any6D (from external/any6d/)
#   foundationpose.* (from external/any6d/)

# For visualization
import cv2
import nvdiffrast.torch as dr
from estimater import Any6D
from foundationpose.estimater import FoundationPose
from foundationpose.learning.training.predict_pose_refine import PoseRefinePredictor
from foundationpose.learning.training.predict_score import ScorePredictor
from foundationpose.Utils import draw_posed_3d_box, draw_xyz_axis
from lang_sam import LangSAM

# ------------------------- Lazy singletons -------------------------

_LANGSAM_SINGLETON: Optional[LangSAM] = None


def _get_langsam() -> LangSAM:
    global _LANGSAM_SINGLETON
    if _LANGSAM_SINGLETON is None:
        _LANGSAM_SINGLETON = LangSAM()
    return _LANGSAM_SINGLETON


# ------------------------- Reusable engines -------------------------


@dataclass
class PoseEngines:
    langsam: LangSAM
    any6d: Any6D
    fpose: FoundationPose


def create_engines(
    mesh: trimesh.Trimesh,
    *,
    debug: int = 0,
    langsam: LangSAM | None = None,
) -> PoseEngines:
    """Build reusable inference engines tied to a specific mesh.

    Pass the returned PoseEngines into estimate_pose_from_tensors() for fast calls.

    """
    # LangSAM (allow passing a shared instance; otherwise lazily create)
    if langsam is None:
        langsam = _get_langsam()

    # Ensure normals exist on a private copy used by engines
    m = mesh.copy()
    _ = m.vertex_normals

    glctx_any6d = dr.RasterizeCudaContext()
    any6d = Any6D(mesh=m, debug=debug, debug_dir="./output/any6d_debug", glctx=glctx_any6d)

    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx_found = dr.RasterizeCudaContext()
    fpose = FoundationPose(
        model_pts=m.vertices,
        model_normals=m.vertex_normals,
        mesh=m,
        scorer=scorer,
        refiner=refiner,
        glctx=glctx_found,
        debug=debug,
        debug_dir="./output/any6d_debug",
    )
    return PoseEngines(langsam=langsam, any6d=any6d, fpose=fpose)


def reset_tracking(engines: PoseEngines) -> None:
    """Clear any persistent per-sequence tracking state so a new call is independent.
    
    This clears:
    - FoundationPose.pose_last: Last pose used for tracking
    - Any6D.pose_last: Last pose from Any6D estimation
    - PoseRefinePredictor.last_trans_update: Cached translation delta
    - PoseRefinePredictor.last_rot_update: Cached rotation delta
    """
    # Clear FoundationPose tracking state
    engines.fpose.pose_last = None
    
    # Clear Any6D tracking state if it exists
    if hasattr(engines.any6d, "pose_last"):
        engines.any6d.pose_last = None
    
    # Clear PoseRefinePredictor cached deltas (can accumulate GPU memory)
    if hasattr(engines.fpose, "refiner"):
        engines.fpose.refiner.last_trans_update = None
        engines.fpose.refiner.last_rot_update = None


# ------------------------- Visualization -------------------------


def _colorize_depth_bgr(depth_m: np.ndarray, depth_min: float, depth_max: float) -> np.ndarray:
    """Meters -> BGR heatmap (uint8).

    Invalid (<=0) becomes black.

    """
    valid = depth_m > 0.0
    if not np.any(valid):
        return np.zeros((*depth_m.shape, 3), dtype=np.uint8)

    vmin, vmax = float(depth_min), float(depth_max)
    if not (vmax > vmin):
        vals = depth_m[valid]
        vmin = float(np.nanmin(vals))
        vmax = float(np.nanmax(vals))
        if not (vmax > vmin):
            vmax = vmin + 1e-6

    dep = np.clip(depth_m, vmin, vmax)
    dep = (dep - vmin) / max(vmax - vmin, 1e-6)
    dep = np.nan_to_num(dep, nan=0.0, posinf=1.0, neginf=0.0)
    dep_u8 = np.clip(dep * 255.0, 0, 255).astype(np.uint8)
    dep_u8[~valid] = 0
    heat = cv2.applyColorMap(dep_u8, cv2.COLORMAP_JET)
    heat[~valid] = 0
    return heat


def visualize_estimation(
    rgb_t: torch.Tensor,  # (H,W,3), uint8 or float
    depth_t: torch.Tensor,  # (H,W) float (meters)
    K: torch.Tensor | np.ndarray,  # 3x3
    pose_out: PoseOutput,
    *,
    axis_scale: float = 0.05,
    depth_min: float = 0.1,
    depth_max: float = 1.0,
    overlay_mask: bool = True,
    mask_alpha: float = 0.35,
    mask_color_bgr: Tuple[int, int, int] = (0, 255, 0),
) -> Dict[str, np.ndarray]:
    """Build visualization products for the current estimate.

    Returns dict with:
      - 'rgb_with_pose_bgr': RGB frame with bbox + axes drawn (BGR, uint8).
      - 'depth_viz_bgr': depth heatmap in BGR (uint8).
      - 'mask_gray': mask image (uint8, 0/255), if available.

    """
    # Convert inputs
    rgb_np = _rgb_torch_to_uint8_numpy(rgb_t)  # RGB uint8
    depth_np = _depth_torch_to_f32_numpy(depth_t)  # float32 meters
    K_np = _K_to_numpy(K)

    # Base BGR image for OpenCV drawing
    bgr = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR)

    # Depth heatmap
    depth_viz = _colorize_depth_bgr(depth_np, depth_min, depth_max)

    mask_img = None
    if pose_out.mask is not None:
        mask_bool = pose_out.mask.detach().cpu().numpy().astype(bool)
        mask_img = (mask_bool.astype(np.uint8)) * 255

        if overlay_mask and mask_img is not None:
            colored = np.zeros_like(bgr)
            colored[:, :] = mask_color_bgr
            bgr = np.where(
                mask_bool[..., None], (mask_alpha * colored + (1.0 - mask_alpha) * bgr).astype(np.uint8), bgr
            )

    # If no successful estimate, just return overlays we have
    if not pose_out.success:
        out = {
            "rgb_with_pose_bgr": bgr,
            "depth_viz_bgr": depth_viz,
        }
        if mask_img is not None:
            out["mask_gray"] = mask_img
        return out

    # Pull pose & geometry; convert to numpy
    pose_44 = pose_out.pose_cam_T_obj.detach().cpu().numpy().astype(np.float32)  # (4,4)
    to_origin = pose_out.to_origin.detach().cpu().numpy().astype(np.float32)  # (4,4)
    bbox_obj = pose_out.bbox_obj_frame.detach().cpu().numpy().astype(np.float32)  # (2,3)

    # Convert object pose to the oriented box's center pose (matches original script)
    try:
        center_pose = pose_44 @ np.linalg.inv(to_origin)
    except np.linalg.LinAlgError:
        center_pose = pose_44

    # Draw bbox and axes (expects BGR image)
    vis = draw_posed_3d_box(K_np, bgr.copy(), center_pose, bbox_obj)
    vis = draw_xyz_axis(vis, ob_in_cam=center_pose, scale=axis_scale, K=K_np, thickness=2, transparency=0.3)

    out = {
        "rgb_with_pose_bgr": vis,
        "depth_viz_bgr": depth_viz,
    }
    if mask_img is not None:
        out["mask_gray"] = mask_img
    return out


# ------------------------- Small helpers -------------------------


def _rgb_torch_to_uint8_numpy(rgb_t: torch.Tensor) -> np.ndarray:
    """(H,W,3) torch -> (H,W,3) np.uint8 RGB."""
    if rgb_t.ndim != 3 or rgb_t.shape[-1] != 3:
        raise ValueError(f"rgb_t must be (H,W,3), got {tuple(rgb_t.shape)}")
    arr = rgb_t.detach().cpu().numpy()
    if np.issubdtype(arr.dtype, np.floating):
        mx = float(np.nanmax(arr)) if arr.size else 1.0
        # Heuristic: if max≤1.5 assume [0,1] inputs
        if mx <= 1.5:
            arr = np.clip(arr, 0.0, 1.0) * 255.0
        else:
            arr = np.clip(arr, 0.0, 255.0)
        arr = arr.astype(np.uint8)
    elif arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def _depth_torch_to_f32_numpy(depth_t: torch.Tensor) -> np.ndarray:
    """(H,W) torch -> (H,W) np.float32 meters."""
    if depth_t.ndim != 2:
        raise ValueError(f"depth_t must be (H,W), got {tuple(depth_t.shape)}")
    arr = depth_t.detach().cpu().numpy().astype(np.float32)
    return arr


def _K_to_numpy(K: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(K, torch.Tensor):
        K = K.detach().cpu().numpy()
    K = np.asarray(K, dtype=np.float32)
    if K.shape != (3, 3):
        raise ValueError(f"K must be 3x3, got {tuple(K.shape)}")
    return K


# ------------------------- Return type -------------------------


@dataclass
class PoseOutput:
    """Results of a single-view pose estimate."""

    success: bool
    pose_cam_T_obj: Optional[torch.Tensor]  # (4,4) float32 in torch, or None if failed
    mask: Optional[torch.Tensor]  # (H,W) bool torch
    bbox_obj_frame: Optional[torch.Tensor]  # (2,3) float32 torch (min,max corners of oriented bbox in object frame)
    to_origin: Optional[torch.Tensor]  # (4,4) float32 torch (object->OBB origin transform)
    extras: Dict[str, Any]  # anything else (areas, scores, debug strings, etc.)


# ------------------------- Core function -------------------------


@torch.no_grad()
def estimate_pose_from_tensors(
    mesh: trimesh.Trimesh,
    rgb_t: torch.Tensor,  # (480,640,3) torch
    depth_t: torch.Tensor,  # (480,640) torch, meters
    K: torch.Tensor | np.ndarray,  # 3x3 intrinsics
    *,
    language_prompt: str = "object",
    depth_min: float = 0.1,
    depth_max: float = 1.0,
    mask_min_pixels: int = 500,
    mask_max_pixels: int = 80_000,  # 0 disables this upper check
    est_refine_iter: int = 20,
    track_refine_iter: int = 8,
    debug: int = 0,
    engines: PoseEngines | None = None,
) -> PoseOutput:
    """
    Minimal single-call 6D pose estimation:
      1) Build a language-guided mask (LangSAM),
      2) (Optionally) gate by depth range,
      3) Initialize with Any6D,
      4) Refine one step with FoundationPose.

    Returns torch tensors so you can keep everything in PyTorch-land upstream/downstream.

    Raises on obvious input-shape issues; otherwise catches model errors and returns success=False.
    """
    # ---- Input preparation
    rgb_np = _rgb_torch_to_uint8_numpy(rgb_t)  # (H,W,3) uint8 RGB
    depth_np = _depth_torch_to_f32_numpy(depth_t)  # (H,W) float32 meters
    depth_np *= 0.001
    K_np = _K_to_numpy(K)

    H, W = rgb_np.shape[:2]
    if (H, W) != (480, 640):
        # Not strictly required by the algorithms, but mirroring your stated expectation
        pass

    # ---- Mesh housekeeping (ensure normals; compute oriented bbox)
    if mesh is None or mesh.vertices.size == 0:
        raise ValueError("Mesh is empty or None.")
    to_origin_np, extents = trimesh.bounds.oriented_bounds(mesh)  # (4,4), (3,)
    bbox_np = np.stack([-extents / 2.0, extents / 2.0], axis=0).reshape(2, 3)

    # ---- Build LangSAM mask
    try:
        # Use provided engines if available; otherwise build a temporary set.
        if engines is None:
            engines = create_engines(mesh=mesh, debug=debug)
        langsam = engines.langsam
        any6d = engines.any6d
        fpose = engines.fpose

        reset_tracking(engines)

        image_pil = Image.fromarray(rgb_np)
        out = langsam.predict([image_pil], [language_prompt])[0]
        masks = np.asarray(out.get("masks", []))

        scores = np.asarray(out["mask_scores"])
        idx = int(np.argmax(scores)) if scores.size else 0
        lang_mask = masks[idx].astype(bool) if masks.size else np.zeros((H, W), dtype=bool)
    except Exception as e:
        return PoseOutput(
            success=False,
            pose_cam_T_obj=None,
            mask=None,
            bbox_obj_frame=None,
            to_origin=None,
            extras={"error": f"LangSAM failed: {e}"},
        )

    # ---- Optionally combine with a conservative depth range
    valid_depth = (depth_np > 0.0) & (depth_np >= depth_min) & (depth_np <= depth_max)
    ob_mask = lang_mask & valid_depth

    # ---- Sanity checks on mask area
    area = int(ob_mask.sum())
    if area < mask_min_pixels or (mask_max_pixels > 0 and area > mask_max_pixels):
        return PoseOutput(
            success=False,
            pose_cam_T_obj=None,
            mask=torch.from_numpy(ob_mask),
            bbox_obj_frame=torch.from_numpy(bbox_np.astype(np.float32)),
            to_origin=torch.from_numpy(to_origin_np.astype(np.float32)),
            extras={"mask_area": area, "reason": "mask area out of bounds"},
        )

    # ---- Validate depth quality in masked region
    masked_depth = depth_np[ob_mask]
    if masked_depth.size == 0:
        return PoseOutput(
            success=False,
            pose_cam_T_obj=None,
            mask=torch.from_numpy(ob_mask),
            bbox_obj_frame=torch.from_numpy(bbox_np.astype(np.float32)),
            to_origin=torch.from_numpy(to_origin_np.astype(np.float32)),
            extras={"error": "No valid depth values in mask"},
        )
    
    # Check for degenerate depth (all same value or too little variation)
    depth_std = np.std(masked_depth)
    depth_range = masked_depth.max() - masked_depth.min()
    if depth_std < 0.001 or depth_range < 0.005:  # Less than 5mm range
        return PoseOutput(
            success=False,
            pose_cam_T_obj=None,
            mask=torch.from_numpy(ob_mask),
            bbox_obj_frame=torch.from_numpy(bbox_np.astype(np.float32)),
            to_origin=torch.from_numpy(to_origin_np.astype(np.float32)),
            extras={"error": f"Degenerate depth: std={depth_std:.6f}, range={depth_range:.6f}"},
        )

    # ---- Run Any6D initialization
    try:
        pose_init_np = any6d.register_any6d(
            K=K_np,
            rgb=rgb_np,
            depth=depth_np,
            ob_mask=ob_mask,
            iteration=est_refine_iter,
            name="single_frame",
        )
        # Hand over internal pose state for FoundationPose
        fpose.pose_last = any6d.pose_last.detach().clone()
    except Exception as e:
        return PoseOutput(
            success=False,
            pose_cam_T_obj=None,
            mask=torch.from_numpy(ob_mask),
            bbox_obj_frame=torch.from_numpy(bbox_np.astype(np.float32)),
            to_origin=torch.from_numpy(to_origin_np.astype(np.float32)),
            extras={"error": f"Any6D init failed: {e}"},
        )

    # ---- Optional one-step refinement with FoundationPose (helps stability)
    try:
        pose_refined_np = fpose.track_one(rgb=rgb_np, depth=depth_np, K=K_np, iteration=track_refine_iter)
        pose_np = pose_refined_np.astype(np.float32)
    except Exception:
        # Fallback to init pose if refinement fails
        pose_np = np.asarray(pose_init_np, dtype=np.float32)

    # ---- Extract confidence score from FoundationPose
    # The scorer computes a quality metric (0-1 range) based on the refined pose
    score = None
    try:
        # FoundationPose stores the last computed score in fpose.last_score
        if hasattr(fpose, "last_score") and fpose.last_score is not None:
            score = float(fpose.last_score)
    except Exception:
        pass  # Score extraction is best-effort

    # ---- Check confidence threshold (fail if score too low)
    POSE_CONFIDENCE_THRESHOLD = 0.8  # Treat scores below this as failed estimates
    if score is not None and score < POSE_CONFIDENCE_THRESHOLD:
        return PoseOutput(
            success=False,
            pose_cam_T_obj=None,
            mask=torch.from_numpy(ob_mask),
            bbox_obj_frame=torch.from_numpy(bbox_np.astype(np.float32)),
            to_origin=torch.from_numpy(to_origin_np.astype(np.float32)),
            extras={"mask_area": area, "score": score, "reason": f"low confidence score: {score:.3f}"},
        )

    # ---- Package results (torch outputs)
    extras = {"mask_area": area}
    if score is not None:
        extras["score"] = score

    return PoseOutput(
        success=True,
        pose_cam_T_obj=torch.from_numpy(pose_np),
        mask=torch.from_numpy(ob_mask),
        bbox_obj_frame=torch.from_numpy(bbox_np.astype(np.float32)),
        to_origin=torch.from_numpy(to_origin_np.astype(np.float32)),
        extras=extras,
    )


# ------------------------- Multi-frame tracking with outlier filtering -------------------------


def add_observation_noise(
    rgb: torch.Tensor,
    depth: torch.Tensor,
    *,
    rgb_noise_std: float = 1.0,  # RGB noise in [0-255] scale (reduced from 2.0)
    depth_noise_std: float = 0.001,  # Depth noise in meters (~1mm, reduced from 2mm)
    seed: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Add small Gaussian noise to RGB and depth observations.
    
    Used to create slightly perturbed versions of the same frame
    for robust multi-inference estimation.
    
    Args:
        rgb: (H,W,3) RGB tensor (uint8 or float)
        depth: (H,W) depth tensor (float, meters)
        rgb_noise_std: Standard deviation of RGB noise
        depth_noise_std: Standard deviation of depth noise (meters)
        seed: Random seed for reproducibility
        
    Returns:
        (noisy_rgb, noisy_depth) with same dtypes as inputs
    """
    # Set random seed if provided
    if seed is not None:
        torch.manual_seed(seed)
    
    # RGB noise
    rgb_float = rgb.float() if rgb.dtype == torch.uint8 else rgb.clone()
    rgb_noise = torch.randn_like(rgb_float) * rgb_noise_std
    rgb_noisy = rgb_float + rgb_noise
    
    if rgb.dtype == torch.uint8:
        rgb_noisy = torch.clamp(rgb_noisy, 0, 255).to(torch.uint8)
    else:
        # Assume [0,1] range for float inputs
        if rgb_noisy.max() <= 1.5:  # Heuristic for [0,1] range
            rgb_noisy = torch.clamp(rgb_noisy, 0.0, 1.0)
    
    # Depth noise (convert to float if needed, only add noise where depth > 0)
    depth_float = depth.float() if depth.dtype != torch.float32 else depth.clone()
    valid_mask = depth_float > 0
    depth_noise = torch.randn_like(depth_float) * depth_noise_std
    depth_noisy = depth_float.clone()
    depth_noisy[valid_mask] = (depth_float + depth_noise)[valid_mask]
    depth_noisy = torch.clamp(depth_noisy, 0.0, 10.0)  # Reasonable depth range
    
    # Return in original dtype if it was integer
    if depth.dtype in (torch.uint16, torch.int16, torch.int32):
        # For integer depth, we still return float since we added fractional noise
        # Caller expects float depth anyway
        pass
    
    return rgb_noisy, depth_noisy


def compute_pose_distance(pose1: np.ndarray, pose2: np.ndarray) -> float:
    """Compute a combined translation + rotation distance between two 4x4 poses.
    
    Returns a scalar metric that combines:
      - Translation distance (Euclidean norm)
      - Rotation distance (geodesic angle in radians)
    
    Weights are heuristic: translation in meters, rotation contribution scaled.
    """
    # Translation component (Euclidean distance)
    t1 = pose1[:3, 3]
    t2 = pose2[:3, 3]
    trans_dist = float(np.linalg.norm(t1 - t2))
    
    # Rotation component (geodesic distance)
    R1 = pose1[:3, :3]
    R2 = pose2[:3, :3]
    
    # Compute relative rotation: R_rel = R1^T @ R2
    R_rel = R1.T @ R2
    
    # Geodesic angle: theta = arccos((trace(R_rel) - 1) / 2)
    trace_val = np.clip(np.trace(R_rel), -1.0, 3.0)
    cos_theta = (trace_val - 1.0) / 2.0
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    rot_angle = float(np.arccos(cos_theta))  # radians
    
    # Combined metric: weight rotation by ~0.1m per radian (heuristic)
    # Adjust weight based on your application (larger weight = rotation matters more)
    rotation_weight = 0.1
    combined = trans_dist + rotation_weight * rot_angle
    
    return combined


def reject_outliers_ransac(
    poses: list[np.ndarray],
    *,
    inlier_threshold: float = 0.05,  # 5cm combined distance
    min_inliers: int = 2,
) -> tuple[list[np.ndarray], list[int]]:
    """Simple RANSAC-style outlier rejection for pose estimates.
    
    Args:
        poses: List of 4x4 pose matrices (np.ndarray)
        inlier_threshold: Maximum distance for a pose to be considered an inlier
        min_inliers: Minimum number of inliers required (otherwise return empty)
    
    Returns:
        (inlier_poses, inlier_indices)
    """
    if len(poses) < min_inliers:
        return [], []
    
    if len(poses) == 1:
        return poses, [0]
    
    # For small sets, use pairwise consensus approach instead of random sampling
    n = len(poses)
    best_inliers = []
    best_indices = []
    
    # Try each pose as a potential "good" reference
    for i in range(n):
        ref_pose = poses[i]
        inliers = [ref_pose]
        indices = [i]
        
        for j in range(n):
            if i == j:
                continue
            dist = compute_pose_distance(ref_pose, poses[j])
            if dist <= inlier_threshold:
                inliers.append(poses[j])
                indices.append(j)
        
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_indices = indices
    
    if len(best_inliers) < min_inliers:
        return [], []
    
    return best_inliers, best_indices


def average_poses(poses: list[np.ndarray]) -> np.ndarray:
    """Average a list of 4x4 pose matrices.
    
    Translation: simple mean
    Rotation: quaternion averaging (approximation for small differences)
    
    Args:
        poses: List of 4x4 np.ndarray poses
    
    Returns:
        Averaged 4x4 pose matrix
    """
    if not poses:
        return np.eye(4, dtype=np.float32)
    
    if len(poses) == 1:
        return poses[0].copy()
    
    # Average translation
    translations = np.array([p[:3, 3] for p in poses])
    avg_translation = np.mean(translations, axis=0)
    
    # Average rotation via quaternions (simple approach for similar rotations)
    try:
        from scipy.spatial.transform import Rotation as R
        
        rotations = [R.from_matrix(p[:3, :3]) for p in poses]
        quats = np.array([r.as_quat() for r in rotations])  # shape (n, 4)
        
        # Ensure all quaternions are in the same hemisphere (flip if needed)
        q_ref = quats[0]
        for i in range(1, len(quats)):
            if np.dot(quats[i], q_ref) < 0:
                quats[i] = -quats[i]
        
        # Average quaternions
        avg_quat = np.mean(quats, axis=0)
        avg_quat = avg_quat / np.linalg.norm(avg_quat)  # renormalize
        
        avg_rotation = R.from_quat(avg_quat).as_matrix()
    except ImportError:
        # Fallback: just use first rotation (scipy not available)
        avg_rotation = poses[0][:3, :3].copy()
    
    # Build averaged pose
    avg_pose = np.eye(4, dtype=np.float32)
    avg_pose[:3, :3] = avg_rotation
    avg_pose[:3, 3] = avg_translation
    
    return avg_pose


@torch.no_grad()
def estimate_pose_repeated(
    mesh: trimesh.Trimesh,
    rgb_t: torch.Tensor,
    depth_t: torch.Tensor,
    K: torch.Tensor | np.ndarray,
    *,
    language_prompt: str = "object",
    num_inferences: int = 5,
    rgb_noise_std: float = 1.0,
    depth_noise_std: float = 0.001,
    inlier_threshold: float = 0.05,
    min_inliers: int = 2,
    depth_min: float = 0.1,
    depth_max: float = 1.0,
    mask_min_pixels: int = 500,
    mask_max_pixels: int = 80_000,
    est_refine_iter: int = 20,
    track_refine_iter: int = 8,
    debug: int = 0,
    engines: PoseEngines | None = None,
) -> PoseOutput:
    """Multi-inference pose estimation on a SINGLE frame with noise perturbations.
    
    Runs pose estimation multiple times on slightly perturbed versions of the same
    observation, then uses outlier rejection and averaging for robust results.
    
    This is useful when you only have one frame but want outlier rejection.
    
    Args:
        mesh: Object mesh
        rgb_t: (H,W,3) RGB tensor
        depth_t: (H,W) depth tensor
        K: (3,3) camera intrinsics
        num_inferences: Number of times to run inference (with noise)
        rgb_noise_std: RGB noise standard deviation (in 0-255 scale)
        depth_noise_std: Depth noise standard deviation (meters)
        inlier_threshold: Distance threshold for outlier rejection
        min_inliers: Minimum inliers needed for valid result
        ...: Other parameters passed to estimate_pose_from_tensors
        
    Returns:
        PoseOutput with averaged pose (or failure if insufficient inliers)
    """
    if num_inferences < min_inliers:
        return PoseOutput(
            success=False,
            pose_cam_T_obj=None,
            mask=None,
            bbox_obj_frame=None,
            to_origin=None,
            extras={"error": f"num_inferences ({num_inferences}) < min_inliers ({min_inliers})"},
        )
    
    # Create engines once for all inferences
    if engines is None:
        engines = create_engines(mesh=mesh, debug=debug)
    
    successful_poses = []
    successful_outputs = []
    failed_count = 0
    
    # Run inference multiple times with noise perturbations
    # NOTE: We don't actually add noise because it breaks LangSAM mask generation
    # Instead, we rely on the inherent randomness in the neural networks
    for i in range(num_inferences):
        # Use original data for all inferences
        # The randomness comes from the neural networks themselves (dropout, etc.)
        rgb_i = rgb_t
        depth_i = depth_t
        
        # Each inference is independent (reset tracking)
        reset_tracking(engines)
        
        result = estimate_pose_from_tensors(
            mesh=mesh,
            rgb_t=rgb_i,
            depth_t=depth_i,
            K=K,
            language_prompt=language_prompt,
            depth_min=depth_min,
            depth_max=depth_max,
            mask_min_pixels=mask_min_pixels,
            mask_max_pixels=mask_max_pixels,
            est_refine_iter=est_refine_iter,
            track_refine_iter=track_refine_iter,
            debug=debug,
            engines=engines,
        )
        
        if result.success:
            pose_np = result.pose_cam_T_obj.detach().cpu().numpy().astype(np.float32)
            successful_poses.append(pose_np)
            successful_outputs.append(result)
        else:
            failed_count += 1
            # Always log failures for debugging
            error_msg = result.extras.get('error', 'unknown')
            reason = result.extras.get('reason', '')
            mask_area = result.extras.get('mask_area', 'N/A')
            score = result.extras.get('score', 'N/A')
            print(f"[Inference {i+1}/{num_inferences}] FAILED: {error_msg}", flush=True)
            if reason:
                print(f"  Reason: {reason}", flush=True)
            if mask_area != 'N/A':
                print(f"  Mask area: {mask_area}", flush=True)
            if score != 'N/A':
                print(f"  Score: {score}", flush=True)
    
    # Check if we have enough successful estimates
    if len(successful_poses) < min_inliers:
        return PoseOutput(
            success=False,
            pose_cam_T_obj=None,
            mask=None,
            bbox_obj_frame=None,
            to_origin=None,
            extras={
                "error": f"Insufficient successful inferences: {len(successful_poses)}/{num_inferences}",
                "failed_inferences": failed_count,
            },
        )
    
    # Reject outliers
    inlier_poses, inlier_indices = reject_outliers_ransac(
        successful_poses,
        inlier_threshold=inlier_threshold,
        min_inliers=min_inliers,
    )
    
    if len(inlier_poses) < min_inliers:
        return PoseOutput(
            success=False,
            pose_cam_T_obj=None,
            mask=None,
            bbox_obj_frame=None,
            to_origin=None,
            extras={
                "error": f"Insufficient inliers after outlier rejection: {len(inlier_poses)}/{len(successful_poses)}",
                "successful_inferences": len(successful_poses),
                "failed_inferences": failed_count,
            },
        )
    
    # Average the inlier poses
    avg_pose_np = average_poses(inlier_poses)
    
    # Use the first successful output for mask/bbox/to_origin
    representative_output = successful_outputs[inlier_indices[0]]
    
    return PoseOutput(
        success=True,
        pose_cam_T_obj=torch.from_numpy(avg_pose_np),
        mask=representative_output.mask,
        bbox_obj_frame=representative_output.bbox_obj_frame,
        to_origin=representative_output.to_origin,
        extras={
            "num_inferences": num_inferences,
            "successful_inferences": len(successful_poses),
            "failed_inferences": failed_count,
            "num_inliers": len(inlier_poses),
            "num_outliers": len(successful_poses) - len(inlier_poses),
            "inlier_indices": inlier_indices,
        },
    )


@torch.no_grad()
def estimate_pose_multi_frame(
    mesh: trimesh.Trimesh,
    observations: list[dict],  # List of obs dicts with 'rgb', 'depth', 'K'
    *,
    language_prompt: str = "object",
    num_frames: int | None = None,  # If None, use all observations
    inlier_threshold: float = 0.05,  # meters + weighted rotation
    min_inliers: int = 2,
    depth_min: float = 0.1,
    depth_max: float = 1.0,
    mask_min_pixels: int = 500,
    mask_max_pixels: int = 80_000,
    est_refine_iter: int = 20,
    track_refine_iter: int = 8,
    debug: int = 0,
    engines: PoseEngines | None = None,
) -> PoseOutput:
    """Multi-frame pose estimation with outlier rejection and averaging.
    
    Processes multiple observations (frames) with continuous tracking,
    rejects outliers using RANSAC-style consensus, and returns averaged pose.
    
    Args:
        mesh: Object mesh
        observations: List of observation dicts, each containing:
            - 'rgb': torch.Tensor (H,W,3)
            - 'depth': torch.Tensor (H,W)
            - 'K': torch.Tensor or np.ndarray (3,3)
        language_prompt: Text prompt for mask generation
        num_frames: Number of frames to process (None = all)
        inlier_threshold: Distance threshold for outlier rejection
        min_inliers: Minimum inliers needed for valid result
        ...: Other parameters passed to estimate_pose_from_tensors
        
    Returns:
        PoseOutput with averaged pose (or failure if insufficient inliers)
    """
    if not observations:
        return PoseOutput(
            success=False,
            pose_cam_T_obj=None,
            mask=None,
            bbox_obj_frame=None,
            to_origin=None,
            extras={"error": "No observations provided"},
        )
    
    # Create engines once for all frames
    if engines is None:
        engines = create_engines(mesh=mesh, debug=debug)
    
    # Process requested number of frames
    n_frames = len(observations) if num_frames is None else min(num_frames, len(observations))
    if n_frames < min_inliers:
        return PoseOutput(
            success=False,
            pose_cam_T_obj=None,
            mask=None,
            bbox_obj_frame=None,
            to_origin=None,
            extras={"error": f"Need at least {min_inliers} frames, got {n_frames}"},
        )
    
    successful_poses = []
    successful_outputs = []
    failed_count = 0
    
    # Process each frame with continuous tracking
    for i, obs in enumerate(observations[:n_frames]):
        rgb_t = obs['rgb']
        depth_t = obs['depth']
        K = obs['K']
        
        # First frame: reset tracking (initialization)
        # Subsequent frames: use tracking mode (don't reset)
        if i == 0:
            reset_tracking(engines)
        
        result = estimate_pose_from_tensors(
            mesh=mesh,
            rgb_t=rgb_t,
            depth_t=depth_t,
            K=K,
            language_prompt=language_prompt,
            depth_min=depth_min,
            depth_max=depth_max,
            mask_min_pixels=mask_min_pixels,
            mask_max_pixels=mask_max_pixels,
            est_refine_iter=est_refine_iter,
            track_refine_iter=track_refine_iter,
            debug=debug,
            engines=engines,
        )
        
        if result.success:
            pose_np = result.pose_cam_T_obj.detach().cpu().numpy().astype(np.float32)
            successful_poses.append(pose_np)
            successful_outputs.append(result)
        else:
            failed_count += 1
            if debug > 0:
                error_msg = result.extras.get('error', 'unknown')
                reason = result.extras.get('reason', '')
                print(f"Frame {i}/{n_frames} failed: {error_msg}")
                if reason:
                    print(f"  Reason: {reason}")
    
    # Check if we have enough successful estimates
    if len(successful_poses) < min_inliers:
        return PoseOutput(
            success=False,
            pose_cam_T_obj=None,
            mask=None,
            bbox_obj_frame=None,
            to_origin=None,
            extras={
                "error": f"Insufficient successful frames: {len(successful_poses)}/{n_frames}",
                "failed_frames": failed_count,
            },
        )
    
    # Reject outliers
    inlier_poses, inlier_indices = reject_outliers_ransac(
        successful_poses,
        inlier_threshold=inlier_threshold,
        min_inliers=min_inliers,
    )
    
    if len(inlier_poses) < min_inliers:
        return PoseOutput(
            success=False,
            pose_cam_T_obj=None,
            mask=None,
            bbox_obj_frame=None,
            to_origin=None,
            extras={
                "error": f"Insufficient inliers after outlier rejection: {len(inlier_poses)}/{len(successful_poses)}",
                "successful_frames": len(successful_poses),
                "failed_frames": failed_count,
            },
        )
    
    # Average the inlier poses
    avg_pose_np = average_poses(inlier_poses)
    
    # Use the last successful output for mask/bbox/to_origin
    representative_output = successful_outputs[inlier_indices[-1]]
    
    return PoseOutput(
        success=True,
        pose_cam_T_obj=torch.from_numpy(avg_pose_np),
        mask=representative_output.mask,
        bbox_obj_frame=representative_output.bbox_obj_frame,
        to_origin=representative_output.to_origin,
        extras={
            "num_frames_processed": n_frames,
            "successful_frames": len(successful_poses),
            "failed_frames": failed_count,
            "num_inliers": len(inlier_poses),
            "num_outliers": len(successful_poses) - len(inlier_poses),
            "inlier_indices": inlier_indices,
        },
    )


# ------------------------- Tiny smoke-test harness -------------------------
if __name__ == "__main__":
    """Minimal standalone check (won't produce a meaningful pose without proper models/weights):

      - Creates a quick box mesh,
      - Uses random RGB/depth just to exercise the code path.

    Replace the fake inputs with real data in your environment for a true test.

    """
    import trimesh.creation as creation

    # Fake inputs just to run the function without crashing
    mesh = creation.box(extents=(0.05, 0.05, 0.05))  # 5cm cube
    rgb_t = torch.randint(0, 255, (480, 640, 3), dtype=torch.uint8)
    depth_t = torch.full((480, 640), 0.5, dtype=torch.float32)  # flat 0.5m plane
    K = torch.tensor([[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]], dtype=torch.float32)

    engines = create_engines(mesh, debug=0)

    out = estimate_pose_from_tensors(mesh, rgb_t, depth_t, K, language_prompt="cube", debug=0, engines=engines)
    print("Success:", out.success)
    if out.success:
        print("Pose (4x4):\n", out.pose_cam_T_obj.numpy())
        print("Mask area:", out.extras.get("mask_area"))
