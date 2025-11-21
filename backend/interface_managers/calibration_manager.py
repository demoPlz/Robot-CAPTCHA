"""
CalibrationManager - Handles all calibration data loading and management.

Manages:
- Camera extrinsics (poses for Three.js visualization)
- Camera intrinsics (undistortion maps and projection matrices)
- Gripper tip calibration
- Sim vs real calibration paths

Thread-safe for gripper calibration updates.
"""

import json
import os
from pathlib import Path
from threading import Lock

import numpy as np


class CalibrationManager:
    """Manages calibration data for cameras and gripper.

    Handles both real and sim calibration modes, loading extrinsics, intrinsics, undistortion maps, and gripper tip
    positions.

    """

    def __init__(self, use_sim: bool, repo_root: Path, real_calib_paths: dict, sim_calib_paths: dict):
        """Initialize calibration manager.

        Args:
            use_sim: Whether to use simulation calibrations
            repo_root: Root directory of the repository
            real_calib_paths: Dict mapping camera names to {"intr": path, "extr": path}
            sim_calib_paths: Dict mapping camera names to sim calibration JSON paths

        """
        self.use_sim = use_sim
        self.repo_root = repo_root
        self.real_calib_paths = real_calib_paths
        self.sim_calib_paths = sim_calib_paths

        # Calibration data
        self._undistort_maps: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._camera_models: dict[str, dict] = {}
        self._camera_poses: dict[str, list] = {}
        self._gripper_tip_calib: dict = {}

        # Thread safety for gripper calibration updates
        self._calib_lock = Lock()

        # Load calibrations on init
        self._camera_poses = self._load_calibrations()
        self._gripper_tip_calib = self._load_gripper_tip_calibration()

    # =========================
    # Public API
    # =========================

    def get_camera_poses(self) -> dict[str, list]:
        """Get camera poses (4x4 transformation matrices) for Three.js visualization.

        Returns:
            Dict mapping "{camera_name}_pose" to 4x4 matrix (list of lists).
            Example: {"front_pose": [[...], [...], [...], [...]]}

        """
        return self._camera_poses

    def get_camera_models(self) -> dict[str, dict]:
        """Get camera intrinsic models for projection and rendering.

        Returns:
            Dict mapping camera name to intrinsic parameters:
            - K/Knew: 3x3 camera matrix
            - width, height: image dimensions
            - model: "pinhole"
            - rectified: whether undistortion maps are available

        """
        return self._camera_models

    def get_undistort_maps(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Get precomputed undistortion maps for real cameras.

        Returns:
            Dict mapping camera name to (map1, map2) for cv2.remap().
            Only available for real mode with fisheye calibrations.

        """
        return self._undistort_maps

    def get_gripper_tip_calib(self) -> dict:
        """Get gripper tip calibration offsets in meters.

        Returns:
            Dict with structure: {"left": {"x": float, "y": float, "z": float},
                                  "right": {"x": float, "y": float, "z": float}}

        """
        return self._gripper_tip_calib

    def save_gripper_tip_calibration(self, calib: dict) -> str:
        """Save gripper tip calibration to manual_gripper_tips.json.

        Thread-safe update that validates input, writes to disk, and updates in-memory cache.

        Args:
            calib: Dict with structure {"left": {"x": val, "y": val, "z": val},
                                       "right": {"x": val, "y": val, "z": val}}
                   All values will be cast to float (supports int/float/str input).

        Returns:
            Absolute path to the written JSON file.

        Raises:
            ValueError: If input structure is invalid or values cannot be cast to float.
            IOError: If file write fails.

        """

        # Validate and sanitize input
        def _validate_side(side: str) -> dict[str, float]:
            s = calib.get(side)
            if not s:
                raise ValueError(f"Missing '{side}' gripper calibration")
            try:
                return {"x": float(s["x"]), "y": float(s["y"]), "z": float(s["z"])}
            except (KeyError, TypeError, ValueError) as e:
                raise ValueError(f"Invalid '{side}' calibration: {e}")

        try:
            cleaned = {"left": _validate_side("left"), "right": _validate_side("right")}
        except Exception as e:
            raise ValueError(f"Invalid gripper_tip_calib payload: {e}")

        # Write to disk with lock
        p = self._calib_dir()
        p.mkdir(parents=True, exist_ok=True)
        path = p / "manual_gripper_tips.json"

        with self._calib_lock:
            try:
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(cleaned, f, indent=2)
                # Update in-memory cache only after successful write
                self._gripper_tip_calib = cleaned
            except Exception as e:
                raise IOError(f"Failed to write gripper calibration to {path}: {e}")

        return str(path)

    # =========================
    # Private Helpers
    # =========================

    def _calib_dir(self) -> Path:
        """Return the calibration directory path."""
        return (self.repo_root / "data" / "calib").resolve()

    def _make_camera_poses(self) -> dict[str, list]:
        """Generate fallback camera poses (4x4 matrices)."""

        def euler_pose(x: float, y: float, z: float, roll: float, pitch: float, yaw: float) -> list[list[float]]:
            """Build a 4x4 **world** matrix (row-major list-of-lists) from T = Trans(x,y,z) · Rz(yaw) · Ry(pitch) ·
            Rx(roll)"""
            cr, sr = np.cos(roll), np.sin(roll)
            cp, sp = np.cos(pitch), np.sin(pitch)
            cy, sy = np.cos(yaw), np.sin(yaw)

            # column-major rotation
            Rrow = np.array(
                [
                    [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                    [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                    [-sp, cp * sr, cp * cr],
                ]
            )

            T = np.eye(4)
            T[:3, :3] = Rrow.T
            T[:3, 3] = [x, y, z]
            return T.tolist()

        return {
            #           x     y     z     roll   pitch   yaw
            "front_pose": euler_pose(0.2, -1.0, 0.15, -np.pi / 2, 0.0, 0.0),
            "left_pose": euler_pose(0.2, -1.0, 0.15, -np.pi / 2, 0.0, 0.0),
            "right_pose": euler_pose(0.2, 1.0, 0.15, np.pi / 2, 0.0, np.pi),
            "top_pose": euler_pose(1.3, 1.0, 1.0, np.pi / 4, -np.pi / 4, -3 * np.pi / 4),
        }

    def _load_calibrations(self) -> dict[str, list]:
        """Load camera calibrations based on mode (sim/real).

        Initializes:
        - self._camera_poses: 4x4 matrices for Three.js
        - self._camera_models: intrinsic parameters (K, width, height)
        - self._undistort_maps: precomputed maps for cv2.remap (real mode only)

        Returns:
            Camera poses dict (also stored in self._camera_poses).
            Fallback poses are used for cameras without calibration files.

        """
        poses = self._make_camera_poses()  # start with fallbacks
        self._undistort_maps = {}
        self._camera_models = {}

        # Base directory for calibration files
        manual_dir = self._calib_dir()

        if self.use_sim:
            # In sim mode: load sim calibrations for the 4 sim views
            self._load_sim_calibrations_for_frontend(poses, manual_dir)
            # ALSO load real calibrations for webcam undistortion (left, front cameras)
            # This populates undistort_maps, camera_models, and camera_poses for real cameras
            self._load_real_calibrations_for_webcams()
            # Return self._camera_poses which includes both sim and webcam poses
            return self._camera_poses
        else:
            # In real mode: load real calibrations + manual overrides for frontend
            return self._load_real_calibrations_for_frontend(poses, manual_dir)

    def _load_sim_calibrations_for_frontend(self, poses: dict, manual_dir: Path) -> dict[str, list]:
        """Load simulation calibrations from JSON files.

        Parses Isaac Sim-exported calibration files containing extrinsics, intrinsics,
        and optional orthographic projection parameters.

        Args:
            poses: Fallback poses dict to be populated
            manual_dir: Path to calibration directory (unused for sim, kept for signature consistency)

        Returns:
            Updated poses dict with loaded sim calibrations.

        """

        for name in ["front", "left", "right", "top"]:
            if name not in self.sim_calib_paths:
                continue

            sim_file = self.sim_calib_paths[name]
            if not os.path.exists(sim_file):
                print(f"⚠️ Sim calibration file not found: {sim_file}")
                continue

            try:
                with open(sim_file, "r", encoding="utf-8") as f:
                    scal = json.load(f)

                intr_s = (scal or {}).get("intrinsics") or {}
                extr_s = (scal or {}).get("extrinsics") or {}

                # Load extrinsics
                if "T_three" in extr_s and isinstance(extr_s["T_three"], list):
                    poses[f"{name}_pose"] = extr_s["T_three"]
                    self._camera_poses[f"{name}_pose"] = extr_s["T_three"]
                    print(f"✓ loaded SIM extrinsics for '{name}' from {sim_file}")

                # Load intrinsics
                if all(k in intr_s for k in ("width", "height", "Knew")):
                    self._camera_models[name] = {
                        "model": "pinhole",
                        "rectified": False,  # Sim calibrations don't have undistort maps
                        "width": int(intr_s["width"]),
                        "height": int(intr_s["height"]),
                        "Knew": intr_s["Knew"],
                    }

                    # Add orthographic projection parameters if present
                    if "projection_type" in scal:
                        self._camera_models[name]["projection_type"] = scal["projection_type"]
                    if "orthographic_width" in intr_s:
                        self._camera_models[name]["orthographic_width"] = intr_s["orthographic_width"]
                    if "orthographic_height" in intr_s:
                        self._camera_models[name]["orthographic_height"] = intr_s["orthographic_height"]
                    if "scale_x" in intr_s:
                        self._camera_models[name]["scale_x"] = intr_s["scale_x"]
                    if "scale_y" in intr_s:
                        self._camera_models[name]["scale_y"] = intr_s["scale_y"]

                    print(
                        f"✓ loaded SIM intrinsics for '{name}' from {sim_file} (projection: {scal.get('projection_type', 'perspective')})"
                    )

            except Exception as e:
                print(f"⚠️ Failed to load sim calibration for '{name}' from {sim_file}: {e}")

        # Load RealSense D455 extrinsics (used for pose estimation even in sim mode)
        realsense_extr = manual_dir / "extrinsics_realsense_d455.npz"
        if realsense_extr.exists():
            try:
                data = np.load(realsense_extr, allow_pickle=True)
                # Use T_base_cam for pose estimation (OpenCV convention, +Z forward)
                # Pose workers output poses in OpenCV camera frame, NOT Three.js frame
                M = data["T_base_cam"]
                poses["realsense_pose"] = M.tolist()
                self._camera_poses["realsense_pose"] = M.tolist()
                print(f"✓ loaded RealSense D455 extrinsics from {realsense_extr.name} (for pose estimation)")
            except Exception as e:
                print(f"⚠️ Failed to load RealSense D455 extrinsics: {e}")
        else:
            print(f"⚠️ RealSense D455 extrinsics not found at {realsense_extr} (pose estimation will fail)")

        return poses

    def _load_real_calibrations_for_frontend(self, poses: dict, manual_dir: Path) -> dict[str, list]:
        """Load real camera calibrations from NPZ files with optional JSON overrides.

        Loading order:
        1. Extrinsics (.npz): T_three or T_base_cam → camera pose
        2. Intrinsics (.npz): K, D, Knew, width, height → undistortion maps + camera model
        3. Manual overrides (manual_calibration_{name}.json): Override any loaded values

        Args:
            poses: Fallback poses dict to be populated
            manual_dir: Path to directory containing manual_calibration_*.json files

        Returns:
            Updated poses dict with loaded real calibrations.

        """

        for name, paths in self.real_calib_paths.items():
            if not paths:
                continue

            # ---- Load extrinsics → camera pose ----
            extr = paths.get("extr")
            if extr:
                extr_path = (self.repo_root / extr).resolve() if not Path(extr).is_absolute() else Path(extr)
                if extr_path.exists():
                    try:
                        data = np.load(extr_path, allow_pickle=True)
                        if "T_three" in data:
                            M = np.asarray(data["T_three"], dtype=np.float64)
                        elif "T_base_cam" in data:
                            # Convert OpenCV cam (Z forward) to Three.js cam (looks -Z)
                            T = np.asarray(data["T_base_cam"], dtype=np.float64)
                            Rflip = np.diag([1.0, -1.0, -1.0])
                            M = np.eye(4, dtype=np.float64)
                            M[:3, :3] = T[:3, :3] @ Rflip
                            M[:3, 3] = T[:3, 3]
                        else:
                            M = None
                        if M is not None:
                            poses[f"{name}_pose"] = M.tolist()
                            print(f"✓ loaded extrinsics for '{name}' from {extr_path}")
                    except Exception as e:
                        print(f"⚠️  failed to load extrinsics for '{name}' ({extr_path}): {e}")

            # ---- Load intrinsics → undistortion maps + Knew for projection ----
            intr = paths.get("intr")
            if intr:
                intr_path = (self.repo_root / intr).resolve() if not Path(intr).is_absolute() else Path(intr)
                if intr_path.exists():
                    try:
                        idata = np.load(intr_path, allow_pickle=True)
                        W = int(idata["width"])
                        H = int(idata["height"])
                        # Prefer rectified Knew (matches undistorted frames)
                        Knew = np.asarray(idata["Knew"], dtype=np.float64)
                        # Optional: precomputed undistort maps
                        if "map1" in idata.files and "map2" in idata.files:
                            # Store undistort maps with base name (left, front) for WebcamManager
                            self._undistort_maps[name] = (idata["map1"], idata["map2"])
                            rectified = True
                            print(f"✓ loaded undistort maps for webcam '{name}' from {intr_path}")
                        else:
                            rectified = False
                            print(f"⚠️  no undistort maps found for webcam '{name}' in {intr_path}")
                        # Expose per-camera intrinsics to the frontend with webcam_ prefix
                        self._camera_models[f"webcam_{name}"] = {
                            "model": "pinhole",
                            "rectified": rectified,
                            "width": W,
                            "height": H,
                            "Knew": Knew.tolist(),
                        }
                        print(f"✓ loaded intrinsics (Knew {W}x{H}) for webcam '{name}' from {intr_path}")
                    except Exception as e:
                        print(f"⚠️  failed to load intrinsics for webcam '{name}' ({intr_path}): {e}")

            # ---- Manual override (JSON) if present ----
            try:
                manual_path = manual_dir / f"manual_calibration_{name}.json"
                if manual_path.exists():
                    with open(manual_path, "r", encoding="utf-8") as f:
                        mcal = json.load(f)
                    intr_m = (mcal or {}).get("intrinsics") or {}
                    extr_m = (mcal or {}).get("extrinsics") or {}
                    # Validate presence of fields we expect
                    if "T_three" in extr_m and isinstance(extr_m["T_three"], list):
                        poses[f"{name}_pose"] = extr_m["T_three"]
                        print(f"✓ applied MANUAL extrinsics for '{name}' from {manual_path}")
                    if all(k in intr_m for k in ("width", "height", "Knew")):
                        # Preserve existing 'rectified' flag if any, otherwise False
                        prev_rect = self._camera_models.get(name, {}).get("rectified", False)
                        self._camera_models[name] = {
                            "model": "pinhole",
                            "rectified": prev_rect,
                            "width": int(intr_m["width"]),
                            "height": int(intr_m["height"]),
                            "Knew": intr_m["Knew"],
                        }
                        print(f"✓ applied MANUAL intrinsics for '{name}' from {manual_path}")
            except Exception as e:
                print(f"⚠️  failed to apply manual calibration for '{name}': {e}")

        # ---- Load RealSense D455 extrinsics (for pose estimation coordinate transform) ----
        realsense_extr = manual_dir / "extrinsics_realsense_d455.npz"
        if realsense_extr.exists():
            try:
                data = np.load(realsense_extr, allow_pickle=True)
                # Use T_base_cam directly (OpenCV convention, +Z forward)
                # Pose workers output poses in OpenCV camera frame, so we need matching extrinsics
                if "T_base_cam" in data:
                    M = np.asarray(data["T_base_cam"], dtype=np.float64)
                    poses["realsense_pose"] = M.tolist()
                    print(f"✓ loaded RealSense D455 extrinsics from {realsense_extr}")
                else:
                    print(f"⚠️  T_base_cam not found in {realsense_extr}")
            except Exception as e:
                print(f"⚠️  failed to load RealSense D455 extrinsics ({realsense_extr}): {e}")
        else:
            print(f"⚠️  RealSense D455 extrinsics not found at {realsense_extr}")
            print(f"    Pose estimation will fail to transform to world frame!")
            print(f"    Save camera pose calibration to: {realsense_extr}")

        return poses

    def _load_real_calibrations_for_webcams(self):
        """Load real camera calibrations for webcam views (left and front only).
        
        This method is called when use_sim=True to load undistortion maps and
        camera models for the real webcam cameras, while sim calibrations are
        used for the 4 sim views.
        
        IMPORTANT: Uses separate keys (webcam_left, webcam_front) to avoid
        overwriting sim calibrations (left, front).
        
        Populates:
        - self._undistort_maps: for cv2.remap() in WebcamManager (keys: "left", "front")
        - self._camera_models: for frontend projection (keys: "webcam_left", "webcam_front")
        - self._camera_poses: for frontend (keys: "webcam_left_pose", "webcam_front_pose")
        """
        # Only load left and front real cameras for webcam views
        webcam_names = ["left", "front"]
        manual_dir = self._calib_dir()
        
        for name in webcam_names:
            # Try manual calibration file first (for intrinsics/extrinsics)
            manual_path = manual_dir / f"manual_calibration_{name}.json"
            if manual_path.exists():
                try:
                    with open(manual_path, "r", encoding="utf-8") as f:
                        mcal = json.load(f)
                    
                    intr_m = mcal.get("intrinsics", {})
                    extr_m = mcal.get("extrinsics", {})
                    
                    # Load intrinsics from manual calibration
                    if all(k in intr_m for k in ("width", "height", "Knew")):
                        # Store camera model with webcam_ prefix to distinguish from sim
                        self._camera_models[f"webcam_{name}"] = {
                            "model": "pinhole",
                            "rectified": False,  # Will be updated if NPZ has undistort maps
                            "width": int(intr_m["width"]),
                            "height": int(intr_m["height"]),
                            "Knew": intr_m["Knew"],
                        }
                        print(f"✓ loaded intrinsics for webcam '{name}' from manual calibration {manual_path}")
                    
                    # Load extrinsics from manual calibration
                    if "T_three" in extr_m and isinstance(extr_m["T_three"], list):
                        # Store pose with webcam_ prefix
                        self._camera_poses[f"webcam_{name}_pose"] = extr_m["T_three"]
                        print(f"✓ loaded extrinsics for webcam '{name}' from manual calibration {manual_path}")
                    
                    # DON'T continue - still need to load undistortion maps from NPZ
                    
                except Exception as e:
                    print(f"⚠️  failed to load manual calibration for webcam '{name}': {e}")
            
            # Load undistortion maps from NPZ files (manual calibrations don't have these)
            if name not in self.real_calib_paths:
                continue
            
            paths = self.real_calib_paths[name]
            if not paths:
                continue
            
            # ---- Load undistortion maps from NPZ (if available) ----
            intr = paths.get("intr")
            if intr:
                intr_path = (self.repo_root / intr).resolve() if not Path(intr).is_absolute() else Path(intr)
                if intr_path.exists():
                    try:
                        idata = np.load(intr_path, allow_pickle=True)
                        # Only load undistortion maps - intrinsics already loaded from manual calibration
                        if "map1" in idata.files and "map2" in idata.files:
                            # Store undistort maps with base name (left, front) for WebcamManager
                            self._undistort_maps[name] = (idata["map1"], idata["map2"])
                            # Update rectified flag in camera model
                            if f"webcam_{name}" in self._camera_models:
                                self._camera_models[f"webcam_{name}"]["rectified"] = True
                            print(f"✓ loaded undistort maps for webcam '{name}' from {intr_path}")
                        else:
                            print(f"⚠️  no undistort maps found for webcam '{name}' in {intr_path}")
                    except Exception as e:
                        print(f"⚠️  failed to load undistort maps for webcam '{name}' ({intr_path}): {e}")

    def _load_gripper_tip_calibration(self) -> dict:
        """Load gripper tip calibration offsets from manual_gripper_tips.json.

        Returns:
            Dict with structure: {"left": {"x": float, "y": float, "z": float},
                                  "right": {"x": float, "y": float, "z": float}}

        Raises:
            FileNotFoundError: If manual_gripper_tips.json doesn't exist.
            KeyError/ValueError: If file format is invalid.

        """
        p = self._calib_dir() / "manual_gripper_tips.json"

        if not p.exists():
            raise FileNotFoundError(
                f"Gripper calibration file not found: {p}\n"
                f"Please create this file with left/right gripper tip offsets."
            )

        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in gripper calibration file: {e}")

        # Validate structure and cast to float
        def _validate_side(side: str) -> dict[str, float]:
            if side not in data:
                raise KeyError(f"Missing '{side}' key in gripper calibration")
            s = data[side]
            try:
                return {
                    "x": float(s["x"]),
                    "y": float(s["y"]),
                    "z": float(s["z"]),
                }
            except (KeyError, TypeError, ValueError) as e:
                raise ValueError(f"Invalid '{side}' gripper calibration: {e}")

        return {"left": _validate_side("left"), "right": _validate_side("right")}
