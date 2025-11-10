#!/usr/bin/env python3
"""
Calibrate drawer ArUco marker positions at closed position (position = 0).

This script solves the problem:
- At closed position (pos=0): only marker ID=1 is visible
- At open position (pos=X): markers ID=0, 1, 2 are all visible
- Goal: Find world-frame positions of markers 0, 1, 2 at closed position

Workflow:
1. Close drawer (pos=0) → capture marker ID=1 position
2. Open drawer to arbitrary position → capture markers ID=0, 1, 2
3. Use rigid-body assumption to compute marker 0 and 2 positions at pos=0
4. Save all three marker positions in world frame

Usage:
    python utility/calibrate_drawer_markers.py --serial 151422252817 --marker-size 0.04
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs


# ArUco dictionary mapping
DICT_MAP = {
    "4X4_50": cv2.aruco.DICT_4X4_50,
    "4X4_100": cv2.aruco.DICT_4X4_100,
    "5X5_50": cv2.aruco.DICT_5X5_50,
    "6X6_50": cv2.aruco.DICT_6X6_50,
    "APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11,
}


def estimate_marker_pose(corners, marker_size, K, D):
    """Estimate marker pose using solvePnP."""
    half_size = marker_size / 2.0
    obj_points = np.array([
        [-half_size,  half_size, 0],
        [ half_size,  half_size, 0],
        [ half_size, -half_size, 0],
        [-half_size, -half_size, 0]
    ], dtype=np.float64)
    
    img_points = corners.reshape(-1, 2).astype(np.float64)
    
    success, rvec, tvec = cv2.solvePnP(
        obj_points, img_points, K, D,
        flags=cv2.SOLVEPNP_IPPE_SQUARE
    )
    
    if success:
        R, _ = cv2.Rodrigues(rvec)
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = tvec.flatten()
        return T
    return None


def capture_markers(pipeline, K, D, ar_dict, marker_size, required_ids, use_new_api):
    """Capture a frame and detect specified markers."""
    print("Capturing frame...")
    frames = pipeline.wait_for_frames(timeout_ms=5000)
    color_frame = frames.get_color_frame()
    
    if not color_frame:
        return None, None
    
    image = np.asanyarray(color_frame.get_data())
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect markers
    if use_new_api:
        detector = cv2.aruco.ArucoDetector(ar_dict)
        corners, ids, _ = detector.detectMarkers(gray)
    else:
        corners, ids, _ = cv2.aruco.detectMarkers(gray, ar_dict)
    
    if ids is None:
        return image, {}
    
    # Extract poses for required markers
    marker_poses = {}
    for i, marker_id in enumerate(ids.flatten()):
        if marker_id in required_ids:
            T_cam_marker = estimate_marker_pose(corners[i], marker_size, K, D)
            if T_cam_marker is not None:
                marker_poses[marker_id] = T_cam_marker
    
    return image, marker_poses


def visualize_markers(image, marker_poses, K, D, marker_size):
    """Draw detected markers and their axes."""
    vis = image.copy()
    
    for marker_id, T_cam_marker in marker_poses.items():
        # Extract rvec and tvec for drawing
        R = T_cam_marker[:3, :3]
        tvec = T_cam_marker[:3, 3].reshape(3, 1)
        rvec, _ = cv2.Rodrigues(R)
        
        # Draw axes
        try:
            cv2.drawFrameAxes(vis, K, D, rvec, tvec, marker_size * 0.5)
        except Exception:
            pass
        
        # Draw ID label
        pt = T_cam_marker[:3, 3]
        if pt[2] > 0:
            x = int(pt[0] / pt[2] * K[0, 0] + K[0, 2])
            y = int(pt[1] / pt[2] * K[1, 1] + K[1, 2])
            cv2.putText(vis, f"ID{marker_id}", (x, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    return vis


def main():
    parser = argparse.ArgumentParser(description="Calibrate drawer ArUco markers at closed position")
    parser.add_argument('--serial', required=True, help='RealSense serial number')
    parser.add_argument('--dict', default='5X5_50', choices=DICT_MAP.keys(),
                       help='ArUco dictionary to use')
    parser.add_argument('--marker-size', type=float, required=True,
                       help='Physical marker size in meters')
    parser.add_argument('--width', type=int, default=640, help='Frame width')
    parser.add_argument('--height', type=int, default=480, help='Frame height')
    parser.add_argument('--fps', type=int, default=30, help='Frame rate')
    parser.add_argument('--output', default='data/calib/drawer_markers_closed.json',
                       help='Output JSON file for marker positions')
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    
    # Load camera intrinsics
    print("Loading calibration files...")
    intr_file = repo_root / 'data/calib/intrinsics_realsense_d455.npz'
    extr_file = repo_root / 'data/calib/extrinsics_realsense_d455.npz'
    
    if not intr_file.exists():
        print(f"❌ Intrinsics file not found: {intr_file}")
        return 1
    
    if not extr_file.exists():
        print(f"❌ Extrinsics file not found: {extr_file}")
        return 1
    
    intr_data = np.load(intr_file)
    K = intr_data['K'].astype(np.float64)
    D = intr_data['D'].astype(np.float64)
    print(f"✓ Loaded intrinsics")
    
    extr_data = np.load(extr_file)
    T_base_cam = extr_data['T_base_cam'].astype(np.float64)
    print(f"✓ Loaded extrinsics")
    print(f"  Camera position: [{T_base_cam[0,3]:.3f}, {T_base_cam[1,3]:.3f}, {T_base_cam[2,3]:.3f}]m")
    
    # Initialize RealSense
    print(f"\n📸 Initializing RealSense {args.serial}...")
    ctx = rs.context()
    for dev in ctx.query_devices():
        if dev.get_info(rs.camera_info.serial_number) == args.serial:
            dev.hardware_reset()
            time.sleep(3)
            break

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(args.serial)
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    
    profile = pipeline.start(config)
    
    # Setup ArUco detection
    ar_dict = cv2.aruco.getPredefinedDictionary(DICT_MAP[args.dict])
    try:
        cv2.aruco.ArucoDetector(ar_dict)
        use_new_api = True
    except (AttributeError, TypeError):
        use_new_api = False
    
    print(f"✓ Using OpenCV ArUco {'new' if use_new_api else 'old'} API")
    print(f"📏 Marker size: {args.marker_size}m ({args.marker_size*100}cm)")
    
    # Warmup
    for _ in range(5):
        pipeline.wait_for_frames(timeout_ms=5000)
    
    try:
        # ===== STEP 1: Capture marker ID=1 at closed position (pos=0) =====
        print("\n" + "="*60)
        print("STEP 1: CLOSE DRAWER (position = 0)")
        print("="*60)
        print("Instructions:")
        print("  - Close the drawer completely")
        print("  - Ensure marker ID=1 is visible")
        print("  - Press SPACE when ready")
        print("="*60)
        
        cv2.namedWindow('Drawer Calibration', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Drawer Calibration', 1280, 720)
        
        marker1_pos0 = None
        while marker1_pos0 is None:
            image, marker_poses = capture_markers(pipeline, K, D, ar_dict, args.marker_size, {1}, use_new_api)
            
            vis = visualize_markers(image, marker_poses, K, D, args.marker_size)
            
            if 1 in marker_poses:
                cv2.putText(vis, "Marker ID=1 detected! Press SPACE to capture", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            else:
                cv2.putText(vis, "Marker ID=1 NOT detected - adjust drawer/camera", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            cv2.imshow('Drawer Calibration', vis)
            
            key = cv2.waitKey(10) & 0xFF
            if key == ord(' ') and 1 in marker_poses:
                marker1_pos0 = marker_poses[1]
                T_world_marker1_pos0 = T_base_cam @ marker1_pos0
                print(f"\n✓ Captured marker ID=1 at closed position")
                print(f"  Camera frame: {marker1_pos0[:3, 3]}")
                print(f"  World frame:  {T_world_marker1_pos0[:3, 3]}")
                break
            elif key == ord('q'):
                print("\nCancelled by user")
                return 0
        
        # ===== STEP 2: Capture markers 0, 1, 2 at open position =====
        print("\n" + "="*60)
        print("STEP 2: OPEN DRAWER (position = X)")
        print("="*60)
        print("Instructions:")
        print("  - Open the drawer to any position")
        print("  - Ensure markers ID=0, 1, 2 are ALL visible")
        print("  - Press SPACE when ready")
        print("="*60)
        
        markers_posX = None
        while markers_posX is None:
            image, marker_poses = capture_markers(pipeline, K, D, ar_dict, args.marker_size, {0, 1, 2}, use_new_api)
            
            vis = visualize_markers(image, marker_poses, K, D, args.marker_size)
            
            detected = [id for id in [0, 1, 2] if id in marker_poses]
            missing = [id for id in [0, 1, 2] if id not in marker_poses]
            
            status_color = (0, 255, 0) if len(detected) == 3 else (0, 0, 255)
            status_text = f"Detected: {detected} | Missing: {missing}"
            cv2.putText(vis, status_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
            
            if len(detected) == 3:
                cv2.putText(vis, "All markers visible! Press SPACE to capture",
                           (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            else:
                cv2.putText(vis, "Adjust drawer/camera to see all markers",
                           (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            cv2.imshow('Drawer Calibration', vis)
            
            key = cv2.waitKey(10) & 0xFF
            if key == ord(' ') and len(detected) == 3:
                markers_posX = marker_poses
                print(f"\n✓ Captured all markers at open position")
                for mid in [0, 1, 2]:
                    print(f"  Marker ID={mid}: {markers_posX[mid][:3, 3]}")
                break
            elif key == ord('q'):
                print("\nCancelled by user")
                return 0
        
        # ===== STEP 3: Compute marker positions at closed position =====
        print("\n" + "="*60)
        print("STEP 3: COMPUTING MARKER POSITIONS AT CLOSED POSITION")
        print("="*60)
        
        # Compute relative transforms (in camera frame at open position)
        T_marker1_marker0 = np.linalg.inv(markers_posX[1]) @ markers_posX[0]
        T_marker1_marker2 = np.linalg.inv(markers_posX[1]) @ markers_posX[2]
        
        print("Relative transforms computed:")
        print(f"  Marker1 → Marker0: {T_marker1_marker0[:3, 3]}")
        print(f"  Marker1 → Marker2: {T_marker1_marker2[:3, 3]}")
        
        # Project to closed position (camera frame)
        T_cam_marker0_pos0 = marker1_pos0 @ T_marker1_marker0
        T_cam_marker2_pos0 = marker1_pos0 @ T_marker1_marker2
        
        print("\nMarker positions at closed position (camera frame):")
        print(f"  Marker ID=0: {T_cam_marker0_pos0[:3, 3]}")
        print(f"  Marker ID=1: {marker1_pos0[:3, 3]}")
        print(f"  Marker ID=2: {T_cam_marker2_pos0[:3, 3]}")
        
        # Convert to world frame
        T_world_marker0_pos0 = T_base_cam @ T_cam_marker0_pos0
        T_world_marker1_pos0 = T_base_cam @ marker1_pos0
        T_world_marker2_pos0 = T_base_cam @ T_cam_marker2_pos0
        
        print("\nMarker positions at closed position (world frame):")
        print(f"  Marker ID=0: {T_world_marker0_pos0[:3, 3]}")
        print(f"  Marker ID=1: {T_world_marker1_pos0[:3, 3]}")
        print(f"  Marker ID=2: {T_world_marker2_pos0[:3, 3]}")
        
        # ===== STEP 4: Save results =====
        output_path = repo_root / args.output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        result = {
            "description": "ArUco marker positions on drawer at closed position (pos=0)",
            "marker_size_m": args.marker_size,
            "dict": args.dict,
            "markers": {
                "0": {
                    "T_world_marker": T_world_marker0_pos0.tolist(),
                    "position": T_world_marker0_pos0[:3, 3].tolist(),
                },
                "1": {
                    "T_world_marker": T_world_marker1_pos0.tolist(),
                    "position": T_world_marker1_pos0[:3, 3].tolist(),
                },
                "2": {
                    "T_world_marker": T_world_marker2_pos0.tolist(),
                    "position": T_world_marker2_pos0[:3, 3].tolist(),
                },
            },
            "T_base_cam": T_base_cam.tolist(),
        }
        
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"\n💾 Saved results to: {output_path}")
        print("\n" + "="*60)
        print("CALIBRATION COMPLETE!")
        print("="*60)
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

    return 0


if __name__ == '__main__':
    sys.exit(main())
