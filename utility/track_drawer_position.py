#!/usr/bin/env python3
"""
Track drawer position in real-time using ArUco markers.
Displays the distance (in meters) the drawer has moved from closed position.

Usage:
    python utility/track_drawer_position.py --serial 151422252817
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


def main():
    parser = argparse.ArgumentParser(description="Track drawer position in real-time")
    parser.add_argument('--serial', required=True, help='RealSense serial number')
    parser.add_argument('--width', type=int, default=640, help='Frame width')
    parser.add_argument('--height', type=int, default=480, help='Frame height')
    parser.add_argument('--fps', type=int, default=30, help='Frame rate')
    parser.add_argument('--calib', default='data/calib/drawer_markers_closed.json',
                       help='Drawer calibration file')
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    
    # Load drawer calibration
    print("Loading calibration files...")
    calib_file = repo_root / args.calib
    if not calib_file.exists():
        print(f"❌ Calibration file not found: {calib_file}")
        print("Run calibrate_drawer_markers.py first!")
        return 1
    
    with open(calib_file, 'r') as f:
        drawer_calib = json.load(f)
    
    marker_size = drawer_calib['marker_size_m']
    dict_name = drawer_calib['dict']
    
    # Extract closed positions (world frame)
    markers_closed = {}
    for marker_id_str, data in drawer_calib['markers'].items():
        marker_id = int(marker_id_str)
        pos_closed = np.array(data['position'], dtype=np.float64)
        markers_closed[marker_id] = pos_closed
    
    print(f"✓ Loaded drawer calibration")
    print(f"  Dictionary: {dict_name}")
    print(f"  Marker size: {marker_size}m ({marker_size*100}cm)")
    print(f"  Markers at closed position: {list(markers_closed.keys())}")
    
    # Load camera intrinsics and extrinsics
    intr_file = repo_root / 'data/calib/intrinsics_realsense_d455.npz'
    extr_file = repo_root / 'data/calib/extrinsics_realsense_d455.npz'
    
    if not intr_file.exists() or not extr_file.exists():
        print(f"❌ Camera calibration files not found")
        return 1
    
    intr_data = np.load(intr_file)
    K = intr_data['K'].astype(np.float64)
    D = intr_data['D'].astype(np.float64)
    
    extr_data = np.load(extr_file)
    T_base_cam = extr_data['T_base_cam'].astype(np.float64)
    
    print(f"✓ Loaded camera calibration")
    
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
    ar_dict = cv2.aruco.getPredefinedDictionary(DICT_MAP[dict_name])
    try:
        detector = cv2.aruco.ArucoDetector(ar_dict)
        use_new_api = True
    except (AttributeError, TypeError):
        use_new_api = False
    
    print(f"✓ Using OpenCV ArUco {'new' if use_new_api else 'old'} API")
    print(f"\n🎯 Tracking drawer position")
    print("Press 'q' to quit\n")
    
    # Warmup
    for _ in range(5):
        pipeline.wait_for_frames(timeout_ms=5000)
    
    cv2.namedWindow('Drawer Position Tracker', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Drawer Position Tracker', 1280, 720)

    try:
        while True:
            # Capture frame
            frames = pipeline.wait_for_frames(timeout_ms=5000)
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            image = np.asanyarray(color_frame.get_data())
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Detect markers
            if use_new_api:
                corners, ids, _ = detector.detectMarkers(gray)
            else:
                corners, ids, _ = cv2.aruco.detectMarkers(gray, ar_dict)

            vis = image.copy()
            drawer_distances = []
            
            if ids is not None and len(ids) > 0:
                # Draw detected markers
                cv2.aruco.drawDetectedMarkers(vis, corners, ids)
                
                # Process each detected marker
                for i, marker_id in enumerate(ids.flatten()):
                    if marker_id not in markers_closed:
                        continue
                    
                    # Estimate current pose in camera frame
                    T_cam_marker = estimate_marker_pose(corners[i], marker_size, K, D)
                    if T_cam_marker is None:
                        continue
                    
                    # Convert to world frame
                    T_world_marker_current = T_base_cam @ T_cam_marker
                    pos_current = T_world_marker_current[:3, 3]
                    
                    # Get closed position
                    pos_closed = markers_closed[marker_id]
                    
                    # Compute distance (drawer slides along one axis, but compute full 3D distance)
                    delta = pos_current - pos_closed
                    distance = np.linalg.norm(delta)
                    
                    drawer_distances.append({
                        'id': marker_id,
                        'distance': distance,
                        'delta': delta,
                        'pos_current': pos_current,
                        'pos_closed': pos_closed
                    })
                    
                    # Draw axis
                    R = T_cam_marker[:3, :3]
                    tvec = T_cam_marker[:3, 3].reshape(3, 1)
                    rvec, _ = cv2.Rodrigues(R)
                    try:
                        cv2.drawFrameAxes(vis, K, D, rvec, tvec, marker_size * 0.5)
                    except Exception:
                        pass
            
            # Display results
            y_offset = 30
            line_height = 35
            
            if drawer_distances:
                # Average distance from all detected markers
                avg_distance = np.mean([d['distance'] for d in drawer_distances])
                
                # Main distance display (large text)
                distance_text = f"Drawer: {avg_distance*100:.1f} cm ({avg_distance:.4f} m)"
                cv2.putText(vis, distance_text, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                y_offset += line_height + 10
                
                # Per-marker details
                cv2.putText(vis, f"Markers detected: {len(drawer_distances)}", (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                y_offset += line_height
                
                for d in drawer_distances:
                    detail_text = f"  ID{d['id']}: {d['distance']*100:.1f} cm"
                    cv2.putText(vis, detail_text, (10, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
                    y_offset += line_height
                
                # Delta vector (shows which direction drawer moved)
                if len(drawer_distances) > 0:
                    avg_delta = np.mean([d['delta'] for d in drawer_distances], axis=0)
                    delta_text = f"Delta: [{avg_delta[0]*100:.1f}, {avg_delta[1]*100:.1f}, {avg_delta[2]*100:.1f}] cm"
                    cv2.putText(vis, delta_text, (10, y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
            else:
                # No markers detected
                cv2.putText(vis, "No drawer markers detected", (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                y_offset += line_height
                cv2.putText(vis, f"Looking for IDs: {list(markers_closed.keys())}", (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.imshow('Drawer Position Tracker', vis)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

    return 0


if __name__ == '__main__':
    sys.exit(main())
