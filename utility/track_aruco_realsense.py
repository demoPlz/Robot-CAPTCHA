#!/usr/bin/env python3
"""
Track ArUco markers (IDs 0, 1, 2, 3) using RealSense D455 camera.
Displays live visualization with marker detection and pose estimation.

Usage:
    python utility/track_aruco_realsense.py --serial 151422252817
    python utility/track_aruco_realsense.py --serial 151422252817 --dict 4X4_50
"""

import argparse
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


def main():
    parser = argparse.ArgumentParser(description="Track ArUco markers with RealSense D455")
    parser.add_argument('--serial', required=True, help='RealSense serial number')
    parser.add_argument('--dict', default=None, choices=list(DICT_MAP.keys()) + [None],
                       help='ArUco dictionary to use (default: try all)')
    parser.add_argument('--marker-size', type=float, default=0.05,
                       help='Physical marker size in meters (for pose estimation)')
    parser.add_argument('--width', type=int, default=640, help='Frame width')
    parser.add_argument('--height', type=int, default=480, help='Frame height')
    parser.add_argument('--fps', type=int, default=30, help='Frame rate')
    parser.add_argument('--track-ids', type=int, nargs='+', default=[0, 1, 2, 3],
                       help='ArUco IDs to track (default: 0 1 2 3)')
    args = parser.parse_args()

    # Load intrinsics if available
    repo_root = Path(__file__).resolve().parent.parent
    intr_file = repo_root / 'data/calib/intrinsics_realsense_d455.npz'
    
    K = None
    D = None
    if intr_file.exists():
        print(f"Loading intrinsics from {intr_file}")
        intr_data = np.load(intr_file)
        K = intr_data['K'].astype(np.float64)
        D = intr_data['D'].astype(np.float64)
        print(f"✓ Loaded camera matrix K and distortion D")
    else:
        print(f"⚠ Intrinsics file not found: {intr_file}")
        print("  Pose estimation will use approximate intrinsics from RealSense")

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
    
    # Get intrinsics from RealSense if not loaded from file
    if K is None:
        color_stream = profile.get_stream(rs.stream.color)
        intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
        K = np.array([
            [intrinsics.fx, 0, intrinsics.ppx],
            [0, intrinsics.fy, intrinsics.ppy],
            [0, 0, 1]
        ], dtype=np.float64)
        D = np.array(intrinsics.coeffs, dtype=np.float64)
        print(f"✓ Using RealSense intrinsics")

    # Setup ArUco detection - try all dictionaries if not specified
    detectors = {}
    dict_names = [args.dict] if args.dict else list(DICT_MAP.keys())
    
    for dict_name in dict_names:
        ar_dict = cv2.aruco.getPredefinedDictionary(DICT_MAP[dict_name])
        
        # Detect if we have new or old OpenCV ArUco API
        try:
            detector = cv2.aruco.ArucoDetector(ar_dict)
            use_new_api = True
        except (AttributeError, TypeError):
            detector = ar_dict  # Old API just uses the dictionary
            use_new_api = False
        
        detectors[dict_name] = (detector, ar_dict)
    
    if len(detectors) == 1:
        print(f"✓ Using OpenCV ArUco {'new' if use_new_api else 'old'} API")
        print(f"📖 Dictionary: {dict_names[0]}")
    else:
        print(f"✓ Using OpenCV ArUco {'new' if use_new_api else 'old'} API")
        print(f"📖 Testing dictionaries: {', '.join(dict_names)}")

    print(f"\n🎯 Tracking ArUco IDs: {args.track_ids}")
    print(f"📏 Marker size: {args.marker_size}m ({args.marker_size*100}cm)")
    print("\nPress 'q' to quit, 's' to save current frame\n")

    cv2.namedWindow('ArUco Tracking', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('ArUco Tracking', 1280, 720)

    frame_count = 0
    track_ids_set = set(args.track_ids)

    try:
        while True:
            # Capture frame
            frames = pipeline.wait_for_frames(timeout_ms=5000)
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            image = np.asanyarray(color_frame.get_data())
            frame_count += 1

            # Convert to grayscale for detection
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Detect ArUco markers with all dictionaries
            all_corners = []
            all_ids = []
            dict_detections = {}
            
            for dict_name, (detector, ar_dict) in detectors.items():
                if use_new_api:
                    corners, ids, rejected = detector.detectMarkers(gray)
                else:
                    corners, ids, rejected = cv2.aruco.detectMarkers(gray, ar_dict)
                
                if ids is not None and len(ids) > 0:
                    dict_detections[dict_name] = (corners, ids)
                    all_corners.extend(corners)
                    all_ids.extend(ids)
            
            # Use the dictionary with most detections, or first one with any
            best_dict = None
            corners = None
            ids = None
            
            if dict_detections:
                # Pick dictionary with most detections
                best_dict = max(dict_detections.keys(), key=lambda d: len(dict_detections[d][1]))
                corners, ids = dict_detections[best_dict]

            # Visualization
            vis = image.copy()
            detected_count = 0
            
            if ids is not None and len(ids) > 0:
                # Draw all detected markers
                cv2.aruco.drawDetectedMarkers(vis, corners, ids)

                # Process tracked IDs
                for i, marker_id in enumerate(ids.flatten()):
                    if marker_id in track_ids_set:
                        detected_count += 1
                        
                        # Estimate pose if we have calibration
                        if K is not None:
                            # Get 3D points of marker corners (in marker coordinate system)
                            half_size = args.marker_size / 2.0
                            obj_points = np.array([
                                [-half_size,  half_size, 0],
                                [ half_size,  half_size, 0],
                                [ half_size, -half_size, 0],
                                [-half_size, -half_size, 0]
                            ], dtype=np.float64)
                            
                            # Get 2D corners
                            img_points = corners[i].reshape(-1, 2).astype(np.float64)
                            
                            # Solve PnP
                            success, rvec, tvec = cv2.solvePnP(
                                obj_points, img_points, K, D,
                                flags=cv2.SOLVEPNP_IPPE_SQUARE
                            )
                            
                            if success:
                                # Draw axis
                                try:
                                    cv2.drawFrameAxes(vis, K, D, rvec, tvec, args.marker_size * 0.5)
                                except Exception:
                                    pass
                                
                                # Display position info
                                corner = corners[i][0]
                                center = corner.mean(axis=0).astype(int)
                                pos_text = f"ID{marker_id}: [{tvec[0][0]:.3f}, {tvec[1][0]:.3f}, {tvec[2][0]:.3f}]m"
                                cv2.putText(vis, pos_text, (center[0] + 15, center[1] - 10),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Status overlay
            status_y = 30
            cv2.putText(vis, f"Frame: {frame_count}", (10, status_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            status_y += 30
            
            # Show which dictionary detected markers
            if best_dict:
                cv2.putText(vis, f"Dict: {best_dict}", (10, status_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                status_y += 30
            
            total_detected = len(ids) if ids is not None else 0
            cv2.putText(vis, f"Markers: {total_detected} total, {detected_count}/{len(args.track_ids)} tracked",
                       (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            status_y += 30

            # Show which IDs are detected
            if detected_count > 0:
                detected_ids = [int(id) for id in ids.flatten() if id in track_ids_set]
                cv2.putText(vis, f"Detected: {sorted(detected_ids)}",
                           (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                missing_ids = sorted(args.track_ids)
                cv2.putText(vis, f"Missing: {missing_ids}",
                           (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            cv2.imshow('ArUco Tracking', vis)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\nQuitting...")
                break
            elif key == ord('s'):
                # Save current frame
                save_path = repo_root / f'aruco_frame_{frame_count:06d}.png'
                cv2.imwrite(str(save_path), vis)
                print(f"💾 Saved frame to {save_path}")

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print(f"\n✓ Processed {frame_count} frames")

    return 0


if __name__ == '__main__':
    sys.exit(main())
