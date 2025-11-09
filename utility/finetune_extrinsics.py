#!/usr/bin/env python3
"""
Interactive extrinsics fine-tuning tool for RealSense D455.

Captures a live image, runs pose estimation, and allows you to interactively
adjust the camera extrinsics to align projected 3D objects with the real image.

Usage:
    python utility/finetune_extrinsics.py --serial 151422252817

Controls:
    Translation (meters):
        w/s: +/- X (forward/back)
        a/d: +/- Y (left/right)
        q/e: +/- Z (up/down)
    
    Rotation (degrees):
        i/k: +/- pitch (around Y)
        j/l: +/- yaw (around Z)
        u/o: +/- roll (around X)
    
    Step size:
        [/]: decrease/increase translation step
        ;/': decrease/increase rotation step
    
    Other:
        r: reset to original calibration
        c: capture new image & re-estimate poses
        space: save current calibration
        esc/q: quit without saving
"""

import argparse
import json
import os
import sys
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs
import torch
from scipy.spatial.transform import Rotation as R


class ExtrinsicsFineTuner:
    """Interactive tool for fine-tuning camera extrinsics."""
    
    # Object specifications (matching isaac_sim_worker.py)
    OBJECTS = {
        'Cube_Blue': {'type': 'cube', 'size': 0.05},  # 5cm cube
        'Cube_Red': {'type': 'cube', 'size': 0.05},   # 5cm cube
        'Tennis': {'type': 'sphere', 'radius': 0.0335},  # Tennis ball
    }
    
    # Colors for visualization (BGR)
    COLORS = {
        'Cube_Blue': (255, 0, 0),    # Blue
        'Cube_Red': (0, 0, 255),     # Red
        'Tennis': (0, 255, 255),     # Yellow
    }
    
    def __init__(self, serial_number, repo_root):
        self.serial = serial_number
        self.repo_root = Path(repo_root)
        
        # Load intrinsics
        self.K, self.D = self._load_intrinsics()
        
        # Load initial extrinsics
        self.T_world_cam_original = self._load_extrinsics()
        self.T_world_cam = self.T_world_cam_original.copy()
        
        # Control parameters
        self.trans_step = 0.01  # 1cm
        self.rot_step = 1.0     # 1 degree
        
        # Captured data
        self.image = None
        self.object_poses_cam = {}  # Object poses in camera frame (4x4 matrices)
        
        # Window setup
        cv2.namedWindow('Extrinsics Fine-Tuning', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Extrinsics Fine-Tuning', 1280, 720)
        
    def _load_intrinsics(self):
        """Load camera intrinsics."""
        intr_file = self.repo_root / 'data' / 'calib' / 'intrinsics_realsense_d455.npz'
        if not intr_file.exists():
            raise FileNotFoundError(f"Intrinsics not found: {intr_file}")
        
        data = np.load(intr_file, allow_pickle=True)
        K = np.asarray(data['K'], dtype=np.float64)
        D = np.asarray(data['D'], dtype=np.float64)
        
        print(f"✓ Loaded intrinsics from {intr_file.name}")
        return K, D
    
    def _load_extrinsics(self):
        """Load camera extrinsics."""
        extr_file = self.repo_root / 'data' / 'calib' / 'extrinsics_realsense_d455.npz'
        if not extr_file.exists():
            raise FileNotFoundError(f"Extrinsics not found: {extr_file}")
        
        data = np.load(extr_file, allow_pickle=True)
        T = np.asarray(data['T_base_cam'], dtype=np.float64)
        
        pos = T[:3, 3]
        print(f"✓ Loaded extrinsics from {extr_file.name}")
        print(f"  Camera position: X={pos[0]:.3f}m, Y={pos[1]:.3f}m, Z={pos[2]:.3f}m")
        return T
    
    def capture_image_and_estimate_poses(self):
        """Capture image from RealSense and run pose estimation."""
        print("\n📸 Capturing image from RealSense...")
        
        # Initialize RealSense pipeline
        pipeline = rs.pipeline()
        config = rs.config()
        
        # Hardware reset helps resolve timeout issues (from intelrealsense.py)
        ctx = rs.context()
        devices = ctx.query_devices()
        for device in devices:
            if device.get_info(rs.camera_info.serial_number) == str(self.serial):
                print(f"  Performing hardware reset on camera {self.serial}")
                device.hardware_reset()
                time.sleep(3)
                break
        
        config.enable_device(str(self.serial))
        config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)  # ENABLE DEPTH!
        
        try:
            print("  Starting pipeline...")
            pipeline.start(config)
            
            # Create alignment object to align depth to color
            align_to_color = rs.align(rs.stream.color)
            
            # Warm up camera - discard first few frames to let auto-exposure stabilize
            print("  Warming up camera (discarding first 5 frames)...")
            for i in range(5):
                try:
                    pipeline.wait_for_frames(timeout_ms=10000)
                except RuntimeError as e:
                    print(f"  ⚠️  Warmup frame {i} failed: {e}, continuing...")
            
            # Capture frame with longer timeout
            print("  Capturing frame...")
            frames = pipeline.wait_for_frames(timeout_ms=10000)
            
            # Align depth to color
            aligned_frames = align_to_color.process(frames)
            
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            
            if not color_frame or not depth_frame:
                raise RuntimeError("Failed to capture color or depth frame")
            
            # Convert to numpy arrays
            rgb_image = np.asanyarray(color_frame.get_data())  # RGB format
            depth_image_uint16 = np.asanyarray(depth_frame.get_data())  # uint16 in millimeters
            
            # Convert depth from uint16 millimeters to float32 meters
            self.depth_map = depth_image_uint16.astype(np.float32) / 1000.0
            
            # Convert RGB to BGR for OpenCV compatibility
            import cv2
            self.image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
            
            print(f"✓ Captured {self.image.shape[1]}x{self.image.shape[0]} image")
            print(f"✓ Captured depth map, range: {self.depth_map.min():.3f}m to {self.depth_map.max():.3f}m")
            
        finally:
            pipeline.stop()
        
        # Save captured image
        img_path = self.repo_root / 'data' / 'calib' / 'finetune_capture.jpg'
        cv2.imwrite(str(img_path), self.image)
        print(f"✓ Saved capture to {img_path.name}")
        
        # Show captured image for verification
        print("\n👁️  Displaying captured image...")
        print("   Close the window to continue with pose estimation")
        cv2.imshow('Captured Image - Verify Objects Visible', self.image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # Run pose estimation
        print("\n🔍 Running pose estimation...")
        self._run_pose_estimation(img_path)
    
    def _run_pose_estimation(self, image_path):
        """Run pose estimation using the any6d workers via disk-based job queue."""
        import uuid
        import torch
        
        # Create pose_jobs directory structure
        pose_jobs_root = self.repo_root / 'data' / 'calib' / 'pose_jobs'
        inbox = pose_jobs_root / 'inbox'
        outbox = pose_jobs_root / 'outbox'
        tmp = pose_jobs_root / 'tmp'
        
        for d in [inbox, outbox, tmp]:
            d.mkdir(parents=True, exist_ok=True)
        
        # Clean old jobs
        for f in inbox.glob("*.json"):
            f.unlink()
        for f in outbox.glob("*.json"):
            f.unlink()
        
        # Save observation in format expected by pose workers
        obs_path = self.repo_root / 'data' / 'calib' / 'finetune_obs.pt'
        
        # Convert image to RGB tensor
        rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        rgb_tensor = torch.from_numpy(rgb).to(torch.uint8)
        
        # Use REAL depth data captured from RealSense
        depth_tensor = torch.from_numpy(self.depth_map).to(torch.float32)
        
        # Use the key names that pose_worker.py expects
        obs = {
            'observation.images.cam_main': rgb_tensor,  # Worker expects this key
            'depth': depth_tensor,  # Worker expects this key - NOW WITH REAL DEPTH!
        }
        torch.save(obs, obs_path)
        print(f"✓ Saved observation to {obs_path.name}")
        
        # Get intrinsics
        K_list = self.K.tolist()
        
        # Object mesh paths (from your project structure)
        object_meshes = {
            'Cube_Blue': self.repo_root / 'public' / 'assets' / 'cube.obj',
            'Cube_Red': self.repo_root / 'public' / 'assets' / 'cube.obj',
            'Tennis': self.repo_root / 'public' / 'assets' / 'sphere.obj',
        }
        
        object_prompts = {
            'Cube_Blue': 'Blue cube',    # Match production exactly
            'Cube_Red': 'Red cube',      # Match production exactly
            'Tennis': 'Tennis ball',     # Match production exactly
        }
        
        # Create jobs
        job_ids = {}
        for obj_name, mesh_path in object_meshes.items():
            if not mesh_path.exists():
                print(f"⚠️  Mesh not found: {mesh_path}")
                continue
            
            job_id = f"finetune_{obj_name}_{uuid.uuid4().hex[:8]}"
            job_ids[obj_name] = job_id
            
            job = {
                'job_id': job_id,
                'episode_id': 0,
                'state_id': 0,
                'object': obj_name,
                'obs_path': str(obs_path),
                'K': K_list,
                'prompt': object_prompts[obj_name],
                'est_refine_iter': 20,
                'track_refine_iter': 8,
            }
            
            tmp_file = tmp / f"{job_id}.json"
            dst_file = inbox / f"{job_id}.json"
            
            with open(tmp_file, 'w') as f:
                json.dump(job, f)
            os.replace(tmp_file, dst_file)
            print(f"   📝 Job: {job_id}")
        
        if not job_ids:
            print("❌ No valid jobs created (mesh files missing?)")
            return
        
        # Wait for results (with timeout)
        print(f"⏳ Waiting for pose estimation results (timeout: 30s)...")
        import time
        deadline = time.time() + 30.0
        results = {}
        
        while len(results) < len(job_ids) and time.time() < deadline:
            for obj_name, job_id in job_ids.items():
                if obj_name in results:
                    continue
                
                result_file = outbox / f"{job_id}.json"
                if result_file.exists():
                    try:
                        with open(result_file, 'r') as f:
                            result = json.load(f)
                        
                        if result.get('success') and result.get('pose_cam_T_obj'):
                            pose_matrix = np.array(result['pose_cam_T_obj'], dtype=np.float64)
                            results[obj_name] = pose_matrix
                            print(f"   ✅ {obj_name}: pose received")
                        else:
                            error_msg = result.get('error', 'unknown error')
                            reason = result.get('reason', '')
                            mask_area = result.get('mask_area', 'N/A')
                            
                            # Build detailed error message
                            details = []
                            if error_msg and error_msg != 'unknown error':
                                details.append(error_msg)
                            if reason:
                                details.append(f"reason={reason}")
                            if mask_area != 'N/A':
                                details.append(f"mask_area={mask_area}")
                            
                            error_str = ', '.join(details) if details else 'pose failed (no details)'
                            print(f"   ❌ {obj_name}: {error_str}")
                            results[obj_name] = None
                        
                        result_file.unlink()
                    except Exception as e:
                        print(f"   ⚠️  {obj_name}: error reading result: {e}")
            
            if len(results) < len(job_ids):
                time.sleep(0.1)
        
        # Store results
        self.object_poses_cam = {k: v for k, v in results.items() if v is not None}
        
        if not self.object_poses_cam:
            print("❌ No successful pose estimations")
            print("   Make sure pose workers are running:")
            print("   conda run -n any6d python backend/any6d/pose_worker.py \\")
            print("     --jobs-dir data/calib/pose_jobs --object Cube_Blue \\")
            print("     --mesh backend/isaac_sim/assets/meshes/cube_blue.obj \\")
            print("     --prompt 'blue cube'")
        else:
            print(f"✓ Estimated poses for {len(self.object_poses_cam)} objects")
    
    def _project_cube(self, T_world_obj, size):
        """Project a cube's corners onto the image."""
        # Cube corners in object frame
        half = size / 2
        corners_obj = np.array([
            [-half, -half, -half],
            [half, -half, -half],
            [half, half, -half],
            [-half, half, -half],
            [-half, -half, half],
            [half, -half, half],
            [half, half, half],
            [-half, half, half],
        ])
        
        # Transform to world frame
        corners_world = (T_world_obj[:3, :3] @ corners_obj.T).T + T_world_obj[:3, 3]
        
        # Transform to camera frame
        T_cam_world = np.linalg.inv(self.T_world_cam)
        corners_cam = (T_cam_world[:3, :3] @ corners_world.T).T + T_cam_world[:3, 3]
        
        # Project to image
        corners_img = []
        for corner in corners_cam:
            # Simple pinhole projection (no distortion for visualization)
            x = corner[0] / corner[2] * self.K[0, 0] + self.K[0, 2]
            y = corner[1] / corner[2] * self.K[1, 1] + self.K[1, 2]
            corners_img.append([x, y])
        
        return np.array(corners_img, dtype=np.int32)
    
    def _project_sphere(self, T_world_obj, radius):
        """Project a sphere's outline onto the image."""
        center_world = T_world_obj[:3, 3]
        
        # Transform to camera frame
        T_cam_world = np.linalg.inv(self.T_world_cam)
        center_cam = (T_cam_world[:3, :3] @ center_world) + T_cam_world[:3, 3]
        
        # Project center
        cx = center_cam[0] / center_cam[2] * self.K[0, 0] + self.K[0, 2]
        cy = center_cam[1] / center_cam[2] * self.K[1, 1] + self.K[1, 2]
        
        # Approximate radius in pixels
        # Project a point at radius distance from center
        radius_point_cam = center_cam + np.array([radius, 0, 0])
        rx = radius_point_cam[0] / radius_point_cam[2] * self.K[0, 0] + self.K[0, 2]
        r_pixels = abs(rx - cx)
        
        return (int(cx), int(cy)), int(r_pixels)
    
    def _draw_cube(self, img, corners, color):
        """Draw a 3D cube on the image."""
        # Draw back face
        cv2.polylines(img, [corners[:4]], True, color, 2)
        # Draw front face
        cv2.polylines(img, [corners[4:]], True, color, 2)
        # Draw connecting edges
        for i in range(4):
            cv2.line(img, tuple(corners[i]), tuple(corners[i+4]), color, 2)
    
    def render_overlay(self):
        """Render projected objects onto the image."""
        if self.image is None:
            return None
        
        overlay = self.image.copy()
        
        for obj_name, T_cam_obj in self.object_poses_cam.items():
            # Transform to world frame
            T_world_obj = self.T_world_cam @ T_cam_obj
            
            color = self.COLORS[obj_name]
            obj_spec = self.OBJECTS[obj_name]
            
            if obj_spec['type'] == 'cube':
                corners = self._project_cube(T_world_obj, obj_spec['size'])
                self._draw_cube(overlay, corners, color)
                
                # Draw label
                center = corners.mean(axis=0).astype(int)
                cv2.putText(overlay, obj_name, tuple(center), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
            elif obj_spec['type'] == 'sphere':
                center, radius = self._project_sphere(T_world_obj, obj_spec['radius'])
                cv2.circle(overlay, center, radius, color, 2)
                cv2.putText(overlay, obj_name, (center[0] - 30, center[1] - radius - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return overlay
    
    def _get_pose_as_xyz_rpy(self):
        """Get current pose as position + Euler angles."""
        pos = self.T_world_cam[:3, 3]
        rot = R.from_matrix(self.T_world_cam[:3, :3])
        rpy = rot.as_euler('xyz', degrees=True)
        return pos, rpy
    
    def _set_pose_from_xyz_rpy(self, pos, rpy_deg):
        """Set pose from position + Euler angles."""
        rot = R.from_euler('xyz', rpy_deg, degrees=True)
        self.T_world_cam[:3, :3] = rot.as_matrix()
        self.T_world_cam[:3, 3] = pos
    
    def adjust_translation(self, axis, delta):
        """Adjust translation along an axis."""
        pos, rpy = self._get_pose_as_xyz_rpy()
        pos[axis] += delta
        self._set_pose_from_xyz_rpy(pos, rpy)
    
    def adjust_rotation(self, axis, delta_deg):
        """Adjust rotation around an axis."""
        pos, rpy = self._get_pose_as_xyz_rpy()
        rpy[axis] += delta_deg
        self._set_pose_from_xyz_rpy(pos, rpy)
    
    def reset_calibration(self):
        """Reset to original calibration."""
        self.T_world_cam = self.T_world_cam_original.copy()
        print("\n🔄 Reset to original calibration")
    
    def save_calibration(self):
        """Save current calibration."""
        extr_file = self.repo_root / 'data' / 'calib' / 'extrinsics_realsense_d455.npz'
        
        # Load existing data
        data = dict(np.load(extr_file, allow_pickle=True))
        
        # Update with new calibration
        data['T_base_cam'] = self.T_world_cam
        
        # Also update T_three (Three.js convention)
        T_three = self.T_world_cam.copy()
        T_three[:3, 1] *= -1  # Flip Y
        T_three[:3, 2] *= -1  # Flip Z
        data['T_three'] = T_three
        
        # Save
        np.savez(extr_file, **data)
        
        pos = self.T_world_cam[:3, 3]
        print(f"\n💾 Saved calibration to {extr_file.name}")
        print(f"   New position: X={pos[0]:.3f}m, Y={pos[1]:.3f}m, Z={pos[2]:.3f}m")
    
    def run(self):
        """Main interaction loop."""
        # Capture initial image and poses
        self.capture_image_and_estimate_poses()
        
        print("\n" + "="*60)
        print("INTERACTIVE EXTRINSICS FINE-TUNING")
        print("="*60)
        print("\nControls:")
        print("  Translation: w/s (X), a/d (Y), q/e (Z)")
        print("  Rotation:    i/k (pitch), j/l (yaw), u/o (roll)")
        print("  Step size:   [/] (trans), ;/' (rot)")
        print("  Actions:     r (reset), c (recapture), space (save), esc/q (quit)")
        print("="*60)
        
        while True:
            # Render overlay
            overlay = self.render_overlay()
            if overlay is None:
                break
            
            # Add info overlay
            pos, rpy = self._get_pose_as_xyz_rpy()
            info_lines = [
                f"Position: X={pos[0]:.3f} Y={pos[1]:.3f} Z={pos[2]:.3f} m",
                f"Rotation: R={rpy[0]:.1f} P={rpy[1]:.1f} Y={rpy[2]:.1f} deg",
                f"Steps: trans={self.trans_step*1000:.1f}mm, rot={self.rot_step:.1f}deg",
            ]
            
            y_offset = 30
            for line in info_lines:
                cv2.putText(overlay, line, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                y_offset += 25
            
            cv2.imshow('Extrinsics Fine-Tuning', overlay)
            
            # Handle keyboard input
            key = cv2.waitKey(10) & 0xFF
            
            if key == 27 or key == ord('q'):  # ESC or 'q'
                print("\n👋 Exiting without saving")
                break
            
            elif key == ord(' '):  # Space - save
                self.save_calibration()
                print("Press ESC to exit or continue adjusting...")
            
            elif key == ord('r'):  # Reset
                self.reset_calibration()
            
            elif key == ord('c'):  # Recapture
                self.capture_image_and_estimate_poses()
            
            # Translation controls
            elif key == ord('w'):
                self.adjust_translation(0, self.trans_step)
            elif key == ord('s'):
                self.adjust_translation(0, -self.trans_step)
            elif key == ord('a'):
                self.adjust_translation(1, -self.trans_step)
            elif key == ord('d'):
                self.adjust_translation(1, self.trans_step)
            elif key == ord('q'):
                self.adjust_translation(2, self.trans_step)
            elif key == ord('e'):
                self.adjust_translation(2, -self.trans_step)
            
            # Rotation controls
            elif key == ord('i'):
                self.adjust_rotation(1, self.rot_step)  # Pitch
            elif key == ord('k'):
                self.adjust_rotation(1, -self.rot_step)
            elif key == ord('j'):
                self.adjust_rotation(2, -self.rot_step)  # Yaw
            elif key == ord('l'):
                self.adjust_rotation(2, self.rot_step)
            elif key == ord('u'):
                self.adjust_rotation(0, self.rot_step)  # Roll
            elif key == ord('o'):
                self.adjust_rotation(0, -self.rot_step)
            
            # Step size controls
            elif key == ord('['):
                self.trans_step = max(0.001, self.trans_step / 2)
                print(f"Translation step: {self.trans_step*1000:.1f}mm")
            elif key == ord(']'):
                self.trans_step = min(0.1, self.trans_step * 2)
                print(f"Translation step: {self.trans_step*1000:.1f}mm")
            elif key == ord(';'):
                self.rot_step = max(0.1, self.rot_step / 2)
                print(f"Rotation step: {self.rot_step:.1f}°")
            elif key == ord("'"):
                self.rot_step = min(10.0, self.rot_step * 2)
                print(f"Rotation step: {self.rot_step:.1f}°")
        
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='Interactive extrinsics fine-tuning tool')
    parser.add_argument('--serial', type=str, required=True,
                       help='RealSense camera serial number')
    parser.add_argument('--repo-root', type=str, default='.',
                       help='Repository root directory (default: current directory)')
    
    args = parser.parse_args()
    
    try:
        tuner = ExtrinsicsFineTuner(args.serial, args.repo_root)
        tuner.run()
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
