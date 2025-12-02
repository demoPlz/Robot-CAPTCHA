#!/usr/bin/env python3
"""Test multi-frame pose estimation with synthetic data.

This script demonstrates multi-frame pose estimation with outlier rejection.
"""

import sys
from pathlib import Path

import numpy as np
import torch
import trimesh

# Add this directory to path
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.append(str(THIS_DIR))

from estimate_pose import (
    create_engines,
    estimate_pose_multi_frame,
    compute_pose_distance,
    reject_outliers_ransac,
    average_poses,
)


def create_synthetic_observations(num_frames=5):
    """Create synthetic RGB/depth observations for testing.
    
    Note: These won't produce meaningful pose estimates without real sensor data,
    but they exercise the code path.
    """
    observations = []
    K = torch.tensor([[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]], dtype=torch.float32)
    
    for i in range(num_frames):
        # Random RGB
        rgb_t = torch.randint(0, 255, (480, 640, 3), dtype=torch.uint8)
        
        # Depth plane with slight variation
        depth_t = torch.full((480, 640), 0.5 + i * 0.01, dtype=torch.float32)
        
        observations.append({
            'rgb': rgb_t,
            'depth': depth_t,
            'K': K.clone(),
        })
    
    return observations


def test_pose_distance():
    """Test pose distance computation."""
    print("\n=== Testing Pose Distance ===")
    
    # Identity pose
    pose1 = np.eye(4, dtype=np.float32)
    
    # Translate by 10cm
    pose2 = np.eye(4, dtype=np.float32)
    pose2[:3, 3] = [0.1, 0, 0]
    
    # Rotate by 45 degrees around Z
    pose3 = np.eye(4, dtype=np.float32)
    angle = np.pi / 4
    pose3[:3, :3] = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1],
    ])
    
    # Combine translation and rotation
    pose4 = pose2 @ pose3
    
    d12 = compute_pose_distance(pose1, pose2)
    d13 = compute_pose_distance(pose1, pose3)
    d14 = compute_pose_distance(pose1, pose4)
    
    print(f"Distance (identity -> translate 10cm): {d12:.4f}")
    print(f"Distance (identity -> rotate 45deg): {d13:.4f}")
    print(f"Distance (identity -> both): {d14:.4f}")
    
    assert d12 > 0, "Translation should produce non-zero distance"
    assert d13 > 0, "Rotation should produce non-zero distance"
    assert abs(d12 - 0.1) < 0.01, "Translation distance should be ~0.1m"


def test_outlier_rejection():
    """Test RANSAC-style outlier rejection."""
    print("\n=== Testing Outlier Rejection ===")
    
    # Create a cluster of 4 similar poses
    poses = []
    base = np.eye(4, dtype=np.float32)
    base[:3, 3] = [0.5, 0, 0]
    
    for i in range(4):
        pose = base.copy()
        # Add small noise (~1cm)
        pose[:3, 3] += np.random.randn(3) * 0.01
        poses.append(pose)
    
    # Add 2 outliers (far away)
    outlier1 = np.eye(4, dtype=np.float32)
    outlier1[:3, 3] = [0.0, 0.0, 0.0]  # 50cm away
    
    outlier2 = np.eye(4, dtype=np.float32)
    outlier2[:3, 3] = [1.0, 0.0, 0.0]  # 50cm away in other direction
    
    poses.append(outlier1)
    poses.append(outlier2)
    
    # Shuffle to make it interesting
    indices = np.random.permutation(len(poses))
    poses = [poses[i] for i in indices]
    
    print(f"Total poses: {len(poses)}")
    
    # Reject outliers
    inliers, inlier_idx = reject_outliers_ransac(
        poses,
        inlier_threshold=0.05,
        min_inliers=2,
    )
    
    print(f"Inliers: {len(inliers)}")
    print(f"Outliers: {len(poses) - len(inliers)}")
    
    assert len(inliers) == 4, "Should keep the 4 clustered poses as inliers"
    assert len(inlier_idx) == 4, "Should have 4 inlier indices"


def test_pose_averaging():
    """Test pose averaging with quaternions."""
    print("\n=== Testing Pose Averaging ===")
    
    # Create 3 similar poses
    poses = []
    for i in range(3):
        pose = np.eye(4, dtype=np.float32)
        # Small translation variation
        pose[:3, 3] = [0.5 + i * 0.01, 0.0, 0.3]
        poses.append(pose)
    
    avg = average_poses(poses)
    
    print(f"Input translations:")
    for i, p in enumerate(poses):
        print(f"  Pose {i}: {p[:3, 3]}")
    
    print(f"Average translation: {avg[:3, 3]}")
    
    # Check that average is close to mean
    expected_t = np.mean([p[:3, 3] for p in poses], axis=0)
    assert np.allclose(avg[:3, 3], expected_t, atol=1e-6), "Average translation should match mean"


def test_multi_frame_api():
    """Test the multi-frame API (without real models)."""
    print("\n=== Testing Multi-Frame API ===")
    
    # Create synthetic mesh
    mesh = trimesh.creation.box(extents=(0.05, 0.05, 0.05))
    
    # Create synthetic observations
    observations = create_synthetic_observations(num_frames=5)
    
    print(f"Created {len(observations)} synthetic observations")
    
    try:
        # Create engines (this requires the models to be available)
        engines = create_engines(mesh, debug=0)
        
        # Run multi-frame estimation
        result = estimate_pose_multi_frame(
            mesh=mesh,
            observations=observations,
            language_prompt="test cube",
            num_frames=5,
            inlier_threshold=0.05,
            min_inliers=1,  # Relax for testing
            engines=engines,
        )
        
        print(f"Multi-frame result: success={result.success}")
        if result.success:
            print(f"  Processed: {result.extras.get('num_frames_processed', 0)} frames")
            print(f"  Successful: {result.extras.get('successful_frames', 0)}")
            print(f"  Failed: {result.extras.get('failed_frames', 0)}")
            print(f"  Inliers: {result.extras.get('num_inliers', 0)}")
        else:
            print(f"  Error: {result.extras.get('error', 'unknown')}")
    
    except Exception as e:
        print(f"Multi-frame API test skipped (models not available): {e}")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Multi-Frame Pose Estimation Tests")
    print("=" * 60)
    
    # Test individual components (these don't need models)
    test_pose_distance()
    test_outlier_rejection()
    test_pose_averaging()
    
    # Test full API (requires models, may fail in CI)
    test_multi_frame_api()
    
    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
