# Pose Estimation with Repeated Inference (Outlier Rejection)

This module provides **robust pose estimation with outlier rejection** using repeated inference on single frames.

**Default Mode: Single Inference** - Standard behavior. Use `--repeated-inference` flag for robustness via outlier rejection.

## Overview

### Single Inference (DEFAULT) ⭐

Runs pose estimation once on the observation - the original behavior. Fastest option but no robustness.

### Repeated Inference

Runs inference multiple times on the **same observation** to get robust results:
1. **Runs independent pose estimation** multiple times on the same input
2. **Relies on inherent randomness** in the neural networks for variation
3. **Rejects outliers** using RANSAC-style consensus based on pose distance
4. **Averages** the inlier poses for robustness

**Why no noise?** Adding noise to RGB/depth breaks LangSAM's mask generation (produces empty masks). Instead, we leverage the inherent stochasticity in the neural networks themselves.

**Settings**: 5 inferences, min 2 inliers required for success

### Multi-Frame Mode (Optional)

Tracks across multiple different observations with continuous tracking. Requires multiple observation files.
1. **Initializes** pose estimation on the first frame using Any6D
2. **Tracks** the object across subsequent frames using FoundationPose
3. **Rejects outliers** using RANSAC-style consensus based on pose distance
4. **Averages** the inlier poses to produce a robust final estimate

### Single-Frame Approach (Legacy)

Original behavior - fastest but least robust, prone to outliers.

This is inspired by `run_demo_realsense.py` which demonstrates continuous tracking across video frames.

## Key Benefits

- **Robustness**: Outlier rejection handles occasional tracking failures
- **Accuracy**: Averaging multiple estimates reduces noise
- **Temporal consistency**: Tracking leverages motion coherence across frames

## API Usage

### Single-Frame (Original)

```python
from estimate_pose import estimate_pose_from_tensors, create_engines

engines = create_engines(mesh)
result = estimate_pose_from_tensors(
    mesh=mesh,
    rgb_t=rgb_tensor,
    depth_t=depth_tensor,
    K=intrinsics,
    language_prompt="blue cube",
    engines=engines,
)
```

### Repeated Inference (New - Single Frame with Robustness) ⭐

```python
from estimate_pose import estimate_pose_repeated, create_engines

engines = create_engines(mesh)
result = estimate_pose_repeated(
    mesh=mesh,
    rgb_t=rgb_tensor,
    depth_t=depth_tensor,
    K=intrinsics,
    language_prompt="blue cube",
    num_inferences=5,  # Run 5 times with different noise
    rgb_noise_std=2.0,  # Small RGB noise (0-255 scale)
    depth_noise_std=0.002,  # ~2mm depth noise
    inlier_threshold=0.05,  # 5cm distance threshold
    min_inliers=3,  # Need at least 3 inliers
    engines=engines,
)

if result.success:
    print(f"Averaged pose from {result.extras['num_inliers']} inliers")
    print(f"Rejected {result.extras['num_outliers']} outliers")
    pose = result.pose_cam_T_obj  # 4x4 averaged pose
```

### Multi-Frame (Multiple Different Observations)

```python
from estimate_pose import estimate_pose_multi_frame, create_engines

# Prepare observations (list of dicts with 'rgb', 'depth', 'K')
observations = [
    {'rgb': rgb_t1, 'depth': depth_t1, 'K': K},
    {'rgb': rgb_t2, 'depth': depth_t2, 'K': K},
    {'rgb': rgb_t3, 'depth': depth_t3, 'K': K},
    # ... more frames
]

engines = create_engines(mesh)
result = estimate_pose_multi_frame(
    mesh=mesh,
    observations=observations,
    language_prompt="blue cube",
    num_frames=5,  # Process first 5 frames
    inlier_threshold=0.05,  # 5cm distance threshold
    min_inliers=2,  # Need at least 2 inliers
    engines=engines,
)

if result.success:
    print(f"Averaged pose from {result.extras['num_inliers']} inliers")
    print(f"Rejected {result.extras['num_outliers']} outliers")
    pose = result.pose_cam_T_obj  # 4x4 averaged pose
```

## Worker Usage

### Standard Usage (Single Inference - DEFAULT)

Job JSON (standard format):
```json
{
  "job_id": "12345",
  "object": "Cube_Blue",
  "obs_path": "/path/to/observation.pt",
  "K": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
  "prompt": "blue cube"
}
```

Start worker (single inference):
```bash
conda run -n any6d python pose_worker.py \
  --jobs-dir /tmp/crowd_obs_cache/pose_jobs \
  --object Cube_Blue \
  --mesh assets/meshes/cube_blue.obj \
  --prompt "blue cube"
```

### Enable Repeated Inference (Robust Mode)

Use `--repeated-inference` for robustness via outlier rejection:
```bash
conda run -n any6d python pose_worker.py \
  --jobs-dir /tmp/crowd_obs_cache/pose_jobs \
  --object Cube_Blue \
  --mesh assets/meshes/cube_blue.obj \
  --prompt "blue cube" \
  --repeated-inference
```

Or in job JSON:
```json
{
  "repeated_inference": true
}
```

### Multi-Frame Mode (Optional)

Job JSON with multiple observations:
```json
{
  "job_id": "12345",
  "object": "Cube_Blue",
  "obs_paths": [
    "/path/to/obs1.pt",
    "/path/to/obs2.pt",
    "/path/to/obs3.pt",
    "/path/to/obs4.pt",
    "/path/to/obs5.pt"
  ],
  "K": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
  "prompt": "blue cube"
}
```

Start worker with multi-frame:
```bash
conda run -n any6d python pose_worker.py \
  --jobs-dir /tmp/crowd_obs_cache/pose_jobs \
  --object Cube_Blue \
  --mesh assets/meshes/cube_blue.obj \
  --prompt "blue cube" \
  --multi-frame \
  --num-frames 5 \
  --min-inliers 2
```

## Parameters

### `estimate_pose_repeated()` (Recommended)

- **`rgb_t`, `depth_t`, `K`**: Single observation to process
- **`num_inferences`**: How many times to run inference with noise (default: 5)
  - More inferences = more robust but slower
  - Recommended: 5-10 for good balance
- **`rgb_noise_std`**: RGB noise standard deviation in 0-255 scale (default: 1.0)
  - Should be small to preserve image content
  - 1.0 ≈ ~0.4% noise on 0-255 scale (very conservative)
- **`depth_noise_std`**: Depth noise standard deviation in meters (default: 0.001 = 1mm)
  - Should match or be less than sensor noise characteristics
  - RealSense D455: ~2-5mm at 0.5m distance (we use 1mm to be conservative)
- **`inlier_threshold`**: Distance threshold for outlier rejection (default: 0.05 = 5cm)
  - Adjust based on object size and expected variation
- **`min_inliers`**: Minimum inliers required for success (default: 2)
  - Higher = more conservative (rejects more uncertain results)

### `estimate_pose_multi_frame()`

- **`observations`**: List of dicts with `'rgb'`, `'depth'`, `'K'` keys
- **`num_frames`**: How many frames to process (None = all)
- **`inlier_threshold`**: Distance threshold for outlier rejection (default: 0.05 = 5cm)
  - Combines translation distance (meters) + weighted rotation distance
  - Adjust based on object size and motion characteristics
- **`min_inliers`**: Minimum inliers required for success (default: 2)
  - If fewer inliers remain after filtering, returns failure
- **`est_refine_iter`**: Refinement iterations for Any6D initialization (default: 20)
- **`track_refine_iter`**: Refinement iterations per FoundationPose tracking step (default: 8)

### Worker Flags

**Common**:
- **`--inlier-threshold T`**: Distance threshold for outlier rejection (default: 0.05)
- **`--min-inliers M`**: Minimum inliers required (default: 2)

**Repeated Inference**:
- **`--repeated-inference`**: Enable repeated inference mode
- **`--num-inferences N`**: Number of inferences to run (default: 5)
- **`--rgb-noise-std S`**: RGB noise std dev (default: 2.0)
- **`--depth-noise-std S`**: Depth noise std dev in meters (default: 0.002)

**Multi-Frame**:
- **`--multi-frame`**: Enable multi-frame mode
- **`--num-frames N`**: Process up to N frames (default: 5)

## Output

### Repeated Inference Mode

The `PoseOutput.extras` includes:

```python
result.extras = {
    'num_inferences': 5,             # How many inferences were run
    'successful_inferences': 4,      # How many had valid pose estimates
    'failed_inferences': 1,          # How many failed
    'num_inliers': 3,                # Inliers after outlier rejection
    'num_outliers': 1,               # Outliers rejected
    'inlier_indices': [0, 1, 3],    # Which inferences were inliers
}
```

### Multi-Frame Mode

The `PoseOutput.extras` includes:

```python
result.extras = {
    'num_frames_processed': 5,      # How many frames were processed
    'successful_frames': 4,          # How many had valid pose estimates
    'failed_frames': 1,              # How many failed
    'num_inliers': 3,                # Inliers after outlier rejection
    'num_outliers': 1,               # Outliers rejected
    'inlier_indices': [0, 1, 3],    # Which frames were inliers
}
```

## Implementation Details

### Pose Distance Metric

Combines translation and rotation distances:
```
distance = ||t1 - t2|| + weight * arccos((trace(R1^T R2) - 1) / 2)
```

Where:
- `||t1 - t2||`: Euclidean translation distance (meters)
- `arccos(...)`: Geodesic rotation angle (radians)
- `weight`: Rotation weight (default: 0.1, meaning ~10cm per radian)

### Outlier Rejection

Uses consensus-based approach (similar to RANSAC):
1. Try each pose as a reference
2. Count how many other poses are within threshold
3. Select the reference with most inliers
4. Return all inliers for that consensus set

### Pose Averaging

- **Translation**: Simple arithmetic mean
- **Rotation**: Quaternion averaging with hemisphere alignment
  - Converts rotation matrices to quaternions
  - Aligns quaternions to same hemisphere
  - Averages quaternions
  - Normalizes and converts back to matrix

Requires `scipy` for quaternion operations. Falls back to using first rotation if `scipy` unavailable.

## Tuning Guidelines

### Inlier Threshold

- **Small objects** (< 5cm): Use 0.02 - 0.03
- **Medium objects** (5-20cm): Use 0.05 (default)
- **Large objects** (> 20cm): Use 0.10 or higher

### Number of Frames

- **Minimum**: 3 frames (allows meaningful outlier rejection)
- **Recommended**: 5-10 frames (good balance of robustness and speed)
- **Maximum**: Limited by memory and computation time

### Minimum Inliers

- **Conservative** (high confidence): 3-4 inliers
- **Balanced**: 2 inliers (default)
- **Aggressive** (accept more uncertainty): 1 inlier (essentially disables filtering)

## Debugging

Set `debug=1` in estimation calls to see per-frame results:
```python
result = estimate_pose_multi_frame(..., debug=1)
```

This prints:
- Frame-by-frame success/failure
- Error messages for failed frames
- Outlier rejection statistics

## Example: Continuous Tracking

See `run_demo_realsense.py` for a complete example of continuous tracking with RealSense camera.

The multi-frame API provides similar functionality in a reusable form.
