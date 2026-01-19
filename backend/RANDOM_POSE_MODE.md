# Random Pose Mode

## Overview

The random pose mode allows you to skip real-time 6D pose estimation and instead use fixed random poses for objects. This is useful for:

- **Testing and development**: Quickly test the system without running heavy pose estimation workers
- **Simulation scenarios**: When object poses don't need to match reality
- **Performance testing**: Reduce computational overhead during system testing

## How It Works

When enabled, the system:

1. Generates one random pose per tracked object at initialization
2. Applies the same pose to each object across all states
3. Skips spawning pose estimation worker processes
4. Bypasses the disk-based job queue system

Each random pose consists of:
- **Position**: Random (x, y, z) coordinates within configurable bounds
- **Orientation**: Random quaternion (uniformly sampled rotation)

## Configuration

### In Code

Edit `crowd_interface_config.py`:

```python
# Enable random pose mode
self.use_random_poses = True

# Configure pose bounds (in meters)
self.random_pose_bounds = {
    "x_min": -0.3, "x_max": 0.3,
    "y_min": -0.3, "y_max": 0.3,
    "z_min": 0.0, "z_max": 0.3
}
```

### Via CLI

When running your data collection script:

```bash
python backend/collect_data.py --use-random-poses
```

## Example Output

When random pose mode is enabled, you'll see:

```
🎲 Random pose mode enabled - skipping pose estimation workers
🎲 Generating random fixed poses for 1 objects...
   Cube_Red: pos=[0.123, -0.045, 0.156], rot=[0.123, 0.456, 0.789, 0.234]
✅ Random fixed poses generated
```

During state processing:

```
🎲 Applied random fixed poses to state (ep=1, state=5)
```

## Implementation Details

### Modified Files

1. **`crowd_interface_config.py`**: Added configuration options
   - `use_random_poses`: Boolean flag to enable/disable the feature
   - `random_pose_bounds`: Dictionary defining position bounds
   - CLI argument `--use-random-poses`

2. **`crowd_interface.py`**: Updated to pass configuration to PoseEstimationManager
   - Added parameters to `__init__`
   - Passes parameters to PoseEstimationManager

3. **`pose_estimation_manager.py`**: Core implementation
   - Skips worker initialization when in random mode
   - `_generate_random_fixed_poses()`: Generates poses at startup
   - `_apply_random_poses_to_state()`: Applies poses to each state
   - `enqueue_pose_jobs_for_state()`: Fast-path for random poses

### Pose Format

Poses are stored in the same format as real pose estimation:

```python
{
    "pos": [x, y, z],        # Position in meters
    "rot": [x, y, z, w]      # Quaternion rotation
}
```

This ensures compatibility with the rest of the system (simulation, dataset saving, etc.).

## Limitations

- **Same pose for all states**: Each object uses the same random pose throughout data collection
- **No tracking**: Object positions don't update based on real-world movement
- **Not suitable for**: Real robot deployment or scenarios requiring accurate pose tracking

## Use Cases

✅ **Good for:**
- System testing and debugging
- Performance benchmarking
- Simulation-only workflows
- UI/UX testing

❌ **Not suitable for:**
- Real robot data collection
- Scenarios requiring accurate object localization
- Multi-object manipulation tasks requiring relative positioning
