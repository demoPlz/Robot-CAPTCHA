# Crowdsourced Dataset Structure

This document describes the structure of datasets saved by the crowdsourcing interface.

## Overview

The dataset stores robot trajectories collected through crowdsourcing, where multiple users submit actions for each critical state. The dataset preserves both the ground truth (what was actually executed) and the crowd diversity (all submitted actions with their approval status).

## Dataset Format

- **Format**: LeRobot HDF5/Parquet dataset
- **Episodes**: Each episode is a complete trajectory from start to end
- **Frames**: Each frame represents one robot state (observation + action + metadata)

## Frame Structure

Each frame in the dataset contains the following fields:

### Core Fields

| Field | Type | Description |
|-------|------|-------------|
| `observation.*` | Various | Robot observations (camera images, joint positions, etc.) |
| `action` | `float32[N*7]` | Concatenated actions from all submissions (N = max submissions per state) |
| `task` | `string` | Task description text |
| `index` | `int64` | Global frame index across all episodes |
| `episode_index` | `int64` | Episode number |
| `frame_index` | `int64` | Frame number within episode |
| `timestamp` | `float32` | Timestamp in seconds |

### Crowd Action Fields

| Field | Type | Description |
|-------|------|-------------|
| `executed_actions` | `float32[N*7]` | Executed actions (padded with NaN) |
| `executed_propensities` | `float32[N]` | Propensity score for each action |
| `executed_approvals` | `float32[N]` | Approval status: 1.0 (approved), -1.0 (rejected), NaN (not executed) |
| `final_executed_action` | `float32[7]` | The single action that was physically executed by the robot |

Where `N` is the maximum number of submissions across all states in the episode (dynamically sized).

## Execution History Structure

Each critical state has an `execution_history` list containing all action submissions:

```python
execution_history = [
    {
        "action": torch.Tensor[7],           # The submitted action (7 joints)
        "propensity": float,                 # P(selecting this action | state)
        "approval": int,                     # 1 = approved, -1 = rejected, None = pending
        "submitted_by": [                    # List of users who submitted this action
            {
                "email": str,
                "name": str,
                "action_index": int          # Which action slot (0-indexed)
            }
        ],
        "executed": bool,                    # True if physically executed (sync mode only)
        "currently_executing": bool,         # True if being executed now (sync mode only)
        "post_execution_approved": bool      # Post-execution approval (sync mode only)
    },
    # ... more submissions
]
```

## Action Shape

The action dimensionality is **dynamic per episode** based on the maximum number of submissions:

- **Single action**: 7 floats (6 arm joints + 1 gripper joint)
- **Concatenated actions**: `N × 7` floats, where `N` = max submissions in any state
- **Padding**: Unused slots filled with `NaN`

Example for max 3 submissions:
```
action = [j0, j1, j2, j3, j4, j5, grip,  # Submission 1
          j0, j1, j2, j3, j4, j5, grip,  # Submission 2
          j0, j1, j2, j3, j4, j5, grip]  # Submission 3
```

## Propensities

Propensities represent importance weights for learning from crowd data:

### Definition
```
propensity = count(action appears) / total_actual_submissions
```

### Purpose
- Correct for selection bias when action selector chooses actions non-uniformly
- Enable unbiased learning from crowd data using importance weighting
- Account for duplicate submissions of the same action

### Example
If state receives 5 submissions: [A, B, A, C, A]
- Action A appears 3 times → propensity = 3/5 = 0.6
- Action B appears 1 time → propensity = 1/5 = 0.2  
- Action C appears 1 time → propensity = 1/5 = 0.2

### Admin Action
The first admin action always has propensity = 1.0 (always selected)

## Approval Status

Each action has an approval status:

| Value | Meaning | Description |
|-------|---------|-------------|
| `1` | Approved | Action was reviewed and approved |
| `-1` | Rejected | Action was reviewed and rejected |
| `None` / `NaN` | Pending | Not yet reviewed (should not appear in saved dataset) |

## Final Executed Action

The `final_executed_action` field stores the **ground truth** - the single action that was physically executed by the robot to reach this state:

- **Async mode**: Always the admin's action (first in execution_history)
- **Sync mode**: The action that was approved post-execution
- **Non-critical states**: `NaN` (auto-labeled, no execution happened at that state)

## Mode Differences

### Synchronous Mode
1. Admin submits action → immediate pre-approval → robot executes → post-execution approval
2. Multiple executions may occur before approval
3. `executed` and `post_execution_approved` flags are used
4. `final_executed_action` = the post-execution approved action

### Asynchronous Mode  
1. Admin submits → auto-approved → robot executes immediately
2. Users label later (async) with per-action review
3. Only pre-execution approval is used (`approval` field)
4. `final_executed_action` = admin's action (first entry)

## Non-Critical States

States that are auto-labeled (non-critical) have:
- `executed_actions`: NaN-filled array
- `executed_propensities`: NaN-filled array
- `executed_approvals`: NaN-filled array
- `final_executed_action`: NaN-filled array
- No `execution_history`

## Data Usage

### For Behavior Cloning
```python
# Use final_executed_action for standard BC
final_action = frame["final_executed_action"]  # Shape: (7,)
```

### For Crowd-Aware Learning
```python
# Use all approved actions with importance weighting
actions = frame["executed_actions"]              # Shape: (N*7,)
propensities = frame["executed_propensities"]    # Shape: (N,)
approvals = frame["executed_approvals"]          # Shape: (N,)

# Filter to approved actions
approved_mask = (approvals == 1.0)
approved_actions = actions.reshape(N, 7)[approved_mask]
approved_weights = propensities[approved_mask]
```

### For Learning from Failures
```python
# Include rejected actions as negative examples
rejections = frame["executed_approvals"] == -1.0
rejected_actions = actions.reshape(N, 7)[rejections]
```

## Episode Timing Metadata

Saved in episode info:
- `episode_start_time`: Unix timestamp
- `episode_end_time`: Unix timestamp  
- `total_episode_duration_seconds`: Duration
- `per_user_stats`: Per-user timing and counts
- `per_state_stats`: Per-state timing and user counts

## File Locations

- **Dataset root**: `~/.cache/huggingface/lerobot/{repo_id}/`
- **Metadata**: `meta/info.json`
- **Episodes**: `data/chunk-{xxx}/` (Parquet format)
- **Videos**: `videos/chunk-{xxx}/` (if enabled)
