# Action Selection Logging Design

## Overview
This document describes how action propensity and approval status are logged for importance-weighted offline RL training.

## What We Log

### 1. Action Propensity
- **Definition**: P(a|s) - The probability of selecting action `a` given state `s`
- **Purpose**: Enable importance weighting for off-policy learning
- **Calculation**: Computed by `ActionSelectorManager.select_action()`
  - Random: `1/n` where n = number of candidate actions
  - Epsilon-greedy: `ε·(1/n) + (1-ε)·P_learned(a|s)`
  - Learned: Softmax probability from neural network
  - Resampled (undo old state): Conditional propensity accounting for already-executed actions

### 2. Approval Status
- **Definition**: Whether administrator approved/rejected the critical state
- **Purpose**: Filter training data, analyze failure modes, label success
- **Values**:
  - `"approved"` - Admin approved, robot proceeded
  - `"rejected"` - Admin rejected, triggered undo
  - `"none"` - Non-critical state (no approval needed)

## Storage Locations

### Location 1: Main Training Dataset (Parquet files)
**Path**: `{data_collection_policy_repo_id}/data/chunk-000/episode_XXXXXX.parquet`

Each frame includes:
```python
{
    # Standard LeRobot fields
    "observation.images.cam_main": torch.Tensor,
    "observation.images.cam_wrist": torch.Tensor,
    "observation.state": torch.Tensor,
    "action": torch.Tensor,  # Shape: (required_responses_per_critical_state * action_dim,)
    "task": str,
    
    # NEW: Action selection metadata
    "action_propensity": np.array([float], dtype=np.float32),  # P(a|s)
    "selector_mode": str,  # "random", "epsilon_greedy", or "learned"
    "approval_status": str,  # "approved", "rejected", or "none"
}
```

**Dataset Features**:
```python
dataset.features = {
    ...
    "action_propensity": {"dtype": "float32", "shape": (1,), "names": None},
    "selector_mode": {"dtype": "string", "shape": (1,), "names": None},
    "approval_status": {"dtype": "string", "shape": (1,), "names": None},
}
```

### Location 2: Detailed Propensity Log (JSONL)
**Path**: `{data_collection_policy_repo_id}/action_propensity_log.jsonl`

Each line contains rich metadata for offline analysis:
```json
{
    "episode_index": 0,
    "state_id": 42,
    "frame_index": 15,
    "timestamp": 0.5,
    "is_critical": true,
    "approval_status": "approved",
    "action_propensity": 0.085,
    "selector_mode": "epsilon_greedy",
    "epsilon": 0.1,
    "resampled": false,
    "num_candidate_actions": 10,
    "num_already_executed": 0
}
```

**Fields**:
- `episode_index`: Episode number in dataset
- `state_id`: Internal state ID (may not be sequential due to undo)
- `frame_index`: Sequential frame index within episode
- `timestamp`: Time in seconds from episode start
- `is_critical`: Whether this was a critical state requiring crowd responses
- `approval_status`: "approved", "rejected", or null
- `action_propensity`: P(a|s) for importance weighting
- `selector_mode`: Which selector was used
- `epsilon`: Epsilon value for epsilon-greedy (if applicable)
- `resampled`: True if action was resampled after undo
- `num_candidate_actions`: How many actions were available to choose from
- `num_already_executed`: How many actions had already been tried (for resampling)

## Data Flow

### 1. Action Selection (record_response)
```python
# StateManager.record_response()
selected_action, propensity, metadata = action_selector.select_action(actions, state_info)

state_info["action_propensity"] = propensity  # Store P(a|s)
state_info["action_selection_metadata"] = metadata  # Store selector details
```

### 2. Approval Recording
```python
# StateManager.approve_critical_state()
state_info["approval_status"] = "approved"

# StateManager.reject_critical_state()
state_info["approval_status"] = "rejected"
```

### 3. Dataset Writing
```python
# DatasetManager.save_episode()
frame = {
    **obs,
    "action": state["action_to_save"],
    "action_propensity": np.array([state.get("action_propensity", 1.0)]),
    "selector_mode": metadata.get("selector_used"),
    "approval_status": state.get("approval_status") or "none",
}
dataset.add_frame(frame)

# Also log to JSONL
log_entry = {...}  # Rich metadata
with open(propensity_log_path, "a") as f:
    f.write(json.dumps(log_entry) + "\n")
```

## Usage Examples

### Example 1: Importance-Weighted Training
```python
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import torch

dataset = LeRobotDataset("yilong/my_dataset")

for batch in dataset:
    action = batch["action"]
    propensity = batch["action_propensity"]
    
    # Compute importance weight
    policy_prob = policy.get_log_prob(batch["observation"], action)
    importance_weight = torch.exp(policy_prob) / propensity
    
    # Clip weights to reduce variance
    importance_weight = torch.clamp(importance_weight, 0, 10)
    
    # Weighted loss
    loss = importance_weight * criterion(predicted_action, action)
```

### Example 2: Filter by Approval Status
```python
# Only train on approved states
dataset = LeRobotDataset("yilong/my_dataset")
approved_mask = dataset.hf_dataset.filter(
    lambda x: x["approval_status"] == "approved"
)
```

### Example 3: Analyze Propensity Log
```python
import pandas as pd

# Load detailed log
log = pd.read_json("action_propensity_log.jsonl", lines=True)

# Analyze rejection rate by selector mode
rejection_rate = log.groupby("selector_mode")["approval_status"].apply(
    lambda x: (x == "rejected").sum() / len(x)
)

# Find low-propensity actions that were approved
successful_exploratory = log[
    (log["action_propensity"] < 0.1) & 
    (log["approval_status"] == "approved")
]
```

## Benefits

1. **Importance Weighting**: Correct for distribution shift when training from off-policy data
2. **Data Quality**: Filter out rejected states during training
3. **Analysis**: Understand which actions work, which selectors perform better
4. **Debugging**: Track down why certain actions were selected
5. **Curriculum Learning**: Train first on high-propensity (safe) actions, then add exploratory ones

## Notes

- Propensity is always stored, even for non-critical states (defaults to 1.0)
- Approval status is only set for critical states
- JSONL log is append-only and never deleted (useful for long-term analysis)
- Both logs are pushed to HuggingFace Hub with the dataset
