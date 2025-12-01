# Action Selection System

This module implements an action selection system for crowdsourced robot teleoperation data collection. It supports multiple selection strategies with proper propensity logging for importance-weighted training.

## Overview

When multiple crowd workers submit different actions for the same robot state, we need to select which action to execute. This system provides three strategies:

1. **Random**: Uniform random selection (baseline)
2. **Epsilon-Greedy**: Mix of random exploration and learned exploitation
3. **Learned**: Pure learned selection using a neural network

All strategies properly log the **propensity** (selection probability) for each chosen action, enabling importance-weighted training to correct for selection bias.

## Architecture

```
action_selector/
├── selector.py              # Base interface (ActionSelector ABC)
├── random_selector.py       # Uniform random selection
├── learned_selector.py      # Neural network-based selection
└── README.md               # This file

interface_managers/
└── action_selector_manager.py  # Manages selector strategy and propensity
```

### Components

#### 1. `ActionSelector` (Base Interface)

All selectors implement this interface:

```python
class ActionSelector(ABC):
    @abstractmethod
    def select(self, actions: list[torch.Tensor], state_info: dict) -> tuple[torch.Tensor, float]:
        """
        Returns:
            (selected_action, propensity)
        """
        pass
```

#### 2. `RandomSelector`

Selects uniformly at random:
- Propensity: `1/n` where `n` is the number of actions

#### 3. `LearnedSelector`

Uses a neural network to score actions and samples via softmax:
- Network takes state features and candidate actions as input
- Outputs logits for each action
- Samples according to softmax probabilities
- Propensity: softmax probability of selected action

**Note**: The neural network architecture is currently a placeholder and needs to be designed based on:
- State observation format (images, joint positions, etc.)
- Action dimensions
- Number of candidate actions

#### 4. `ActionSelectorManager`

Manages the selection strategy for each session (program run):

```python
manager = ActionSelectorManager(
    mode="epsilon_greedy",  # or "random", "learned"
    epsilon=0.1,            # exploration rate
    learned_model_path="/path/to/model.pt",
    device="cpu"
)

selected_action, propensity, metadata = manager.select_action(actions, state_info)
```

## Propensity Calculation

Propensity is critical for unbiased learning from selective data.

### Random Mode
```
P(a|s) = 1/n
```

### Learned Mode
```
P(a|s) = softmax(f(s, a))_a
```
where `f` is the learned network.

### Epsilon-Greedy Mode
The full propensity combines both selectors:

```
P(a|s) = ε * P_random(a|s) + (1-ε) * P_learned(a|s)
       = ε * (1/n) + (1-ε) * softmax(f(s, a))_a
```

This is computed correctly regardless of which selector actually chose the action:
- If random selector chose it: we query learned selector for its probability
- If learned selector chose it: we know random would give it `1/n` probability

## Usage

### Configuration

Set action selection strategy via command-line arguments:

```bash
# Random selection (default)
python backend/collect_data.py --action-selector-mode random

# Epsilon-greedy with 10% exploration
python backend/collect_data.py \
    --action-selector-mode epsilon_greedy \
    --action-selector-epsilon 0.1 \
    --action-selector-model-path /path/to/model.pt

# Pure learned selection
python backend/collect_data.py \
    --action-selector-mode learned \
    --action-selector-model-path /path/to/model.pt
```

### Accessing Propensity in Training

Propensities are logged in the state metadata:

```python
state_info = {
    "actions": [...],
    "action_propensity": 0.0856,  # P(selected_action | state)
    "action_selection_metadata": {
        "mode": "epsilon_greedy",
        "epsilon": 0.1,
        "selector_used": "learned",  # which selector actually chose
        "num_actions": 5,
        "random_propensity": 0.2,
        "learned_propensity": 0.0695,
        "propensity": 0.0856,  # ε * 0.2 + (1-ε) * 0.0695
    }
}
```

Use propensity for importance weighting:

```python
# Inverse propensity weighting
weight = 1.0 / state_info["action_propensity"]
loss = weight * compute_loss(predicted, actual)
```

## Training the Learned Selector

The learned selector requires training data from previous collections. Training workflow:

1. **Collect initial data** using random selection
2. **Train selector** to predict which actions lead to successful episodes
3. **Deploy learned model** using epsilon-greedy for exploration
4. **Iterate**: Retrain periodically with new data

### Training Data Format

Each training example should contain:
- State observations (images, joint positions, etc.)
- Candidate actions (all submissions)
- Selected action (ground truth)
- Episode outcome (success/failure)

### Model Architecture (TODO)

The placeholder in `learned_selector.py` needs to be replaced with:

```python
class ActionSelectorModel(nn.Module):
    def __init__(self, obs_dim, action_dim, num_candidates):
        super().__init__()
        # TODO: Define architecture
        # - Encode observations (e.g., CNN for images)
        # - Encode each candidate action
        # - Combine state and action features
        # - Output logit for each candidate
        pass
    
    def forward(self, state_features, candidate_actions):
        # Returns: [num_candidates] logits
        pass
```

Recommended approach:
- Vision encoder: ResNet/EfficientNet for camera images
- Action encoder: MLP for action vectors
- Fusion: Attention or cross-product between state and actions
- Output: Scalar score per action → softmax → probabilities

## Implementation Details

### Thread Safety

Action selection happens within `StateManager.record_response()`, which is protected by `state_lock`. The action selector itself doesn't need additional locking.

### Performance

- Random selector: O(1)
- Learned selector: O(n) where n is number of actions (typically small, ~5-10)
- Epsilon-greedy: O(n) (needs learned propensities for all actions)

### Extensibility

To add a new selector:

1. Create a new class inheriting from `ActionSelector`
2. Implement `select()` method with proper propensity calculation
3. Add it to `ActionSelectorManager` mode options
4. Update CLI arguments in `crowd_interface_config.py`

## Future Enhancements

- [ ] Implement proper neural network architecture for learned selector
- [ ] Add training script for learned selector
- [ ] Support different exploration strategies (UCB, Thompson sampling)
- [ ] Add selector performance metrics/logging
- [ ] Support online learning (update selector during collection)
- [ ] GPU support for learned selector inference
- [ ] Batch processing for multiple states
- [ ] Contextual bandits approach with streaming updates

## References

- Importance weighting: [Doubly Robust Off-Policy Evaluation](https://arxiv.org/abs/1103.4601)
- Epsilon-greedy: Classic RL exploration strategy
- Propensity scoring: [Causal inference in statistics](https://www.hsph.harvard.edu/miguel-hernan/causal-inference-book/)
