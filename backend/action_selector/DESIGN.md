# Action Selection System - Design Overview

## Summary

I've designed and implemented a complete action selection system for your crowdsourced robot teleoperation platform. The system supports:

1. **Random selection** - Uniform baseline
2. **Epsilon-greedy** - Mix of random exploration and learned exploitation  
3. **Learned selection** - Pure neural network-based selection

All modes properly log **propensity** (selection probability) for importance-weighted training.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    CrowdInterface                            │
│  - Initializes ActionSelectorManager with CLI config        │
│  - Passes to StateManager                                   │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│               ActionSelectorManager                          │
│  - Manages selector strategy (random/epsilon/learned)       │
│  - Computes propensities for all modes                      │
│  - Handles epsilon-greedy mixing logic                      │
└─────────────────────────────────────────────────────────────┘
                            │
              ┌─────────────┴─────────────┐
              ▼                           ▼
    ┌──────────────────┐        ┌──────────────────┐
    │ RandomSelector   │        │ LearnedSelector  │
    │ - Uniform random │        │ - Neural network │
    │ - P(a) = 1/n     │        │ - Softmax sample │
    └──────────────────┘        └──────────────────┘
```

## Key Components

### 1. Base Interface (`selector.py`)

```python
class ActionSelector(ABC):
    def select(self, actions, state_info) -> (action, propensity)
```

All selectors must return both the selected action and its selection probability.

### 2. Random Selector (`random_selector.py`)

- Selects uniformly at random
- Propensity: `1/n` where n = number of actions
- Simple baseline for comparison

### 3. Learned Selector (`learned_selector.py`)

- Uses neural network to score actions
- Applies softmax and samples according to probabilities
- Propensity: softmax probability of selected action
- **Architecture is placeholder** - needs to be designed based on your obs/action format

### 4. Action Selector Manager (`action_selector_manager.py`)

Central component that:
- Initializes the appropriate selector(s) based on mode
- Handles epsilon-greedy mixing logic
- Computes correct propensities for all modes
- Returns selection metadata for logging

**Epsilon-greedy propensity calculation:**
```
P(a|s) = ε · P_random(a|s) + (1-ε) · P_learned(a|s)
       = ε · (1/n) + (1-ε) · softmax(network(s,a))_a
```

This is computed correctly regardless of which selector actually chose the action.

## Integration Points

### Configuration (`crowd_interface_config.py`)

Added three new parameters:
```python
--action-selector-mode {random,epsilon_greedy,learned}
--action-selector-epsilon 0.1  # for epsilon_greedy mode
--action-selector-model-path /path/to/model.pt
```

### State Manager (`state_manager.py`)

Modified `record_response()` to use action selector instead of `random.choice()`:

```python
# Old:
a_execute_index = random.randint(0, required_responses - 1)

# New:
selected_action, propensity, metadata = self.action_selector.select_action(
    state_info["actions"][:required_responses], 
    state_info
)
```

Stores selection metadata in `state_info`:
- `action_propensity`: P(selected_action | state)
- `action_selection_metadata`: Full details (mode, epsilon, selector used, etc.)

## Usage Examples

### Random (Default)
```bash
python backend/collect_data.py \
    --action-selector-mode random \
    # ... other args
```

### Epsilon-Greedy (10% exploration)
```bash
python backend/collect_data.py \
    --action-selector-mode epsilon_greedy \
    --action-selector-epsilon 0.1 \
    --action-selector-model-path /path/to/trained_model.pt \
    # ... other args
```

### Pure Learned
```bash
python backend/collect_data.py \
    --action-selector-mode learned \
    --action-selector-model-path /path/to/trained_model.pt \
    # ... other args
```

## Propensity Logging

After each critical state completion, the following is logged in `state_info`:

```python
{
    "action_propensity": 0.0856,  # For importance weighting
    "action_selection_metadata": {
        "mode": "epsilon_greedy",
        "epsilon": 0.1,
        "selector_used": "learned",  # Which selector actually chose
        "num_actions": 5,
        "random_propensity": 0.2,     # 1/5
        "learned_propensity": 0.0695,  # From softmax
        "propensity": 0.0856,          # 0.1*0.2 + 0.9*0.0695
    }
}
```

### Using Propensity in Training

For importance-weighted learning:
```python
# Inverse propensity weighting
weight = 1.0 / state_info["action_propensity"]
weighted_loss = weight * loss
```

## Testing

Comprehensive test suite in `test_action_selection.py`:
- ✅ Random selector propensity correctness
- ✅ Learned selector propensity validity
- ✅ Epsilon-greedy propensity calculation
- ✅ Mode switching works correctly
- ✅ Uniform coverage for random selector

Run tests:
```bash
cd backend && conda run -n csui python action_selector/test_action_selection.py
```

## Next Steps

### 1. Define Neural Network Architecture

The learned selector currently uses a placeholder model. You need to design:

```python
class ActionSelectorModel(nn.Module):
    def __init__(self):
        # Vision encoder for observations (e.g., ResNet)
        # Action encoder (e.g., MLP)
        # Fusion mechanism
        # Output: scalar score per action
        
    def forward(self, state_obs, candidate_actions):
        # Returns: [num_candidates] logits
```

Recommended approach:
- Extract visual features from observations
- Encode each candidate action
- Compute compatibility score between state and each action
- Apply softmax for probabilities

### 2. Training Pipeline

To train the learned selector:

1. **Collect data** using random mode
2. **Label episodes** by success/failure
3. **Train selector** to predict which actions led to success
4. **Deploy** using epsilon-greedy for continued exploration
5. **Iterate** with new data

### 3. Model Checkpointing

Add to your training script:
```python
selector = LearnedSelector()
# ... training ...
selector.save_model("selector_checkpoint.pt")
```

Then use in data collection:
```bash
--action-selector-model-path selector_checkpoint.pt
```

## Files Changed/Created

### New Files
- `backend/action_selector/selector.py` - Base interface
- `backend/action_selector/random_selector.py` - Random implementation
- `backend/action_selector/learned_selector.py` - Learned implementation  
- `backend/interface_managers/action_selector_manager.py` - Manager
- `backend/action_selector/test_action_selection.py` - Tests
- `backend/action_selector/README.md` - Documentation

### Modified Files
- `backend/crowd_interface_config.py` - Added CLI args
- `backend/crowd_interface.py` - Initialize manager
- `backend/interface_managers/state_manager.py` - Use manager for selection

## Design Decisions

### Why ActionSelectorManager?

Separates concerns:
- Individual selectors are simple (just implement selection logic)
- Manager handles mode switching and propensity calculation complexity
- Easy to add new selector types

### Why Session-Level Configuration?

Your requirement: "for EACH SESSION (every time this program is ran), the user can choose..."

This is cleaner than:
- Per-episode switching (would complicate propensity logging)
- Dynamic switching (would require handling model loading mid-session)

### Propensity Calculation for Epsilon-Greedy

The key insight: regardless of which selector actually chose the action, the propensity must account for BOTH selectors' probabilities:

```
P(a|s) = P(choose random) · P_random(a|s) + P(choose learned) · P_learned(a|s)
       = ε · (1/n) + (1-ε) · P_learned(a|s)
```

This ensures correct importance weights for unbiased learning.

## Verification

All tests pass:
```
✅ Random selector: 1/n propensity ✓
✅ Learned selector: valid softmax probabilities ✓  
✅ Epsilon-greedy: correct mixing ✓
✅ Mode switching: all modes work ✓
✅ Uniform coverage: actions selected fairly ✓
```

System is ready for use!
