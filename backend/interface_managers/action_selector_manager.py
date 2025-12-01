"""Action Selector Manager.

Manages action selection strategy for the crowd interface. Supports:
- Random selection (uniform)
- Epsilon-greedy (mix of random and learned)
- Pure learned selection (epsilon=0)

Handles propensity logging for importance-weighted training.

"""

import random
import sys
from pathlib import Path
from typing import Any, Literal

import torch

# Add backend to path to support both absolute and relative imports
backend_path = Path(__file__).parent.parent
if str(backend_path) not in sys.path:
    sys.path.insert(0, str(backend_path))

from action_selector.learned_selector import LearnedSelector
from action_selector.random_selector import RandomSelector
from action_selector.selector import ActionSelector


class ActionSelectorManager:
    """Manages action selection strategy and propensity logging.

    Supports three modes:
    1. "random": Pure uniform random selection
    2. "epsilon_greedy": Mix of random (epsilon) and learned (1-epsilon)
    3. "learned": Pure learned selection (epsilon=0)

    For epsilon-greedy, propensity is computed as:
        P(a|s) = epsilon * (1/n) + (1-epsilon) * P_learned(a|s)
    where n is the number of actions.

    """

    def __init__(
        self,
        mode: Literal["random", "epsilon_greedy", "learned"] = "random",
        epsilon: float = 0.1,
        learned_model_path: str | None = None,
        device: str = "cpu",
    ):
        """Initialize action selector manager.

        Args:
            mode: Selection strategy ('random', 'epsilon_greedy', or 'learned')
            epsilon: Exploration rate for epsilon-greedy (0.0 to 1.0)
            learned_model_path: Path to learned model weights (required for epsilon_greedy/learned)
            device: Device for learned model ('cpu' or 'cuda')

        """
        self.mode = mode
        self.epsilon = epsilon
        self.device = device

        # Validate epsilon
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError(f"Epsilon must be in [0, 1], got {epsilon}")

        # Initialize selectors based on mode
        self.random_selector = RandomSelector()
        self.learned_selector = None

        if mode in ["epsilon_greedy", "learned"]:
            self.learned_selector = LearnedSelector(model_path=learned_model_path, device=device)
            print(f"✓ Initialized learned selector (mode={mode}, epsilon={epsilon})")
        elif mode == "random":
            print(f"✓ Using random action selection")
        else:
            raise ValueError(f"Unknown mode: {mode}. Must be 'random', 'epsilon_greedy', or 'learned'")

    def select_action(
        self, actions: list[torch.Tensor], state_info: dict[str, Any]
    ) -> tuple[torch.Tensor, float, dict[str, Any]]:
        """Select an action and compute its propensity.

        Args:
            actions: List of action tensors from crowd workers
            state_info: State information (observations, metadata, etc.)

        Returns:
            Tuple of:
                - selected_action: The chosen action tensor
                - propensity: P(selected_action | state)
                - metadata: Additional info about selection (selector used, epsilon, etc.)

        """
        if len(actions) == 0:
            raise ValueError("Cannot select from empty action list")

        metadata = {
            "mode": self.mode,
            "epsilon": self.epsilon,
            "num_actions": len(actions),
        }

        if self.mode == "random":
            # Pure random selection
            selected_action, propensity = self.random_selector.select(actions, state_info)
            metadata["selector_used"] = "random"

        elif self.mode == "learned":
            # Pure learned selection
            selected_action, propensity = self.learned_selector.select(actions, state_info)
            metadata["selector_used"] = "learned"

        elif self.mode == "epsilon_greedy":
            # Epsilon-greedy: random with prob epsilon, learned with prob (1-epsilon)
            use_random = random.random() < self.epsilon

            if use_random:
                # Random exploration
                selected_action, random_propensity = self.random_selector.select(actions, state_info)
                # Get learned propensity for the selected action to compute full propensity
                learned_action, learned_propensity = self._get_learned_propensity_for_action(
                    actions, state_info, selected_action
                )
                metadata["selector_used"] = "random"
            else:
                # Learned exploitation
                selected_action, learned_propensity = self.learned_selector.select(actions, state_info)
                random_propensity = 1.0 / len(actions)
                metadata["selector_used"] = "learned"

            # Combined propensity: P(a|s) = epsilon * P_random(a|s) + (1-epsilon) * P_learned(a|s)
            propensity = self.epsilon * random_propensity + (1.0 - self.epsilon) * learned_propensity
            metadata["random_propensity"] = random_propensity
            metadata["learned_propensity"] = learned_propensity

        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        metadata["propensity"] = propensity
        return selected_action, propensity, metadata

    def _get_learned_propensity_for_action(
        self, actions: list[torch.Tensor], state_info: dict[str, Any], target_action: torch.Tensor
    ) -> tuple[torch.Tensor, float]:
        """Get learned selector's propensity for a specific action.

        Used in epsilon-greedy when random selector chooses an action,
        but we need learned propensity for the full propensity calculation.

        Args:
            actions: All available actions
            state_info: State information
            target_action: The action to get propensity for

        Returns:
            Tuple of (target_action, its learned propensity)

        """
        with torch.no_grad():
            # Get all learned propensities
            state_features = torch.zeros(1).to(self.device)  # Placeholder
            action_tensor = torch.stack(actions).to(self.device)
            logits = self.learned_selector.model(state_features, action_tensor)
            probs = torch.nn.functional.softmax(logits, dim=0)

            # Find the target action's index
            for idx, action in enumerate(actions):
                if torch.equal(action, target_action):
                    return target_action, probs[idx].item()

            # Shouldn't happen, but fallback
            return target_action, 1.0 / len(actions)

    def get_info(self) -> dict[str, Any]:
        """Get current selector configuration info."""
        return {
            "mode": self.mode,
            "epsilon": self.epsilon,
            "device": self.device,
            "has_learned_model": self.learned_selector is not None,
        }
