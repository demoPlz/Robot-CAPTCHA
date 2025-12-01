"""Random action selector using uniform distribution.

Selects actions uniformly at random from all submissions, providing a baseline for comparison with learned selectors.

"""

import random
from typing import Any

import torch

from .selector import ActionSelector


class RandomSelector(ActionSelector):
    """Selects actions uniformly at random.

    Propensity is always 1/n where n is the number of actions.

    """

    def select(self, actions: list[torch.Tensor], state_info: dict[str, Any]) -> tuple[torch.Tensor, float]:
        """Select an action uniformly at random.

        Args:
            actions: List of action tensors
            state_info: State information (unused for random selection)

        Returns:
            Tuple of (selected_action, propensity)

        """
        n = len(actions)
        selected_idx = random.randint(0, n - 1)
        propensity = 1.0 / n

        return actions[selected_idx], propensity

    def get_name(self) -> str:
        return "random"
