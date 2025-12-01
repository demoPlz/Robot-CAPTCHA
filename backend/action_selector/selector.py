"""Base interface for action selectors.

Defines the contract that all action selectors must implement for selecting actions from a set of crowdsourced
submissions.

"""

from abc import ABC, abstractmethod
from typing import Any

import torch


class ActionSelector(ABC):
    """Base class for action selection strategies.

    All selectors must implement select() which returns both the selected action and its propensity (probability of
    selection) for importance weighting in training.

    """

    @abstractmethod
    def select(self, actions: list[torch.Tensor], state_info: dict[str, Any]) -> tuple[torch.Tensor, float]:
        """Select an action from the list of crowdsourced submissions.

        Args:
            actions: List of action tensors submitted by crowd workers
            state_info: Dictionary containing state information (observations, metadata, etc.)
                       Can be used by learned selectors for context-aware selection

        Returns:
            Tuple of:
                - selected_action: The chosen action tensor
                - propensity: Probability with which this action was selected (for importance weighting)
                             For uniform random: 1/len(actions)
                             For learned: softmax probability
                             For epsilon-greedy: epsilon/len(actions) + (1-epsilon) * learned_prob

        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Return a human-readable name for this selector."""
        pass
