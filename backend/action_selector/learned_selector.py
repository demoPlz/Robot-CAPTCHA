"""Learned action selector using neural network with softmax.

Uses a neural network to score actions and selects via softmax sampling.
The network architecture and training are defined separately.
"""

import random
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .selector import ActionSelector


class LearnedSelector(ActionSelector):
    """Selects actions using learned preferences from a neural network.
    
    The network takes state information and candidate actions as input,
    outputs logits for each action, applies softmax, and samples accordingly.
    
    Propensity is the softmax probability of the selected action.
    """

    def __init__(self, model_path: str | None = None, device: str = "cpu"):
        """Initialize learned selector.
        
        Args:
            model_path: Path to saved model weights. If None, uses random initialization.
            device: Device to run model on ('cpu' or 'cuda')
        """
        self.device = device
        self.model = self._create_model()
        
        if model_path is not None and Path(model_path).exists():
            self.load_model(model_path)
        
        self.model.to(self.device)
        self.model.eval()  # Always in eval mode for inference

    def _create_model(self) -> nn.Module:
        """Create the neural network architecture.
        
        TODO: Define proper architecture based on:
        - State observations (images, joint positions, etc.)
        - Action dimensions
        - Number of candidate actions to score
        
        For now, returns a placeholder that will be properly designed later.
        """
        # Placeholder - will be replaced with actual architecture
        class PlaceholderModel(nn.Module):
            def __init__(self):
                super().__init__()
                # This is a placeholder that accepts arbitrary input
                # Real implementation will process observations and actions
                self.fc = nn.Linear(1, 1)
            
            def forward(self, state_features, action_features):
                # Returns random logits for now
                num_actions = action_features.size(0)
                return torch.randn(num_actions)
        
        return PlaceholderModel()

    def load_model(self, model_path: str):
        """Load model weights from disk.
        
        Args:
            model_path: Path to model checkpoint
        """
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint)
        print(f"✓ Loaded learned selector model from {model_path}")

    def save_model(self, model_path: str):
        """Save model weights to disk.
        
        Args:
            model_path: Path to save checkpoint
        """
        torch.save(self.model.state_dict(), model_path)
        print(f"✓ Saved learned selector model to {model_path}")

    def select(self, actions: list[torch.Tensor], state_info: dict[str, Any]) -> tuple[torch.Tensor, float]:
        """Select an action using learned network and softmax sampling.
        
        Args:
            actions: List of action tensors from crowd workers
            state_info: State information including observations
        
        Returns:
            Tuple of (selected_action, propensity)
        """
        with torch.no_grad():
            # Extract features from state
            # TODO: Properly process state_info to extract relevant features
            # For now, use placeholder
            state_features = torch.zeros(1).to(self.device)
            
            # Stack actions for batch processing
            action_tensor = torch.stack(actions).to(self.device)
            
            # Get logits from model
            logits = self.model(state_features, action_tensor)
            
            # Compute softmax probabilities
            probs = F.softmax(logits, dim=0)
            
            # Sample according to probabilities
            # Use torch.multinomial for proper sampling
            selected_idx = torch.multinomial(probs, num_samples=1).item()
            
            selected_action = actions[selected_idx]
            propensity = probs[selected_idx].item()
            
            return selected_action, propensity

    def get_name(self) -> str:
        return "learned"
