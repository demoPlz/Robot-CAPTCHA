#!/usr/bin/env python3
"""Test script for action selection system.

Verifies propensity calculations for random, learned, and epsilon-greedy modes.
"""

import sys
import os

# Add backend directory to path for imports
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, backend_dir)

import torch
from action_selector.random_selector import RandomSelector
from action_selector.learned_selector import LearnedSelector
from interface_managers.action_selector_manager import ActionSelectorManager


def create_test_actions(num_actions=5, action_dim=7):
    """Create dummy action tensors for testing."""
    return [torch.randn(action_dim) for _ in range(num_actions)]


def create_test_state():
    """Create dummy state info for testing."""
    return {
        "state_id": 0,
        "episode_id": 0,
        "joint_positions": {},
        # Add more fields as needed
    }


def test_random_selector():
    """Test random selector propensity."""
    print("\n" + "="*60)
    print("Testing Random Selector")
    print("="*60)
    
    selector = RandomSelector()
    actions = create_test_actions(num_actions=5)
    state_info = create_test_state()
    
    # Test multiple selections
    propensities = []
    for i in range(10):
        selected_action, propensity = selector.select(actions, state_info)
        propensities.append(propensity)
        assert abs(propensity - 0.2) < 1e-6, f"Expected propensity 0.2, got {propensity}"
    
    print(f"✓ Random selector correctly returns 1/n = {propensities[0]:.4f}")
    print(f"✓ Tested {len(propensities)} selections, all correct")


def test_learned_selector():
    """Test learned selector propensity."""
    print("\n" + "="*60)
    print("Testing Learned Selector")
    print("="*60)
    
    selector = LearnedSelector(model_path=None, device="cpu")
    actions = create_test_actions(num_actions=5)
    state_info = create_test_state()
    
    # Test multiple selections
    propensities = []
    for i in range(10):
        selected_action, propensity = selector.select(actions, state_info)
        propensities.append(propensity)
        # Propensity should be in (0, 1)
        assert 0 < propensity < 1, f"Propensity out of range: {propensity}"
    
    # Sum of all propensities should be ~1 (softmax)
    # We can't easily verify this without recomputing, but we can check plausibility
    avg_propensity = sum(propensities) / len(propensities)
    print(f"✓ Learned selector propensities in valid range (0, 1)")
    print(f"✓ Average propensity: {avg_propensity:.4f} (expected ~0.2 for uniform)")
    print(f"✓ Tested {len(propensities)} selections")


def test_epsilon_greedy_propensity():
    """Test epsilon-greedy propensity calculation."""
    print("\n" + "="*60)
    print("Testing Epsilon-Greedy Propensity")
    print("="*60)
    
    epsilon = 0.1
    manager = ActionSelectorManager(
        mode="epsilon_greedy",
        epsilon=epsilon,
        learned_model_path=None,
        device="cpu"
    )
    
    actions = create_test_actions(num_actions=5)
    state_info = create_test_state()
    n = len(actions)
    
    # Collect statistics over many trials
    num_trials = 1000
    propensities = []
    random_count = 0
    learned_count = 0
    
    for i in range(num_trials):
        selected_action, propensity, metadata = manager.select_action(actions, state_info)
        propensities.append(propensity)
        
        # Track which selector was used
        if metadata["selector_used"] == "random":
            random_count += 1
        else:
            learned_count += 1
        
        # Verify propensity is combination of both
        random_prop = metadata["random_propensity"]
        learned_prop = metadata["learned_propensity"]
        expected_prop = epsilon * random_prop + (1 - epsilon) * learned_prop
        
        assert abs(propensity - expected_prop) < 1e-6, \
            f"Propensity mismatch: {propensity} != {expected_prop}"
        
        # Random propensity should always be 1/n
        assert abs(random_prop - 1.0/n) < 1e-6, \
            f"Random propensity should be {1.0/n}, got {random_prop}"
    
    # Check that random selector was used ~epsilon fraction of the time
    random_fraction = random_count / num_trials
    print(f"✓ Propensity correctly combines random and learned components")
    print(f"✓ Random selector used: {random_count}/{num_trials} ({random_fraction:.1%})")
    print(f"  Expected: ~{epsilon:.1%}, Actual: {random_fraction:.1%}")
    print(f"✓ Average propensity: {sum(propensities)/len(propensities):.4f}")
    
    # Should be within reasonable margin of epsilon (allow for randomness)
    assert abs(random_fraction - epsilon) < 0.05, \
        f"Random fraction {random_fraction} too far from epsilon {epsilon}"


def test_action_selector_manager_modes():
    """Test all ActionSelectorManager modes."""
    print("\n" + "="*60)
    print("Testing ActionSelectorManager Modes")
    print("="*60)
    
    actions = create_test_actions(num_actions=5)
    state_info = create_test_state()
    
    # Test random mode
    manager = ActionSelectorManager(mode="random")
    selected, propensity, metadata = manager.select_action(actions, state_info)
    assert metadata["mode"] == "random"
    assert abs(propensity - 0.2) < 1e-6
    print("✓ Random mode works")
    
    # Test learned mode
    manager = ActionSelectorManager(mode="learned", learned_model_path=None)
    selected, propensity, metadata = manager.select_action(actions, state_info)
    assert metadata["mode"] == "learned"
    assert 0 < propensity < 1
    print("✓ Learned mode works")
    
    # Test epsilon-greedy mode
    manager = ActionSelectorManager(mode="epsilon_greedy", epsilon=0.2, learned_model_path=None)
    selected, propensity, metadata = manager.select_action(actions, state_info)
    assert metadata["mode"] == "epsilon_greedy"
    assert metadata["epsilon"] == 0.2
    assert 0 < propensity < 1
    print("✓ Epsilon-greedy mode works")
    
    # Test get_info
    info = manager.get_info()
    assert info["mode"] == "epsilon_greedy"
    assert info["epsilon"] == 0.2
    print("✓ Manager info retrieval works")


def test_propensity_coverage():
    """Test that propensities are reasonable across all actions."""
    print("\n" + "="*60)
    print("Testing Propensity Coverage")
    print("="*60)
    
    manager = ActionSelectorManager(mode="random")
    actions = create_test_actions(num_actions=5)
    state_info = create_test_state()
    
    # Each action should be selected roughly equally
    selection_counts = [0] * len(actions)
    num_trials = 1000
    
    for _ in range(num_trials):
        selected, propensity, metadata = manager.select_action(actions, state_info)
        # Find which action was selected
        for i, action in enumerate(actions):
            if torch.equal(action, selected):
                selection_counts[i] += 1
                break
    
    # Each action should be selected ~200 times (20% of 1000)
    expected_count = num_trials / len(actions)
    print(f"Selection counts: {selection_counts}")
    print(f"Expected per action: ~{expected_count:.0f}")
    
    for i, count in enumerate(selection_counts):
        # Allow 30% deviation due to randomness
        assert abs(count - expected_count) < expected_count * 0.3, \
            f"Action {i} selected {count} times, expected ~{expected_count}"
    
    print("✓ Actions selected uniformly as expected")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("Action Selection System Test Suite")
    print("="*60)
    
    try:
        test_random_selector()
        test_learned_selector()
        test_action_selector_manager_modes()
        test_epsilon_greedy_propensity()
        test_propensity_coverage()
        
        print("\n" + "="*60)
        print("✅ All tests passed!")
        print("="*60 + "\n")
    except Exception as e:
        print("\n" + "="*60)
        print(f"❌ Test failed: {e}")
        print("="*60 + "\n")
        raise
