"""Dataset Manager Module.

Manages LeRobot dataset operations for the crowd interface. Handles dataset initialization, episode saving, and
observation loading/cleanup.

"""

import json
import os
import time
from pathlib import Path

import datasets
import numpy as np
import torch
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.robot_devices.control_utils import (
    sanity_check_dataset_name,
    sanity_check_dataset_robot_compatibility,
)


class DatasetManager:
    """Manages LeRobot dataset operations.

    Responsibilities:
    - Dataset initialization (create or resume)
    - Episode saving (frames and episodes)
    - Observation loading from disk cache
    - Observation cleanup after saving
    - Dataset action shape updates for crowd responses

    Attributes:
        dataset: LeRobotDataset instance
        task_text: Task text used for dataset frames
        required_responses_per_critical_state: Number of responses per critical state (for action shape)
        obs_cache_root: Root directory for observation cache

    """

    def __init__(
        self,
        required_responses_per_critical_state: int,
        obs_cache_root: Path,
    ):
        """Initialize dataset manager.

        Args:
            required_responses_per_critical_state: Number of responses per critical state (for action shape)
            obs_cache_root: Root directory for observation cache

        """
        self.required_responses_per_critical_state = required_responses_per_critical_state
        self._obs_cache_root = obs_cache_root

        # Dataset state
        self.dataset = None
        self.task_text = None

    # =========================
    # Dataset Initialization
    # =========================

    def init_dataset(self, cfg, robot):
        """Initialize dataset for data collection policy training."""
        if cfg.resume:
            self.dataset = LeRobotDataset(cfg.data_collection_policy_repo_id, root=cfg.root)
            self.dataset.start_image_writer(
                num_processes=cfg.num_image_writer_processes,
                num_threads=cfg.num_image_writer_threads_per_camera * len(robot.cameras),
            )
            sanity_check_dataset_robot_compatibility(self.dataset, robot, cfg.fps, cfg.video)

        else:
            sanity_check_dataset_name(cfg.data_collection_policy_repo_id, cfg.policy)
            self.dataset = LeRobotDataset.create(
                cfg.data_collection_policy_repo_id,
                cfg.fps,
                root=cfg.root,
                robot=robot,
                use_videos=cfg.video,
                image_writer_processes=cfg.num_image_writer_processes,
                image_writer_threads=cfg.num_image_writer_threads_per_camera * len(robot.cameras),
            )

        # For UI fallback and dataset writes, always use cfg.single_task
        self.task_text = getattr(cfg, "single_task", None)

        # Update dataset action shape to accommodate crowd responses
        self._update_dataset_action_shape()

        return self.task_text

    def _update_dataset_action_shape(self):
        """Update the dataset's action feature shape to include crowd responses dimension."""
        if self.dataset is not None and "action" in self.dataset.features:
            from datasets import Features, Sequence, Value
            from lerobot.common.datasets.utils import get_hf_features_from_features

            original_action_dim = self.dataset.features["action"]["shape"][-1]  # Get the last dimension (joint count)
            new_action_shape = (self.required_responses_per_critical_state * original_action_dim,)

            # Update both the dataset features and metadata
            self.dataset.features["action"]["shape"] = new_action_shape
            self.dataset.meta.features["action"]["shape"] = new_action_shape

            # Add fields for executed actions with their propensities and approvals
            # All three arrays have matching indices:
            # executed_actions[i] has executed_propensities[i] and executed_approvals[i]
            if "executed_actions" not in self.dataset.features:
                self.dataset.features["executed_actions"] = {
                    "dtype": "float32",
                    "shape": (self.required_responses_per_critical_state * original_action_dim,),
                    "names": None,
                }
                self.dataset.meta.features["executed_actions"] = self.dataset.features["executed_actions"]
            
            if "executed_propensities" not in self.dataset.features:
                self.dataset.features["executed_propensities"] = {
                    "dtype": "float32",
                    "shape": (self.required_responses_per_critical_state,),
                    "names": None,
                }
                self.dataset.meta.features["executed_propensities"] = self.dataset.features["executed_propensities"]

            if "executed_approvals" not in self.dataset.features:
                self.dataset.features["executed_approvals"] = {
                    "dtype": "float32",  # Using float to allow NaN for non-executed
                    "shape": (self.required_responses_per_critical_state,),
                    "names": None,
                }
                self.dataset.meta.features["executed_approvals"] = self.dataset.features["executed_approvals"]

            # Recreate the HF dataset with updated features
            if self.dataset.hf_dataset is not None:
                # Get new HF features from the updated self.features
                new_hf_features = get_hf_features_from_features(self.dataset.features)

                # Create a new empty dataset with the correct features
                ft_dict = {col: [] for col in new_hf_features}
                new_hf_dataset = datasets.Dataset.from_dict(ft_dict, features=new_hf_features, split="train")

                # Apply the same transform
                from lerobot.common.datasets.utils import hf_transform_to_torch

                new_hf_dataset.set_transform(hf_transform_to_torch)

                # Replace the old dataset
                self.dataset.hf_dataset = new_hf_dataset

            # Clear any existing episode buffer so it gets recreated with new features
            if hasattr(self.dataset, "episode_buffer") and self.dataset.episode_buffer is not None:
                self.dataset.episode_buffer = None

            print(
                f"📐 Updated dataset action shape to {new_action_shape} (crowd_responses={self.required_responses_per_critical_state}, joints={original_action_dim})"
            )

    # =========================
    # Episode Saving
    # =========================

    def log_execution_attempt(
        self,
        episode_index: int,
        state_id: int,
        execution_index: int,
        action: torch.Tensor,
        propensity: float,
        selector_metadata: dict,
    ) -> None:
        """Log an individual execution attempt immediately when it happens."""
        propensity_log_path = self.dataset.root / "action_propensity_log.jsonl"
        
        log_entry = {
            "type": "execution_attempt",
            "episode_index": episode_index,
            "state_id": state_id,
            "execution_index": execution_index,
            "timestamp": time.time(),
            "propensity": propensity,
            "selector": selector_metadata.get("selector_used"),
            "mode": selector_metadata.get("mode"),
            "resampled": selector_metadata.get("resampled", False),
            "epsilon": selector_metadata.get("epsilon"),
        }
        
        with open(propensity_log_path, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
    
    def log_approval_summary(
        self,
        episode_index: int,
        state_id: int,
        num_executions: int,
        execution_propensities: list[float],
        approved: bool,
    ) -> None:
        """Log a summary when an action is approved/rejected and we proceed."""
        propensity_log_path = self.dataset.root / "action_propensity_log.jsonl"
        
        log_entry = {
            "type": "approval_summary",
            "episode_index": episode_index,
            "state_id": state_id,
            "timestamp": time.time(),
            "approved": approved,
            "num_executions": num_executions,
            "execution_propensities": execution_propensities,
        }
        
        with open(propensity_log_path, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

    def save_episode(self, buffer):
        """Save episode from completed states buffer to dataset."""
        episode_index = self.dataset.meta.total_episodes
        propensity_log_path = self.dataset.root / "action_propensity_log.jsonl"

        for state_id in sorted(buffer.keys()):
            state = buffer[state_id]
            obs = self.load_obs_from_disk(state["obs_path"])
            if "depth" in obs:
                del obs["depth"]  # delete the depth tensor

            # Build execution history arrays with matching indices
            execution_history = state.get("execution_history", [])
            action_dim = len(state["action_to_save"]) // self.required_responses_per_critical_state
            
            # Initialize arrays with NaN
            executed_actions = np.full(
                self.required_responses_per_critical_state * action_dim, np.nan, dtype=np.float32
            )
            executed_propensities = np.full(self.required_responses_per_critical_state, np.nan, dtype=np.float32)
            executed_approvals = np.full(self.required_responses_per_critical_state, np.nan, dtype=np.float32)
            
            # Fill in data for each execution (matching indices)
            for i, execution in enumerate(execution_history[:self.required_responses_per_critical_state]):
                # Store the executed action
                action_tensor = execution["action"]
                start_idx = i * action_dim
                end_idx = start_idx + action_dim
                executed_actions[start_idx:end_idx] = action_tensor.numpy()
                
                # Store propensity
                executed_propensities[i] = execution["propensity"]
                
                # Store approval: 1 = approved, -1 = rejected, None = pending (use NaN)
                if execution["approval"] is not None:
                    executed_approvals[i] = float(execution["approval"])

            # Construct frame with action selection metadata
            frame = {
                **obs,
                "action": state["action_to_save"],
                "task": state["task_text"],
                "executed_actions": executed_actions,
                "executed_propensities": executed_propensities,
                "executed_approvals": executed_approvals,
            }

            self.dataset.add_frame(frame)
            self._delete_obs_from_disk(state.get("obs_path"))

        self.dataset.save_episode()

    # =========================
    # Observation Cache Management
    # =========================

    def load_obs_from_disk(self, path: str | None) -> dict:
        """Load observations from disk cache."""
        if not path:
            return {}
        try:
            return torch.load(path, map_location="cpu")
        except Exception as e:
            print(f"⚠️  failed to load obs from {path}: {e}")
            return {}

    def _delete_obs_from_disk(self, path: str | None):
        """Delete observation file from disk cache after saving."""
        if not path:
            return
        try:
            os.remove(path)
        except Exception:
            pass
