"""Dataset Manager Module.

Manages LeRobot dataset operations for the crowd interface. Handles dataset initialization, episode saving, and
observation loading/cleanup.

"""

import json
import os
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

            # Add new features for action selection logging to BOTH features dicts
            if "action_propensity" not in self.dataset.features:
                self.dataset.features["action_propensity"] = {"dtype": "float32", "shape": (1,), "names": None}
                self.dataset.meta.features["action_propensity"] = self.dataset.features["action_propensity"]

            if "selector_mode" not in self.dataset.features:
                self.dataset.features["selector_mode"] = {"dtype": "string", "shape": (1,), "names": None}
                self.dataset.meta.features["selector_mode"] = self.dataset.features["selector_mode"]

            if "approval_status" not in self.dataset.features:
                self.dataset.features["approval_status"] = {"dtype": "string", "shape": (1,), "names": None}
                self.dataset.meta.features["approval_status"] = self.dataset.features["approval_status"]

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

    def save_episode(self, buffer):
        """Save episode from completed states buffer to dataset."""
        episode_index = self.dataset.meta.total_episodes
        propensity_log_path = self.dataset.root / "action_propensity_log.jsonl"

        for state_id in sorted(buffer.keys()):
            state = buffer[state_id]
            obs = self.load_obs_from_disk(state["obs_path"])
            if "depth" in obs:
                del obs["depth"]  # delete the depth tensor

            # Construct frame with action selection metadata
            frame = {
                **obs,
                "action": state["action_to_save"],
                "task": state["task_text"],
                "action_propensity": np.array([state.get("action_propensity", 1.0)], dtype=np.float32),
                "selector_mode": state.get("action_selection_metadata", {}).get("selector_used", "unknown"),
                "approval_status": state.get("approval_status") or "none",  # "approved", "rejected", or "none"
            }

            self.dataset.add_frame(frame)
            self._delete_obs_from_disk(state.get("obs_path"))

            # Log detailed propensity info to separate JSONL file for offline analysis
            action_metadata = state.get("action_selection_metadata", {})
            log_entry = {
                "episode_index": episode_index,
                "state_id": state_id,
                "frame_index": self.dataset.episode_buffer["size"] - 1,  # Just added
                "timestamp": (
                    self.dataset.episode_buffer["timestamp"][-1] if self.dataset.episode_buffer["timestamp"] else None
                ),
                "is_critical": state.get("critical", False),
                "approval_status": state.get("approval_status"),
                "action_propensity": float(state.get("action_propensity", 1.0)),
                "selector_mode": action_metadata.get("selector_used"),
                "epsilon": action_metadata.get("epsilon"),
                "resampled": action_metadata.get("resampled", False),
                "num_candidate_actions": (
                    action_metadata.get("num_remaining_actions")
                    if action_metadata.get("resampled")
                    else len(state.get("actions", []))
                ),
                "num_already_executed": action_metadata.get("num_already_executed", 0),
            }

            # Append to JSONL log
            with open(propensity_log_path, "a") as f:
                f.write(json.dumps(log_entry) + "\n")

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
