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
        asynchronous_mode: bool = False,
    ):
        """Initialize dataset manager.

        Args:
            required_responses_per_critical_state: Number of responses per critical state (for action shape)
            obs_cache_root: Root directory for observation cache
            asynchronous_mode: Whether async mode is enabled (changes logging behavior)

        """
        self.required_responses_per_critical_state = required_responses_per_critical_state
        self._obs_cache_root = obs_cache_root
        self.asynchronous_mode = asynchronous_mode

        # Dataset state
        self.dataset = None
        self.task_text = None

    # =========================
    # Dataset Initialization
    # =========================

    def init_dataset(self, cfg, robot, phase1_resumed: bool = False):
        """Initialize dataset for data collection policy training.
        
        Args:
            cfg: RecordControlConfig
            robot: Robot instance
            phase1_resumed: If True, the main dataset was auto-resumed from a checkpoint.
                           The DCP dataset should also be opened in resume mode.
        """
        from pathlib import Path
        
        # Check if we're continuing from a previous dataset
        from crowd_interface_config import CrowdInterfaceConfig
        crowd_cfg = CrowdInterfaceConfig()
        continue_from = crowd_cfg.continue_from_dataset
        
        # Auto-rename output dataset if it matches the continue_from dataset (prevent overwrite)
        if continue_from and cfg.data_collection_policy_repo_id == continue_from:
            original_repo_id = cfg.data_collection_policy_repo_id
            cfg.data_collection_policy_repo_id = f"{continue_from}_continue"
            print(f"⚠️  Output dataset would overwrite source dataset!")
            print(f"   Auto-renaming: {original_repo_id} → {cfg.data_collection_policy_repo_id}")
        
        # Auto-resume DCP dataset when main dataset was resumed
        dataset_root = Path(cfg.root) if cfg.root else (Path.home() / ".cache" / "huggingface" / "lerobot")
        dcp_path = dataset_root / cfg.data_collection_policy_repo_id

        if cfg.resume or phase1_resumed:
            if dcp_path.exists() and (dcp_path / "meta" / "info.json").exists():
                print(f"🔄 DCP dataset auto-resume: {dcp_path}")
                self.dataset = LeRobotDataset(cfg.data_collection_policy_repo_id, root=cfg.root)
                self.dataset.start_image_writer(
                    num_processes=cfg.num_image_writer_processes,
                    num_threads=cfg.num_image_writer_threads_per_camera * len(robot.cameras),
                )
                sanity_check_dataset_robot_compatibility(self.dataset, robot, cfg.fps, cfg.video)
            else:
                # DCP dataset doesn't exist yet (e.g. no episodes finalized before crash)
                # Create it fresh
                print(f"📦 Creating DCP dataset (not yet saved): {cfg.data_collection_policy_repo_id}")
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

        else:
            # Check if dataset already exists and auto-rename to prevent overwrite
            if dcp_path.exists() and (dcp_path / "meta" / "info.json").exists():
                # Dataset already exists - find a unique name
                original_repo_id = cfg.data_collection_policy_repo_id
                counter = 1
                while True:
                    new_repo_id = f"{original_repo_id}_new{counter}"
                    new_path = dataset_root / new_repo_id
                    if not new_path.exists():
                        cfg.data_collection_policy_repo_id = new_repo_id
                        break
                    counter += 1
                
                print(f"⚠️  Dataset already exists at: {dcp_path}")
                print(f"   Auto-renaming to prevent overwrite: {original_repo_id} → {cfg.data_collection_policy_repo_id}")
            
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
        
        # If continuing from previous dataset, copy all old frames
        if continue_from:
            print(f"🔄 Continue mode: copying frames from {continue_from}...")
            # If continue_from is an absolute path, use it directly; otherwise use root
            from pathlib import Path
            continue_path = Path(continue_from)
            if continue_path.is_absolute():
                # Full path provided - use directly
                self.copy_old_dataset_to_new(continue_from, None)
            else:
                # Repo ID provided - use with root
                self.copy_old_dataset_to_new(continue_from, cfg.root)
            
            # IMPORTANT: Clear stats after copying old dataset
            # Old dataset has different action shapes, stats will mismatch
            print(f"🔄 Clearing stats after copying old dataset (action shapes differ)")
            self.dataset.meta.stats = None

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
            
            # Add final_executed_action field (the single action that was executed and approved)
            if "final_executed_action" not in self.dataset.features:
                self.dataset.features["final_executed_action"] = {
                    "dtype": "float32",
                    "shape": (original_action_dim,),  # Just 7 floats (one action)
                    "names": None,
                }
                self.dataset.meta.features["final_executed_action"] = self.dataset.features["final_executed_action"]

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

    def _update_dataset_action_shape_dynamic(self, max_action_count: int):
        """Update dataset action shape dynamically for async mode.
        
        Args:
            max_action_count: Maximum number of actions across all states in the episode
        """
        if self.dataset is not None and "action" in self.dataset.features:
            from datasets import Features, Sequence, Value
            from lerobot.common.datasets.utils import get_hf_features_from_features

            # Get original single action dimension from final_executed_action (which is always 7)
            # This is safer than trying to divide the current action shape
            if "final_executed_action" in self.dataset.features:
                original_action_dim = self.dataset.features["final_executed_action"]["shape"][0]
            else:
                # Fallback: assume 7 joints (6 arm + 1 gripper)
                original_action_dim = 7
            
            new_action_shape = (max_action_count * original_action_dim,)

            # CRITICAL: Clear stats if shape is changing
            # Old episodes have different action shapes, aggregating will fail
            current_action_shape = self.dataset.features["action"]["shape"]
            if current_action_shape != new_action_shape:
                print(f"🔄 Action shape changing: {current_action_shape} → {new_action_shape}")
                print(f"🔄 Clearing dataset stats to prevent shape mismatch")
                self.dataset.meta.stats = None

            # Update all relevant shapes
            self.dataset.features["action"]["shape"] = new_action_shape
            self.dataset.meta.features["action"]["shape"] = new_action_shape
            
            self.dataset.features["executed_actions"]["shape"] = new_action_shape
            self.dataset.meta.features["executed_actions"]["shape"] = new_action_shape
            
            self.dataset.features["executed_propensities"]["shape"] = (max_action_count,)
            self.dataset.meta.features["executed_propensities"]["shape"] = (max_action_count,)
            
            self.dataset.features["executed_approvals"]["shape"] = (max_action_count,)
            self.dataset.meta.features["executed_approvals"]["shape"] = (max_action_count,)
            
            # Recreate the HF dataset with updated features
            if self.dataset.hf_dataset is not None:
                from lerobot.common.datasets.utils import get_hf_features_from_features, hf_transform_to_torch
                
                new_hf_features = get_hf_features_from_features(self.dataset.features)
                
                # Create a new empty dataset with the correct features
                ft_dict = {col: [] for col in new_hf_features}
                new_hf_dataset = datasets.Dataset.from_dict(ft_dict, features=new_hf_features, split="train")
                new_hf_dataset.set_transform(hf_transform_to_torch)
                
                # Replace the old dataset
                self.dataset.hf_dataset = new_hf_dataset
            
            print(
                f"📐 Dynamically updated dataset action shape to {new_action_shape} (max_actions={max_action_count}, joints={original_action_dim})"
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
    
    def log_user_approvals_for_state(
        self,
        episode_index: int,
        state_id: int,
        execution_history: list[dict],
        asynchronous_mode: bool = False,
    ) -> None:
        """Log which users' submissions were approved/rejected for a state.
        
        Args:
            episode_index: Episode index
            state_id: State ID
            execution_history: List of execution history entries
            asynchronous_mode: If True, skip logging (async mode has separate logger)
        """
        # Skip in async mode - use AsyncUserLogger instead
        if asynchronous_mode:
            return
        
        # Use dataset name in log filename for unique naming per run
        dataset_name = self.dataset.repo_id.replace('/', '_')
        user_approval_log_path = self.dataset.root / f"user_approval_log_{dataset_name}.jsonl"
        
        # Group by approval status
        accepted_users = []
        rejected_users = []
        
        for execution in execution_history:
            submitted_by = execution.get("submitted_by", [])
            approval = execution.get("approval")
            
            for user in submitted_by:
                user_info = {
                    "name": user.get("name"),
                    "email": user.get("email"),
                }
                
                if approval == 1:  # Approved
                    accepted_users.append(user_info)
                elif approval == -1:  # Rejected
                    rejected_users.append(user_info)
        
        log_entry = {
            "type": "state_approval",
            "episode_index": episode_index,
            "state_id": state_id,
            "timestamp": time.time(),
            "accepted_users": accepted_users,
            "rejected_users": rejected_users,
            "num_accepted": len(accepted_users),
            "num_rejected": len(rejected_users),
            "acceptance_rate": len(accepted_users) / (len(accepted_users) + len(rejected_users)) if (len(accepted_users) + len(rejected_users)) > 0 else None,
        }
        
        with open(user_approval_log_path, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
        
        # Print summary
        accepted_names = [u["name"] for u in accepted_users if u.get("name")]
        rejected_names = [u["name"] for u in rejected_users if u.get("name")]
        print(f"📊 State {state_id}: Accepted: {accepted_names}, Rejected: {rejected_names}, {log_entry['num_accepted']}/{log_entry['num_accepted'] + log_entry['num_rejected']} accepted ({log_entry['acceptance_rate']:.1%})")
    
    def log_episode_user_summary(self, episode_index: int, buffer: dict, episode_timing: dict = None) -> None:
        """Log per-user and overall acceptance rates for an episode, plus timing statistics.
        
        Args:
            episode_index: Episode number
            buffer: Dict of state_id -> state_info
            episode_timing: Optional dict with timing stats from state_manager
        """
        # Skip in async mode - use AsyncUserLogger instead
        if self.asynchronous_mode:
            return
        
        # Use dataset name in log filename for unique naming per run
        dataset_name = self.dataset.repo_id.replace('/', '_')
        user_approval_log_path = self.dataset.root / f"user_approval_log_{dataset_name}.jsonl"
        
        # Track stats per user
        user_stats = {}  # email -> {name, accepted, rejected}
        
        for state_id in sorted(buffer.keys()):
            state = buffer[state_id]
            execution_history = state.get("execution_history", [])
            
            for execution in execution_history:
                submitted_by = execution.get("submitted_by", [])
                approval = execution.get("approval")
                
                for user in submitted_by:
                    email = user.get("email", "unknown")
                    name = user.get("name", "Unknown")
                    
                    if email not in user_stats:
                        user_stats[email] = {"name": name, "accepted": 0, "rejected": 0}
                    
                    if approval == 1:
                        user_stats[email]["accepted"] += 1
                    elif approval == -1:
                        user_stats[email]["rejected"] += 1
        
        # Compute acceptance rates
        user_rates = []
        total_accepted = 0
        total_rejected = 0
        
        for email, stats in user_stats.items():
            total = stats["accepted"] + stats["rejected"]
            rate = stats["accepted"] / total if total > 0 else None
            
            user_entry = {
                "name": stats["name"],
                "email": email,
                "accepted": stats["accepted"],
                "rejected": stats["rejected"],
                "total": total,
                "acceptance_rate": rate,
            }
            
            # Add timing stats if available
            if episode_timing and email in episode_timing.get("per_user_stats", {}):
                user_timing = episode_timing["per_user_stats"][email]
                user_entry.update({
                    "total_time_seconds": user_timing["total_time"],
                    "avg_time_seconds": user_timing["avg_time"],
                    "num_submissions": user_timing["num_submissions"],
                    "min_time_seconds": user_timing["min_time"],
                    "max_time_seconds": user_timing["max_time"],
                })
            
            user_rates.append(user_entry)
            total_accepted += stats["accepted"]
            total_rejected += stats["rejected"]
        
        # Overall rate
        overall_total = total_accepted + total_rejected
        overall_rate = total_accepted / overall_total if overall_total > 0 else None
        
        log_entry = {
            "type": "episode_summary",
            "episode_index": episode_index,
            "timestamp": time.time(),
            "user_stats": user_rates,
            "overall_accepted": total_accepted,
            "overall_rejected": total_rejected,
            "overall_acceptance_rate": overall_rate,
        }
        
        # Add episode timing stats if available
        if episode_timing:
            log_entry["timing"] = {
                "episode_start_time": episode_timing["episode_start_time"],
                "episode_start_time_iso": episode_timing["episode_start_time_iso"],
                "episode_end_time": episode_timing["episode_end_time"],
                "episode_end_time_iso": episode_timing["episode_end_time_iso"],
                "total_episode_duration_seconds": episode_timing["total_episode_duration_seconds"],
                "overall_avg_submission_time": episode_timing["overall_avg_submission_time"],
                "overall_avg_state_duration": episode_timing["overall_avg_state_duration"],
                "per_state_stats": episode_timing["per_state_stats"],
            }
            
            # Print timing summary
            print(f"\n⏱️  === Episode {episode_index} Timing Summary ===")
            print(f"📅 Duration: {episode_timing['total_episode_duration_seconds']:.1f}s ({episode_timing['episode_start_time_iso']} → {episode_timing['episode_end_time_iso']})")
            print(f"⚡ Avg submission time: {episode_timing['overall_avg_submission_time']:.1f}s")
            print(f"📊 Avg state completion: {episode_timing['overall_avg_state_duration']:.1f}s")
            print(f"\n👥 Per-user timing:")
            for user_entry in user_rates:
                if "avg_time_seconds" in user_entry:
                    print(f"  • {user_entry['name']}: {user_entry['avg_time_seconds']:.1f}s avg ({user_entry['num_submissions']} submissions)")
        
        with open(user_approval_log_path, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
        
        # Print summary
        print(f"\n📊 Episode {episode_index} User Approval Summary:")
        for user in user_rates:
            print(f"   {user['name']} ({user['email']}): {user['accepted']}/{user['total']} accepted ({user['acceptance_rate']:.1%})")
        print(f"   Overall: {total_accepted}/{overall_total} accepted ({overall_rate:.1%})\n")

    def save_episode(self, buffer, episode_timing=None):
        """Save episode from completed states buffer to dataset.
        
        Args:
            buffer: Dict of state_id -> state_info
            episode_timing: Optional dict with episode timing statistics
        """
        episode_index = self.dataset.meta.total_episodes
        propensity_log_path = self.dataset.root / "action_propensity_log.jsonl"
        
        # In async mode: dynamically determine max action count across all states
        if self.asynchronous_mode:
            critical_states = [state for state in buffer.values() if state.get("critical", False)]
            if critical_states:
                # Calculate max and collect per-state stats
                state_action_counts = {}
                max_action_count = 0
                max_state_id = None
                
                for state in critical_states:
                    state_id = state.get("state_id")
                    exec_history = state.get("execution_history", [])
                    count = len(exec_history)
                    state_action_counts[state_id] = count
                    
                    if count > max_action_count:
                        max_action_count = count
                        max_state_id = state_id
                
                # Ensure minimum of required_responses_per_critical_state
                if max_action_count == 0:
                    print(f"⚠️  All states have empty execution_history, using required_responses_per_critical_state={self.required_responses_per_critical_state}")
                    max_action_count = self.required_responses_per_critical_state
                else:
                    print(f"📊 Async mode dataset sizing:")
                    print(f"   States: {len(critical_states)} critical states")
                    for s_id, count in sorted(state_action_counts.items()):
                        exec_hist = buffer[s_id].get("execution_history", [])
                        num_approved = sum(1 for e in exec_hist if e.get("approval") == 1)
                        num_rejected = sum(1 for e in exec_hist if e.get("approval") == -1)
                        marker = "📌" if s_id == max_state_id else "  "
                        print(f"   {marker} State {s_id}: {count} total ({num_approved} approved, {num_rejected} rejected)")
                        
                        # Debug: show each execution entry
                        for i, e in enumerate(exec_hist):
                            submitted = e.get("submitted_by", [])
                            submitter_info = f"{submitted[0].get('name', 'Unknown')}" if submitted else "Auto-filled"
                            approval_str = "✓" if e.get("approval") == 1 else "✗" if e.get("approval") == -1 else "?"
                            print(f"      [{i}] {approval_str} by {submitter_info}")
                            
                    print(f"   Max: {max_action_count} actions (state {max_state_id})")
            else:
                max_action_count = self.required_responses_per_critical_state
            
            # Skip dynamic shape update if we're in batch save mode (schema already set globally)
            if not getattr(self, '_batch_save_in_progress', False):
                self._update_dataset_action_shape_dynamic(max_action_count)
                action_capacity = max_action_count
            else:
                # In batch save mode: derive action_capacity from current schema
                # Schema shape is (max_actions * action_dim), so divide by action_dim
                action_dim = self.dataset.features["final_executed_action"]["shape"][0]
                action_capacity = self.dataset.features["action"]["shape"][0] // action_dim
        else:
            # Sync mode: use fixed capacity
            action_capacity = self.required_responses_per_critical_state

        for state_id in sorted(buffer.keys()):
            state = buffer[state_id]
            obs = self.load_obs_from_disk(state["obs_path"])
            if "depth" in obs:
                del obs["depth"]  # delete the depth tensor

            # Build execution history arrays with matching indices
            execution_history = state.get("execution_history", [])
            # Get action dimension from the dataset features (accounts for dynamic sizing)
            if self.asynchronous_mode:
                action_dim = self.dataset.features["final_executed_action"]["shape"][-1]
            else:
                action_dim = len(state["action_to_save"]) // self.required_responses_per_critical_state
            
            # Initialize arrays with NaN using action_capacity (dynamic in async, fixed in sync)
            executed_actions = np.full(
                action_capacity * action_dim, np.nan, dtype=np.float32
            )
            executed_propensities = np.full(action_capacity, np.nan, dtype=np.float32)
            executed_approvals = np.full(action_capacity, np.nan, dtype=np.float32)
            
            # Fill in data for each execution (matching indices)
            for i, execution in enumerate(execution_history[:action_capacity]):
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
            
            # Build action_to_save
            if self.asynchronous_mode:
                # Async mode: always build from execution_history with padding to capacity
                all_actions_list = []
                for execution in execution_history:
                    action_tensor = execution["action"]
                    action_np = action_tensor.numpy() if hasattr(action_tensor, "numpy") else np.array(action_tensor)
                    all_actions_list.append(action_np)
                
                # Concatenate all actions
                if all_actions_list:
                    all_actions = np.concatenate(all_actions_list)
                else:
                    all_actions = np.array([], dtype=np.float32)
                
                # Pad to capacity if needed
                if len(execution_history) < action_capacity:
                    padding_size = (action_capacity - len(execution_history)) * action_dim
                    if padding_size > 0:
                        padding = np.full(padding_size, np.nan, dtype=np.float32)
                        all_actions = np.concatenate([all_actions, padding])
                
                action_to_save = all_actions
            else:
                # Sync mode: use pre-built action_to_save
                action_to_save = state["action_to_save"]

            # Construct frame with action selection metadata
            frame = {
                **obs,
                "action": action_to_save,
                "task": state["task_text"],
                "executed_actions": executed_actions,
                "executed_propensities": executed_propensities,
                "executed_approvals": executed_approvals,
            }
            
            # Add final_executed_action if present (7 floats - the one executed action that was approved)
            if state.get("final_executed_action") is not None:
                final_action = state["final_executed_action"]
                # Convert to numpy array if needed
                if isinstance(final_action, list):
                    final_action = np.array(final_action, dtype=np.float32)
                elif hasattr(final_action, "numpy"):
                    final_action = final_action.numpy()
                frame["final_executed_action"] = final_action
            else:
                # Fill with NaN if not set (non-critical or end state)
                frame["final_executed_action"] = np.full(action_dim, np.nan, dtype=np.float32)

            self.dataset.add_frame(frame)
            self._delete_obs_from_disk(state.get("obs_path"))
            
            # Log user approval stats for each state
            if execution_history:
                self.log_user_approvals_for_state(episode_index, state_id, execution_history, asynchronous_mode=self.asynchronous_mode)

        # Log episode-wide user summary with timing
        self.log_episode_user_summary(episode_index, buffer, episode_timing)

        self.dataset.save_episode()

    def get_last_critical_state_from_dataset(self, dataset_repo_id: str, root: Path | None) -> dict | None:
        """Load dataset and return last critical state with joint positions.
        
        Args:
            dataset_repo_id: Either a repo ID (e.g., "user/dataset") or full path to dataset
            root: Root directory for datasets. Ignored if dataset_repo_id is an absolute path.
        
        Returns dict with keys: joint_positions, episode_index, frame_index
        Or None if dataset is empty.
        """
        try:
            # If dataset_repo_id is absolute path, use it directly; otherwise combine with root
            from pathlib import Path
            dataset_path = Path(dataset_repo_id)
            if dataset_path.is_absolute():
                dataset = LeRobotDataset(dataset_repo_id)
            else:
                dataset = LeRobotDataset(dataset_repo_id, root=root)
                
            if len(dataset) == 0:
                return None
            
            # Get last frame
            last_frame = dataset[-1]
            
            return {
                "joint_positions": last_frame["observation.state"].numpy().tolist(),
                "episode_index": int(last_frame["episode_index"]),
                "frame_index": int(last_frame["frame_index"]),
            }
        except Exception as e:
            print(f"❌ Error loading continue dataset: {e}")
            import traceback
            traceback.print_exc()
            return None

    def copy_old_dataset_to_new(self, old_dataset_repo_id: str, root: Path | None):
        """Copy all frames from old dataset into current dataset.
        
        This loads each episode from the old dataset and saves it to the new one,
        properly maintaining episode boundaries and metadata.
        
        Args:
            old_dataset_repo_id: Either a repo ID (e.g., "user/dataset") or full path to dataset
            root: Root directory for datasets. Ignored if old_dataset_repo_id is an absolute path.
        """
        try:
            # If old_dataset_repo_id is absolute path, use it directly; otherwise combine with root
            from pathlib import Path
            dataset_path = Path(old_dataset_repo_id)
            if dataset_path.is_absolute():
                old_dataset = LeRobotDataset(old_dataset_repo_id)
            else:
                old_dataset = LeRobotDataset(old_dataset_repo_id, root=root)
            
            print(f"📂 Loaded dataset from: {old_dataset.root}")
            print(f"📊 Dataset metadata: total_episodes={old_dataset.meta.total_episodes}, total_frames={old_dataset.meta.total_frames}")
            print(f"📋 Copying {old_dataset.meta.total_episodes} episodes from {old_dataset_repo_id}...")
            
            if old_dataset.meta.total_episodes == 0:
                print(f"⚠️  Old dataset is empty, nothing to copy")
                return
            
            # IMPORTANT: Clear stats since we're changing action shapes
            # The old dataset has action shape (7,) but new dataset has (N*7,)
            # Aggregating stats with different shapes will fail
            print(f"🔄 Clearing dataset stats (action shapes differ between old and new)")
            self.dataset.meta.stats = None
            
            # IMPORTANT: When continuing, we want to resume the LAST episode, not start a new one
            # So we only copy the last episode's frames WITHOUT calling save_episode()
            # This keeps the episode buffer open for continued recording
            
            last_episode_idx = old_dataset.meta.total_episodes - 1
            print(f"📝 Continuing episode {last_episode_idx} (will not finalize, keeping buffer open)")
            
            # Get last episode bounds
            from_idx = old_dataset.episode_data_index["from"][last_episode_idx].item()
            to_idx = old_dataset.episode_data_index["to"][last_episode_idx].item()
            
            # Add all frames from last episode to buffer (without finalizing)
            for frame_idx in range(from_idx, to_idx):
                    frame = dict(old_dataset[frame_idx])
                    
                    # Transform frame to match new dataset format
                    # Remove metadata fields that are auto-generated
                    for key in ['episode_index', 'index', 'task_index', 'frame_index', 'timestamp']:
                        frame.pop(key, None)
                    
                    # Transform image shapes from (C, H, W) to (H, W, C)
                    import torch
                    import numpy as np
                    for key in list(frame.keys()):
                        if key.startswith('observation.images.'):
                            img = frame[key]
                            if isinstance(img, torch.Tensor):
                                if img.ndim == 3 and img.shape[0] == 3:
                                    # Transpose from (C, H, W) to (H, W, C)
                                    frame[key] = img.permute(1, 2, 0).numpy()
                                else:
                                    frame[key] = img.numpy()
                            elif isinstance(img, np.ndarray):
                                if img.ndim == 3 and img.shape[0] == 3:
                                    # Transpose from (C, H, W) to (H, W, C)
                                    frame[key] = np.transpose(img, (1, 2, 0))
                    
                    self.dataset.add_frame(frame)
            
            # DO NOT call save_episode() - keep buffer open for continued recording
            print(f"   ✓ Loaded {to_idx - from_idx} frames into episode buffer (not finalized)")
            print(f"✅ Ready to continue recording episode {last_episode_idx}")
            print(f"📊 Current episode has {to_idx - from_idx} frames, will continue adding more")
            
        except Exception as e:
            print(f"❌ Error copying old dataset: {e}")
            import traceback
            traceback.print_exc()

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
