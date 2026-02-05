#!/usr/bin/env python

"""
Merge crowd and expert datasets for flexible_diffusion training.

This script:
1. Adds dataset_id to each sample: [1,0] for expert, [0,1] for crowd
2. For crowd data: adds action_is_pad mask (True for padded timesteps)
3. For expert data: adds action_type=0 if missing, action_is_pad=False
4. Optionally interleaves episodes (doesn't matter for batch training)

The resulting dataset can be used for:
- Co-training: merged dataset with both sources
- Pre-training: single dataset (crowd only) with dataset_id for later fine-tuning
- Fine-tuning: single dataset (expert only) with dataset_id

Usage:
    # Co-training: merge crowd and expert
    python scripts/merge_flexible_datasets.py \
        --crowd-repo-id yilong/async_sess0_full_train \
        --expert-repo-id yilong/drawer_teleop_50 \
        --target-repo-id yilong/merged_crowd_expert \
        --horizon 16

    # Pre-training: crowd only with dataset_id
    python scripts/merge_flexible_datasets.py \
        --crowd-repo-id yilong/async_sess0_full_train \
        --target-repo-id yilong/crowd_for_pretrain \
        --horizon 16

    # Fine-tuning: expert only with dataset_id  
    python scripts/merge_flexible_datasets.py \
        --expert-repo-id yilong/drawer_teleop_50 \
        --target-repo-id yilong/expert_for_finetune \
        --horizon 16
"""

import logging
import numpy as np
from pathlib import Path

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.utils.utils import init_logging


def merge_flexible_datasets(
    crowd_repo_id: str | None,
    expert_repo_id: str | None,
    target_repo_id: str,
    horizon: int = 16,
    root: str | Path | None = None,
    interleave: bool = False,
) -> LeRobotDataset:
    """Merge crowd and/or expert datasets with dataset_id and proper padding.
    
    Args:
        crowd_repo_id: Crowd dataset repo ID (can be None if expert-only)
        expert_repo_id: Expert dataset repo ID (can be None if crowd-only)
        target_repo_id: Target dataset repo ID
        horizon: Target horizon for actions (crowd data will be padded to this)
        root: Root directory for datasets
        interleave: Whether to interleave episodes (doesn't matter for training)
        
    Returns:
        Merged LeRobotDataset
    """
    if not crowd_repo_id and not expert_repo_id:
        raise ValueError("Must provide at least one of crowd_repo_id or expert_repo_id")
    
    logging.info(f"Creating flexible dataset: {target_repo_id}")
    logging.info(f"  Crowd source: {crowd_repo_id or 'None'}")
    logging.info(f"  Expert source: {expert_repo_id or 'None'}")
    logging.info(f"  Target horizon: {horizon}")
    
    # Load source datasets
    crowd_dataset = LeRobotDataset(crowd_repo_id, root=root) if crowd_repo_id else None
    expert_dataset = LeRobotDataset(expert_repo_id, root=root) if expert_repo_id else None
    
    # Get reference dataset for schema
    ref_dataset = crowd_dataset or expert_dataset
    
    # Build unified features dict from both datasets
    features = dict(ref_dataset.features)
    
    # Merge features from the other dataset if it exists
    other_dataset = expert_dataset if crowd_dataset else None
    if crowd_dataset and expert_dataset:
        other_dataset = expert_dataset
        for key, value in other_dataset.features.items():
            if key not in features:
                features[key] = value
    
    # Remove source-specific fields that don't generalize
    # These are dataset-specific and will be replaced by dataset_id
    keys_to_remove = ["original_frame_index", "frame_type"]
    for key in keys_to_remove:
        if key in features:
            del features[key]
    
    # Ensure action_type exists in features
    if "action_type" not in features:
        features["action_type"] = {"dtype": "int64", "shape": (1,), "names": None}
    
    # Add dataset_id feature
    features["dataset_id"] = {"dtype": "float32", "shape": (2,), "names": ["is_expert", "is_crowd"]}
    
    # Create target dataset
    logging.info(f"Creating target dataset: {target_repo_id}")
    target_dataset = LeRobotDataset.create(
        repo_id=target_repo_id,
        fps=ref_dataset.fps,
        root=root,
        features=features,
        use_videos=len(ref_dataset.meta.video_keys) > 0,
    )
    
    target_dataset.start_image_writer(num_processes=8)
    
    total_episodes = 0
    total_frames = 0
    
    # Build episode list for optional interleaving
    episodes_to_process = []
    
    if crowd_dataset:
        for ep_idx in range(crowd_dataset.num_episodes):
            episodes_to_process.append(("crowd", crowd_dataset, ep_idx))
    
    if expert_dataset:
        for ep_idx in range(expert_dataset.num_episodes):
            episodes_to_process.append(("expert", expert_dataset, ep_idx))
    
    if interleave and crowd_dataset and expert_dataset:
        # Interleave: alternate between crowd and expert
        crowd_eps = [e for e in episodes_to_process if e[0] == "crowd"]
        expert_eps = [e for e in episodes_to_process if e[0] == "expert"]
        episodes_to_process = []
        max_len = max(len(crowd_eps), len(expert_eps))
        for i in range(max_len):
            if i < len(crowd_eps):
                episodes_to_process.append(crowd_eps[i])
            if i < len(expert_eps):
                episodes_to_process.append(expert_eps[i])
    
    # Process each episode
    for idx, (source_type, source_dataset, ep_idx) in enumerate(episodes_to_process):
        logging.info(f"[{idx+1}/{len(episodes_to_process)}] Processing {source_type} episode {ep_idx}")
        
        # Get episode frame range
        ep_start = source_dataset.episode_data_index["from"][ep_idx].item()
        ep_end = source_dataset.episode_data_index["to"][ep_idx].item()
        
        # Create new episode buffer
        target_dataset.create_episode_buffer()
        
        # Dataset ID: expert=[1,0], crowd=[0,1]
        if source_type == "expert":
            dataset_id = np.array([1.0, 0.0], dtype=np.float32)
        else:
            dataset_id = np.array([0.0, 1.0], dtype=np.float32)
        
        # Copy all frames in episode
        for frame_idx in range(ep_start, ep_end):
            frame = source_dataset[frame_idx]
            
            # Keys to skip (auto-generated or source-specific)
            skip_keys = {"index", "episode_index", "frame_index", "timestamp", "task_index", 
                        "original_frame_index", "frame_type"}
            
            # Convert all tensors to numpy and remove auto-generated keys
            frame_dict = {}
            for key, value in frame.items():
                if key in skip_keys:
                    continue
                if hasattr(value, "numpy"):
                    np_value = value.numpy()
                    # Images: CHW -> HWC
                    if key.startswith("observation.images.") and np_value.ndim == 3:
                        np_value = np_value.transpose(1, 2, 0)
                    # Scalars need shape (1,)
                    elif np_value.ndim == 0:
                        np_value = np_value.reshape(1)
                    frame_dict[key] = np_value
                else:
                    frame_dict[key] = value
            
            # Add task from metadata
            task_idx = source_dataset.hf_dataset[frame_idx]["task_index"]
            if hasattr(task_idx, "item"):
                task_idx = task_idx.item()
            frame_dict["task"] = source_dataset.meta.tasks[task_idx]
            
            # Add dataset_id
            frame_dict["dataset_id"] = dataset_id
            
            # Handle action_type
            # Crowd action_types: 0=non_critical, 1=critical_final, 2=critical_crowd, 
            #                     3=critical_single_mode, 4=non_critical_negative, 5=critical_negative
            # Expert action_type: 6=expert
            if source_type == "expert":
                # Expert data: use action_type=6
                frame_dict["action_type"] = np.array([6], dtype=np.int64)
            elif "action_type" not in frame_dict:
                # Crowd data without action_type (shouldn't happen): use 0 (non_critical)
                frame_dict["action_type"] = np.array([0], dtype=np.int64)
            elif isinstance(frame_dict["action_type"], np.ndarray):
                # Ensure shape is (1,)
                if frame_dict["action_type"].ndim == 0:
                    frame_dict["action_type"] = frame_dict["action_type"].reshape(1)
            
            target_dataset.add_frame(frame_dict)
        
        # Save episode
        target_dataset.save_episode()
        target_dataset.clear_episode_buffer()
        total_episodes += 1
        total_frames += (ep_end - ep_start)
    
    target_dataset.stop_image_writer()
    
    logging.info(f"\n✅ Merge complete!")
    logging.info(f"   Total episodes: {total_episodes}")
    logging.info(f"   Total frames: {total_frames}")
    logging.info(f"   Target dataset: {target_repo_id}")
    
    if crowd_dataset:
        logging.info(f"   Crowd episodes: {crowd_dataset.num_episodes}")
    if expert_dataset:
        logging.info(f"   Expert episodes: {expert_dataset.num_episodes}")
    
    return target_dataset


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Merge crowd and expert datasets for flexible_diffusion")
    parser.add_argument(
        "--crowd-repo-id",
        help="Crowd dataset repo ID (optional if expert-only)",
    )
    parser.add_argument(
        "--expert-repo-id",
        help="Expert dataset repo ID (optional if crowd-only)",
    )
    parser.add_argument(
        "--target-repo-id",
        required=True,
        help="Target dataset repo ID",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=16,
        help="Target horizon for actions (default: 16)",
    )
    parser.add_argument(
        "--root",
        help="Root directory for datasets (default: ~/.cache/huggingface/lerobot)",
    )
    parser.add_argument(
        "--interleave",
        action="store_true",
        help="Interleave episodes from crowd and expert (doesn't matter for training)",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push merged dataset to Hugging Face Hub",
    )
    
    args = parser.parse_args()
    
    if not args.crowd_repo_id and not args.expert_repo_id:
        parser.error("Must provide at least one of --crowd-repo-id or --expert-repo-id")
    
    init_logging()
    
    target_dataset = merge_flexible_datasets(
        crowd_repo_id=args.crowd_repo_id,
        expert_repo_id=args.expert_repo_id,
        target_repo_id=args.target_repo_id,
        horizon=args.horizon,
        root=args.root,
        interleave=args.interleave,
    )
    
    if args.push_to_hub:
        logging.info("Pushing to Hugging Face Hub...")
        target_dataset.push_to_hub()
        logging.info("✅ Pushed to hub!")


if __name__ == "__main__":
    main()
