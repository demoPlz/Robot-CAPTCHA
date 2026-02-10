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
        --crowd-repo-id /home/yilong/.cache/huggingface/lerobot/yilong/async_sess0_full_train \
        --expert-repo-id /home/yilong/.cache/huggingface/lerobot/yilong/drawer_teleop_50 \
        --target-repo-id /home/yilong/.cache/huggingface/lerobot/yilong/merged_crowd_expert

    # Pre-training: crowd only with dataset_id
    python scripts/merge_flexible_datasets.py \
        --crowd-repo-id /home/yilong/.cache/huggingface/lerobot/yilong/async_sess0_full_train \
        --target-repo-id /home/yilong/.cache/huggingface/lerobot/yilong/crowd_for_pretrain

    # Fine-tuning: expert only with dataset_id  
    python scripts/merge_flexible_datasets.py \
        --expert-repo-id /home/yilong/.cache/huggingface/lerobot/yilong/drawer_teleop_50 \
        --target-repo-id /home/yilong/.cache/huggingface/lerobot/yilong/expert_for_finetune
"""

import logging
import numpy as np
from pathlib import Path

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.utils.utils import init_logging


def _split_path(full_path: str | Path) -> tuple[Path, str]:
    """Split a full dataset path into (root, repo_id).
    
    root is the full path itself (LeRobotDataset uses root directly, not root/repo_id).
    repo_id is the last two path components (namespace/dataset_name).
    
    E.g. '/home/user/.cache/huggingface/lerobot/yilong/my_dataset'
      -> (Path('/home/user/.cache/huggingface/lerobot/yilong/my_dataset'), 'yilong/my_dataset')
    """
    p = Path(full_path)
    repo_id = f"{p.parent.name}/{p.name}"
    return p, repo_id


def merge_flexible_datasets(
    crowd_path: str | None,
    expert_path: str | None,
    target_path: str,
    interleave: bool = False,
) -> LeRobotDataset:
    """Merge crowd and/or expert datasets with dataset_id and proper padding.
    
    All paths should be full filesystem paths to the dataset directories.
    E.g. /home/user/.cache/huggingface/lerobot/yilong/my_dataset
    
    Args:
        crowd_path: Full path to crowd dataset (can be None if expert-only)
        expert_path: Full path to expert dataset (can be None if crowd-only)
        target_path: Full path for target merged dataset
        interleave: Whether to interleave episodes (doesn't matter for training)
        
    Returns:
        Merged LeRobotDataset
    """
    if not crowd_path and not expert_path:
        raise ValueError("Must provide at least one of crowd_path or expert_path")
    
    logging.info(f"Creating flexible dataset: {target_path}")
    logging.info(f"  Crowd source: {crowd_path or 'None'}")
    logging.info(f"  Expert source: {expert_path or 'None'}")
    
    # Load source datasets (split full paths into root + repo_id)
    if crowd_path:
        crowd_root, crowd_repo_id = _split_path(crowd_path)
        crowd_dataset = LeRobotDataset(crowd_repo_id, root=crowd_root)
    else:
        crowd_dataset = None
    
    if expert_path:
        expert_root, expert_repo_id = _split_path(expert_path)
        expert_dataset = LeRobotDataset(expert_repo_id, root=expert_root)
    else:
        expert_dataset = None
    
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
    # Keep original_frame_index for proper temporal sequencing with multi-action data
    keys_to_remove = ["frame_type"]
    for key in keys_to_remove:
        if key in features:
            del features[key]
    
    # Ensure original_frame_index exists in features (for multi-action temporal lookup)
    if "original_frame_index" not in features:
        features["original_frame_index"] = {"dtype": "int64", "shape": (1,), "names": None}
    
    # Ensure action_type exists in features
    if "action_type" not in features:
        features["action_type"] = {"dtype": "int64", "shape": (1,), "names": None}
    
    # Add dataset_id feature
    features["dataset_id"] = {"dtype": "float32", "shape": (2,), "names": ["is_expert", "is_crowd"]}
    
    # Create target dataset
    target_root, target_repo_id = _split_path(target_path)
    logging.info(f"Creating target dataset: {target_repo_id} at {target_root}")
    target_dataset = LeRobotDataset.create(
        repo_id=target_repo_id,
        fps=ref_dataset.fps,
        root=target_root,
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
            # Keep original_frame_index for crowd data, will add it for expert data
            skip_keys = {"index", "episode_index", "frame_index", "timestamp", "task_index", 
                        "frame_type"}
            
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
            
            # Handle original_frame_index for proper temporal sequencing
            # Crowd data: has original_frame_index (preserves temporal identity across multi-action duplicates)
            # Expert data: original_frame_index = relative frame index (no duplicates, sequential)
            if source_type == "expert":
                # Expert: sequential frames, original_frame_index = relative position in episode
                relative_frame_idx = frame_idx - ep_start
                frame_dict["original_frame_index"] = np.array([relative_frame_idx], dtype=np.int64)
            else:
                # Crowd: preserve original_frame_index from source
                if "original_frame_index" in frame_dict:
                    if isinstance(frame_dict["original_frame_index"], np.ndarray):
                        if frame_dict["original_frame_index"].ndim == 0:
                            frame_dict["original_frame_index"] = frame_dict["original_frame_index"].reshape(1)
                    else:
                        frame_dict["original_frame_index"] = np.array([frame_dict["original_frame_index"]], dtype=np.int64)
                else:
                    # Fallback: use relative frame index
                    relative_frame_idx = frame_idx - ep_start
                    frame_dict["original_frame_index"] = np.array([relative_frame_idx], dtype=np.int64)
            
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
    logging.info(f"   Target dataset: {target_path}")
    
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
        help="Full path to crowd dataset (optional if expert-only)",
    )
    parser.add_argument(
        "--expert-repo-id",
        help="Full path to expert dataset (optional if expert-only)",
    )
    parser.add_argument(
        "--target-repo-id",
        required=True,
        help="Full path for target merged dataset",
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
        crowd_path=args.crowd_repo_id,
        expert_path=args.expert_repo_id,
        target_path=args.target_repo_id,
        interleave=args.interleave,
    )
    
    if args.push_to_hub:
        logging.info("Pushing to Hugging Face Hub...")
        target_dataset.push_to_hub()
        logging.info("✅ Pushed to hub!")


if __name__ == "__main__":
    main()
