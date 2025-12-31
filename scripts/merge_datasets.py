#!/usr/bin/env python

"""
Simple script to merge multiple LeRobot datasets into one.

Preserves all custom fields (executed_actions, executed_propensities, executed_approvals, 
final_executed_action) and correctly updates all metadata (episode counts, frame counts, tasks).

Usage:
    python scripts/merge_datasets.py \
        --source-repo-ids yilong/dataset1 yilong/dataset2 yilong/dataset3 \
        --target-repo-id yilong/merged_dataset \
        --root /home/yilong/.cache/huggingface/lerobot
"""

import logging
from pathlib import Path

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.utils.utils import init_logging


def merge_datasets(
    source_repo_ids: list[str],
    target_repo_id: str,
    root: str | Path | None = None,
) -> LeRobotDataset:
    """Merge multiple LeRobot datasets into one.
    
    Args:
        source_repo_ids: List of source dataset repo IDs to merge
        target_repo_id: Target dataset repo ID
        root: Root directory for datasets
        
    Returns:
        Merged LeRobotDataset
    """
    if not source_repo_ids:
        raise ValueError("Must provide at least one source dataset")
    
    logging.info(f"Merging {len(source_repo_ids)} datasets into {target_repo_id}")
    
    # Load first source to get schema
    logging.info(f"Loading first source dataset: {source_repo_ids[0]}")
    first_source = LeRobotDataset(source_repo_ids[0], root=root)
    
    # Create target dataset with same schema
    logging.info(f"Creating target dataset: {target_repo_id}")
    target_dataset = LeRobotDataset.create(
        repo_id=target_repo_id,
        fps=first_source.fps,
        root=root,
        features=first_source.features,
        use_videos=first_source.video,
    )
    
    # Start image writer for parallel processing
    target_dataset.start_image_writer(num_processes=8)
    
    total_episodes = 0
    total_frames = 0
    
    # Process each source dataset
    for src_idx, src_repo_id in enumerate(source_repo_ids):
        logging.info(f"\n[{src_idx+1}/{len(source_repo_ids)}] Processing {src_repo_id}")
        
        # Load source (reuse first_source if it's the first one)
        source = first_source if src_idx == 0 else LeRobotDataset(src_repo_id, root=root)
        
        logging.info(f"  Episodes: {source.num_episodes}, Frames: {source.num_frames}")
        
        # Copy each episode
        for ep_idx in range(source.num_episodes):
            logging.info(f"  Copying episode {ep_idx}/{source.num_episodes-1}")
            
            # Get episode frame range
            ep_start = source.episode_data_index["from"][ep_idx].item()
            ep_end = source.episode_data_index["to"][ep_idx].item()
            
            # Create new episode buffer
            target_dataset.create_episode_buffer()
            
            # Copy all frames in episode
            for frame_idx in range(ep_start, ep_end):
                frame = source[frame_idx]
                
                # Convert all tensors to numpy and remove dataset index keys
                frame_dict = {}
                for key, value in frame.items():
                    if key in ["index", "episode_index", "frame_index", "timestamp", "task_index"]:
                        # Skip auto-generated keys - they'll be recreated
                        continue
                    if hasattr(value, "numpy"):
                        frame_dict[key] = value.numpy()
                    else:
                        frame_dict[key] = value
                
                # Add task from metadata
                task_idx = source.hf_dataset[frame_idx]["task_index"]
                frame_dict["task"] = source.meta.tasks[task_idx]
                
                # Add timestamp
                frame_dict["timestamp"] = source.hf_dataset[frame_idx]["timestamp"]
                
                target_dataset.add_frame(frame_dict)
            
            # Save episode
            target_dataset.save_episode()
            target_dataset.clear_episode_buffer()
            total_episodes += 1
            total_frames += (ep_end - ep_start)
    
    # Stop image writer
    target_dataset.stop_image_writer()
    
    logging.info(f"\n✅ Merge complete!")
    logging.info(f"   Total episodes: {total_episodes}")
    logging.info(f"   Total frames: {total_frames}")
    logging.info(f"   Target dataset: {target_repo_id}")
    
    return target_dataset


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Merge multiple LeRobot datasets")
    parser.add_argument(
        "--source-repo-ids",
        nargs="+",
        required=True,
        help="Source dataset repo IDs to merge (space-separated)",
    )
    parser.add_argument(
        "--target-repo-id",
        required=True,
        help="Target dataset repo ID",
    )
    parser.add_argument(
        "--root",
        help="Root directory for datasets (default: ~/.cache/huggingface/lerobot)",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push merged dataset to Hugging Face Hub",
    )
    
    args = parser.parse_args()
    
    init_logging()
    
    target_dataset = merge_datasets(
        source_repo_ids=args.source_repo_ids,
        target_repo_id=args.target_repo_id,
        root=args.root,
    )
    
    if args.push_to_hub:
        logging.info("Pushing to Hugging Face Hub...")
        target_dataset.push_to_hub()
        logging.info("✅ Pushed to hub!")


if __name__ == "__main__":
    main()
