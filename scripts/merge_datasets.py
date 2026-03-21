#!/usr/bin/env python

"""
Simple script to merge multiple LeRobot datasets into one.

Preserves all custom fields (executed_actions, executed_propensities, executed_approvals, 
final_executed_action) and correctly updates all metadata (episode counts, frame counts, tasks).

Usage:
    python scripts/merge_datasets.py \
        --source-paths \
            /home/yilong/.cache/huggingface/lerobot/insertion/teleop_25_1_mar14 \
            /home/yilong/.cache/huggingface/lerobot/insertion/teleop_25_2_mar14 \
        --target-path /home/yilong/.cache/huggingface/lerobot/insertion/teleop_merged
"""

import logging
from pathlib import Path

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.utils.utils import init_logging


def _split_path(full_path: str | Path) -> tuple[Path, str]:
    """Split a full dataset path into (root, repo_id).
    
    E.g. '/home/user/.cache/huggingface/lerobot/insertion/teleop_25_1_mar14'
      -> (Path('/home/user/.cache/huggingface/lerobot/insertion/teleop_25_1_mar14'), 'insertion/teleop_25_1_mar14')
    """
    p = Path(full_path)
    repo_id = f"{p.parent.name}/{p.name}"
    return p, repo_id


def merge_datasets(
    source_paths: list[str],
    target_path: str,
) -> LeRobotDataset:
    """Merge multiple LeRobot datasets into one.
    
    Args:
        source_paths: List of full filesystem paths to source datasets
        target_path: Full filesystem path for target dataset
        
    Returns:
        Merged LeRobotDataset
    """
    if not source_paths:
        raise ValueError("Must provide at least one source dataset")
    
    logging.info(f"Merging {len(source_paths)} datasets into {target_path}")
    
    # Load first source to get schema
    first_root, first_repo_id = _split_path(source_paths[0])
    logging.info(f"Loading first source dataset: {first_repo_id} at {first_root}")
    first_source = LeRobotDataset(first_repo_id, root=first_root)
    
    # Create target dataset with same schema
    target_root, target_repo_id = _split_path(target_path)
    logging.info(f"Creating target dataset: {target_repo_id} at {target_root}")
    target_dataset = LeRobotDataset.create(
        repo_id=target_repo_id,
        fps=first_source.fps,
        root=target_root,
        features=first_source.features,
        use_videos=len(first_source.meta.video_keys) > 0,
    )
    
    # Start image writer for parallel processing
    target_dataset.start_image_writer(num_processes=8)
    
    total_episodes = 0
    total_frames = 0
    
    # Process each source dataset
    for src_idx, src_path in enumerate(source_paths):
        src_root, src_repo_id = _split_path(src_path)
        logging.info(f"\n[{src_idx+1}/{len(source_paths)}] Processing {src_repo_id}")
        
        # Load source (reuse first_source if it's the first one)
        source = first_source if src_idx == 0 else LeRobotDataset(src_repo_id, root=src_root)
        
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
                        np_value = value.numpy()
                        # Images come as CHW from dataset, need to convert to HWC for add_frame
                        if key.startswith("observation.images.") and np_value.ndim == 3:
                            np_value = np_value.transpose(1, 2, 0)  # CHW -> HWC
                        # Scalars need to be 1D arrays with shape (1,) for validation
                        elif np_value.ndim == 0:
                            np_value = np_value.reshape(1)
                        frame_dict[key] = np_value
                    else:
                        frame_dict[key] = value
                
                # Add task from metadata
                task_idx = source.hf_dataset[frame_idx]["task_index"]
                if hasattr(task_idx, "item"):
                    task_idx = task_idx.item()
                frame_dict["task"] = source.meta.tasks[task_idx]
                
                # Don't add timestamp - let add_frame auto-generate it from frame_index/fps
                # This avoids validation issues
                
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
    logging.info(f"   Target dataset: {target_path}")
    
    return target_dataset


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Merge multiple LeRobot datasets")
    parser.add_argument(
        "--source-paths",
        nargs="+",
        required=True,
        help="Full filesystem paths to source datasets (space-separated)",
    )
    parser.add_argument(
        "--target-path",
        required=True,
        help="Full filesystem path for target merged dataset",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push merged dataset to Hugging Face Hub",
    )
    
    args = parser.parse_args()
    
    init_logging()
    
    target_dataset = merge_datasets(
        source_paths=args.source_paths,
        target_path=args.target_path,
    )
    
    if args.push_to_hub:
        logging.info("Pushing to Hugging Face Hub...")
        target_dataset.push_to_hub()
        logging.info("✅ Pushed to hub!")


if __name__ == "__main__":
    main()
