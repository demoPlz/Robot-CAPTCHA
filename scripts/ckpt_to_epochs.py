#!/usr/bin/env python3
import sys
import json
from pathlib import Path

def main():
    if len(sys.argv) < 2:
        print("Usage: python checkpoint_to_epochs.py /path/to/checkpoints/0110000")
        sys.exit(1)
        
    ckpt_path = Path(sys.argv[1]).resolve()
    # Strip commas in case user inputs things like 400,000
    step_str = ckpt_path.name.replace(',', '')
    try:
        step = int(step_str)
    except ValueError:
        print(f"Error: Folder name '{step_str}' is not a valid step number.")
        sys.exit(1)
        
    cfg_path = ckpt_path / "pretrained_model" / "train_config.json"
    
    # If this specific checkpoint doesn't exist, we can borrow the config from the "last" checkpoint or any sibling
    if not cfg_path.exists():
        checkpoints_dir = ckpt_path.parent
        if (checkpoints_dir / "last" / "pretrained_model" / "train_config.json").exists():
            cfg_path = checkpoints_dir / "last" / "pretrained_model" / "train_config.json"
        else:
            # try to find any
            sibling_configs = list(checkpoints_dir.glob("*/pretrained_model/train_config.json"))
            if sibling_configs:
                cfg_path = sibling_configs[0]
            else:
                print(f"Error: Training config not found in {checkpoints_dir}")
                sys.exit(1)
        
    with open(cfg_path) as f:
        cfg = json.load(f)
        
    batch_size = cfg.get("batch_size")
    if not batch_size:
        print("Error: Could not find 'batch_size' in train_config.")
        sys.exit(1)
        
    dataset_repo = cfg.get("dataset", {}).get("repo_id")
    if not dataset_repo:
        print("Error: Could not find dataset 'repo_id' in train_config.")
        sys.exit(1)
        
    # Resolve the dataset's info.json location
    repo_path = Path(dataset_repo)
    if not repo_path.is_absolute():
        repo_path = Path.home() / ".cache" / "huggingface" / "lerobot" / dataset_repo
        
    info_path = repo_path / "meta" / "info.json"
    
    if not info_path.exists():
        print(f"Error: Dataset info not found at {info_path}.")
        print("Ensure the dataset is available locally on this machine.")
        sys.exit(1)
        
    with open(info_path) as f:
        info = json.load(f)
        
    total_frames = info.get("total_frames")
    if not total_frames:
        print("Error: Could not find 'total_frames' in info.json")
        sys.exit(1)
        
    # Calculation
    epochs = (step * batch_size) / total_frames
    
    print("=" * 60)
    print(f"Checkpoint:   {ckpt_path.parent.parent.name} -> {ckpt_path.name}")
    print(f"Dataset:      {dataset_repo}")
    print("-" * 60)
    print(f"Step:         {step:,}")
    print(f"Batch Size:   {batch_size}")
    print(f"Total Frames: {total_frames:,}")
    print("-" * 60)
    print(f"Elapsed:      {epochs:.2f} Epochs")
    print("=" * 60)


if __name__ == "__main__":
    main()
