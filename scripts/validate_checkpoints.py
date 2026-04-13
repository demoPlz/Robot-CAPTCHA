#!/usr/bin/env python3
import os
import glob
from pathlib import Path
import shutil

try:
    import safetensors.torch
except ImportError:
    print("Error: safetensors not installed. Run this script in the lerobot conda environment.")
    exit(1)

checkpoints_base = Path("/mnt/data_drive/crowdmaps/outputs/pour")

prefixes = [
    "pour_c15_k*",
    "pour_c20_k*",
    "pour_c30_k*",
    "pour_c45_k*",
    "pour_e*_mar10",
    "pour_e*_mar31"
]

def find_highest_valid_checkpoint(run_path):
    ckpt_dir = run_path / "checkpoints"
    if not ckpt_dir.exists():
        return None
    
    # Get all numeric checkpoint directories
    ckpts = [d for d in ckpt_dir.iterdir() if d.is_dir() and d.name.isdigit()]
    
    # Sort them by their actual integer value
    ckpts.sort(key=lambda d: int(d.name), reverse=True)
    
    deleted_some = False
    
    for c in ckpts:
        safetensor_path = c / "training_state" / "optimizer_state.safetensors"
        
        # Sometimes training just started saving so maybe optimizer_state isn't there yet
        if not safetensor_path.exists():
            print(f"[{run_path.name}] Missing optimizer_state for {c.name}. Considering it corrupted/unfinished.")
            deleted_some = True
            # We don't delete immediately yet manually, but mark it bad
            continue
            
        try:
            # Check the integrity of the safetensors file
            safetensors.torch.load_file(str(safetensor_path))
            return c.name, deleted_some
        except Exception as e:
            print(f"[{run_path.name}] CORRUPTED checkpoint {c.name}: {e}")
            deleted_some = True
            
    return None, deleted_some

def main():
    runs = []
    for prefix in prefixes:
        runs.extend(glob.glob(str(checkpoints_base / prefix)))
        
    for run_str in runs:
        run_path = Path(run_str)
        if not run_path.is_dir():
            continue
            
        highest_valid, deleted_some = find_highest_valid_checkpoint(run_path)
        
        if highest_valid:
            print(f"{run_path.name:30} -> Valid Highest: {highest_valid}  (Cleaned up higher: {deleted_some})")
            
            # Update local last symlink just in case
            last_link = run_path / "checkpoints" / "last"
            if last_link.is_symlink():
                last_link.unlink()
            
            # Create relative symlink
            cwd = os.getcwd()
            os.chdir(run_path / "checkpoints")
            os.symlink(highest_valid, "last")
            os.chdir(cwd)
        else:
            print(f"{run_path.name:30} -> NO VALID CHECKPOINTS FOUND")

if __name__ == "__main__":
    main()
