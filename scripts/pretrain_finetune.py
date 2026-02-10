#!/usr/bin/env python

"""
Pretrain on crowd data, then finetune on expert data — in one command.

This script runs two sequential training phases:
1. Pretrain: Train flexible_diffusion on a crowd dataset for N steps
2. Finetune: Resume from pretrain checkpoint, switch to expert dataset, train for M more steps

Both phases use dataset_id conditioning ([0,1] for crowd, [1,0] for expert) so the model
learns to distinguish data sources.

IMPORTANT: Both datasets should be prepared with merge_flexible_datasets.py (even for
single-source datasets) so they have the required fields (dataset_id, action_type,
original_frame_index). Example:

    # Prepare crowd-only dataset for pretraining
    python scripts/merge_flexible_datasets.py \
        --crowd-repo-id /path/to/crowd_dataset \
        --target-repo-id /path/to/crowd_for_pretrain

    # Prepare expert-only dataset for finetuning
    python scripts/merge_flexible_datasets.py \
        --expert-repo-id /path/to/expert_dataset \
        --target-repo-id /path/to/expert_for_finetune

    # Then run the pipeline
    python scripts/pretrain_finetune.py \
        --pretrain-dataset /path/to/crowd_for_pretrain \
        --finetune-dataset /path/to/expert_for_finetune \
        --pretrain-steps 100000 \
        --finetune-steps 50000 \
        --output-dir outputs/my_run
"""

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path


def find_last_checkpoint(output_dir: Path) -> Path | None:
    """Find the last checkpoint directory in the output."""
    checkpoints_dir = output_dir / "checkpoints"
    last_link = checkpoints_dir / "last"
    if last_link.exists():
        # 'last' is a symlink or directory pointing to the latest checkpoint
        return last_link / "pretrained_model"
    
    # Fallback: find the highest numbered checkpoint
    if not checkpoints_dir.exists():
        return None
    
    checkpoint_dirs = sorted(
        [d for d in checkpoints_dir.iterdir() if d.is_dir() and d.name.isdigit()],
        key=lambda d: int(d.name)
    )
    if checkpoint_dirs:
        return checkpoint_dirs[-1] / "pretrained_model"
    return None


def run_training(args: list[str], phase_name: str) -> None:
    """Run a training phase as a subprocess."""
    cmd = [sys.executable] + args
    logging.info(f"{'='*60}")
    logging.info(f"Starting {phase_name}")
    logging.info(f"Command: {' '.join(cmd)}")
    logging.info(f"{'='*60}")
    
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        logging.error(f"{phase_name} failed with return code {result.returncode}")
        sys.exit(result.returncode)
    
    logging.info(f"✅ {phase_name} complete!")


def parse_episodes(spec: str | None) -> list[int] | None:
    """Parse an episode specification string.
    
    Supports:
        None          -> None (all episodes)
        '[0,1,2,3]'   -> [0, 1, 2, 3]
        '0,1,2,3'     -> [0, 1, 2, 3]
        '0-9'         -> [0, 1, 2, ..., 9]
        '0-4,10-14'   -> [0, 1, 2, 3, 4, 10, 11, 12, 13, 14]
    """
    if spec is None:
        return None
    
    spec = spec.strip().strip("[]")
    episodes = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-", 1)
            episodes.extend(range(int(start), int(end) + 1))
        else:
            episodes.append(int(part))
    return sorted(set(episodes))


def main():
    parser = argparse.ArgumentParser(
        description="Pretrain on crowd data, then finetune on expert data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    # Required arguments
    parser.add_argument(
        "--pretrain-dataset",
        required=True,
        help="Full path to crowd/pretrain dataset",
    )
    parser.add_argument(
        "--finetune-dataset",
        required=True,
        help="Full path to expert/finetune dataset",
    )
    parser.add_argument(
        "--pretrain-steps",
        type=int,
        required=True,
        help="Number of training steps for pretraining phase",
    )
    parser.add_argument(
        "--finetune-steps",
        type=int,
        required=True,
        help="Number of training steps for finetuning phase",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Base output directory (pretrain goes to <dir>/pretrain, finetune to <dir>/finetune)",
    )
    
    # Optional arguments
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--save-freq", type=int, default=2000)
    parser.add_argument("--log-freq", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--pretrain-episodes",
        type=str,
        default=None,
        help="Episodes to use for pretraining, e.g. '[0,1,2,3,4]' or '0-9' (default: all)",
    )
    parser.add_argument(
        "--finetune-episodes",
        type=str,
        default=None,
        help="Episodes to use for finetuning, e.g. '[0,1,2,3,4]' or '0-9' (default: all)",
    )
    parser.add_argument(
        "--policy-type",
        default="flexible_diffusion",
        help="Policy type (default: flexible_diffusion)",
    )
    
    # Extra args to forward to train.py
    parser.add_argument(
        "extra_args",
        nargs="*",
        help="Additional args to pass to train.py (e.g. --policy.horizon=16)",
    )
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    
    # Resolve the train.py script path
    script_dir = Path(__file__).resolve().parent.parent
    train_script = script_dir / "external" / "lerobot_trossen" / "lerobot" / "scripts" / "train.py"
    if not train_script.exists():
        logging.error(f"Could not find train.py at {train_script}")
        sys.exit(1)
    
    output_dir = Path(args.output_dir)
    pretrain_dir = output_dir / "pretrain"
    finetune_dir = output_dir / "finetune"
    
    # =========================================================================
    # PHASE 1: PRETRAIN on crowd data
    # =========================================================================
    pretrain_episodes = parse_episodes(args.pretrain_episodes)
    finetune_episodes = parse_episodes(args.finetune_episodes)
    
    pretrain_args = [
        str(train_script),
        f"--dataset.repo_id={args.pretrain_dataset}",
        f"--policy.type={args.policy_type}",
        f"--output_dir={pretrain_dir}",
        f"--steps={args.pretrain_steps}",
        f"--batch_size={args.batch_size}",
        f"--save_freq={args.save_freq}",
        f"--log_freq={args.log_freq}",
        f"--seed={args.seed}",
        f"--num_workers={args.num_workers}",
        "--eval_freq=0",  # No eval during pretrain (no sim env)
        "--save_checkpoint=true",
    ]
    if pretrain_episodes is not None:
        pretrain_args.append(f"--dataset.episodes={pretrain_episodes}")
    pretrain_args.extend(args.extra_args)
    
    run_training(pretrain_args, "Phase 1: PRETRAIN on crowd data")
    
    # =========================================================================
    # Find pretrain checkpoint
    # =========================================================================
    pretrain_checkpoint = find_last_checkpoint(pretrain_dir)
    if pretrain_checkpoint is None:
        logging.error(f"No checkpoint found in {pretrain_dir}")
        sys.exit(1)
    
    logging.info(f"Using pretrain checkpoint: {pretrain_checkpoint}")
    
    # =========================================================================
    # PHASE 2: FINETUNE on expert data
    # =========================================================================
    finetune_args = [
        str(train_script),
        f"--dataset.repo_id={args.finetune_dataset}",
        f"--policy.path={pretrain_checkpoint}",
        f"--output_dir={finetune_dir}",
        f"--steps={args.finetune_steps}",
        f"--batch_size={args.batch_size}",
        f"--save_freq={args.save_freq}",
        f"--log_freq={args.log_freq}",
        f"--seed={args.seed}",
        f"--num_workers={args.num_workers}",
        "--eval_freq=0",
        "--save_checkpoint=true",
    ]
    if finetune_episodes is not None:
        finetune_args.append(f"--dataset.episodes={finetune_episodes}")
    finetune_args.extend(args.extra_args)
    
    run_training(finetune_args, "Phase 2: FINETUNE on expert data")
    
    # =========================================================================
    # Summary
    # =========================================================================
    finetune_checkpoint = find_last_checkpoint(finetune_dir)
    logging.info(f"\n{'='*60}")
    logging.info("🎉 Pretrain → Finetune pipeline complete!")
    logging.info(f"  Pretrain: {args.pretrain_steps} steps on {args.pretrain_dataset}")
    logging.info(f"  Finetune: {args.finetune_steps} steps on {args.finetune_dataset}")
    logging.info(f"  Final checkpoint: {finetune_checkpoint}")
    logging.info(f"{'='*60}")


if __name__ == "__main__":
    main()
