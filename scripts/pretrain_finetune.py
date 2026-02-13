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

    # Start a new run
    python scripts/pretrain_finetune.py \
        --pretrain-dataset /path/to/crowd_for_pretrain \
        --finetune-dataset /path/to/expert_for_finetune \
        --pretrain-steps 100000 \
        --finetune-steps 50000 \
        --output-dir outputs/my_run

    # Resume an interrupted run (only --output-dir needed)
    python scripts/pretrain_finetune.py --resume --output-dir outputs/my_run
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


def get_checkpoint_step(output_dir: Path) -> int | None:
    """Get the training step from the last checkpoint, or None if no checkpoint exists."""
    checkpoint = find_last_checkpoint(output_dir)
    if checkpoint is None:
        return None
    
    # training_step.json is in the training_state/ sibling of pretrained_model/
    training_step_file = checkpoint.parent / "training_state" / "training_step.json"
    if training_step_file.exists():
        with open(training_step_file) as f:
            data = json.load(f)
        return data.get("training_step", None)
    return None


def load_train_config(output_dir: Path) -> dict | None:
    """Load train_config.json from the last checkpoint of a phase directory."""
    checkpoint = find_last_checkpoint(output_dir)
    if checkpoint is None:
        return None
    config_file = checkpoint / "train_config.json"
    if config_file.exists():
        with open(config_file) as f:
            return json.load(f)
    return None


def phase_is_complete(output_dir: Path, target_steps: int) -> bool:
    """Check if a training phase has completed (reached target steps)."""
    step = get_checkpoint_step(output_dir)
    return step is not None and step >= target_steps


def phase_is_started(output_dir: Path) -> bool:
    """Check if a training phase has started (has any checkpoint)."""
    return find_last_checkpoint(output_dir) is not None


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
    
    # These are required for new runs, but optional when --resume is used
    parser.add_argument(
        "--pretrain-dataset",
        help="Full path to crowd/pretrain dataset (required for new runs)",
    )
    parser.add_argument(
        "--finetune-dataset",
        help="Full path to expert/finetune dataset (required for new runs)",
    )
    parser.add_argument(
        "--pretrain-steps",
        type=int,
        help="Number of training steps for pretraining phase (required for new runs)",
    )
    parser.add_argument(
        "--finetune-steps",
        type=int,
        help="Number of training steps for finetuning phase (required for new runs)",
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
        "--resume",
        action="store_true",
        help="Resume from where training left off. Only --output-dir is required.",
    )
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
    # RESUME MODE: recover config from saved checkpoints
    # =========================================================================
    if args.resume:
        # Try to recover missing args from saved train_config.json
        pretrain_cfg = load_train_config(pretrain_dir)
        finetune_cfg = load_train_config(finetune_dir)
        
        if args.pretrain_dataset is None:
            if pretrain_cfg:
                args.pretrain_dataset = pretrain_cfg["dataset"]["repo_id"]
            elif finetune_cfg:
                # Finetune started but pretrain config not found — not critical,
                # pretrain is already done if finetune has started
                args.pretrain_dataset = "(completed)"
            else:
                parser.error("--resume: no checkpoints found in --output-dir. Cannot recover config.")
        
        if args.finetune_dataset is None:
            if finetune_cfg:
                args.finetune_dataset = finetune_cfg["dataset"]["repo_id"]
            elif pretrain_cfg:
                # Pretrain done, finetune not started yet — need finetune dataset
                parser.error("--resume: finetune hasn't started yet. Provide --finetune-dataset.")
        
        if args.pretrain_steps is None:
            if pretrain_cfg:
                args.pretrain_steps = pretrain_cfg["steps"]
            else:
                # If pretrain is done (finetune started), we just need any value >= checkpoint step
                step = get_checkpoint_step(pretrain_dir)
                if step is not None:
                    args.pretrain_steps = step
                else:
                    parser.error("--resume: cannot determine --pretrain-steps. Provide it explicitly.")
        
        if args.finetune_steps is None:
            if finetune_cfg:
                args.finetune_steps = finetune_cfg["steps"]
            else:
                parser.error("--resume: cannot determine --finetune-steps. Provide it explicitly.")
        
        logging.info("Resuming pretrain_finetune pipeline...")
        logging.info(f"  Output dir: {output_dir}")
        logging.info(f"  Pretrain dataset: {args.pretrain_dataset}")
        logging.info(f"  Finetune dataset: {args.finetune_dataset}")
        logging.info(f"  Pretrain steps: {args.pretrain_steps}")
        logging.info(f"  Finetune steps: {args.finetune_steps}")
    else:
        # New run: validate all required args are present
        missing = []
        if args.pretrain_dataset is None:
            missing.append("--pretrain-dataset")
        if args.finetune_dataset is None:
            missing.append("--finetune-dataset")
        if args.pretrain_steps is None:
            missing.append("--pretrain-steps")
        if args.finetune_steps is None:
            missing.append("--finetune-steps")
        if missing:
            parser.error(f"the following arguments are required: {', '.join(missing)}")
    
    pretrain_episodes = parse_episodes(args.pretrain_episodes)
    finetune_episodes = parse_episodes(args.finetune_episodes)
    
    # =========================================================================
    # PHASE 1: PRETRAIN on crowd data
    # =========================================================================
    if phase_is_complete(pretrain_dir, args.pretrain_steps):
        logging.info(f"Phase 1 already complete ({args.pretrain_steps} steps). Skipping pretrain.")
    elif args.resume and phase_is_started(pretrain_dir):
        pretrain_step = get_checkpoint_step(pretrain_dir)
        logging.info(f"Resuming Phase 1 from step {pretrain_step}...")
        pretrain_checkpoint = find_last_checkpoint(pretrain_dir)
        pretrain_args = [
            str(train_script),
            f"--config_path={pretrain_checkpoint}",
            "--resume=true",
        ]
        run_training(pretrain_args, "Phase 1: PRETRAIN on crowd data (RESUMED)")
    else:
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
            "--eval_freq=0",
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
    if phase_is_complete(finetune_dir, args.finetune_steps):
        logging.info(f"Phase 2 already complete ({args.finetune_steps} steps). Skipping finetune.")
    elif args.resume and phase_is_started(finetune_dir):
        finetune_step = get_checkpoint_step(finetune_dir)
        logging.info(f"Resuming Phase 2 from step {finetune_step}...")
        finetune_checkpoint = find_last_checkpoint(finetune_dir)
        finetune_args = [
            str(train_script),
            f"--config_path={finetune_checkpoint}",
            "--resume=true",
        ]
        run_training(finetune_args, "Phase 2: FINETUNE on expert data (RESUMED)")
    else:
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
    if finetune_checkpoint:
        logging.info(f"  Final checkpoint: {finetune_checkpoint}")
    logging.info(f"{'='*60}")


if __name__ == "__main__":
    main()
