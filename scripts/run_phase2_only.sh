#!/bin/bash
# Quick launch script for Phase 2 only (async user labeling without robot)
#
# Usage:
#   ./scripts/run_phase2_only.sh /path/to/phase1_checkpoint.json [--output-repo-id <repo_id>]
#
# Examples:
#   # Use same repo_id as checkpoint (auto-renames if exists)
#   ./scripts/run_phase2_only.sh ~/.cache/huggingface/lerobot/yilong/demo_dcp/phase1_checkpoint.json
#
#   # Specify custom output repo_id
#   ./scripts/run_phase2_only.sh checkpoint.json --output-repo-id yilong/demo_dcp_run2
#
# The checkpoint is reusable - run Phase 2 multiple times with different --output-repo-id to
# collect multiple datasets from the same Phase 1 states.

set -e

if [ -z "$1" ]; then
    echo "❌ Error: Missing checkpoint path"
    echo ""
    echo "Usage: ./scripts/run_phase2_only.sh <path_to_phase1_checkpoint.json> [--output-repo-id <repo_id>]"
    echo ""
    echo "Examples:"
    echo "  ./scripts/run_phase2_only.sh ~/.cache/huggingface/lerobot/yilong/demo_dcp/phase1_checkpoint.json"
    echo "  ./scripts/run_phase2_only.sh checkpoint.json --output-repo-id yilong/demo_dcp_run2"
    exit 1
fi

CHECKPOINT_PATH="$1"

if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "❌ Error: Checkpoint file not found: $CHECKPOINT_PATH"
    exit 1
fi

# Auto-cleanup zombie processes from previous runs
echo "🧹 Running cleanup..."
./scripts/cleanup_zombies.sh

echo ""
echo "🚀 Starting Phase 2 (async user labeling only)..."
echo "   Loading checkpoint: $CHECKPOINT_PATH"
echo "   (Flask will auto-select an available port)"
echo "   (Cloudflared tunnel will be started automatically)"
echo ""

conda run -n csui --no-capture-output python backend/collect_data.py \
  --phase2-only "$CHECKPOINT_PATH" \
  --robot.type=trossen_ai_single_arm \
  --control.type=record \
  --control.fps=30 \
  --control.single_task="Pour the content of the red cup into the teal container" \
  --task-name=pour \
  --control.repo_id=$USER/phase2_dummy \
  --control.num_episodes=1 \
  --control.push_to_hub=false \
  --show-demo-videos \
  "${@:2}"
