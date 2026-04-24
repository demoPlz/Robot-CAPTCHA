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
shift

if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "❌ Error: Checkpoint file not found: $CHECKPOINT_PATH"
    exit 1
fi

# Parse bash-level flags
ARGS=()
for arg in "$@"; do
  if [ "$arg" == "--netlify-backup" ]; then
    export NETLIFY_SITE_ID="c5cc0ab4-1d06-483a-b2b7-c2d744477b16"
    echo "🌐 Using alternate Netlify site: robot-captcha-backup.netlify.app"
  else
    ARGS+=("$arg")
  fi
done

# Auto-cleanup zombie processes from previous runs
echo "🧹 Running cleanup..."
./scripts/cleanup_zombies.sh

echo ""
echo "🚀 Starting Phase 2 (async user labeling only)..."
echo "   Loading checkpoint: $CHECKPOINT_PATH"
echo "   (Flask will auto-select an available port)"
echo "   (Cloudflared tunnel will be started automatically)"
echo ""

# Dynamically read task config from crowd_interface_config.py
TASK_NAME=$(python3 -c "import sys; sys.path.insert(0,'backend'); from crowd_interface_config import CrowdInterfaceConfig; print(CrowdInterfaceConfig().task_name)")
TASK_TEXT=$(python3 -c "import sys; sys.path.insert(0,'backend'); from crowd_interface_config import CrowdInterfaceConfig; print(CrowdInterfaceConfig().task_text)")
echo "   Task: $TASK_NAME ($TASK_TEXT)"

conda run -n csui --no-capture-output python backend/collect_data.py \
  --phase2-only "$CHECKPOINT_PATH" \
  --robot.type=trossen_ai_single_arm \
  --control.type=record \
  --control.fps=30 \
  --control.single_task="${TASK_TEXT}" \
  --task-name=${TASK_NAME} \
  --control.repo_id=${TASK_NAME}/phase2_dummy \
  --control.num_episodes=1 \
  --control.push_to_hub=false \
  --show-demo-videos \
  "${ARGS[@]}"
