#!/bin/bash
# Quick launch script for record data collection

# Auto-cleanup zombie processes from previous runs
echo "🧹 Running cleanup..."
./scripts/cleanup_zombies.sh

echo ""
echo "🚀 Starting data collection..."
echo "   (Flask will auto-select an available port)"
echo "   (Cloudflared tunnel will be started automatically)"
echo ""

python backend/collect_data.py \
  --robot.type=trossen_ai_single_arm \
  --robot.max_relative_target=null \
  --control.type=record \
  --control.fps=30 \
  --control.single_task="Put the objects on the desk into the middle drawer" \
  --task-name=drawer \
  --control.repo_id=$USER/async_test \
  --control.data_collection_policy_repo_id=$USER/async_test_dcp \
  --control.tags='["tutorial"]' \
  --control.warmup_time_s=5 \
  --control.num_episodes=1 \
  --control.push_to_hub=false \
  --control.num_image_writer_processes=8 \
  --control.play_sound=false \
  --show-demo-videos \
  "$@"
