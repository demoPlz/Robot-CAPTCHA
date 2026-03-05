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
  --robot.use_depth_main_camera=true \
  --robot.max_relative_target=null \
  --control.type=record \
  --control.fps=30 \
  --control.single_task="Pour the content of the red cup into the teal container" \
  --task-name=pour \
  --control.repo_id=pour/pour_10_phase1_mar4 \
  --control.data_collection_policy_repo_id=pour/pour_10_phase1_mar4_dcp \
  --control.tags='["tutorial"]' \
  --control.warmup_time_s=5 \
  --control.num_episodes=10 \
  --control.push_to_hub=false \
  --control.num_image_writer_processes=8 \
  --control.play_sound=false \
  --show-demo-videos \
  --robot.use_depth_main_camera=true \
  "$@"
