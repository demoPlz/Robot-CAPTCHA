#!/bin/bash
# Quick launch script for record data collection

python backend/collect_data.py \
  --robot.type=trossen_ai_single_arm \
  --robot.max_relative_target=null \
  --control.type=record \
  --control.fps=30 \
  --control.single_task="Put the objects on the desk into the middle drawer" \
  --task-name=drawer \
  --control.repo_id=$USER/test_1 \
  --control.data_collection_policy_repo_id=$USER/test_1_dcp \
  --control.tags='["tutorial"]' \
  --control.warmup_time_s=5 \
  --control.num_episodes=2 \
  --control.push_to_hub=false \
  --control.num_image_writer_processes=8 \
  --control.play_sound=false \
  --show-demo-videos \
  "$@"
