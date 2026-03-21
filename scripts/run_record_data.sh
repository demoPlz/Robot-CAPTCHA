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

TASK_NAME=$(python -c "import sys; sys.path.append('backend'); from crowd_interface_config import CrowdInterfaceConfig; print(CrowdInterfaceConfig().task_name)")
TASK_TEXT=$(python -c "import sys; sys.path.append('backend'); from crowd_interface_config import CrowdInterfaceConfig; print(CrowdInterfaceConfig().task_text)")

echo "📌 Auto-detected Task Name: ${TASK_NAME}"
echo "📝 Auto-detected Task Text: ${TASK_TEXT}"
echo ""

python backend/collect_data.py \
  --robot.type=trossen_ai_single_arm \
  --robot.use_depth_main_camera=true \
  --robot.max_relative_target=null \
  --control.type=record \
  --control.fps=30 \
  --control.single_task="${TASK_TEXT}" \
  --task-name=${TASK_NAME} \
  --control.repo_id=${TASK_NAME}/${TASK_NAME}_30_mar20 \
  --control.data_collection_policy_repo_id=${TASK_NAME}/${TASK_NAME}_30_mar20_dcp \
  --control.tags='["tutorial"]' \
  --control.warmup_time_s=5 \
  --control.num_episodes=30 \
  --control.push_to_hub=false \
  --control.num_image_writer_processes=8 \
  --control.play_sound=false \
  --show-demo-videos \
  --robot.use_depth_main_camera=true \
  "$@"
