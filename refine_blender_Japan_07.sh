#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
RENDER_CONFIG="${RENDER_CONFIG:-./config/config_RealTime_render_1_Japan_07.json}"
POSE_PRIOR="${POSE_PRIOR:-./data/UAVD4L-LoD/Japan_07/GPS_pose_new_all.txt}"
PT_BASE_PATH="${PT_BASE_PATH:-./Ins_data/Japan_07/PT_640_360_09091800/conf_0.3}"

"$PYTHON_BIN" refine_pose_realtime_area.py \
  --render_config "$RENDER_CONFIG" \
  --sampler rand_yaw_or_pitch \
  --name Japan_07 \
  --pose_prior "$POSE_PRIOR" \
  --pt_base_path "$PT_BASE_PATH"

"$PYTHON_BIN" refine_pose_realtime_score.py \
  --render_config "$RENDER_CONFIG" \
  --sampler rand_yaw_or_pitch \
  --name Japan_07 \
  --pose_prior "$POSE_PRIOR" \
  --pt_base_path "$PT_BASE_PATH"
