#!/bin/bash

set -euo pipefail

source /data/apps/miniforge3/25.11.0-1/etc/profile.d/conda.sh
conda activate drivelaw_video

ROOT=/data/run01/scyb955/sunyuhan/projects/SD/DriveLaW-Act
export NAVSIM_DEVKIT_ROOT="${ROOT}"
export NAVSIM_EXP_ROOT=/data/home/scyb955/run/sunyuhan/projects/exp
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

PROBE_ROOT="${PROBE_ROOT:-${NAVSIM_EXP_ROOT}/joint_video_trajectory_probe_2s_h200_navtrain}"
TRAJECTORY_DIR="${TRAJECTORY_DIR:-${PROBE_ROOT}/trajectories}"
METRIC_CACHE_PATH="${METRIC_CACHE_PATH:-${NAVSIM_EXP_ROOT}/metric_cache}"
TRAJECTORY_KEY="${TRAJECTORY_KEY:-world_model_trajectory}"
OUTPUT_CSV="${OUTPUT_CSV:-${PROBE_ROOT}/pdm_scores_${TRAJECTORY_KEY}.csv}"

cd "${ROOT}"

python navsim/planning/script/run_probe_trajectory_pdm_score.py \
  --trajectory-dir "${TRAJECTORY_DIR}" \
  --metric-cache-path "${METRIC_CACHE_PATH}" \
  --trajectory-key "${TRAJECTORY_KEY}" \
  --output-csv "${OUTPUT_CSV}"
