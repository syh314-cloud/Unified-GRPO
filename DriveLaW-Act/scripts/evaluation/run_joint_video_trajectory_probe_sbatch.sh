#!/bin/bash
#SBATCH -J joint_vtraj
#SBATCH -p gpu_a800
#SBATCH --gres=gpu:1
#SBATCH -t 02:00:00
#SBATCH -o /data/run01/scyb955/sunyuhan/projects/SD/DriveLaW-Act/joint_vtraj.%j.out
#SBATCH -e /data/run01/scyb955/sunyuhan/projects/SD/DriveLaW-Act/joint_vtraj.%j.err

set -euo pipefail

source /data/apps/miniforge3/25.11.0-1/etc/profile.d/conda.sh
conda activate drivelaw_video

ROOT=/data/run01/scyb955/sunyuhan/projects/SD/DriveLaW-Act
export NAVSIM_DEVKIT_ROOT="${ROOT}"
export NAVSIM_EXP_ROOT=/data/home/scyb955/run/sunyuhan/projects/exp
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-/data/home/scyb955/run/sunyuhan/navsim_workspace/dataset}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-${OPENSCENE_DATA_ROOT}/maps}"

cd "${ROOT}"

python navsim/planning/script/run_joint_video_trajectory_probe.py \
  train_test_split="${TRAIN_TEST_SPLIT:-navtrain}" \
  experiment_name=drivelaw_act_raw_decoded_probe \
  agent=videodrive_agent \
  agent.config_file="${ROOT}/navsim/agents/videodrive/configs/ltx_model/video_model_infer_navsim.yaml" \
  +probe.output_dir="${NAVSIM_EXP_ROOT}/drivelaw_act_raw_decoded_probe" \
  +probe.limit=4
