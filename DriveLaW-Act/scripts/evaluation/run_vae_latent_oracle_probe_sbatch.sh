#!/bin/bash
#SBATCH -J vae_oracle
#SBATCH -p gpu_a800
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH -t 00:30:00
#SBATCH -o /data/home/scyb955/slurm_logs/vae_oracle.%j.out
#SBATCH -e /data/home/scyb955/slurm_logs/vae_oracle.%j.err

set -euo pipefail

ROOT=/data/run01/scyb955/sunyuhan/projects/SD/DriveLaW-Act
OUTPUT_DIR=${VAE_ORACLE_OUTPUT_DIR:-/data/home/scyb955/run/sunyuhan/projects/exp/vae_latent_oracle}
CACHE_PATH=/data/run01/scyb955/sunyuhan/projects/SD/DriveLaW-Act/videodrive_agent_cache_dir_final_front
CONFIG_PATH=${ROOT}/navsim/agents/videodrive/configs/ltx_model/video_model_joint_video_action_adapt.yaml

mkdir -p /data/home/scyb955/slurm_logs "${OUTPUT_DIR}"

source /data/apps/miniforge3/25.11.0-1/etc/profile.d/conda.sh
conda activate drivelaw_video
module load cuda/12.4

export NAVSIM_DEVKIT_ROOT="${ROOT}"
export NAVSIM_EXP_ROOT=/data/home/scyb955/run/sunyuhan/projects/exp
export OPENSCENE_DATA_ROOT=/data/home/scyb955/run/sunyuhan/navsim_workspace/dataset
export NUPLAN_MAPS_ROOT=${OPENSCENE_DATA_ROOT}/maps
export NUPLAN_MAP_VERSION=nuplan-maps-v1.0
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false

cd "${ROOT}"

python navsim/planning/script/run_vae_latent_oracle_probe.py \
  agent=videodrive_agent \
  agent.config_file="${CONFIG_PATH}" \
  train_test_split=navtrain \
  cache_path="${CACHE_PATH}" \
  output_dir="${OUTPUT_DIR}" \
  use_deepspeed=false \
  dataloader.params.batch_size=1 \
  dataloader.params.num_workers=0 \
  +probe.output_dir="${OUTPUT_DIR}" \
  +probe.index="${VAE_ORACLE_INDEX:-0}"
