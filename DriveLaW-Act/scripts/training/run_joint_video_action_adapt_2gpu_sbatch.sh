#!/bin/bash
#SBATCH -J dlw_vact_2g
#SBATCH -p gpu_a800
#SBATCH --gres=gpu:2
#SBATCH -t 3-00:00:00
#SBATCH -o /data/home/scyb955/slurm_logs/dlw_vact_2g.%j.out
#SBATCH -e /data/home/scyb955/slurm_logs/dlw_vact_2g.%j.err

set -euo pipefail

REPO=/data/run01/scyb955/sunyuhan/projects/SD/DriveLaW-Act
OUTPUT_ROOT=/data/home/scyb955/run/sunyuhan/projects/exp/drivelaw_joint_video_action_adapt
CACHE_PATH=/data/run01/scyb955/sunyuhan/projects/SD/DriveLaW-Act/videodrive_agent_cache_dir_final_front
CONFIG_PATH=${REPO}/navsim/agents/videodrive/configs/ltx_model/video_model_joint_video_action_adapt.yaml

mkdir -p /data/home/scyb955/slurm_logs "${OUTPUT_ROOT}"

source /data/apps/miniforge3/25.11.0-1/etc/profile.d/conda.sh
conda activate drivelaw_video
module load cuda/12.4

export NAVSIM_DEVKIT_ROOT="${REPO}"
export NAVSIM_EXP_ROOT=/data/home/scyb955/run/sunyuhan/projects/exp
export OPENSCENE_DATA_ROOT=/data/home/scyb955/run/sunyuhan/navsim_workspace/dataset
export NUPLAN_MAPS_ROOT=${OPENSCENE_DATA_ROOT}/maps
export NUPLAN_MAP_VERSION=nuplan-maps-v1.0
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=4
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TOKENIZERS_PARALLELISM=false

cd "${REPO}"

torchrun --standalone --nproc_per_node=2 \
  navsim/agents/videodrive/run_training_joint_video_action.py \
  agent=videodrive_agent \
  agent.config_file="${CONFIG_PATH}" \
  experiment_name=drivelaw_joint_video_action_adapt \
  train_test_split=navtrain \
  cache_path="${CACHE_PATH}" \
  output_dir="${OUTPUT_ROOT}" \
  use_cache_without_dataset=true \
  use_deepspeed=true \
  dataloader.params.batch_size=1 \
  dataloader.params.num_workers=4 \
  gradient_accumulation_steps=1
