#!/bin/bash
#SBATCH -J idm_s1_f0
#SBATCH -p gpu_a800
#SBATCH -o /data/run01/scyb955/sunyuhan/projects/SD/DriveLaW-Act/idm_s1_f0.%j.out
#SBATCH -e /data/run01/scyb955/sunyuhan/projects/SD/DriveLaW-Act/idm_s1_f0.%j.err
#SBATCH -t 08:00:00
#SBATCH --gres=gpu:3

set -euo pipefail

source /data/apps/miniforge3/25.11.0-1/etc/profile.d/conda.sh
conda activate xvo118_pip
module load cuda/11.8

export CUDA_HOME=/data/apps/cuda/11.8
export PATH="${CUDA_HOME}/bin:${PATH}"
export PYTHONPATH="/data/run01/scyb955/sunyuhan/projects/SD/XVO-main:/data/run01/scyb955/sunyuhan/projects/SD/DriveLaW-Act:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="/data/run01/scyb955/.conda/envs/xvo118_pip/lib/python3.9/site-packages/torch/lib:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

export XVO_ROOT=/data/run01/scyb955/sunyuhan/projects/SD/XVO-main
export XVO_FLOWNET_ROOT=/data/run01/scyb955/sunyuhan/projects/models/DriveLaW_weights/DriveLaW-Act/init_weights
export XVO_CHECKPOINT_PATH=/data/run01/scyb955/sunyuhan/projects/models/DriveLaW_weights/DriveLaW-Act/saved_models/xvo_complete_kitti_sl_b8_lr0005/checkpoint-2.pt

# The cluster assigns CPUs automatically with each requested GPU (CpusPerTres=gpu:8).
# Do not add #SBATCH -c: its submission Lua hook rejects that explicit CPU request.
export OMP_NUM_THREADS=4
export NCCL_ASYNC_ERROR_HANDLING=1

cd /data/run01/scyb955/sunyuhan/projects/SD/DriveLaW-Act

torchrun --standalone --nproc_per_node=3 \
  navsim/agents/videodrive/run_training_idm_stage1.py \
  --cache-path /data/run01/scyb955/sunyuhan/projects/SD/DriveLaW-Act/videodrive_agent_cache_dir_final_front \
  --output-dir /data/run01/scyb955/sunyuhan/projects/exp/idm_stage1_f0_speed \
  --tensorboard-dir /data/run01/scyb955/sunyuhan/projects/exp/idm_stage1_f0_speed/tensorboard \
  --batch-size 8 \
  --epochs 50 \
  --lr 1e-4 \
  --weight-decay 1e-2 \
  --num-workers 4 \
  --interval-length 0.5 \
  --num-waypoints 8 \
  --lambda-pos 1.0 \
  --lambda-yaw 0.2 \
  --lambda-speed 0.1 \
  --keep-last 3 \
  --save-every 1 \
  --resume auto \
  --xvo-root "${XVO_ROOT}" \
  --flownet-root "${XVO_FLOWNET_ROOT}" \
  --xvo-checkpoint-path "${XVO_CHECKPOINT_PATH}"
