#!/bin/bash
#SBATCH -J idm_s1_smoke
#SBATCH -p gpu_a800
#SBATCH -o /data/run01/scyb955/sunyuhan/projects/SD/DriveLaW-Act/idm_smoke.%j.out
#SBATCH -e /data/run01/scyb955/sunyuhan/projects/SD/DriveLaW-Act/idm_smoke.%j.err
#SBATCH -t 00:30:00
#SBATCH --gres=gpu:1

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
export OMP_NUM_THREADS=4

cd /data/run01/scyb955/sunyuhan/projects/SD/DriveLaW-Act

torchrun --standalone --nproc_per_node=1 \
  navsim/agents/videodrive/run_training_idm_stage1.py \
  --cache-path /data/run01/scyb955/sunyuhan/projects/SD/DriveLaW-Act/videodrive_agent_cache_dir_final_front \
  --output-dir /data/run01/scyb955/sunyuhan/projects/exp/idm_stage1_smoke \
  --tensorboard-dir /data/run01/scyb955/sunyuhan/projects/exp/idm_stage1_smoke/tensorboard \
  --batch-size 1 \
  --epochs 1 \
  --lr 1e-4 \
  --weight-decay 1e-2 \
  --num-workers 2 \
  --interval-length 0.5 \
  --num-waypoints 8 \
  --lambda-pos 1.0 \
  --lambda-yaw 0.2 \
  --max-train-steps 2 \
  --max-val-steps 1 \
  --keep-last 1 \
  --save-every 1 \
  --xvo-root "${XVO_ROOT}" \
  --flownet-root "${XVO_FLOWNET_ROOT}" \
  --xvo-checkpoint-path "${XVO_CHECKPOINT_PATH}"
