#!/bin/bash
#SBATCH -J drivelaw_hcache
#SBATCH -o hcache.%j.out
#SBATCH -e hcache.%j.err
#SBATCH -t 3-00:00:00

source /data/apps/miniforge3/25.11.0-1/etc/profile.d/conda.sh
conda activate drivelaw_act310
export HYDRA_FULL_ERROR=1
export PYTHONPATH="/data/home/scyb955/run/sunyuhan/projects/SD/DriveLaW-Act:${PYTHONPATH}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"
export RESOURCE_GPU=1
export WORLD_SIZE=1
export RANK=0
export MASTER_ADDR=localhost
export MASTER_PORT=29500

ray stop --force 2>/dev/null || true
sleep 3

cd /data/home/scyb955/run/sunyuhan/projects/SD/DriveLaW-Act
bash scripts/evaluation/run_caching_videodrive_hidden_state.sh
