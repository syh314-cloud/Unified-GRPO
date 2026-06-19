#!/bin/bash
#SBATCH -J drivelaw_hcache
#SBATCH -o hcache.%j.out
#SBATCH -e hcache.%j.err
#SBATCH -t 3-00:00:00

source ~/.bashrc
conda activate drivelaw_act310
export PYTHONNOUSERSITE=1
export RESOURCE_GPU=1
export WORLD_SIZE=1
export RANK=0
export MASTER_ADDR=localhost
export MASTER_PORT=29500

cd /data/home/scyb955/run/sunyuhan/projects/SD/DriveLaW-Act
bash scripts/evaluation/run_caching_videodrive_hidden_state.sh
