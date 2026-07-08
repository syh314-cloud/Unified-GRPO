export NAVSIM_DEVKIT_ROOT="/path/to/DriveLaW-Act"
export NAVSIM_EXP_ROOT="/path/to/DriveLaW-Act/experiments"

CACHE_PATH="/path/to/DriveLaW-Act/videodrive_agent_cache_dir_final_front"
OUTPUT_DIR="${NAVSIM_EXP_ROOT}/idm_stage1"
SPLIT_YAML="${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/config/training/default_train_val_test_log_split.yaml"

python "${NAVSIM_DEVKIT_ROOT}/navsim/agents/videodrive/run_training_idm_stage1.py" \
  --cache-path "${CACHE_PATH}" \
  --split-yaml "${SPLIT_YAML}" \
  --output-dir "${OUTPUT_DIR}" \
  --batch-size 8 \
  --epochs 20 \
  --lr 1e-4 \
  --weight-decay 1e-2 \
  --num-workers 4 \
  --interval-length 0.5 \
  --num-waypoints 8 \
  --lambda-pos 1.0 \
  --lambda-yaw 0.2 \
  --keep-last 3 \
  --save-every 1 \
  --resume auto
