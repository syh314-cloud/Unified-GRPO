#!/usr/bin/env bash
# Paper Stage-3 joint Action training, with the project's real-future video loss.
set -euo pipefail

REPO=${REPO:?Set REPO to the absolute DriveLaW-Act path.}
OPENSCENE_DATA_ROOT=${OPENSCENE_DATA_ROOT:?Set OPENSCENE_DATA_ROOT to the dataset root.}
DRIVELAW_VIDEO_ROOT=${DRIVELAW_VIDEO_ROOT:?Set DRIVELAW_VIDEO_ROOT to Stage-2 DriveLaW-Video.}
CACHE_PATH=${CACHE_PATH:?Set CACHE_PATH to the completed NAVSIM cache.}
OUTPUT_ROOT=${OUTPUT_ROOT:?Set OUTPUT_ROOT for checkpoints and TensorBoard logs.}

RESOURCE_GPU=${RESOURCE_GPU:-8}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-1}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-24}
NUM_WORKERS=${NUM_WORKERS:-4}

if [ ! -f "$DRIVELAW_VIDEO_ROOT/transformer/diffusion_pytorch_model.safetensors" ]; then
  echo "Missing Stage-2 Video transformer under: $DRIVELAW_VIDEO_ROOT" >&2
  exit 1
fi

export NUPLAN_MAP_VERSION=${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}
export NUPLAN_MAPS_ROOT=${NUPLAN_MAPS_ROOT:-$OPENSCENE_DATA_ROOT/maps}
export NAVSIM_DEVKIT_ROOT="$REPO"
export NAVSIM_EXP_ROOT=${NAVSIM_EXP_ROOT:-$REPO}
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}
export TOKENIZERS_PARALLELISM=false

mkdir -p "$OUTPUT_ROOT"
CONFIG="$OUTPUT_ROOT/video_model_stage3_joint_video_action.local.yaml"

sed \
  -e "s#path/to/DriveLaW-Act#$REPO#g" \
  -e "s#path/to/DriveLaW-Video#$DRIVELAW_VIDEO_ROOT#g" \
  -e "s#path/to/output#$OUTPUT_ROOT#g" \
  "$REPO/navsim/agents/videodrive/configs/ltx_model/video_model_stage3_joint_video_action.yaml" \
  > "$CONFIG"

GLOBAL_BATCH=$((RESOURCE_GPU * PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))
echo "Effective global batch: $GLOBAL_BATCH"
if [ "$GLOBAL_BATCH" -ne 192 ]; then
  echo "Warning: paper Stage-3 uses global batch 192." >&2
fi

cd "$REPO"
torchrun --standalone --nproc_per_node="$RESOURCE_GPU" \
  navsim/agents/videodrive/run_training_stage3_joint_video_action.py \
  agent=videodrive_agent \
  agent.config_file="$CONFIG" \
  experiment_name=stage3_joint_video_action \
  train_test_split=navtrain \
  cache_path="$CACHE_PATH" \
  output_dir="$OUTPUT_ROOT" \
  use_cache_without_dataset=true \
  use_deepspeed=true \
  dataloader.params.batch_size="$PER_DEVICE_BATCH_SIZE" \
  dataloader.params.num_workers="$NUM_WORKERS" \
  gradient_accumulation_steps="$GRADIENT_ACCUMULATION_STEPS"
