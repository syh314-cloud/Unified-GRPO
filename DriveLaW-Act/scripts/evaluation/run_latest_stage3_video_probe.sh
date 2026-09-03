#!/usr/bin/env bash
# Run one full decoded-video probe from the newest completed Stage-3 checkpoint.
set -euo pipefail

REPO=${REPO:?Set REPO to the absolute DriveLaW-Act path.}
OPENSCENE_DATA_ROOT=${OPENSCENE_DATA_ROOT:?Set OPENSCENE_DATA_ROOT to the dataset root.}
DRIVELAW_VIDEO_ROOT=${DRIVELAW_VIDEO_ROOT:?Set DRIVELAW_VIDEO_ROOT to Stage-2 DriveLaW-Video.}
TRAIN_OUTPUT_ROOT=${TRAIN_OUTPUT_ROOT:?Set TRAIN_OUTPUT_ROOT to the Stage-3 training output.}
PROBE_LIMIT=${PROBE_LIMIT:-1}
TRAIN_TEST_SPLIT=${TRAIN_TEST_SPLIT:-navtrain}

CKPT=$(find "$TRAIN_OUTPUT_ROOT" -type f \
  -path '*/step_*/diffusion_pytorch_model.safetensors' \
  -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)

if [ -z "$CKPT" ]; then
  echo "No completed Stage-3 checkpoint found below: $TRAIN_OUTPUT_ROOT" >&2
  echo "The first checkpoint is written at step_1000." >&2
  exit 1
fi

STEP_DIR=$(dirname "$CKPT")
PROBE_OUTPUT_DIR=${PROBE_OUTPUT_DIR:-"$TRAIN_OUTPUT_ROOT/probes/$(basename "$STEP_DIR")"}
CONFIG_SOURCE="$REPO/navsim/agents/videodrive/configs/ltx_model/video_model_infer_navsim.yaml"
CONFIG_PATH="$PROBE_OUTPUT_DIR/video_model_infer.local.yaml"

mkdir -p "$PROBE_OUTPUT_DIR"

sed \
  -e "s#^pretrained_model_name_or_path: .*#pretrained_model_name_or_path: $DRIVELAW_VIDEO_ROOT#" \
  -e "s#^diffusion_model_class_path: .*#diffusion_model_class_path: $REPO/navsim/agents/videodrive/video_models/ltx_models/transformer_ltx.py#" \
  -e "s#^pipeline_class_path: .*#pipeline_class_path: $REPO/navsim/agents/videodrive/video_models/pipeline/pipeline_ltx_condition.py#" \
  -e "s#^  model_path: .*#  model_path: $CKPT#" \
  "$CONFIG_SOURCE" > "$CONFIG_PATH"

export NAVSIM_DEVKIT_ROOT="$REPO"
export NAVSIM_EXP_ROOT="$TRAIN_OUTPUT_ROOT"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export NUPLAN_MAP_VERSION=${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}
export NUPLAN_MAPS_ROOT=${NUPLAN_MAPS_ROOT:-$OPENSCENE_DATA_ROOT/maps}

echo "Using checkpoint: $CKPT"
echo "Writing probe: $PROBE_OUTPUT_DIR"

cd "$REPO"
python navsim/planning/script/run_joint_video_trajectory_probe.py \
  train_test_split="$TRAIN_TEST_SPLIT" \
  experiment_name=stage3_joint_video_probe \
  agent=videodrive_agent \
  agent.config_file="$CONFIG_PATH" \
  +probe.output_dir="$PROBE_OUTPUT_DIR" \
  +probe.limit="$PROBE_LIMIT"
