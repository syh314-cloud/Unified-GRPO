set -x
TRAIN_TEST_SPLIT=${TRAIN_TEST_SPLIT:-navtest}
export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-$HOME/navsim_workspace/dataset}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-$OPENSCENE_DATA_ROOT/maps}"
export NAVSIM_DEVKIT_ROOT="${NAVSIM_DEVKIT_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-$NAVSIM_DEVKIT_ROOT}"
export DRIVELAW_VIDEO_ROOT="${DRIVELAW_VIDEO_ROOT:-$HOME/navsim_workspace/weights/DriveLaW-Video}"
export DRIVELAW_ACT_CKPT="${DRIVELAW_ACT_CKPT:-$DRIVELAW_VIDEO_ROOT/transformer/diffusion_pytorch_model.safetensors}"
export RESOURCE_GPU="${RESOURCE_GPU:-1}"
export WORLD_SIZE="${WORLD_SIZE:-1}"
export RANK="${RANK:-0}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29500}"
# Ensure navsim package is importable (fixes "Error locating target VideoDriveAgent")
export PYTHONPATH="${NAVSIM_DEVKIT_ROOT}:${PYTHONPATH}"
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=0
export NCCL_TIMEOUT=1800000

if [ -z "${VIDEODRIVE_CONFIG:-}" ]; then
    mkdir -p "$NAVSIM_EXP_ROOT/generated_configs"
    VIDEODRIVE_CONFIG="$NAVSIM_EXP_ROOT/generated_configs/video_model_infer_navsim_eval.local.yaml"
    sed \
        -e "s#/path/to/DriveLaW-Video#$DRIVELAW_VIDEO_ROOT#g" \
        -e "s#path/to/DriveLaW-Act#$NAVSIM_DEVKIT_ROOT#g" \
        -e "s#/path/to/DriveLaW-Act#$NAVSIM_DEVKIT_ROOT#g" \
        -e "s#path/to/DriveLaW-Video#$DRIVELAW_VIDEO_ROOT#g" \
        -e "s#model_path: .*#model_path: $DRIVELAW_ACT_CKPT#g" \
        "$NAVSIM_DEVKIT_ROOT/navsim/agents/videodrive/configs/ltx_model/video_model_infer_navsim_eval.yaml" \
        > "$VIDEODRIVE_CONFIG"
fi

torchrun --nproc_per_node=$RESOURCE_GPU \
         --nnodes=$WORLD_SIZE \
         --node_rank=$RANK \
         --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score_videodrive.py \
    train_test_split=$TRAIN_TEST_SPLIT \
    agent=videodrive_agent \
    agent.config_file="$VIDEODRIVE_CONFIG" \
    cache_path="${CACHE_PATH:-$NAVSIM_EXP_ROOT/videodrive_agent_cache_dir_final_front}" \
    use_cache_without_dataset=False \
    experiment_name=videodrive_agent_eval 
