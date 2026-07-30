TRAIN_TEST_SPLIT=${TRAIN_TEST_SPLIT:-navtrain}
export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-$HOME/navsim_workspace/dataset}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-$OPENSCENE_DATA_ROOT/maps}"
export NAVSIM_DEVKIT_ROOT="${NAVSIM_DEVKIT_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-$NAVSIM_DEVKIT_ROOT}"
export DRIVELAW_VIDEO_ROOT="${DRIVELAW_VIDEO_ROOT:-$HOME/navsim_workspace/weights/DriveLaW-Video}"
export DRIVELAW_ACT_CKPT="${DRIVELAW_ACT_CKPT:-$DRIVELAW_VIDEO_ROOT/transformer/diffusion_pytorch_model.safetensors}"
export SENSOR_BLOBS_TRAINVAL="${SENSOR_BLOBS_TRAINVAL:-$OPENSCENE_DATA_ROOT/sensor_blobs/trainval}"
export NAVSIM_LOGS_TRAINVAL="${NAVSIM_LOGS_TRAINVAL:-$OPENSCENE_DATA_ROOT/navsim_logs/trainval}"
CACHE_PATH=${CACHE_PATH:-$NAVSIM_EXP_ROOT/videodrive_agent_cache_dir_final_front}

export HYDRA_FULL_ERROR=1
export PYTHONPATH="${NAVSIM_DEVKIT_ROOT}:${PYTHONPATH}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"

export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=0
export NCCL_TIMEOUT=1800000

if [ -z "${VIDEODRIVE_CONFIG:-}" ]; then
    mkdir -p "$NAVSIM_EXP_ROOT/generated_configs"
    VIDEODRIVE_CONFIG="$NAVSIM_EXP_ROOT/generated_configs/video_model_infer_navsim.local.yaml"
    sed \
        -e "s#/data/home/scyb955/run/sunyuhan/projects/models/DriveLaW_weights/DriveLaW-Video#$DRIVELAW_VIDEO_ROOT#g" \
        -e "s#path/to/DriveLaW-Act#$NAVSIM_DEVKIT_ROOT#g" \
        -e "s#/path/to/DriveLaW-Act#$NAVSIM_DEVKIT_ROOT#g" \
        -e "s#path/to/DriveLaW-Video#$DRIVELAW_VIDEO_ROOT#g" \
        -e "s#model_path: .*#model_path: $DRIVELAW_ACT_CKPT#g" \
        "$NAVSIM_DEVKIT_ROOT/navsim/agents/videodrive/configs/ltx_model/video_model_infer_navsim.yaml" \
        > "$VIDEODRIVE_CONFIG"
fi

# 单卡 cache 用 python 即可，错误信息更完整（不要用 torchrun）
python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_dataset_caching.py \
    agent=videodrive_agent \
    agent.config_file="$VIDEODRIVE_CONFIG" \
    experiment_name=videodrive_agent_cache \
    train_test_split=$TRAIN_TEST_SPLIT \
    cache_path=$CACHE_PATH \
    sensor_blobs_path=$SENSOR_BLOBS_TRAINVAL \
    navsim_log_path=$NAVSIM_LOGS_TRAINVAL \
    worker.threads_per_node=2
