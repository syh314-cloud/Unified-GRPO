TRAIN_TEST_SPLIT=navtrain
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/data/home/scyb955/run/sunyuhan/navsim_workspace/dataset/maps"
export NAVSIM_EXP_ROOT="/data/home/scyb955/run/sunyuhan/projects/SD/DriveLaW-Act"
export NAVSIM_DEVKIT_ROOT="/data/home/scyb955/run/sunyuhan/projects/SD/DriveLaW-Act"
export OPENSCENE_DATA_ROOT="/data/home/scyb955/run/sunyuhan/navsim_workspace/dataset"
export SENSOR_BLOBS_TRAINVAL="/data/home/scyb955/run/sunyuhan/navsim_workspace/navsim/download/trainval_sensor_blobs/trainval"
export NAVSIM_LOGS_TRAINVAL="/data/home/scyb955/run/sunyuhan/navsim_workspace/navsim/download/trainval_navsim_logs/trainval"
CACHE_PATH=$NAVSIM_EXP_ROOT/videodrive_agent_cache_dir_final_front

export HYDRA_FULL_ERROR=1
export PYTHONPATH="${NAVSIM_DEVKIT_ROOT}:${PYTHONPATH}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"

export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=0
export NCCL_TIMEOUT=1800000

# 单卡 cache 用 python 即可，错误信息更完整（不要用 torchrun）
python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_dataset_caching.py \
    agent=videodrive_agent \
    agent.config_file="${NAVSIM_DEVKIT_ROOT}/navsim/agents/videodrive/configs/ltx_model/video_model_infer_navsim.yaml" \
    experiment_name=videodrive_agent_cache \
    train_test_split=$TRAIN_TEST_SPLIT \
    cache_path=$CACHE_PATH \
    sensor_blobs_path=$SENSOR_BLOBS_TRAINVAL \
    navsim_log_path=$NAVSIM_LOGS_TRAINVAL \
    worker.threads_per_node=2
