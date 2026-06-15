TRAIN_TEST_SPLIT=navtest
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/data/home/scyb955/run/sunyuhan/navsim_workspace/dataset/maps"
export NAVSIM_EXP_ROOT="/data/home/scyb955/run/sunyuhan/projects/SD/DriveLaW-Act"
export NAVSIM_DEVKIT_ROOT="/data/home/scyb955/run/sunyuhan/projects/SD/DriveLaW-Act"
export OPENSCENE_DATA_ROOT="/data/home/scyb955/run/sunyuhan/navsim_workspace/dataset"
CACHE_PATH=$NAVSIM_EXP_ROOT/metric_cache
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=0
export NCCL_TIMEOUT=1800000

torchrun --nproc_per_node=$RESOURCE_GPU \
         --nnodes=$WORLD_SIZE \
         --node_rank=$RANK \
         --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_metric_caching.py \
    train_test_split=$TRAIN_TEST_SPLIT \
    cache.cache_path=$CACHE_PATH