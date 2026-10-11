# DriveLaW 项目交接现况

## 1. 当前代码状态

DriveLaW-Act 已恢复到论文代码的初始版本。当前与初始官方版本相比，仅保留 IDM 相关内容：

    DriveLaW-Act/navsim/agents/videodrive/idm_xvo_style.py
    DriveLaW-Act/navsim/agents/videodrive/run_training_idm_stage1.py
    DriveLaW-Act/navsim/agents/videodrive/videodrive_features.py
    DriveLaW-Act/navsim/planning/script/run_generated_video_idm_probe.py
    DriveLaW-Act/scripts/training/run_idm_stage1_training.sh
    DriveLaW-Act/scripts/training/run_idm_stage1_sbatch.sh
    DriveLaW-Act/scripts/training/run_idm_stage1_smoke_sbatch.sh

已移除的改动包括：联合 Video + Action Stage-3 训练、视频 flow/RGB loss、视频训练配置、联合视频 probe、VAE oracle probe、自定义视频 pipeline 改动、DeepSpeed 相关视频训练兼容性改动。

原论文：

- DriveLaW: Unifying Planning and Video Generation in a Latent Driving World，CVPR 2026。
- arXiv：https://arxiv.org/abs/2512.23421
- 官方代码：https://github.com/xiaomi-research/drivelaw
- 官方权重：https://huggingface.co/tz2026/DriveLaW

## 2. IDM 当前状态

IDM 的目标是从视频帧预测 2 Hz、4 秒的 8 点轨迹，用于比较：

    生成视频 -> IDM trajectory
    World Model trajectory
    GT trajectory

IDM 已训练过，但没有完成严格的有效性验证。尚未完成真实视频上的 held-out 轨迹误差评估，也未完成生成视频上的可靠性验证，因此不能用 IDM 输出或相关 PDM 分数证明方法有效。

北京超级云上的 IDM 训练结果：

    /data/home/scyb955/run/sunyuhan/projects/exp/idm_stage1_f0_speed/

其中包含 checkpoints/best.pt、train.log、metrics.jsonl 和 TensorBoard event。该 IDM checkpoint 不在 H20 的当前训练输出目录中。

## 3. H20 上已训练的权重

以下均为本项目训练产生的 Transformer checkpoint，单个 safetensors 文件约 4.1 GB。

### 3.1 首次 Stage-3 联合 Video + Action 训练，完成 44k steps

训练特点：联合 Video + Action；未加入后续的 2 Hz target RGB loss。

    /rdata/sunyuhan/projects/exp/stage3_joint_video_action_4gpu_run2/
      2026_09_15_18_46_29/
      step_1000/ ... step_44000/

最终权重：

    /rdata/sunyuhan/projects/exp/stage3_joint_video_action_4gpu_run2/2026_09_15_18_46_29/step_44000/diffusion_pytorch_model.safetensors

checkpoint 数量：44。目录总大小约 169 GB。

### 3.2 2 Hz target RGB loss smoke test，1 step

训练特点：验证新增 2 Hz target latent 与 RGB loss 路径是否可运行，不是可用模型。

    /rdata/sunyuhan/projects/exp/stage3_joint_video_action_4gpu_2hz_rgb_smoke/
      2026_09_22_09_44_11/step_1/diffusion_pytorch_model.safetensors

checkpoint 数量：1。目录总大小约 3.9 GB。

### 3.3 2 Hz target RGB loss pilot，训练中断于 9k steps

训练特点：4 张 H20、每卡 batch 1、实际 global batch 4；加入 video flow loss、RGB loss 和 Action loss。该 run 因 NCCL 问题中断。

    /rdata/sunyuhan/projects/exp/stage3_joint_video_action_4gpu_2hz_rgb_pilot_b4_44k/
      2026_09_22_10_03_45/
      step_1000/ ... step_9000/

最新权重：

    /rdata/sunyuhan/projects/exp/stage3_joint_video_action_4gpu_2hz_rgb_pilot_b4_44k/2026_09_22_10_03_45/step_9000/diffusion_pytorch_model.safetensors

checkpoint 数量：9。目录总大小约 36 GB。

### 3.4 2 Hz target RGB loss pilot 重跑，完成 44k steps

训练特点：与 3.3 相同；4 张 H20、每卡 batch 1、实际 global batch 4，完成 44k steps、3 epochs。该实验视频输出仍为无结构块状图，未解决视频生成问题。

    /rdata/sunyuhan/projects/exp/stage3_joint_video_action_4gpu_2hz_rgb_pilot_b4_rerun/
      2026_09_27_12_59_57/
      step_1000/ ... step_44000/

最终权重：

    /rdata/sunyuhan/projects/exp/stage3_joint_video_action_4gpu_2hz_rgb_pilot_b4_rerun/2026_09_27_12_59_57/step_44000/diffusion_pytorch_model.safetensors

checkpoint 数量：44。目录总大小约 169 GB。

### 3.5 无 checkpoint 的失败启动目录

以下目录存在，但没有生成任何模型 checkpoint：

    /rdata/sunyuhan/projects/exp/stage3_joint_video_action_4gpu_2hz_rgb_run1/
    /rdata/sunyuhan/projects/exp/stage3_joint_video_action_4gpu_2hz_rgb_run2/

### 3.6 官方下载的初始权重，不是本项目训练产物

    /rdata/sunyuhan/projects/models/DriveLaW_weights/DriveLaW-Video/

该目录是训练时使用的公开 DriveLaW-Video 初始化权重。

## 4. 服务器数据和代码位置

### H20

    NAVSIM 数据：
    /rdata/sunyuhan/navsim_workspace/dataset/

    NAVSIM cache：
    /rdata/sunyuhan/navsim_workspace/cache/videodrive_agent_cache_dir_final_front

    当前代码仓库：
    Unified-GRPO/
    其中 DriveLaW-Act/ 为论文代码加 IDM 模块。

### 北京超级云

    DriveLaW-Act 代码：
    /data/run01/scyb955/sunyuhan/projects/SD/DriveLaW-Act/

    NAVSIM 数据：
    /data/home/scyb955/run/sunyuhan/navsim_workspace/dataset/

    IDM 训练输出：
    /data/home/scyb955/run/sunyuhan/projects/exp/idm_stage1_f0_speed/

