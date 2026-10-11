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

## 2. 新方法：三项强化学习奖励

本项目的创新方法是基于 GRPO 的强化学习优化，不是视频生成训练。对同一驾驶场景采样多条候选轨迹及其对应未来视频，并以三项奖励信号构造优化目标：

1. PDM 驾驶奖励：使用 NAVSIM 的 PDM scorer 对候选轨迹进行闭环安全、进度、舒适性和驾驶方向等评估。现有基础实现位于：

       DriveLaW-Act/navsim/agents/videodrive/video_models/planner/diffusion_planner.py

2. IDM 轨迹一致性奖励：将生成视频输入逆动力学模块得到轨迹，与采样的 Action trajectory 比较位置和航向误差，奖励为该加权误差的相反数。实现位于：

       DriveLaW-Act/navsim/agents/videodrive/idm_xvo_style.py

3. MUSIQ 视频质量奖励：对生成未来视频的中间帧和末帧计算 MUSIQ 分数并取平均，用于约束生成结果的感知质量。实现位于：

       DriveLaW-Act/navsim/agents/videodrive/idm_xvo_style.py

这三项 reward 的 GRPO 联合训练、尺度归一化及有效性验证尚未完成；当前保留的是 IDM 模块及相关工具代码。此前进行的视频生成训练只是为确认 DriveLaW 基线能否提供可用视频输入，与上述创新方法无关。当前瓶颈：联合模型无法生成可用视频。

## 3. IDM 当前状态

IDM 的目标是从视频帧预测 2 Hz、4 秒的 8 点轨迹，用于比较：

    生成视频 -> IDM trajectory
    World Model trajectory
    GT trajectory

IDM 已训练过，但没有完成严格的有效性验证。尚未完成真实视频上的 held-out 轨迹误差评估，也未完成生成视频上的可靠性验证，因此不能用 IDM 输出或相关 PDM 分数证明方法有效。

北京超级云上的 IDM 训练结果：

    /data/home/scyb955/run/sunyuhan/projects/exp/idm_stage1_f0_speed/

其中包含 checkpoints/best.pt、train.log、metrics.jsonl 和 TensorBoard event。该 IDM checkpoint 不在 H20 的当前训练输出目录中。

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
