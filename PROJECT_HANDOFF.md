# DriveLaW 项目交接

## 1. 原项目

论文：DriveLaW: Unifying Planning and Video Generation in a Latent Driving World，CVPR 2026。

- 论文：https://arxiv.org/abs/2512.23421
- 官方代码：https://github.com/xiaomi-research/drivelaw
- 官方权重：https://huggingface.co/tz2026/DriveLaW

项目包含两个部分：

- `DriveLaW-Video/`：视频世界模型的训练和推理代码。
- `DriveLaW-Act/`：在 NAVSIM 上执行轨迹规划、评测和后续强化学习的代码。

H20 上的官方初始化权重：

    /rdata/sunyuhan/projects/models/DriveLaW_weights/DriveLaW-Video/

北京超级云上的项目代码：

    /data/run01/scyb955/sunyuhan/projects/SD/DriveLaW-Act/

H20 上的项目代码目录：

    Unified-GRPO/DriveLaW-Act/

## 2. 新方法：基于 GRPO 的三项奖励优化

本项目的新方法是在 DriveLaW-Act 的 GRPO 框架中，以三项奖励共同优化规划轨迹和对应未来视频的一致性与质量。视频生成训练本身不是新方法。

1. PDM 驾驶奖励：使用 NAVSIM PDM scorer 评估轨迹的安全性、进度、舒适性和驾驶方向等闭环驾驶指标。

2. IDM 轨迹一致性奖励：将生成视频输入逆动力学模块，预测 2 Hz、4 秒的 8 点轨迹；该轨迹与 Action 分支采样轨迹的加权位置误差和航向误差共同构成一致性代价，奖励取其相反数。

3. MUSIQ 视频质量奖励：对生成未来视频的中间帧和末帧计算 MUSIQ 图像质量分数，并取平均作为视频质量奖励。

其中 PDM 奖励依赖 DriveLaW-Act 原有的 GRPO/PDM 实现；IDM 一致性奖励和 MUSIQ 奖励为新增模块。目前尚未完成三项奖励的 GRPO 联合训练、奖励尺度处理及最终有效性验证。

## 3. 已完成部分：逆动力学模块

已完成 IDM 的训练与接入准备。IDM 的输入为视频帧，输出为 2 Hz、4 秒的 8 点轨迹，用于比较：

    生成视频 -> IDM trajectory
    World Model trajectory
    GT trajectory

北京超级云上的 IDM 训练输出：

    /data/home/scyb955/run/sunyuhan/projects/exp/idm_stage1_f0_speed/

该目录中包含：

    checkpoints/best.pt
    train.log
    metrics.jsonl
    tensorboard/

IDM 尚未完成严格的有效性验证：没有完成真实视频上的 held-out 轨迹误差评估，也没有完成生成视频条件下的可靠性验证。因此，当前不能据 IDM 输出或相关 PDM 分数证明新方法有效。

## 4. 新增代码

以下是为 IDM 和三项奖励方法保留的新增代码：

    DriveLaW-Act/navsim/agents/videodrive/idm_xvo_style.py
        IDM 网络、IDM 轨迹一致性奖励和 MUSIQ 视频质量奖励。

    DriveLaW-Act/navsim/agents/videodrive/run_training_idm_stage1.py
        IDM Stage-1 训练入口。

    DriveLaW-Act/navsim/agents/videodrive/videodrive_features.py
        为 IDM 训练和推理提供视频特征处理。

    DriveLaW-Act/navsim/planning/script/run_generated_video_idm_probe.py
        对生成视频运行 IDM，并输出 IDM 轨迹、World Model 轨迹和 GT 轨迹的检查脚本。

    DriveLaW-Act/scripts/training/run_idm_stage1_training.sh
    DriveLaW-Act/scripts/training/run_idm_stage1_sbatch.sh
    DriveLaW-Act/scripts/training/run_idm_stage1_smoke_sbatch.sh
        IDM 训练和环境检查脚本。

PDM 奖励的现有基础实现位于：

    DriveLaW-Act/navsim/agents/videodrive/video_models/planner/diffusion_planner.py

## 5. 当前瓶颈

当前联合模型无法生成可用视频。

该问题阻断了新方法中的两项视频相关奖励：IDM 轨迹一致性奖励需要可信的生成视频作为输入，MUSIQ 视频质量奖励也需要生成视频具有可评估的内容。因此，在视频生成问题解决前，不能完成三项奖励的 GRPO 联合训练，也不能验证新方法的实际效果。

## 6. 已尝试但失败的工作

以下工作均用于排查联合模型的视频输出问题，与第 2 节的新方法无关。

1. 联合 Video + Action Stage-3 训练：以公开 DriveLaW-Video 权重为初始化，在 NAVSIM 上同时训练 Video DiT 与 Action/Planning 分支，按 44k steps 完成训练。训练 loss 下降，但最终联合模型仍无法生成可用视频。

2. 2 Hz RGB 监督试验：在上述联合训练目标外，加入基于 NAVSIM 当前帧和未来采样帧的 RGB 监督。先完成 1 step smoke test 验证训练路径可运行；随后进行 4 卡、每卡 batch 1 的 pilot。一个 pilot 在约 9k steps 时因多卡 NCCL 通信超时中断；另一个 rerun 完成 44k steps，但仍无法生成可用视频。

3. 推理与解码检查：对训练中间 checkpoint 和完成 checkpoint 分别进行了视频推理、逐帧导出以及 VAE latent 解码检查。结果没有得到可用于 IDM 的未来视频，因此不能据此开展 IDM 一致性奖励或 MUSIQ 奖励的强化学习验证。

4. 训练资源配置试验：曾使用 H20 多卡训练，并尝试按论文公开的 global batch 规模配置梯度累积；受单次训练耗时和多卡通信稳定性影响，没有完成严格等价于论文 batch 配方的可用视频复现。

以上试验产生的训练权重均不作为可用模型记录。相关联合训练脚本、联合推理脚本、视频训练配置和临时检查脚本已从当前代码中移除，不应作为后续实验基础。

## 7. 数据与缓存位置

### H20

    NAVSIM 数据：
    /rdata/sunyuhan/navsim_workspace/dataset/

    NAVSIM cache：
    /rdata/sunyuhan/navsim_workspace/cache/videodrive_agent_cache_dir_final_front

    DriveLaW-Video 官方初始化权重：
    /rdata/sunyuhan/projects/models/DriveLaW_weights/DriveLaW-Video/

### 北京超级云

    DriveLaW-Act 代码：
    /data/run01/scyb955/sunyuhan/projects/SD/DriveLaW-Act/

    NAVSIM 数据：
    /data/home/scyb955/run/sunyuhan/navsim_workspace/dataset/

    IDM 训练输出和权重：
    /data/home/scyb955/run/sunyuhan/projects/exp/idm_stage1_f0_speed/
