# DriveLaW 项目交接现况

## 1. 原项目

- 论文：DriveLaW: Unifying Planning and Video Generation in a Latent Driving World，CVPR 2026。
- arXiv：https://arxiv.org/abs/2512.23421
- 官方代码：https://github.com/xiaomi-research/drivelaw
- 官方权重：https://huggingface.co/tz2026/DriveLaW

原项目由两部分组成：

- DriveLaW-Video：生成未来驾驶视频的世界模型。
- DriveLaW-Act：读取 Video DiT latent，生成 2 Hz、4 秒的 8 个轨迹点的扩散规划器。

当前 H20 上运行的代码仓库为 Unified-GRPO，其中 DriveLaW-Act/ 是本项目的主要改动目录，DriveLaW-Video/ 保留官方视频训练和推理代码。

## 2. 当前方法

当前尝试的总方法是在 DriveLaW 的联合 Video + Action 框架中，同时优化视频世界模型和轨迹规划分支。新增的总损失由三部分组成：

    L_total = L_video_flow + 0.1 * L_video_rgb + L_action

### 2.1 Video flow-matching loss

Video 分支使用真实未来视频构造 latent target，并对 Video DiT 的 flow-matching 预测进行监督。

输入为 4 张历史前视图；历史图分别编码为 memory latent，只作为条件。以当前帧加 8 张 NAVSIM future 图构成 9 帧、2 Hz 的 target clip，经过 Temporal VAE 编码为 2 个 temporal target latent。

### 2.2 RGB reconstruction loss

将 Video DiT 预测出的两个 target latent 通过冻结的 Temporal VAE 解码为 9 帧，与当前帧加 8 张真实未来图计算 RGB L1 loss。其目的，是让 latent-space 的监督与可见视频质量直接关联。

### 2.3 Action loss

Action / Planning DiT 保持论文 Stage-3 的扩散轨迹训练目标，学习生成 2 Hz、4 秒的 8 个轨迹点。该部分与视频 loss 联合训练。

## 3. 逆动力学模块（IDM）

已实现 IDM，用视频帧预测轨迹，原计划用于比较：

    生成视频 -> IDM trajectory
    World Model trajectory
    GT trajectory

代码位置：

    DriveLaW-Act/navsim/agents/videodrive/idm_xvo_style.py
    DriveLaW-Act/navsim/agents/videodrive/run_training_idm_stage1.py
    DriveLaW-Act/scripts/training/run_idm_stage1_training.sh
    DriveLaW-Act/scripts/training/run_idm_stage1_sbatch.sh
    DriveLaW-Act/navsim/planning/script/run_generated_video_idm_probe.py

IDM 已训练过，但尚未完成有效性验证。未完成真实视频上的 held-out 轨迹误差评估，也未完成生成视频上的可靠性验证，因此目前不能使用 IDM 输出或相关 PDM 分数证明方法有效。

北京超级云上旧的 IDM 训练输出路径：

    /data/home/scyb955/run/sunyuhan/projects/exp/idm_stage1_f0_speed/

其中包含 checkpoints/best.pt、train.log、metrics.jsonl 和 TensorBoard event。

## 4. 当前代码位置

### 4.1 联合训练

    DriveLaW-Act/navsim/agents/videodrive/run_training_joint_video_action.py
      - 三个 loss 的计算
      - history memory latent 和 future target latent 的构造
      - 保留论文 Action 分支的训练逻辑

    DriveLaW-Act/navsim/agents/videodrive/run_training_stage3_joint_video_action.py
      - 联合训练入口

    DriveLaW-Act/navsim/agents/videodrive/configs/ltx_model/video_model_stage3_joint_video_action.yaml
      - loss 权重、优化器、DeepSpeed 和训练配置

    DriveLaW-Act/scripts/training/run_stage3_joint_video_action.sh
      - 联合训练启动脚本

### 4.2 联合推理与可视化

    DriveLaW-Act/navsim/agents/videodrive/video_models/pipeline/pipeline_ltx_condition.py
      - 返回最终 latent

    DriveLaW-Act/navsim/agents/videodrive/videodrive_agent.py
      - 仅选取最后两个 temporal target latent 进行 VAE decode

    DriveLaW-Act/navsim/planning/script/run_joint_video_trajectory_probe.py
      - 保存生成视频、World trajectory 和 GT trajectory

    DriveLaW-Act/scripts/evaluation/run_latest_stage3_video_probe.sh
      - 根据最新或指定 checkpoint 运行视频 probe

### 4.3 兼容性修改

    DriveLaW-Act/navsim/agents/videodrive/run_training_videodrive.py
      - DeepSpeed gradient accumulation 同步修复

## 5. 服务器数据、缓存和权重位置

### H20 当前服务器

    NAVSIM 数据：
    /rdata/sunyuhan/navsim_workspace/dataset/

    NAVSIM cache：
    /rdata/sunyuhan/navsim_workspace/cache/videodrive_agent_cache_dir_final_front

    DriveLaW-Video 初始权重：
    /rdata/sunyuhan/projects/models/DriveLaW_weights/DriveLaW-Video/

### 北京超级云旧服务器

    代码：
    /data/run01/scyb955/sunyuhan/projects/SD/DriveLaW-Act/

    NAVSIM 数据：
    /data/home/scyb955/run/sunyuhan/navsim_workspace/dataset/

    IDM 训练输出：
    /data/home/scyb955/run/sunyuhan/projects/exp/idm_stage1_f0_speed/

## 6. 已完成训练与权重

已完成一次联合 Video + Action 的 RGB-loss pilot：

    /rdata/sunyuhan/projects/exp/stage3_joint_video_action_4gpu_2hz_rgb_pilot_b4_rerun/
      2026_09_27_12_59_57/step_44000/

最终权重：

    /rdata/sunyuhan/projects/exp/stage3_joint_video_action_4gpu_2hz_rgb_pilot_b4_rerun/2026_09_27_12_59_57/step_44000/diffusion_pytorch_model.safetensors

该训练使用 4 张 H20、每卡 batch 1，实际 global batch 为 4，共 44,000 steps、3 epochs。权重由发布的 DriveLaW-Video Transformer 初始化，不是由发布的 DriveLaW-Act checkpoint 初始化。

最终 loss：

    total=0.11017
    video=0.10159
    flow=0.09382
    rgb=0.07769
    action=0.00858

对应的视频 probe 输出：

    /rdata/sunyuhan/projects/exp/stage3_joint_video_action_4gpu_2hz_rgb_pilot_b4_rerun/probes/step_44000_target2_decode/

    视频：
    videos/0000_0000be0b1dc65be3_raw_decoded_all_frames.mp4

    逐帧 PNG：
    raw_decoded_frames/0000_0000be0b1dc65be3/

## 7. 已尝试但失败的方法

### 7.1 直接解码 6 个 latent 为 41 帧

早期将 4 个独立 history memory latent 和 2 个 future target latent 连续拼接，作为 6 个 temporal latent 直接送入 VAE decode，得到 41 帧视频。

该方法不成立：4 个 history latent 是分别单帧编码的 memory，不是连续 Temporal VAE 视频 latent；41 帧没有正确时间语义。该路径已废弃。

### 7.2 仅使用 latent flow-matching loss 的联合训练

已完成一次 44k steps 联合训练，视频仍无法生成有效内容。仅看 flow loss 下降不能保证扩散采样视频正确。

### 7.3 加入 RGB reconstruction loss 的联合训练

在原 flow loss 和 Action loss 上加入两个 target latent 的 RGB L1 loss，完成了上述 44k steps pilot。

结果仍失败：最新 checkpoint 生成的 9 帧视频全部为无结构、块状图像。说明当前问题不是视频导出 FPS，也不是 41 帧错误拼接造成的。

### 7.4 修改导出 FPS

将 9 帧 target video 的导出 FPS 改为 2，以正确显示 t=0, 0.5, ..., 4.0s 的时间范围。该修改只修正播放速度，不改变图像内容，不能解决视频为噪声的问题。

## 8. 当前瓶颈

当前核心瓶颈是：Video DiT 在联合训练后仍无法从条件 latent 生成可用的 future target latent，VAE decode 后得到全程块状噪声视频。

当前结果不能说明视频世界模型具备生成能力，也不能开展基于生成视频的 IDM 验证。

虽然训练 loss 下降，但多步采样结果失败。最可能的问题类别包括：

- 训练阶段和推理阶段的 scheduler、sigma 或 timestep 定义不一致。
- 训练和推理的 memory latent、mask、token 排序或 latent frame layout 不一致。
- 当前 4 GPU、global batch 4 的 pilot 与论文 global batch 192 的优化规模差异较大。
- RGB loss 权重为 0.1，无法单独约束最终多步采样质量。

