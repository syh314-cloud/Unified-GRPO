# DriveLaW 项目交接文档

更新日期：2026-10-11  
当前本地代码提交：`7b72958`  
H20 运行仓库提交：`7b72958`

## 1. 原项目

- 论文：**DriveLaW: Unifying Planning and Video Generation in a Latent Driving World**，CVPR 2026。
- arXiv：https://arxiv.org/abs/2512.23421
- 官方代码：https://github.com/xiaomi-research/drivelaw
- 权重：https://huggingface.co/tz2026/DriveLaW

DriveLaW 将视频世界模型和轨迹规划器放入同一个扩散 Transformer：Video DiT 生成未来视频 latent，Planning/Action DiT 读取 Video DiT latent 并生成 2 Hz、4 秒的 8 个轨迹点。

代码目录：

```text
/home/sunyuhan/projects/Unified-GRPO
  DriveLaW-Video/Infer/     # 官方视频模型推理
  DriveLaW-Video/Train/     # 官方视频模型训练
  DriveLaW-Act/             # NAVSIM 规划、联合模型和本项目改动
```

H20 是嵌套 Docker，正确进入方式：

```bash
# 外层 ltk-server
docker exec -it 1226_privileged_ubuntu22.04 bash
# 内层容器
docker exec -it sunyuhan bash
conda activate drivelaw_act
cd /home/sunyuhan/projects/Unified-GRPO/DriveLaW-Act
```

不要在 H20 外层直接运行 `docker exec sunyuhan ...`。

## 2. 当前新增方法

### 2.1 联合 Video + Action Stage-3 训练

目标：从发布的 **DriveLaW-Video** 权重开始，在 NAVSIM 上重新执行论文的 Stage-3 联合训练，同时加入真实未来视频监督。

当前视频 target 的定义：

```text
4 张历史前视图
  -> 分别单帧 VAE encode
  -> 4 个 memory latent，只作为条件

当前图 + 8 张 NAVSIM future 图（2 Hz，覆盖 4 秒）
  -> 9 帧连续 target clip
  -> Temporal VAE encode
  -> 2 个 temporal target latent

Video DiT 预测 2 个 target latent
  -> flow-matching latent loss
  -> VAE decode 为 9 帧
  -> RGB L1 辅助 loss
```

Action 分支保留论文 Stage-3 的 Action noise、target 和 conditioning 训练逻辑。

总损失：

```text
L = 1.0 * (L_video_flow + 0.1 * L_video_rgb) + 1.0 * L_action
```

RGB loss 是辅助项，不保证多步采样的视频质量。

### 2.2 逆动力学模块（IDM）

已实现由视频帧预测轨迹的 IDM，用于比较“生成视频 -> IDM 轨迹”和 World Model / GT 轨迹。

代码：

```text
DriveLaW-Act/navsim/agents/videodrive/idm_xvo_style.py
DriveLaW-Act/navsim/agents/videodrive/run_training_idm_stage1.py
DriveLaW-Act/scripts/training/run_idm_stage1_training.sh
DriveLaW-Act/scripts/training/run_idm_stage1_sbatch.sh
DriveLaW-Act/navsim/planning/script/run_generated_video_idm_probe.py
```

旧服务器上的一次 IDM 输出：

```text
/data/home/scyb955/run/sunyuhan/projects/exp/idm_stage1_f0_speed/
```

其中有 `checkpoints/best.pt`、`train.log`、`metrics.jsonl` 和 TensorBoard event；该目录不保证已迁移到 H20。

IDM **尚未做严格有效性验证**。目前没有完成 held-out 轨迹误差、真实视频鲁棒性或生成视频上的系统比较，因此不能将 IDM 输出或相关 PDM 分数作为方法有效的结论。

## 3. 已完成部分

### 3.1 H20 数据、权重和缓存

```text
数据根目录：
/rdata/sunyuhan/navsim_workspace/dataset
  sensor_blobs/trainval
  navsim_logs
  maps

VideoDrive cache：
/rdata/sunyuhan/navsim_workspace/cache/videodrive_agent_cache_dir_final_front

DriveLaW-Video 权重：
/rdata/sunyuhan/projects/models/DriveLaW_weights/DriveLaW-Video
```

NAVTRAIN cache 已完成，历史日志显示 103258/103258 场景成功。

### 3.2 已完成的 Stage-3 pilot

输出目录：

```text
/rdata/sunyuhan/projects/exp/stage3_joint_video_action_4gpu_2hz_rgb_pilot_b4_rerun/
  2026_09_27_12_59_57/step_44000/
```

配置：

- 4 张 H20。
- 每卡 batch 1，实际 global batch = 4。
- 44,000 optimizer steps，3 epochs。
- AdamW，learning rate `3e-5`，weight decay `1e-5`，bf16，DeepSpeed ZeRO-1。
- Transformer 从发布的 **DriveLaW-Video** 初始化，不是发布的 DriveLaW-Act checkpoint。
- Action expert 在模型构造时初始化；Action loss 保持论文 Stage-3 实现。

最终日志：

```text
step 44000:
total=0.11017
video=0.10159
flow=0.09382
rgb=0.07769
action=0.00858
```

这不是论文的严格规模复现。论文报告 global batch 192、44k steps，但未公开 GPU 数、每卡 batch 和梯度累积；本实验实际 global batch 为 4。

### 3.3 已生成的视频 probe

```text
/rdata/sunyuhan/projects/exp/stage3_joint_video_action_4gpu_2hz_rgb_pilot_b4_rerun/probes/step_44000_target2_decode/

视频：
videos/0000_0000be0b1dc65be3_raw_decoded_all_frames.mp4

逐帧 PNG：
raw_decoded_frames/0000_0000be0b1dc65be3/frame_0000.png ... frame_0008.png
```

当前 probe 只解码最后两个 temporal target latent，得到当前帧加 8 张未来帧，共 9 帧。视频以 2 FPS 导出，对应 `t=0, 0.5, ..., 4.0s`。旧的 41 帧拼接视频没有有效时间语义，不能作为结果。

## 4. 当前阻塞点

**step_44000 的 9 帧视频全部为无结构块状图，视频生成失败。**

这不是 MP4 FPS 设置问题，也不是把 4 个 history latent 与 target latent 错误连续拼接为 41 帧造成的问题：当前推理只解码模型采样出的两个 target latent，结果仍然是噪声。

因此当前 checkpoint 不能作为可用世界模型；在生成视频清晰且时序正确前，不能用它验证 IDM 或宣称视频-轨迹一致性。

loss 下降不代表多步视频采样正确。flow-matching loss 约束随机噪声级别的局部向量场；若训练时的 noise/sigma/timestep 定义和推理 scheduler 不一致，或条件 token layout 不一致，loss 可以下降而 30 步采样保持噪声。

不要直接继续另一轮 44k 训练，先定位训练-推理 contract。

## 5. 新增和修改的关键代码

### 联合训练与视频 target

```text
DriveLaW-Act/navsim/agents/videodrive/run_training_joint_video_action.py
  - 4 memory latent + 2 target latent 构造
  - video flow-matching loss
  - 9 帧 target clip 的 RGB L1 辅助 loss
  - 论文 Stage-3 Action 分支

DriveLaW-Act/navsim/agents/videodrive/run_training_stage3_joint_video_action.py
  - Stage-3 训练入口

DriveLaW-Act/navsim/agents/videodrive/configs/ltx_model/video_model_stage3_joint_video_action.yaml
  - 损失、优化器、DeepSpeed 和 Stage-3 配置

DriveLaW-Act/scripts/training/run_stage3_joint_video_action.sh
  - 训练启动脚本

DriveLaW-Act/scripts/training/wait_and_run_stage3_joint_video_action.sh
  - 等待 GPU 显存满足阈值后启动训练
```

### 联合推理与视频导出

```text
DriveLaW-Act/navsim/agents/videodrive/video_models/pipeline/pipeline_ltx_condition.py
  - 返回最终 latent，避免 pipeline 内直接 VAE decode

DriveLaW-Act/navsim/agents/videodrive/videodrive_agent.py
  - 仅选最后 2 个 target latent 后 VAE decode
  - 不再把 4 个独立 history memory latent 与 target latent 连续 decode

DriveLaW-Act/navsim/planning/script/run_joint_video_trajectory_probe.py
  - 保存单样本视频、World trajectory 和 GT trajectory
  - probe 导出 FPS 为 2，模型内部 frame_rate=8 未改动

DriveLaW-Act/scripts/evaluation/run_latest_stage3_video_probe.sh
  - 自动寻找最新 checkpoint 并运行 probe
```

### 其他兼容性修改

```text
DriveLaW-Act/navsim/agents/videodrive/run_training_videodrive.py
  - DeepSpeed gradient accumulation 同步修复
```

## 6. 接手后的优先步骤

按下面顺序做；不要先重训。

1. **GT target VAE round-trip。**  
   固定一个 token，用训练时相同方式构造 9 帧 target clip，VAE encode 成 2 个 latent 后立刻 decode。结果应清晰且与 GT 对齐。已有 oracle：
   ```text
   DriveLaW-Act/navsim/planning/script/run_vae_latent_oracle_probe.py
   DriveLaW-Act/scripts/evaluation/run_vae_latent_oracle_probe_sbatch.sh
   ```

2. **单步 x0 诊断。**  
   对同一 checkpoint 和训练 sigma，计算 `z0_hat = z_sigma - sigma * v_hat`，只 decode 最后两个 target latent。保存并比较：
   - GT target latent decode
   - 单步 x0 decode
   - 30 步 pipeline decode
   - latent MSE 与 RGB L1

   判定：
   - 单步 x0 清晰、30 步采样为噪声：问题在 scheduler / sigma / timestep 推理 contract。
   - 单步 x0 也是噪声：问题在训练 target、条件布局、初始化或优化规模。

3. **逐项核对训练和推理条件布局。**  
   核查 history memory latent、mask、token 排序、`latent_frames=6`、`pixel_wise_timestep` 和 scheduler timestep。重点文件：
   ```text
   run_training_joint_video_action.py
   pipeline_ltx_condition.py
   videodrive_agent.py
   ```

4. **定位完成后才重训。**  
   先跑短诊断训练并保存固定样本可视化。若要对齐论文 global batch 192，需解决资源与梯度累积；现有 batch 4 pilot 不能视为论文复现。

5. **最后验证 IDM。**  
   先在真实视频上评估 IDM，再在生成视频上评估，报告轨迹误差和 PDM 指标，不能仅展示个例。

## 7. 常用命令

查看 checkpoint 和 probe：

```bash
TRAIN=/rdata/sunyuhan/projects/exp/stage3_joint_video_action_4gpu_2hz_rgb_pilot_b4_rerun
find "$TRAIN" -path '*/step_*/diffusion_pytorch_model.safetensors' -print
find "$TRAIN/probes/step_44000_target2_decode" -type f | sort
```

重跑指定 checkpoint 的单样本 probe：

```bash
export REPO=/home/sunyuhan/projects/Unified-GRPO/DriveLaW-Act
export OPENSCENE_DATA_ROOT=/rdata/sunyuhan/navsim_workspace/dataset
export DRIVELAW_VIDEO_ROOT=/rdata/sunyuhan/projects/models/DriveLaW_weights/DriveLaW-Video
export TRAIN_OUTPUT_ROOT=/rdata/sunyuhan/projects/exp/stage3_joint_video_action_4gpu_2hz_rgb_pilot_b4_rerun
export CHECKPOINT_PATH="$TRAIN_OUTPUT_ROOT/2026_09_27_12_59_57/step_44000/diffusion_pytorch_model.safetensors"
export PROBE_OUTPUT_DIR="$TRAIN_OUTPUT_ROOT/probes/manual_step_44000"
export PROBE_LIMIT=1
export TRAIN_TEST_SPLIT=navtrain
bash "$REPO/scripts/evaluation/run_latest_stage3_video_probe.sh"
```

## 8. Git 注意事项

本地提交 `7b72958` 已通过 Git bundle 同步到 H20 运行仓库。GitHub SSH/网络认证尚未恢复，`origin/main` 仍停在 `4384ac5`；不要假设 `git pull` 能拿到后续提交。

接手前检查：

```bash
cd /home/sunyuhan/projects/Unified-GRPO
git log --oneline -5
git status --short
```

恢复 GitHub SSH 认证后，再将 `7b72958` 及后续提交推送到远端。

