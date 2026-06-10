# DriveLaW-Act 复现操作指南（中文）

这份文档当前只服务于一个目标：

**先跑通评估或复现 `DriveLaW-Act` 的 planner 指标。**

也就是说，这份文档默认你当前**优先准备 `DriveLaW-Act`**，而不是先展开 `DriveLaW-Video` 的完整训练链。

说明：

- 这份文档基于仓库里的这些文件整理：
  - [README.md](/Users/sunyuhan/Downloads/drivelaw-main/README.md)
  - [DriveLaW-Act/README.md](/Users/sunyuhan/Downloads/drivelaw-main/DriveLaW-Act/README.md)
  - [DriveLaW-Act/docs/install.md](/Users/sunyuhan/Downloads/drivelaw-main/DriveLaW-Act/docs/install.md)
- 这份文档尽量按 README 组织，但会补足实际执行时最容易遗漏的步骤。
- 如果你的目标是**尽量对齐论文里的 planner 数字**，除了按文档执行，还要尽量保持：
  - 相同数据划分
  - 相同训练分辨率
  - 相同预训练初始化
  - 相同缓存流程

---

## 1. 当前复现目标

你现在的目标是：

- 先跑通 `DriveLaW-Act`
- 先复现 planner 评估结果
- 先尽量对齐论文里 `DriveLaW (Ours)` 这一行对应的 **planner 指标**

所以当前只需要盯住：

1. `DriveLaW-Act` 代码
2. `DriveLaW` 官方权重
3. `NAVSIM/OpenScene + nuPlan maps`
4. `DriveLaW-Act` 的环境变量
5. `hidden-state cache`
6. `metric cache`
7. `DriveLaW-Act` 的训练与评估脚本

**这一版文档不把 `DriveLaW-Video` 训练当作前置条件。**

---

## 2. 代码仓库准备

假设你已经把代码放到服务器的：

```bash
~/projects/drivelaw-main
```

进入仓库：

```bash
cd ~/projects/drivelaw-main
```

建议先确认代码版本稳定：

```bash
git status
git branch
git log --oneline -n 5
```

如果你是从自己的 Git 仓库同步的，建议固定一个分支做复现，例如：

```bash
git checkout -b repro
```

---

## 3. Python / Conda 环境（只准备 DriveLaW-Act）

顶层 README 写的是：

- 基础安装推荐 `python==3.10`

但 `DriveLaW-Act/docs/install.md` 又给了：

- `conda env create --name navsim -f environment.yml`

所以实际建议是：

### 3.1 为 `DriveLaW-Act` 单独建环境

```bash
cd ~/projects/drivelaw-main/DriveLaW-Act
conda env create --name drivelaw_act -f environment.yml
conda activate drivelaw_act
pip install -e .
pip install -r requirements.txt
```

### 3.2 补安装你当前扩展所需依赖

如果你要跑你已经改过的奖励项，还要额外有：

```bash
pip install pyiqa
```

不过如果你当前目标只是“先按原始仓库跑 planner 评估”，这一步不是原始仓库必须项。

---

## 4. 权重准备（只准备 DriveLaW-Act 所需）

顶层 README 给了官方权重：

- [DriveLaW HuggingFace](https://huggingface.co/tz2026/DriveLaW)

你至少要准备：

### 4.1 DriveLaW 官方权重

建议目录：

```bash
~/models/DriveLaW
```

需要确认里面至少有：

- `DriveLaW-Act/...`
- `DriveLaW-Video/transformer/...`
- `DriveLaW-Video/vae/...`
- `DriveLaW-Video/tokenizer/...`
- `DriveLaW-Video/text_encoder/...`

### 4.2 当前阶段权重结论

如果你现在只做 `DriveLaW-Act`：

- **必须有** `DriveLaW` 官方权重
- **暂时不用**单独准备 `DriveLaW-Video` 训练底座模型
- **暂时不用**展开世界模型训练权重链

---

## 5. 数据准备总览（只看 DriveLaW-Act）

当前阶段 `DriveLaW-Act` 需要：

- `nuPlan maps`
- `NAVSIM / OpenScene logs`
- `sensor_blobs`
- 可能还包括：
  - `warmup_two_stage`
  - `navhard_two_stage`
  - `private_test_hard_two_stage`

参考：
[DriveLaW-Act/docs/install.md](/Users/sunyuhan/Downloads/drivelaw-main/DriveLaW-Act/docs/install.md)

如果你只是先跑通评估：

- 最小可以先准备 `mini`

如果你想尽量对齐论文里的 planner 指标：

- 最后至少要准备 `trainval`
- 以及对应评估 split

---

## 6. `DriveLaW-Act` 数据目录要求

按照安装文档，建议组织成：

```bash
~/navsim_workspace
├── navsim
├── exp
└── dataset
    ├── maps
    ├── navsim_logs
    │   ├── mini
    │   ├── trainval
    │   ├── test
    │   └── private_test_hard
    ├── sensor_blobs
    │   ├── mini
    │   ├── trainval
    │   ├── test
    │   └── private_test_hard
    ├── navhard_two_stage
    ├── warmup_two_stage
    └── private_test_hard_two_stage
```

如果你只是最小跑通：

```bash
~/navsim_workspace/dataset/maps
~/navsim_workspace/dataset/navsim_logs/mini
~/navsim_workspace/dataset/sensor_blobs/mini
```

如果你要复现论文级 planner 结果，至少要有：

- `trainval`
- `maps`
- `metric cache`
- `hidden-state cache`

---

## 7. `DriveLaW-Act` 环境变量

把这些加到 `~/.bashrc` 或 `~/.zshrc`：

```bash
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="$HOME/navsim_workspace/dataset/maps"
export NAVSIM_EXP_ROOT="$HOME/navsim_workspace/exp"
export NAVSIM_DEVKIT_ROOT="$HOME/navsim_workspace/navsim"
export OPENSCENE_DATA_ROOT="$HOME/navsim_workspace/dataset"
```

然后执行：

```bash
source ~/.bashrc
```

或：

```bash
source ~/.zshrc
```

检查是否生效：

```bash
echo $NUPLAN_MAPS_ROOT
echo $OPENSCENE_DATA_ROOT
```

---

## 8. `DriveLaW-Act` 训练前必须做的缓存

这是 README 里明确要求的，**不能跳**。

在 planner 训练前，先缓存：

### 8.1 hidden state cache

```bash
cd ~/projects/drivelaw-main/DriveLaW-Act
sh scripts/evaluation/run_caching_videodrive_hidden_state.sh
```

### 8.2 metric cache

```bash
cd ~/projects/drivelaw-main/DriveLaW-Act
sh scripts/evaluation/run_metric_caching.sh
```

这些缓存的作用：

- `hidden state cache`
  - 加速训练
  - 为 planner 提供预计算特征
- `metric cache`
  - 为 `PDMS` 评估和训练使用

如果你要对齐论文结果，这一步必须严格跑通。

---

## 9. `DriveLaW-Video` 数据准备

如果你只做 planner 评估，可以跳过本节。  
如果你要完整复现世界模型训练，需要按下面准备。

### 9.1 NuPlan 数据整理

参考：
[Data-preparation.md](/Users/sunyuhan/Downloads/drivelaw-main/DriveLaW-Video/Train/docs/Data-preparation.md)

需要类似结构：

```bash
<nuplan-data-root>
├── splits
│   ├── mini
│   └── trainval
└── sensor_blobs
    ├── <scene_1>
    ├── <scene_2>
    └── ...
```

### 9.2 视频预处理

训练前必须先做 dataset preprocessing。

例如：

```bash
cd ~/projects/drivelaw-main/DriveLaW-Video/Train
python scripts/preprocess_dataset.py /path/to/dataset \
    --resolution-buckets "1280x704x33"
```

或者直接：

```bash
bash preprocess.sh
```

预处理完成后应生成：

```bash
dataset/
└── .precomputed/
    ├── latents/
    └── conditions/
```

### 9.3 训练 bucket 约束

README 里写得很明确：

- 宽高必须是 `32` 的倍数
- 帧数必须满足 `8n + 1`

例如：

- `33`
- `41`
- `49`

这是因为 VAE 的时空压缩结构决定的。

---

## 10. 分辨率要求

对 planner 结果尤其重要。

`DriveLaW-Act/README.md` 明确说明：

- 论文中的指标是在 `1344×768` 下得到的

所以如果你要尽量对齐论文结果：

- **训练分辨率**
- **推理分辨率**

都尽量保持 `1344×768`

如果你降分辨率，只能作为“跑通验证”，不要期望对齐论文表格数值。

---

## 11. 开始训练前的检查清单

在真正开始训练前，你至少要逐项确认下面这些。

### 11.1 代码

- [ ] `drivelaw-main` 已在服务器上
- [ ] 代码版本固定
- [ ] 本地与服务器 Git 可同步

### 11.2 环境

- [ ] `conda` 环境创建成功
- [ ] `pip install -e .` 成功
- [ ] `torch / diffusers / transformers / deepspeed / timm` 可导入

检查：

```bash
python -c "import torch, diffusers, transformers, timm; print('ok')"
```

### 11.3 数据

- [ ] `maps` 目录存在
- [ ] `navsim_logs/trainval` 存在
- [ ] `sensor_blobs/trainval` 存在
- [ ] 如果先做最小验证，`mini` 数据可用

### 11.4 环境变量

- [ ] `NUPLAN_MAPS_ROOT` 正确
- [ ] `OPENSCENE_DATA_ROOT` 正确
- [ ] `NAVSIM_DEVKIT_ROOT` 正确

### 11.5 权重

- [ ] `DriveLaW` 官方权重已下载
- [ ] `LTX-Video-0.9.5` 已下载
- [ ] 配置文件中权重路径已改对

### 11.6 缓存

- [ ] hidden-state cache 已完成
- [ ] metric cache 已完成

### 11.7 资源

- [ ] 有足够磁盘空间
- [ ] 有足够 GPU
- [ ] 有日志与输出目录

建议提前建好：

```bash
mkdir -p ~/logs/drivelaw
mkdir -p ~/outputs/drivelaw
```

---

## 12. 推荐启动顺序（当前只做 DriveLaW-Act）

推荐你按这两个阶段走：

### 阶段 A：最小跑通

1. 准备 `mini + maps`
2. 配好 `DriveLaW-Act` 环境
3. 跑 hidden-state cache
4. 跑 metric cache
5. 跑 1 次 evaluation
6. 确认路径、日志、输出都正常

### 阶段 B：planner 指标复现

1. 准备 `trainval + maps`
2. 用正式 split 重新做完整缓存
3. 跑 `DriveLaW-Act` 训练
4. 跑 planner evaluation

如果你时间紧，先完成阶段 A，再进阶段 B。

---

## 13. 官方脚本顺序

### 13.1 DriveLaW-Act

```bash
cd ~/projects/drivelaw-main/DriveLaW-Act

# 缓存
sh scripts/evaluation/run_caching_videodrive_hidden_state.sh
sh scripts/evaluation/run_metric_caching.sh

# 训练
sh scripts/training/run_videodrive_train.sh

# 评估
sh scripts/evaluation/run_videodrive_agent_pdm_score_evaluation.sh
```

## 14. 你最容易卡住的地方

### 14.1 外网下载

如果服务器连不上：

- Hugging Face
- GitHub
- 数据下载源

那就改成：

1. 本地下载
2. 上传服务器

### 14.2 数据量

即使 `mini` 也不算特别小。  
如果是完整复现，磁盘和 I/O 压力都会很大。

### 14.3 配置路径

最容易错的是：

- 模型权重路径
- 地图路径
- OpenScene 根目录
- `.precomputed` 数据根目录

### 14.4 分辨率

如果训练或推理分辨率和论文不一致，最后结果通常对不上。

---

## 15. 推荐的服务器目录组织

建议你统一成这样：

```bash
~/projects/
  drivelaw-main/

~/models/
  DriveLaW/

~/navsim_workspace/
  navsim/
  exp/
  dataset/
    maps/
    navsim_logs/
    sensor_blobs/

~/logs/
  drivelaw/

~/outputs/
  drivelaw/
```

这样后面改配置最省事。

---

## 16. 如果你的目标只是先复现论文里的 planner 表格数据

那你可以先只准备这些：

1. `DriveLaW-Act` 代码
2. `DriveLaW` 官方权重
3. `maps`
4. `navsim_logs/trainval`
5. `sensor_blobs/trainval`
6. hidden-state cache
7. metric cache
8. 训练脚本与评估脚本

也就是说，**先不展开 `DriveLaW-Video` 训练**，优先把 planner 结果跑出来。

---

## 17. 最后一句结论

如果你是以“先复现论文里 `DriveLaW (Ours)` 对应 planner 指标”为目标，那么在训练开始前，你至少必须准备好：

- 正确的代码版本
- 可用的 Python/Conda 环境
- `DriveLaW` 官方权重
- `NAVSIM/OpenScene + nuPlan maps`
- 正确的环境变量
- `DriveLaW-Act` 所需的 hidden-state / metric cache
- 与论文一致的训练/推理分辨率

在这些都到位之前，不建议直接开训。

---

## 附：如果以后你要补做 DriveLaW-Video

如果后面你的目标从“planner 指标复现”升级成“整个项目完整复现”，再额外补准备：

- `DriveLaW-Video/Train` 环境
- `LTX-Video-0.9.5`
- NuPlan 原始视频数据
- 视频预处理
- `.precomputed/latents`
- `.precomputed/conditions`

但这些都不是你当前阶段的前置条件。
