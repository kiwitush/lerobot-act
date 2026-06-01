# CALVIN 数据集准备说明

## 概述

本项目使用 LeRobot 格式的 CALVIN 数据集。数据保存在 HuggingFace Hub 上的 `lerobot/calvin` 仓库中，可通过 LeRobotDataset 直接加载。

## 1. 直接从 HuggingFace Hub 加载 (推荐)

训练脚本默认从 HuggingFace Hub 自动下载，无需手动准备:

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset
dataset = LeRobotDataset("lerobot/calvin", split="train")
```

首次运行时 HuggingFace 会自动缓存到 `~/.cache/huggingface/`。

## 2. 下载到本地

若网络环境受限，可提前下载:

```bash
# 方式 A: 通过 huggingface-cli
pip install huggingface_hub
huggingface-cli download lerobot/calvin --repo-type dataset --local-dir ./data/calvin_lerobot

# 方式 B: 通过 Python
python -c "
from huggingface_hub import snapshot_download
snapshot_download('lerobot/calvin', repo_type='dataset', local_dir='./data/calvin_lerobot')
"
```

下载后在配置中指定本地路径:

```yaml
# configs/act_baseline.yaml
data:
  repo_id: "lerobot/calvin"
  local_dir: "./data/calvin_lerobot"  # 回退路径
```

## 3. 从 CALVIN 原始 .npz 格式转换

如果你已经下载了原始 CALVIN 数据集 (.npz 格式)，可以转换为 LeRobot 格式:

```bash
python scripts/prepare_data.py \
    --input ./data/calvin_raw/task_ABC_D \
    --output ./data/calvin_lerobot \
    --envs A B C
```

### 原始 CALVIN 数据结构

```
calvin/dataset/task_ABC_D/
├── training/
│   ├── episode_00000/
│   │   ├── scene_000000.npz    # 含 rgb_static, rgb_gripper, robot_obs, actions
│   │   └── ...
│   └── ...
└── validation/
    └── ...
```

### LeRobot 格式结构

```
data/calvin_lerobot/
├── meta/
│   ├── info.json               # 特征定义、相机名称等
│   ├── stats.json              # 归一化统计量 (mean/std)
│   └── episodes/               # 片段元数据
├── data/
│   └── chunk-000/
│       └── file-000.parquet    # 状态/动作数据 (observation.state, action 等)
└── videos/
    ├── rgb_static/
    │   └── chunk-000/
    │       └── file-000.mp4    # 静态相机视频
    └── rgb_gripper/
        └── chunk-000/
            └── file-000.mp4    # 夹爪相机视频
```

## 4. 环境筛选

`lerobot/calvin` 数据集在 episode 元数据中包含环境标签，训练脚本通过 `data.train_envs` 筛选:

```yaml
# Baseline: 仅环境 A
data:
  train_envs: ["A"]

# Joint: 环境 A+B+C
data:
  train_envs: ["A", "B", "C"]
```

验证集 (validation split) 默认对应环境 D，用于零样本评估。

## 5. 验证数据加载

```bash
python -c "
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset('lerobot/calvin', split='train')
print(f'Episodes: {ds.num_episodes}, Frames: {len(ds)}')
print(f'Cameras: {ds.camera_names}')
print(f'Features: {list(ds.features.keys())}')

# 查看一个样本
sample = ds[0]
for k, v in sample.items():
    if hasattr(v, 'shape'):
        print(f'  {k}: shape={v.shape}, dtype={v.dtype}')
    else:
        print(f'  {k}: {v}')
"
```
