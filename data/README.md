# CALVIN 数据集准备说明

## 概述

本项目使用 LeRobot 格式的 CALVIN 数据集。数据保存在 HuggingFace Hub 上的 `xiaoma26/calvin-lerobot` 仓库中，可通过 LeRobotDataset 直接加载。

## 1. 直接从 HuggingFace Hub 加载 (推荐)

训练脚本默认从 HuggingFace Hub 自动下载，无需手动准备:

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset
dataset = LeRobotDataset("xiaoma26/calvin-lerobot", split="train")
```

首次运行时 HuggingFace 会自动缓存到 `~/.cache/huggingface/`。

## 2. 下载到本地

若网络环境受限，可提前下载:

```bash
# 方式 A: 通过 huggingface-cli
pip install huggingface_hub
huggingface-cli download xiaoma26/calvin-lerobot --repo-type dataset --local-dir ./data/calvin_lerobot

# 方式 B: 通过 Python
python -c "
from huggingface_hub import snapshot_download
snapshot_download('xiaoma26/calvin-lerobot', repo_type='dataset', local_dir='./data/calvin_lerobot')
"
```

下载后在配置中指定本地路径:

```yaml
# configs/act_baseline.yaml
data:
  repo_id: "xiaoma26/calvin-lerobot"
  local_dir: "./data/calvin_lerobot"  # 回退路径
```


## 3. 环境筛选

`xiaoma26/calvin-lerobot` 数据集在 episode 元数据中包含环境标签，训练脚本通过 `data.train_envs` 筛选:

```yaml
# Baseline: 仅环境 A
data:
  train_envs: ["A"]

# Joint: 环境 A+B+C
data:
  train_envs: ["A", "B", "C"]
```

验证集 (validation split) 默认对应环境 D，用于零样本评估。

## 4. 验证数据加载

```bash
python -c "
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset('xiaoma26/calvin-lerobot', split='train')
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
