# LeRobot-ACT: 具身智能跨环境泛化挑战

基于 [LeRobot](https://github.com/huggingface/lerobot) 框架内置的 ACT (Action Chunking with Transformers) 策略，在 [CALVIN](https://github.com/mees/calvin) 基准上探索跨环境视觉泛化能力。

## 项目概述

本项目聚焦于具身智能中的动作策略学习与环境泛化问题，采用轻量级的 ACT 算法，通过以下三个实验阶段系统评估模型的泛化能力：

| 阶段 | 描述 | 训练数据 | 测试环境 |
|---|---|---|---|
| **基础策略训练 (Baseline)** | 单环境视觉-动作策略学习 | 环境 A | 环境 A (验证) |
| **多环境联合训练 (Joint)** | 混合多环境数据的联合训练 | 环境 A + B + C | 环境 D (验证) |
| **零样本跨环境测试 (Zero-shot)** | 在未见环境中评估泛化能力 | — | 环境 D |

## 环境配置

### 依赖安装

```bash
# 克隆仓库
git clone https://github.com/<your-username>/lerobot-act.git
cd lerobot-act

# 创建 Conda 环境
conda env create -f environment.yml
conda activate lerobot-act

# 或使用 pip
pip install -r requirements.txt
```

### 系统要求

- Python 3.10
- CUDA 12.1+ (GPU 训练推荐)
- 至少 16GB GPU 显存 (推荐 24GB+)
- MuJoCo 2.3.0 (仅评估时需要，用于 CALVIN 仿真环境)

## 数据准备

本项目使用 HuggingFace Hub 上的 `lerobot/calvin` 数据集（LeRobot 格式，已对齐）。

### 方式一：直接从 HuggingFace Hub 加载（推荐）

训练脚本默认从 HuggingFace Hub 自动下载，无需手动准备数据：

```bash
python scripts/train.py --config configs/act_baseline.yaml
```

首次运行时 HuggingFace 会自动缓存数据集到本地。

### 方式二：本地 LeRobot 格式数据

若网络受限，可提前下载数据集到本地：

```bash
# 通过 huggingface-cli 下载
huggingface-cli download lerobot/calvin --repo-type dataset --local-dir ./data/calvin_lerobot

# 训练时指定本地路径
python scripts/train.py --config configs/act_baseline.yaml \
    data.local_dir=./data/calvin_lerobot
```

### 方式三：从 CALVIN 原始 .npz 转换

若已有 CALVIN 原始数据，可使用转换脚本：

```bash
python scripts/prepare_data.py \
    --input ./data/calvin_raw/task_ABC_D \
    --output ./data/calvin_lerobot
```

## 项目结构

```
lerobot-act/
├── configs/                     # YAML 实验配置
│   ├── act_baseline.yaml        # 单环境基线训练配置
│   ├── act_joint.yaml           # 多环境联合训练配置
│   └── act_eval.yaml            # 零样本评估配置
├── scripts/                     # 可执行脚本
│   ├── train.py                 # 训练入口 (使用 LeRobot ACTPolicy)
│   ├── eval.py                  # 零样本评估
│   ├── visualize.py             # 结果可视化
│   └── prepare_data.py          # CALVIN → LeRobot 格式转换 (可选)
├── src/                         # 核心源码
│   ├── calvin_loader.py         # CALVIN 数据加载 (封装 LeRobotDataset)
│   ├── calvin_env.py            # CALVIN Gym 环境封装
│   └── utils/
│       ├── logger.py            # WandB/SwanLab 日志接口
│       └── metrics.py           # Action L1 Loss 等指标计算
├── notebooks/                   # Jupyter 分析笔记
│   ├── 01_data_exploration.ipynb
│   ├── 02_training_analysis.ipynb
│   └── 03_zero_shot_results.ipynb
├── outputs/                     # 训练产出
│   ├── checkpoints/
│   └── logs/
└── data/                        # 本地数据集目录 (可选)
```

## 训练

### 基础策略训练 (Baseline) — 单环境 A

```bash
python scripts/train.py --config configs/act_baseline.yaml
```

### 多环境联合训练 (Joint) — 环境 A + B + C

```bash
python scripts/train.py --config configs/act_joint.yaml
```

### WandB / SwanLab 日志

训练日志默认使用 WandB。切换为 SwanLab（国内用户推荐）:

```bash
python scripts/train.py --config configs/act_baseline.yaml \
    experiment.backend=swanlab
```

禁用在线日志:

```bash
python scripts/train.py --config configs/act_baseline.yaml \
    experiment.backend=none
```

## 评估

### 零样本跨环境测试

在未见环境 D 上评估训练好的模型:

```bash
python scripts/eval.py --config configs/act_eval.yaml
```

指定自定义 checkpoint:

```bash
python scripts/eval.py --config configs/act_eval.yaml \
    --baseline-ckpt ./outputs/checkpoints/act_baseline/best_model.pt \
    --joint-ckpt ./outputs/checkpoints/act_joint/best_model.pt
```

### 可视化结果

```bash
python scripts/visualize.py \
    --logs-dir ./outputs/logs \
    --output-dir ./outputs/figures \
    --eval-results ./outputs/logs/eval_results.json
```

## 实验结果

### 关键指标

| 指标 | 描述 | 目标 |
|---|---|---|
| Action L1 Loss | 预测动作与真实动作的 L1 距离 | 越低越好 |
| Success Rate | CALVIN 长序列任务成功率 | 越高越好 |
| Avg Steps | 平均 Episode 步数 | 越高越好 (完成更多子任务) |

### 超参数配置

| 参数 | 值 |
|---|---|
| **Network Architecture** | ResNet-18 + Transformer Encoder-Decoder |
| **Hidden Dimension** | 512 |
| **Encoder Layers** | 4 |
| **Decoder Layers** | 7 |
| **Attention Heads** | 8 |
| **Feedforward Dim** | 2048 |
| **Dropout** | 0.1 |
| **Action Chunk Size** | 100 |
| **Batch Size** | 32 |
| **Learning Rate (Transformer)** | 1e-4 |
| **Learning Rate (Backbone)** | 1e-5 |
| **Weight Decay** | 1e-4 |
| **Optimizer** | AdamW |
| **Epochs** | 200 |
| **Loss Function** | L1 Loss + KL Divergence (VAE) |

## 模型权重

训练好的模型权重可通过以下链接下载：

- **Baseline Model** (环境 A): [Google Drive / 百度网盘链接]
- **Joint Model** (环境 A+B+C): [Google Drive / 百度网盘链接]

> 提取码（如有）将在实验报告中标注。

## 引用

```bibtex
@inproceedings{zhao2023act,
  title={Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware},
  author={Zhao, Tony Z. and Kumar, Vikash and Levine, Sergey and Finn, Chelsea},
  booktitle={Robotics: Science and Systems (RSS)},
  year={2023}
}

@article{mees2022calvin,
  title={CALVIN: A Benchmark for Language-Conditioned Policy Learning for Long-Horizon Robot Manipulation Tasks},
  author={Mees, Oier and Hermann, Lukas and Rosete-Beas, Erick and Burgard, Wolfram},
  journal={IEEE Robotics and Automation Letters (RA-L)},
  year={2022}
}

@misc{lerobot2024,
  title={LeRobot: Open-Source Robot Learning},
  author={Hugging Face Robotics},
  year={2024},
  publisher={GitHub},
  url={https://github.com/huggingface/lerobot}
}
```

## 许可证

本项目仅用于学术研究目的。CALVIN 数据集和 LeRobot 框架分别遵循其各自的许可证。
