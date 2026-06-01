"""
ACT 策略训练脚本，使用 LeRobot 框架内置的 ACTPolicy 和 LeRobotDataset。

通过 YAML 配置文件驱动，支持:
  - Baseline: 单环境 (A) 训练
  - Joint: 多环境 (A+B+C) 联合训练
  - WandB / SwanLab 日志记录

用法:
    python scripts/train.py --config configs/act_baseline.yaml
    python scripts/train.py --config configs/act_joint.yaml
"""

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.calvin_loader import build_calvin_dataloader, build_calvin_val_dataloader
from src.utils.logger import get_logger_safe
from src.utils.policy_utils import build_act_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_policy(config: Dict, device: str):
    """使用 LeRobot 内置 ACTPolicy 创建策略。"""
    from lerobot.policies.act.modeling_act import ACTPolicy

    policy_cfg = config["policy"]
    act_cfg = build_act_config(policy_cfg)
    policy = ACTPolicy(act_cfg)
    policy.to(device)

    n_params = sum(p.numel() for p in policy.parameters())
    n_trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    logger.info("ACTPolicy 创建成功: dim_model=%d, enc=%d, dec=%d, ff=%d",
                policy_cfg["hidden_dim"], policy_cfg["n_encoder_layers"],
                policy_cfg["n_decoder_layers"], policy_cfg["dim_feedforward"])
    logger.info("参数量: 总计 %.2fM, 可训练 %.2fM", n_params / 1e6, n_trainable / 1e6)

    return policy


def create_optimizer(policy: torch.nn.Module, config: Dict):
    """创建优化器，视觉 backbone 使用更小的学习率。"""
    training_cfg = config["training"]
    lr = training_cfg["lr"]
    lr_backbone = training_cfg.get("lr_backbone", lr * 0.1)

    backbone_params = []
    other_params = []

    for name, param in policy.named_parameters():
        if not param.requires_grad:
            continue
        if any(k in name for k in ["backbone", "vision_encoder", "resnet", "trunk"]):
            backbone_params.append(param)
        else:
            other_params.append(param)

    param_groups = []
    if other_params:
        param_groups.append({"params": other_params, "lr": lr})
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": lr_backbone})

    optimizer = torch.optim.AdamW(
        param_groups,
        weight_decay=training_cfg.get("weight_decay", 1e-4),
    )

    return optimizer


def train_one_epoch(policy, dataloader, optimizer, accelerator, grad_clip_norm: float, step_begin: int, log_interval: int, exp_logger):
    """单轮训练，返回平均损失。"""
    policy.train()
    total_loss = 0.0
    n_batches = 0

    for batch in dataloader:
        with accelerator.autocast():
            output = policy.forward(batch)
            loss = output["loss"] if isinstance(output, dict) else output[0]

        accelerator.backward(loss)

        if grad_clip_norm > 0:
            accelerator.clip_grad_norm_(policy.parameters(), grad_clip_norm)

        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()
        n_batches += 1

        step = step_begin + n_batches
        if n_batches % log_interval == 0:
            exp_logger.log({
                "train/loss": loss.item(),
                "train/lr": optimizer.param_groups[0]["lr"],
            }, step=step)

    return total_loss / n_batches if n_batches > 0 else float("inf")


@torch.no_grad()
def validate(policy, dataloader, accelerator, exp_logger, step: int):
    """在验证集上计算平均损失。"""
    policy.eval()
    total_loss = 0.0
    n_batches = 0

    for batch in dataloader:
        with accelerator.autocast():
            output = policy.forward(batch)
            loss = output["loss"] if isinstance(output, dict) else output[0]
        total_loss += loss.item()
        n_batches += 1

    avg_loss = total_loss / n_batches if n_batches > 0 else float("inf")
    exp_logger.log({"val/loss": avg_loss}, step=step)
    return avg_loss


def save_checkpoint(policy, optimizer, epoch: int, loss: float, output_dir: Path, exp_name: str, accelerator=None, is_best: bool = False):
    """保存检查点，按实验名隔离避免 baseline/joint 互相覆盖。"""
    ckpt_dir = output_dir / "checkpoints" / exp_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # 保存 unwrapped model 的 state_dict，避免 DDP 下 key 带 module. 前缀
    model = accelerator.unwrap_model(policy) if accelerator is not None else policy
    ckpt = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }

    if is_best:
        path = ckpt_dir / "best_model.pt"
        torch.save(ckpt, path)
        logger.info("最优模型已保存: %s (loss=%.6f)", path, loss)
    else:
        path = ckpt_dir / f"epoch_{epoch:04d}.pt"
        torch.save(ckpt, path)


def load_config(config_path: str) -> Dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="ACT 策略训练 (基于 LeRobot)")
    parser.add_argument("--config", type=str, required=True, help="YAML 配置文件路径")
    parser.add_argument("--resume", type=str, default=None, help="从检查点恢复训练")
    parser.add_argument("--device", type=str, default="cuda", help="训练设备")
    args = parser.parse_args()

    config = load_config(args.config)
    exp_cfg = config["experiment"]
    data_cfg = config["data"]
    training_cfg = config["training"]
    log_cfg = config.get("logging", {})

    set_seed(exp_cfg.get("seed", 42))

    output_dir = Path(exp_cfg.get("output_dir", "./outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(exist_ok=True)
    (output_dir / "logs").mkdir(exist_ok=True)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    logger.info("设备: %s", device)

    # 初始化日志器
    exp_logger = get_logger_safe(
        project=exp_cfg.get("project", "lerobot-act"),
        name=exp_cfg["name"],
        config=config,
        backend=exp_cfg.get("backend", "none"),
    )

    # 构建数据加载器
    train_envs = data_cfg.get("train_envs", None)
    logger.info("训练环境: %s", train_envs if train_envs else "全部")

    train_loader = build_calvin_dataloader(
        repo_id=data_cfg.get("repo_id", "lerobot/calvin"),
        split="train",
        envs=train_envs,
        batch_size=training_cfg["batch_size"],
        num_workers=training_cfg.get("num_workers", 4),
        shuffle=True,
        local_dir=data_cfg.get("local_dir", None),
    )

    val_loader = build_calvin_val_dataloader(
        repo_id=data_cfg.get("repo_id", "lerobot/calvin"),
        batch_size=training_cfg["batch_size"],
        num_workers=training_cfg.get("num_workers", 0),
        local_dir=data_cfg.get("local_dir", None),
    )

    # 创建策略
    policy = create_policy(config, device)

    # 创建优化器
    optimizer = create_optimizer(policy, config)

    # HuggingFace Accelerator
    from accelerate import Accelerator
    accelerator = Accelerator(mixed_precision="fp16")

    start_epoch = 0
    best_val_loss = float("inf")
    patience_epochs = 0

    epochs = training_cfg["epochs"]
    val_every = training_cfg.get("val_every", 5)
    save_every = training_cfg.get("save_every", 20)
    early_stop_patience = training_cfg.get("early_stop_patience", 30)
    grad_clip_norm = training_cfg.get("grad_clip_norm", 1.0)
    log_interval = log_cfg.get("log_interval", 50)
    exp_name = exp_cfg["name"]

    # 恢复训练 (必须在 accelerator.prepare 之前，避免 state_dict key 错位)
    if args.resume:
        logger.info("从检查点恢复: %s", args.resume)
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        policy.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("loss", float("inf"))

    policy, optimizer, train_loader, val_loader = accelerator.prepare(
        policy, optimizer, train_loader, val_loader
    )

    # 训练指标记录
    train_losses = []
    val_losses = []

    logger.info("开始训练: %s, %d epochs, batch_size=%d",
                exp_name, epochs, training_cfg["batch_size"])
    logger.info("检查点目录: %s", output_dir / "checkpoints" / exp_name)

    for epoch in range(start_epoch, epochs):
        step_begin = epoch * len(train_loader)
        train_loss = train_one_epoch(
            policy, train_loader, optimizer, accelerator,
            grad_clip_norm, step_begin, log_interval, exp_logger,
        )
        train_losses.append(train_loss)

        exp_logger.log({"epoch": epoch, "train/epoch_loss": train_loss}, step=step_begin + len(train_loader))

        if epoch % 2 == 0 or epoch == start_epoch:
            logger.info("Epoch %d/%d | train_loss=%.6f", epoch + 1, epochs, train_loss)

        # 验证
        if epoch % val_every == 0:
            val_loss = validate(
                policy, val_loader, accelerator,
                exp_logger, step=step_begin + len(train_loader),
            )
            val_losses.append(val_loss)

            logger.info("Epoch %d/%d | val_loss=%.6f", epoch + 1, epochs, val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_epochs = 0
                save_checkpoint(policy, optimizer, epoch, val_loss, output_dir, exp_name, accelerator, is_best=True)
            else:
                patience_epochs += val_every
                if patience_epochs >= early_stop_patience:
                    logger.info("早停触发，epoch=%d, best_val_loss=%.6f", epoch + 1, best_val_loss)
                    break

        # 定期保存检查点
        if epoch > 0 and epoch % save_every == 0:
            save_checkpoint(policy, optimizer, epoch, train_loss, output_dir, exp_name, accelerator)

    # 最终保存 (使用实际最后 epoch 编号)
    last_epoch = epoch if patience_epochs >= early_stop_patience else epochs - 1
    save_checkpoint(policy, optimizer, last_epoch, train_loss, output_dir, exp_name, accelerator)

    # 保存训练指标
    metrics = {
        "train_loss": train_losses,
        "val_loss": val_losses,
        "best_val_loss": best_val_loss,
        "config": {k: v for k, v in config.items() if k != "policy"},
    }
    metrics_path = output_dir / "logs" / f"{exp_cfg['name']}_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logger.info("训练指标已保存至 %s", metrics_path)

    exp_logger.finish()
    logger.info("训练完成。最优验证损失: %.6f", best_val_loss)


if __name__ == "__main__":
    main()
