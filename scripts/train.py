"""
ACT 策略训练脚本，使用 LeRobot 内置的 ACTPolicy 和 LeRobotDataset。

通过 YAML 配置文件驱动，支持:
  - Baseline: 单环境 B 训练
  - Joint: 多环境 (A+B+C) 联合训练
  - WandB / SwanLab 日志记录
"""

import argparse
import json
import logging
import random
import sys
from collections import defaultdict
from importlib import import_module
from pathlib import Path
from typing import Callable, Dict

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.calvin_loader import (
    DEFAULT_TRAIN_REPO_ID,
    get_dataset_stats,
    infer_dataset_spec,
    load_calvin_dataset,
)
from src.utils.logger import get_logger_safe
from src.utils.policy_utils import build_act_config, extract_policy_forward_metrics, resolve_policy_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_policy(policy_cfg: Dict, dataset_spec: Dict, device: str):
    """使用 LeRobot 内置 ACTPolicy 创建策略。"""
    from lerobot.policies.act.modeling_act import ACTPolicy

    resolved_cfg = resolve_policy_config(policy_cfg, dataset_spec)
    act_cfg = build_act_config(resolved_cfg)
    policy = ACTPolicy(act_cfg)
    policy.to(device)

    n_params = sum(p.numel() for p in policy.parameters())
    n_trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    logger.info(
        "ACTPolicy 创建成功: dim_model=%d, enc=%d, dec=%d, ff=%d, cameras=%s",
        resolved_cfg["hidden_dim"],
        resolved_cfg["n_encoder_layers"],
        resolved_cfg["n_decoder_layers"],
        resolved_cfg["dim_feedforward"],
        resolved_cfg["camera_names"],
    )
    logger.info("参数量: 总计 %.2fM, 可训练 %.2fM", n_params / 1e6, n_trainable / 1e6)

    return policy, resolved_cfg


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

    return torch.optim.AdamW(
        param_groups,
        weight_decay=training_cfg.get("weight_decay", 1e-4),
    )


def create_preprocessor(policy, dataset_stats) -> Callable:
    """构建与 LeRobot 版本兼容的 batch 预处理器。"""
    if dataset_stats is None:
        logger.warning("未拿到 dataset stats，跳过 LeRobot 预处理。")
        return lambda batch: batch

    config_obj = getattr(policy, "config", getattr(policy, "cfg", None))
    if config_obj is None:
        logger.warning("策略对象无 config/cfg，跳过 LeRobot 预处理。")
        return lambda batch: batch

    import_errors = []
    for module_name in [
        "lerobot.policies.factory",
        "lerobot.common.policies.factory",
    ]:
        try:
            module = import_module(module_name)
            make_pre_post_processors = getattr(module, "make_pre_post_processors")
            preprocess, _ = make_pre_post_processors(config_obj, dataset_stats=dataset_stats)
            logger.info("启用 LeRobot 预处理器: %s.make_pre_post_processors", module_name)
            return preprocess
        except Exception as exc:
            import_errors.append(f"{module_name}: {exc}")

    logger.warning("未能初始化 LeRobot 预处理器，回退为 no-op。详情: %s", " | ".join(import_errors))
    return lambda batch: batch


def build_dataloader(dataset, batch_size: int, num_workers: int, shuffle: bool):
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=shuffle,
    )


def train_one_epoch(
    policy,
    dataloader,
    optimizer,
    accelerator,
    preprocess_batch: Callable,
    grad_clip_norm: float,
    step_begin: int,
    log_interval: int,
    exp_logger,
):
    """单轮训练，返回平均指标。"""
    policy.train()
    metric_sums = defaultdict(float)
    n_batches = 0

    for batch in dataloader:
        batch = preprocess_batch(batch)
        with accelerator.autocast():
            output = policy.forward(batch)
            loss_tensor, batch_metrics = extract_policy_forward_metrics(output)

        accelerator.backward(loss_tensor)

        if grad_clip_norm > 0:
            accelerator.clip_grad_norm_(policy.parameters(), grad_clip_norm)

        optimizer.step()
        optimizer.zero_grad()

        for key, value in batch_metrics.items():
            metric_sums[key] += value
        n_batches += 1

        step = step_begin + n_batches
        if n_batches % log_interval == 0:
            exp_logger.log(
                {
                    "train/loss": batch_metrics["loss"],
                    "train/action_l1_loss": batch_metrics.get("action_l1_loss", batch_metrics["loss"]),
                    "train/kl_loss": batch_metrics.get("kl_loss", 0.0),
                    "train/lr": optimizer.param_groups[0]["lr"],
                },
                step=step,
            )

    if n_batches == 0:
        return {"loss": float("inf"), "action_l1_loss": float("inf"), "kl_loss": float("inf")}

    return {key: value / n_batches for key, value in metric_sums.items()}


@torch.no_grad()
def validate(policy, dataloader, accelerator, preprocess_batch: Callable, exp_logger, step: int):
    """在验证集上计算平均损失。"""
    policy.eval()
    metric_sums = defaultdict(float)
    n_batches = 0

    for batch in dataloader:
        batch = preprocess_batch(batch)
        with accelerator.autocast():
            output = policy.forward(batch)
            _, batch_metrics = extract_policy_forward_metrics(output)
        for key, value in batch_metrics.items():
            metric_sums[key] += value
        n_batches += 1

    if n_batches == 0:
        raise RuntimeError("验证集为空，无法计算验证指标。")

    avg_metrics = {key: value / n_batches for key, value in metric_sums.items()}
    exp_logger.log(
        {
            "val/loss": avg_metrics["loss"],
            "val/action_l1_loss": avg_metrics.get("action_l1_loss", avg_metrics["loss"]),
            "val/kl_loss": avg_metrics.get("kl_loss", 0.0),
        },
        step=step,
    )
    return avg_metrics


def save_checkpoint(
    policy,
    optimizer,
    epoch: int,
    metric_value: float,
    output_dir: Path,
    exp_name: str,
    accelerator=None,
    is_best: bool = False,
    policy_config: Dict = None,
    dataset_spec: Dict = None,
    dataset_stats=None,
    metric_name: str = "loss",
):
    """保存检查点，按实验名隔离避免 baseline/joint 互相覆盖。"""
    ckpt_dir = output_dir / "checkpoints" / exp_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    model = accelerator.unwrap_model(policy) if accelerator is not None else policy
    ckpt = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "selection_metric": metric_value,
        "selection_metric_name": metric_name,
        "policy_config": policy_config,
        "dataset_spec": dataset_spec,
        "dataset_stats": dataset_stats,
    }

    if is_best:
        path = ckpt_dir / "best_model.pt"
        torch.save(ckpt, path)
        logger.info("最优模型已保存: %s (%s=%.6f)", path, metric_name, metric_value)
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

    exp_logger = get_logger_safe(
        project=exp_cfg.get("project", "lerobot-act"),
        name=exp_cfg["name"],
        config=config,
        backend=exp_cfg.get("backend", "none"),
    )

    repo_id = data_cfg.get("repo_id", DEFAULT_TRAIN_REPO_ID)
    train_envs = data_cfg.get("train_envs")
    val_envs = data_cfg.get("val_envs", train_envs)
    strict_env_filter = data_cfg.get("strict_env_filter", True)
    env_episode_map_path = data_cfg.get("env_episode_map_path")
    local_dir = data_cfg.get("local_dir")
    val_split_ratio = data_cfg.get("val_split_ratio", 0.0)

    logger.info("训练环境: %s", train_envs if train_envs else "全部")
    logger.info("验证环境: %s", val_envs if val_envs else "全部")

    full_dataset = load_calvin_dataset(
        repo_id=repo_id,
        split="train",
        envs=train_envs,
        local_dir=local_dir,
        strict_env_filter=strict_env_filter,
        env_episode_map_path=env_episode_map_path,
    )

    # LeRobot 0.5+ 无 split 参数；若配置了 val_split_ratio 则手动划分
    if val_split_ratio > 0:
        n_total = len(full_dataset)
        n_train = int(n_total * (1 - val_split_ratio))
        n_val = n_total - n_train
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [n_train, n_val],
            generator=torch.Generator().manual_seed(exp_cfg.get("seed", 42)),
        )
        logger.info("手动划分 train/val: %d/%d 帧", n_train, n_val)
    else:
        train_dataset = full_dataset
        val_dataset = load_calvin_dataset(
            repo_id=repo_id,
            split="validation",
            envs=val_envs,
            local_dir=local_dir,
            strict_env_filter=strict_env_filter,
            env_episode_map_path=env_episode_map_path,
        )

    dataset_spec = infer_dataset_spec(full_dataset, preferred_camera_names=config["policy"].get("camera_names"))
    dataset_stats = get_dataset_stats(full_dataset)

    train_loader = build_dataloader(
        train_dataset,
        batch_size=training_cfg["batch_size"],
        num_workers=training_cfg.get("num_workers", 4),
        shuffle=True,
    )
    val_loader = build_dataloader(
        val_dataset,
        batch_size=training_cfg["batch_size"],
        num_workers=training_cfg.get("num_workers", 0),
        shuffle=False,
    )

    policy, resolved_policy_cfg = create_policy(config["policy"], dataset_spec, device)
    optimizer = create_optimizer(policy, config)
    preprocess_batch = create_preprocessor(policy, dataset_stats)

    from accelerate import Accelerator

    mixed_precision = "fp16" if device == "cuda" else "no"
    accelerator = Accelerator(mixed_precision=mixed_precision)

    start_epoch = 0
    best_val_metric = float("inf")
    best_metric_name = "action_l1_loss"
    patience_epochs = 0

    epochs = training_cfg["epochs"]
    val_every = training_cfg.get("val_every", 5)
    save_every = training_cfg.get("save_every", 20)
    early_stop_patience = training_cfg.get("early_stop_patience", 30)
    grad_clip_norm = training_cfg.get("grad_clip_norm", 1.0)
    log_interval = log_cfg.get("log_interval", 50)
    exp_name = exp_cfg["name"]

    if args.resume:
        logger.info("从检查点恢复: %s", args.resume)
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        policy.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val_metric = ckpt.get("selection_metric", float("inf"))
        best_metric_name = ckpt.get("selection_metric_name", best_metric_name)

    policy, optimizer, train_loader, val_loader = accelerator.prepare(
        policy, optimizer, train_loader, val_loader
    )

    history = {
        "train_loss": [],
        "train_action_l1_loss": [],
        "train_kl_loss": [],
        "val_loss": [],
        "val_action_l1_loss": [],
        "val_kl_loss": [],
    }

    logger.info("开始训练: %s, %d epochs, batch_size=%d", exp_name, epochs, training_cfg["batch_size"])
    logger.info("检查点目录: %s", output_dir / "checkpoints" / exp_name)

    stopped_early = False
    last_epoch = start_epoch - 1

    for epoch in range(start_epoch, epochs):
        last_epoch = epoch
        step_begin = epoch * len(train_loader)
        train_metrics = train_one_epoch(
            policy=policy,
            dataloader=train_loader,
            optimizer=optimizer,
            accelerator=accelerator,
            preprocess_batch=preprocess_batch,
            grad_clip_norm=grad_clip_norm,
            step_begin=step_begin,
            log_interval=log_interval,
            exp_logger=exp_logger,
        )

        history["train_loss"].append(train_metrics["loss"])
        history["train_action_l1_loss"].append(train_metrics.get("action_l1_loss", train_metrics["loss"]))
        history["train_kl_loss"].append(train_metrics.get("kl_loss", 0.0))

        exp_logger.log(
            {
                "epoch": epoch,
                "train/epoch_loss": train_metrics["loss"],
                "train/epoch_action_l1_loss": train_metrics.get("action_l1_loss", train_metrics["loss"]),
                "train/epoch_kl_loss": train_metrics.get("kl_loss", 0.0),
            },
            step=step_begin + len(train_loader),
        )

        if epoch % 2 == 0 or epoch == start_epoch:
            logger.info(
                "Epoch %d/%d | train_loss=%.6f | train_action_l1=%.6f",
                epoch + 1,
                epochs,
                train_metrics["loss"],
                train_metrics.get("action_l1_loss", train_metrics["loss"]),
            )

        if epoch % val_every == 0:
            val_metrics = validate(
                policy=policy,
                dataloader=val_loader,
                accelerator=accelerator,
                preprocess_batch=preprocess_batch,
                exp_logger=exp_logger,
                step=step_begin + len(train_loader),
            )

            history["val_loss"].append(val_metrics["loss"])
            history["val_action_l1_loss"].append(val_metrics.get("action_l1_loss", val_metrics["loss"]))
            history["val_kl_loss"].append(val_metrics.get("kl_loss", 0.0))

            logger.info(
                "Epoch %d/%d | val_loss=%.6f | val_action_l1=%.6f",
                epoch + 1,
                epochs,
                val_metrics["loss"],
                val_metrics.get("action_l1_loss", val_metrics["loss"]),
            )

            current_metric_name = "action_l1_loss" if "action_l1_loss" in val_metrics else "loss"
            current_metric_value = val_metrics[current_metric_name]

            if current_metric_value < best_val_metric:
                best_val_metric = current_metric_value
                best_metric_name = current_metric_name
                patience_epochs = 0
                save_checkpoint(
                    policy=policy,
                    optimizer=optimizer,
                    epoch=epoch,
                    metric_value=current_metric_value,
                    output_dir=output_dir,
                    exp_name=exp_name,
                    accelerator=accelerator,
                    is_best=True,
                    policy_config=resolved_policy_cfg,
                    dataset_spec=dataset_spec,
                    dataset_stats=dataset_stats,
                    metric_name=current_metric_name,
                )
            else:
                patience_epochs += val_every
                if patience_epochs >= early_stop_patience:
                    logger.info(
                        "早停触发，epoch=%d, best_%s=%.6f",
                        epoch + 1,
                        best_metric_name,
                        best_val_metric,
                    )
                    stopped_early = True
                    break

        if epoch > 0 and epoch % save_every == 0:
            save_checkpoint(
                policy=policy,
                optimizer=optimizer,
                epoch=epoch,
                metric_value=train_metrics.get("action_l1_loss", train_metrics["loss"]),
                output_dir=output_dir,
                exp_name=exp_name,
                accelerator=accelerator,
                policy_config=resolved_policy_cfg,
                dataset_spec=dataset_spec,
                dataset_stats=dataset_stats,
                metric_name="action_l1_loss" if "action_l1_loss" in train_metrics else "loss",
            )

    final_epoch = last_epoch if stopped_early else max(last_epoch, epochs - 1)
    final_metric = history["train_action_l1_loss"][-1] if history["train_action_l1_loss"] else float("inf")
    save_checkpoint(
        policy=policy,
        optimizer=optimizer,
        epoch=final_epoch,
        metric_value=final_metric,
        output_dir=output_dir,
        exp_name=exp_name,
        accelerator=accelerator,
        policy_config=resolved_policy_cfg,
        dataset_spec=dataset_spec,
        dataset_stats=dataset_stats,
        metric_name="action_l1_loss",
    )

    metrics = {
        **history,
        "best_val_metric": best_val_metric,
        "best_val_metric_name": best_metric_name,
        "dataset_spec": dataset_spec,
        "resolved_policy_config": resolved_policy_cfg,
        "config": {k: v for k, v in config.items() if k != "policy"},
    }
    metrics_path = output_dir / "logs" / f"{exp_cfg['name']}_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    logger.info("训练指标已保存至 %s", metrics_path)

    exp_logger.finish()
    logger.info("训练完成。最优 %s: %.6f", best_metric_name, best_val_metric)


if __name__ == "__main__":
    main()
