"""
ACT 训练与评估指标。

包含 Action L1 Loss、chunk 内时间衰减分析，以及 baseline vs. joint 对比工具。
"""

from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F


def compute_action_l1_loss(
    pred_actions: torch.Tensor,
    target_actions: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """计算预测动作块与目标动作块之间的 L1 损失。"""
    loss = F.l1_loss(pred_actions, target_actions, reduction="none")  # (B, chunk, dim)

    if mask is not None:
        mask = mask.unsqueeze(-1).float()
        loss = loss * mask
        return loss.sum() / (mask.sum() * loss.shape[-1] + 1e-8)

    return loss.mean()


def compute_chunk_breakdown(
    pred_actions: torch.Tensor,
    target_actions: torch.Tensor,
) -> Dict[str, float]:
    """按 chunk 时间步分解 L1 误差，分析远/中/近期预测精度衰减。"""
    errors = F.l1_loss(pred_actions, target_actions, reduction="none")  # (B, chunk, dim)
    mean_error = errors.mean(dim=(0, 2))  # (chunk,)

    breakdown = {}
    chunk_size = errors.shape[1]
    for i in range(chunk_size):
        breakdown[f"action_l1_step_{i:03d}"] = mean_error[i].item()

    # 按时间远近分组
    b3 = chunk_size // 3
    breakdown["action_l1_near"] = errors[:, :b3].mean().item()
    breakdown["action_l1_mid"]  = errors[:, b3 : 2 * b3].mean().item()
    breakdown["action_l1_far"]  = errors[:, 2 * b3 :].mean().item()

    return breakdown


def compare_runs(
    baseline_metrics: Dict[str, List[float]],
    joint_metrics: Dict[str, List[float]],
) -> Dict[str, Dict]:
    """对比 baseline 与 joint 训练指标，返回最优验证损失、收敛轮次和 delta 差值。"""
    result = {}

    for name, metrics in [("baseline", baseline_metrics), ("joint", joint_metrics)]:
        val_loss = np.array(metrics.get("val_loss", [0]))
        train_loss = np.array(metrics.get("train_loss", [0]))

        result[name] = {
            "best_val_loss": float(val_loss.min()),
            "best_val_epoch": int(val_loss.argmin()),
            "final_train_loss": float(train_loss[-1]) if len(train_loss) > 0 else 0.0,
            "convergence_epoch": _find_convergence(val_loss),
        }

    delta = result["baseline"]["best_val_loss"] - result["joint"]["best_val_loss"]
    result["delta"] = {
        "best_val_loss": delta,
        "convergence_gap": result["baseline"]["convergence_epoch"] - result["joint"]["convergence_epoch"],
    }

    return result


def _find_convergence(val_loss: np.ndarray, patience: int = 10, threshold: float = 1e-4) -> int:
    """基于验证损失平台期检测收敛轮次。"""
    if len(val_loss) < patience + 1:
        return len(val_loss)
    best = val_loss.min()
    for i in range(len(val_loss) - patience):
        window = val_loss[i : i + patience]
        if (window.max() - window.min()) < threshold and abs(window[-1] - best) < threshold * 10:
            return i
    return len(val_loss)
