"""
训练结果可视化，使用真实训练输出的 metrics JSON。

生成用于实验报告的对比图:
  - Baseline vs. Joint 训练/验证总损失曲线
  - Baseline vs. Joint 训练/验证 Action L1 曲线
  - 零样本评估结果柱状图
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def smooth_curve(values: np.ndarray, window: int = 10) -> np.ndarray:
    """滑动平均平滑曲线。"""
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def _plot_metric(ax, baseline_metrics, joint_metrics, metric_key: str, title: str, ylabel: str, smooth_window: int):
    colors = {"baseline": "#2196F3", "joint": "#FF5722"}
    for label, metrics, color in [
        ("Baseline", baseline_metrics, colors["baseline"]),
        ("Joint", joint_metrics, colors["joint"]),
    ]:
        values = np.array(metrics.get(metric_key, []), dtype=float)
        if len(values) == 0:
            continue
        if len(values) > smooth_window:
            smooth = smooth_curve(values, smooth_window)
            ax.plot(smooth, color=color, label=label, linewidth=1.5)
            ax.plot(values, color=color, alpha=0.15, linewidth=0.5)
        else:
            ax.plot(values, color=color, label=label, linewidth=1.5, marker="o", markersize=3)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend()


def plot_training_curves(
    baseline_metrics: Dict[str, List[float]],
    joint_metrics: Dict[str, List[float]],
    output_path: str,
    title: str = "ACT 训练曲线",
    smooth_window: int = 10,
):
    """绘制 baseline vs. joint 训练曲线"""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.error("matplotlib 未安装。安装命令: pip install matplotlib")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    _plot_metric(axes[0, 0], baseline_metrics, joint_metrics, "train_loss", "训练总损失", "Total Loss", smooth_window)
    _plot_metric(axes[0, 1], baseline_metrics, joint_metrics, "val_loss", "验证总损失", "Total Loss", smooth_window)
    _plot_metric(
        axes[1, 0],
        baseline_metrics,
        joint_metrics,
        "train_action_l1_loss",
        "训练 Action L1",
        "Action L1 Loss",
        smooth_window,
    )
    _plot_metric(
        axes[1, 1],
        baseline_metrics,
        joint_metrics,
        "val_action_l1_loss",
        "验证 Action L1",
        "Action L1 Loss",
        smooth_window,
    )

    for ax in axes[1]:
        ax.set_xlabel("Epoch / Val Step")

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("训练曲线已保存至 %s", output_path)


def plot_eval_results(eval_results_path: str, output_path: str):
    """绘制零样本评估对比柱状图。"""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.error("matplotlib 未安装。")
        return

    with open(eval_results_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    baseline = results.get("baseline", {})
    joint = results.get("joint", {})

    if baseline.get("metric") == "success_rate" and joint.get("metric") == "success_rate":
        metric_label = "成功率 (%)"
        values = [baseline.get("success_rate", 0) * 100, joint.get("success_rate", 0) * 100]
        title = "任务成功率"
    else:
        metric_label = "Action L1 Loss"
        values = [baseline.get("action_l1_loss", 0), joint.get("action_l1_loss", 0)]
        title = "离线动作误差"

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    fig.suptitle(f"环境 {results.get('env', 'D')} 零样本评估", fontsize=13, fontweight="bold")

    colors = {"baseline": "#2196F3", "joint": "#FF5722"}
    models = ["Baseline", "Joint"]
    bars = ax.bar(models, values, color=[colors["baseline"], colors["joint"]])
    ax.set_ylabel(metric_label)
    ax.set_title(title)
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + (max(values) * 0.02 if max(values) > 0 else 0.02),
            f"{val:.3f}" if metric_label == "Action L1 Loss" else f"{val:.1f}%",
            ha="center",
            fontweight="bold",
        )
    if max(values) > 0:
        ax.set_ylim(0, max(values) * 1.25)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("评估结果已保存至 %s", output_path)


def main():
    parser = argparse.ArgumentParser(description="可视化 ACT 训练结果")
    parser.add_argument("--logs-dir", type=str, default="./outputs/logs")
    parser.add_argument("--output-dir", type=str, default="./outputs/figures")
    parser.add_argument("--eval-results", type=str, default="./outputs/logs/eval_results.json")
    parser.add_argument("--smooth", type=int, default=10, help="平滑窗口大小")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_metrics_path = Path(args.logs_dir) / "act_baseline_metrics.json"
    joint_metrics_path = Path(args.logs_dir) / "act_joint_metrics.json"

    if not baseline_metrics_path.exists() or not joint_metrics_path.exists():
        raise FileNotFoundError(
            f"未找到真实训练指标: {baseline_metrics_path} 或 {joint_metrics_path}"
        )

    with open(baseline_metrics_path, encoding="utf-8") as f:
        baseline_metrics = json.load(f)
    with open(joint_metrics_path, encoding="utf-8") as f:
        joint_metrics = json.load(f)
    logger.info("加载真实训练指标。")

    plot_training_curves(
        baseline_metrics,
        joint_metrics,
        str(output_dir / "training_curves.png"),
        title="ACT 训练: Baseline vs. Joint (基于 LeRobot)",
        smooth_window=args.smooth,
    )

    if Path(args.eval_results).exists():
        plot_eval_results(args.eval_results, str(output_dir / "eval_comparison.png"))
    else:
        logger.info("评估结果文件未找到 (%s)，跳过评估图。", args.eval_results)

    logger.info("所有图表已保存至 %s", output_dir)

    from src.utils.metrics import compare_runs

    cmp = compare_runs(baseline_metrics, joint_metrics)
    logger.info("Baseline best val loss: %.6f (epoch %d)", cmp["baseline"]["best_val_loss"], cmp["baseline"]["best_val_epoch"])
    logger.info("Joint best val loss:    %.6f (epoch %d)", cmp["joint"]["best_val_loss"], cmp["joint"]["best_val_epoch"])
    logger.info("Delta (baseline - joint): %.6f", cmp["delta"]["best_val_loss"])


if __name__ == "__main__":
    main()
