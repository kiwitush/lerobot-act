"""
零样本跨环境评估脚本。

加载训练好的 baseline 和 joint 模型，在未见环境 D 上评估并对比。
使用 LeRobot ACTPolicy 的 select_action 接口进行推理。

用法:
    python scripts/eval.py --config configs/act_eval.yaml
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

from src.calvin_env import evaluate_policy_on_calvin
from src.utils.policy_utils import build_act_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_policy(checkpoint_path: str, config: Dict, device: torch.device):
    """从检查点加载 LeRobot ACTPolicy。"""
    from lerobot.policies.act.modeling_act import ACTPolicy

    act_cfg = build_act_config(config["policy"])
    policy = ACTPolicy(act_cfg)
    policy.to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    policy.load_state_dict(ckpt["model_state_dict"])
    policy.eval()

    logger.info("已加载检查点: %s (epoch %d)", checkpoint_path, ckpt.get("epoch", -1))
    return policy


def run_zero_shot_eval(policy, eval_cfg: Dict, device: torch.device) -> Dict:
    """运行零样本评估，若 CALVIN 环境不可用则返回占位结果。"""
    try:
        return evaluate_policy_on_calvin(
            policy=policy,
            env_name=eval_cfg["env_name"],
            num_episodes=eval_cfg["num_episodes"],
            max_steps=eval_cfg["max_steps"],
            device=str(device),
        )
    except (ImportError, RuntimeError) as e:
        logger.warning("CALVIN 仿真环境不可用 (%s)，返回占位结果。", e)
        return {
            "success_rate": 0.0,
            "avg_steps": 0.0,
            "note": "CALVIN 环境不可用 — 需要 Mujoco + CALVIN 仿真。",
        }


def main():
    parser = argparse.ArgumentParser(description="CALVIN 环境 D 零样本评估 (基于 LeRobot)")
    parser.add_argument("--config", type=str, default="configs/act_eval.yaml", help="YAML 配置文件路径")
    parser.add_argument("--baseline-ckpt", type=str, default=None, help="覆盖 baseline 检查点路径")
    parser.add_argument("--joint-ckpt", type=str, default=None, help="覆盖 joint 检查点路径")
    args = parser.parse_args()

    config_path = args.config
    if not Path(config_path).exists():
        logger.error("配置文件未找到: %s", config_path)
        sys.exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    eval_cfg = config["eval"]
    seed = eval_cfg.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    device = torch.device(eval_cfg["device"] if torch.cuda.is_available() else "cpu")
    logger.info("使用设备: %s", device)

    results = {}

    # 评估 Baseline
    baseline_ckpt = args.baseline_ckpt or config["model"]["baseline_checkpoint"]
    logger.info("评估 BASELINE — 环境 %s", eval_cfg["env_name"])

    if not Path(baseline_ckpt).exists():
        logger.error("Baseline 检查点未找到: %s", baseline_ckpt)
        results["baseline"] = {"error": f"Checkpoint not found: {baseline_ckpt}"}
    else:
        baseline_policy = load_policy(baseline_ckpt, config, device)
        results["baseline"] = run_zero_shot_eval(baseline_policy, eval_cfg, device)
        logger.info("Baseline 成功率: %.2f%%", results["baseline"].get("success_rate", 0) * 100)

    # 评估 Joint
    joint_ckpt = args.joint_ckpt or config["model"]["joint_checkpoint"]
    logger.info("评估 JOINT — 环境 %s", eval_cfg["env_name"])

    if not Path(joint_ckpt).exists():
        logger.error("Joint 检查点未找到: %s", joint_ckpt)
        results["joint"] = {"error": f"Checkpoint not found: {joint_ckpt}"}
    else:
        joint_policy = load_policy(joint_ckpt, config, device)
        results["joint"] = run_zero_shot_eval(joint_policy, eval_cfg, device)
        logger.info("Joint 成功率: %.2f%%", results["joint"].get("success_rate", 0) * 100)

    # 对比
    logger.info("=" * 50)
    logger.info("BASELINE vs JOINT 零样本对比")
    logger.info("=" * 50)
    logger.info("Baseline — 成功率: %.2f%%", results["baseline"].get("success_rate", 0) * 100)
    logger.info("Joint    — 成功率: %.2f%%", results["joint"].get("success_rate", 0) * 100)

    # 保存
    output = {
        "env": eval_cfg["env_name"],
        "num_episodes": eval_cfg["num_episodes"],
        "baseline": results["baseline"],
        "joint": results["joint"],
    }

    output_dir = Path(config["experiment"].get("output_dir", "./outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "logs").mkdir(exist_ok=True)

    result_path = output_dir / "logs" / "eval_results.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    logger.info("结果已保存至 %s", result_path)


if __name__ == "__main__":
    main()
