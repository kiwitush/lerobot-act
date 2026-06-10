"""
零样本跨环境评估脚本。

优先在未见环境 D 上进行仿真 SuccessRate 评估；
若仿真环境不可用，则回退到环境 D 数据集上的离线动作误差评估。
"""

import argparse
import json
import logging
import random
import sys
from collections import defaultdict
from importlib import import_module
from pathlib import Path
from typing import Callable, Dict, Tuple

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.calvin_env import evaluate_policy_on_calvin
from src.calvin_loader import DEFAULT_EVAL_REPO_ID, get_dataset_stats, load_calvin_dataset
from src.utils.policy_utils import build_act_config, extract_policy_forward_metrics, resolve_policy_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_preprocessor(policy, dataset_stats) -> Callable:
    if dataset_stats is None:
        return lambda batch: batch

    config_obj = getattr(policy, "config", getattr(policy, "cfg", None))
    if config_obj is None:
        return lambda batch: batch

    for module_name in [
        "lerobot.policies.factory",
        "lerobot.common.policies.factory",
    ]:
        try:
            module = import_module(module_name)
            make_pre_post_processors = getattr(module, "make_pre_post_processors")
            preprocess, _ = make_pre_post_processors(config_obj, dataset_stats=dataset_stats)
            return preprocess
        except Exception:
            continue

    logger.warning("未能初始化 LeRobot 预处理器，回退为 no-op。")
    return lambda batch: batch


def load_policy(checkpoint_path: str, config: Dict, device: torch.device) -> Tuple[torch.nn.Module, Dict, Dict, Dict]:
    """从检查点加载 LeRobot ACTPolicy。"""
    from lerobot.policies.act.modeling_act import ACTPolicy

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    dataset_spec = ckpt.get("dataset_spec")
    policy_cfg = ckpt.get("policy_config") or resolve_policy_config(config["policy"], dataset_spec)

    act_cfg = build_act_config(policy_cfg)
    policy = ACTPolicy(act_cfg)
    policy.to(device)
    policy.load_state_dict(ckpt["model_state_dict"])
    policy.eval()

    logger.info("已加载检查点: %s (epoch %d)", checkpoint_path, ckpt.get("epoch", -1))
    return policy, ckpt, policy_cfg, dataset_spec or {}


@torch.no_grad()
def evaluate_policy_on_dataset(
    policy,
    dataloader,
    preprocess_batch: Callable,
) -> Dict:
    """在离线数据集上评估动作误差。"""
    policy.eval()
    metric_sums = defaultdict(float)
    n_batches = 0

    for batch in dataloader:
        batch = preprocess_batch(batch)
        output = policy.forward(batch)
        _, metrics = extract_policy_forward_metrics(output)
        for key, value in metrics.items():
            metric_sums[key] += value
        n_batches += 1

    if n_batches == 0:
        raise RuntimeError("离线评估数据为空，无法计算动作误差。")

    avg_metrics = {key: value / n_batches for key, value in metric_sums.items()}
    return {
        "metric": "action_l1_loss",
        "action_l1_loss": avg_metrics.get("action_l1_loss", avg_metrics["loss"]),
        "loss": avg_metrics["loss"],
        "kl_loss": avg_metrics.get("kl_loss", 0.0),
        "num_batches": n_batches,
    }


def run_zero_shot_eval(
    policy,
    config: Dict,
    policy_cfg: Dict,
    dataset_spec: Dict,
    preprocess_batch: Callable,
    device: torch.device,
) -> Dict:
    """优先仿真 SuccessRate；失败则在环境 D 数据集上做动作误差评估。"""
    eval_cfg = config["eval"]
    data_cfg = config.get("data", {})
    mode = eval_cfg.get("mode", "auto").lower()
    last_error = None

    if mode in {"auto", "sim"}:
        try:
            return evaluate_policy_on_calvin(
                policy=policy,
                env_name=eval_cfg["env_name"],
                num_episodes=eval_cfg["num_episodes"],
                max_steps=eval_cfg["max_steps"],
                device=str(device),
                preprocess_batch=preprocess_batch,
                camera_names=policy_cfg["camera_names"],
                camera_shapes=policy_cfg.get("camera_shapes", dataset_spec.get("camera_shapes", {})),
                camera_aliases=eval_cfg.get("camera_aliases"),
                state_dim=policy_cfg["state_dim"],
            )
        except Exception as exc:
            last_error = exc
            logger.warning("仿真评估不可用，准备回退到离线动作误差评估: %s", exc)
            if mode == "sim":
                raise

    if mode in {"auto", "offline"}:
        eval_repo_id = data_cfg.get("eval_repo_id", DEFAULT_EVAL_REPO_ID)
        eval_split = data_cfg.get("eval_split", "train")
        eval_envs = data_cfg.get("eval_envs")
        eval_local_dir = data_cfg.get("eval_local_dir")
        strict_eval_filter = data_cfg.get("strict_eval_env_filter", False)
        eval_env_episode_map_path = data_cfg.get("eval_env_episode_map_path")

        eval_dataset = load_calvin_dataset(
            repo_id=eval_repo_id,
            split=eval_split,
            envs=eval_envs,
            local_dir=eval_local_dir,
            strict_env_filter=strict_eval_filter,
            env_episode_map_path=eval_env_episode_map_path,
            chunk_size=config.get("policy", {}).get("chunk_size", 1),
        )
        eval_loader = torch.utils.data.DataLoader(
            eval_dataset,
            batch_size=eval_cfg.get("batch_size", 32),
            shuffle=False,
            num_workers=eval_cfg.get("num_workers", 0),
            pin_memory=True,
            drop_last=False,
        )

        eval_stats = get_dataset_stats(eval_dataset)
        offline_preprocess = preprocess_batch
        if offline_preprocess is None:
            offline_preprocess = create_preprocessor(policy, eval_stats if eval_stats is not None else None)
        return evaluate_policy_on_dataset(policy, eval_loader, offline_preprocess)

    if last_error is not None:
        raise last_error
    raise ValueError(f"不支持的评估模式: {mode}")


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

    baseline_ckpt = args.baseline_ckpt or config["model"]["baseline_checkpoint"]
    logger.info("评估 BASELINE — 环境 %s", eval_cfg["env_name"])

    if not Path(baseline_ckpt).exists():
        logger.error("Baseline 检查点未找到: %s", baseline_ckpt)
        results["baseline"] = {"error": f"Checkpoint not found: {baseline_ckpt}"}
    else:
        baseline_policy, baseline_meta, baseline_policy_cfg, baseline_dataset_spec = load_policy(
            baseline_ckpt,
            config,
            device,
        )
        baseline_preprocess = create_preprocessor(
            baseline_policy,
            baseline_meta.get("dataset_stats"),
        )
        results["baseline"] = run_zero_shot_eval(
            baseline_policy,
            config,
            baseline_policy_cfg,
            baseline_dataset_spec,
            baseline_preprocess,
            device,
        )
        logger.info("Baseline 结果: %s", results["baseline"])

    joint_ckpt = args.joint_ckpt or config["model"]["joint_checkpoint"]
    logger.info("评估 JOINT — 环境 %s", eval_cfg["env_name"])

    if not Path(joint_ckpt).exists():
        logger.error("Joint 检查点未找到: %s", joint_ckpt)
        results["joint"] = {"error": f"Checkpoint not found: {joint_ckpt}"}
    else:
        joint_policy, joint_meta, joint_policy_cfg, joint_dataset_spec = load_policy(
            joint_ckpt,
            config,
            device,
        )
        joint_preprocess = create_preprocessor(
            joint_policy,
            joint_meta.get("dataset_stats"),
        )
        results["joint"] = run_zero_shot_eval(
            joint_policy,
            config,
            joint_policy_cfg,
            joint_dataset_spec,
            joint_preprocess,
            device,
        )
        logger.info("Joint 结果: %s", results["joint"])

    logger.info("=" * 50)
    logger.info("BASELINE vs JOINT 零样本对比")
    logger.info("=" * 50)
    logger.info("Baseline — %s", results["baseline"])
    logger.info("Joint    — %s", results["joint"])

    output = {
        "env": eval_cfg["env_name"],
        "mode": eval_cfg.get("mode", "auto"),
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
