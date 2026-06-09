"""ACT 策略工具函数，供 train.py 和 eval.py 共用。"""

from copy import deepcopy
from typing import Any, Dict, Optional


def resolve_policy_config(policy_cfg: Dict[str, Any], dataset_spec: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """用数据集实际特征覆盖静态配置，避免相机名/维度与数据不一致。"""
    resolved = deepcopy(policy_cfg)

    if dataset_spec:
        for key in ["camera_names", "camera_shapes", "state_dim", "action_dim"]:
            if key in dataset_spec and dataset_spec[key] is not None:
                resolved[key] = dataset_spec[key]

    if not resolved.get("camera_names"):
        raise ValueError("policy.camera_names 为空，且未能从数据集自动推断。")
    if not resolved.get("state_dim") or not resolved.get("action_dim"):
        raise ValueError("policy.state_dim / action_dim 缺失，且未能从数据集自动推断。")

    return resolved


def build_act_config(policy_cfg: Dict[str, Any], dataset_spec: Optional[Dict[str, Any]] = None):
    """根据配置字典构建 ACTConfig，train 和 eval 共用。"""
    from lerobot.configs.policies import PolicyFeature
    from lerobot.configs.types import FeatureType
    from lerobot.policies.act.configuration_act import ACTConfig

    resolved_cfg = resolve_policy_config(policy_cfg, dataset_spec)
    camera_names = resolved_cfg["camera_names"]
    camera_shapes = resolved_cfg.get("camera_shapes", {})

    input_features = {
        "observation.state": PolicyFeature(
            type=FeatureType.STATE,
            shape=(resolved_cfg["state_dim"],),
        ),
    }
    for cam in camera_names:
        input_features[f"observation.images.{cam}"] = PolicyFeature(
            type=FeatureType.VISUAL,
            shape=tuple(camera_shapes.get(cam, (3, 200, 200))),
        )

    output_features = {
        "action": PolicyFeature(
            type=FeatureType.ACTION,
            shape=(resolved_cfg["action_dim"],),
        ),
    }

    return ACTConfig(
        input_features=input_features,
        output_features=output_features,
        chunk_size=resolved_cfg["chunk_size"],
        n_obs_steps=1,
        n_action_steps=resolved_cfg["chunk_size"],
        n_encoder_layers=resolved_cfg["n_encoder_layers"],
        n_decoder_layers=resolved_cfg["n_decoder_layers"],
        dim_model=resolved_cfg["hidden_dim"],
        n_heads=resolved_cfg["nheads"],
        dim_feedforward=resolved_cfg["dim_feedforward"],
        dropout=resolved_cfg.get("dropout", 0.1),
        use_vae=resolved_cfg.get("use_vae", True),
        latent_dim=resolved_cfg.get("latent_dim", 32),
        kl_weight=resolved_cfg.get("kl_weight", 10.0),
        vision_backbone=resolved_cfg.get("vision_backbone", "resnet18"),
        pretrained_backbone_weights=resolved_cfg.get(
            "pretrained_backbone_weights",
            "ResNet18_Weights.IMAGENET1K_V1",
        ),
        feedforward_activation=resolved_cfg.get("feedforward_activation", "relu"),
        pre_norm=resolved_cfg.get("pre_norm", False),
        temporal_ensemble_coeff=resolved_cfg.get("temporal_ensemble_coeff"),
    )


def extract_policy_forward_metrics(output) -> tuple[Any, Dict[str, float]]:
    """兼容不同 LeRobot 版本的 forward 返回格式，并提取关键指标。"""
    loss_tensor = None
    aux = {}

    if isinstance(output, tuple):
        loss_tensor = output[0]
        if len(output) > 1 and isinstance(output[1], dict):
            aux = output[1]
    elif isinstance(output, dict):
        loss_tensor = output.get("loss")
        aux = output
    else:
        loss_tensor = output

    if loss_tensor is None:
        raise RuntimeError("策略 forward 未返回 loss。")

    metrics = {"loss": _to_float(loss_tensor)}

    aliases = {
        "action_l1_loss": ["action_l1_loss", "l1_loss", "loss_l1", "reconstruction_loss"],
        "kl_loss": ["kl_loss", "loss_kl", "kld_loss"],
    }

    for normalized_key, candidates in aliases.items():
        for candidate in candidates:
            if candidate in aux:
                metrics[normalized_key] = _to_float(aux[candidate])
                break

    for key, value in aux.items():
        if key in {"loss"} or key in metrics:
            continue
        if isinstance(value, (int, float)) or hasattr(value, "item"):
            metrics[key] = _to_float(value)

    return loss_tensor, metrics


def _to_float(value: Any) -> float:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)
