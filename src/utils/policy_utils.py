"""ACT 策略工具函数，供 train.py 和 eval.py 共用。"""

from typing import Dict


def build_act_config(policy_cfg: Dict):
    """根据配置字典构建 ACTConfig，train 和 eval 共用。

    LeRobot 的 ACTPolicy 根据 input_features/output_features
    自动构建视觉编码器和动作头，无需手动指定网络结构。
    """
    from lerobot.policies.act.configuration_act import ACTConfig

    camera_names = policy_cfg["camera_names"]
    input_features = {
        "observation.state": {
            "shape": (policy_cfg["state_dim"],),
            "dtype": "float32",
        },
    }
    for cam in camera_names:
        input_features[f"observation.images.{cam}"] = {
            "shape": (3, 200, 200),
            "dtype": "image",
        }

    output_features = {
        "action": {
            "shape": (policy_cfg["action_dim"],),
            "dtype": "float32",
        },
    }

    return ACTConfig(
        input_features=input_features,
        output_features=output_features,
        chunk_size=policy_cfg["chunk_size"],
        n_obs_steps=1,
        n_action_steps=policy_cfg["chunk_size"],
        n_encoder_layers=policy_cfg["n_encoder_layers"],
        n_decoder_layers=policy_cfg["n_decoder_layers"],
        dim_model=policy_cfg["hidden_dim"],
        n_heads=policy_cfg["nheads"],
        dim_feedforward=policy_cfg["dim_feedforward"],
        dropout=policy_cfg.get("dropout", 0.1),
        use_vae=True,
        latent_dim=32,
        kl_weight=10.0,
        vision_backbone="resnet18",
        pretrained_backbone_weights="ResNet18_Weights.IMAGENET1K_V1",
        feedforward_activation="relu",
        pre_norm=False,
        temporal_ensemble_coeff=None,
    )
