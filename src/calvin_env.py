"""
CALVIN Gym 环境封装，用于策略零样本评估。

将 CALVIN 原始观测转为 LeRobot ACTPolicy 期望的格式，
并在评估期显式检查 success 字段，避免输出伪造的成功率。
"""

import logging
import os
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class CalvinEnv:
    """CALVIN 仿真环境轻量封装，用于策略评估。"""

    def __init__(
        self,
        env_name: str = "D",
        task_cfg_path: Optional[str] = None,
        render_mode: str = "rgb_array",
        max_steps: int = 360,
        device: str = "cuda",
        camera_names: Optional[List[str]] = None,
        camera_shapes: Optional[Dict[str, tuple]] = None,
        camera_aliases: Optional[Dict[str, str]] = None,
        state_dim: Optional[int] = None,
    ):
        self.env_name = env_name
        self.max_steps = max_steps
        self.device = device
        self.camera_names = camera_names or ["rgb_static", "rgb_gripper"]
        self.camera_shapes = camera_shapes or {}
        self.camera_aliases = camera_aliases or {}
        self.state_dim = state_dim

        try:
            from calvin_env.envs.play_table_env import get_env

            self._env = get_env(
                task_cfg_path or self._default_task_cfg(),
                env_name=env_name,
                render_mode=render_mode,
            )
        except ImportError:
            logger.error(
                "calvin_env 未安装。安装命令: "
                "pip install calvin @ git+https://github.com/mees/calvin.git"
            )
            raise
        except Exception as e:
            logger.warning("CALVIN 环境初始化失败: %s", e)
            self._env = None

        self._step_count = 0
        self._state_dim_warned = False

    @staticmethod
    def _default_task_cfg() -> str:
        import calvin_env

        pkg_dir = os.path.dirname(calvin_env.__file__)
        cfg = os.path.join(
            pkg_dir,
            "..",
            "calvin_models",
            "conf",
            "callbacks",
            "rollout",
            "tasks",
            "tasks_ABCD_D.yaml",
        )
        if not os.path.exists(cfg):
            cfg = os.path.join(pkg_dir, "conf", "tasks.yaml")
        return cfg

    def reset(self) -> Dict[str, torch.Tensor]:
        self._step_count = 0
        if self._env is None:
            raise RuntimeError("CALVIN 环境未初始化。")
        raw = self._env.reset()
        raw_obs = raw[0] if isinstance(raw, tuple) else raw
        return self._process_obs(raw_obs)

    def step(self, action: np.ndarray):
        if self._env is None:
            raise RuntimeError("CALVIN 环境未初始化。")

        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()
        action = np.squeeze(action)

        transition = self._env.step(action)
        if len(transition) == 5:
            raw_obs, reward, terminated, truncated, info = transition
            done = terminated or truncated
        else:
            raw_obs, reward, done, info = transition

        self._step_count += 1
        if self._step_count >= self.max_steps:
            done = True

        return self._process_obs(raw_obs), reward, done, info

    def _resolve_camera_key(self, raw_obs: dict, camera_name: str) -> str:
        rgb_obs = raw_obs["rgb_obs"]
        alias_candidates = [
            self.camera_aliases.get(camera_name),
            camera_name,
        ]

        known_aliases = {
            "agentview": ["rgb_static"],
            "rgb_static": ["rgb_static", "agentview"],
            "static": ["rgb_static"],
            "eye_in_hand": ["rgb_gripper"],
            "rgb_gripper": ["rgb_gripper", "eye_in_hand"],
            "wrist": ["rgb_gripper"],
        }
        alias_candidates.extend(known_aliases.get(camera_name, []))

        for candidate in alias_candidates:
            if candidate and candidate in rgb_obs:
                return candidate

        raise KeyError(f"无法为相机 {camera_name} 在 CALVIN 观测中找到对应图像键。可用键: {list(rgb_obs)}")

    def _process_obs(self, raw_obs: dict) -> Dict[str, torch.Tensor]:
        """将 CALVIN 原始观测转为 LeRobot 标准 observation dict。"""
        robot_state = torch.from_numpy(raw_obs["robot_obs"]).float()
        if self.state_dim is not None and robot_state.shape[-1] != self.state_dim:
            if robot_state.shape[-1] < self.state_dim:
                raise RuntimeError(
                    f"CALVIN robot_obs 维度 {robot_state.shape[-1]} 小于策略期望的 {self.state_dim}。"
                )
            if not self._state_dim_warned:
                logger.warning(
                    "CALVIN robot_obs 维度 %d 与策略 state_dim=%d 不一致，已裁剪前 %d 维。",
                    robot_state.shape[-1],
                    self.state_dim,
                    self.state_dim,
                )
                self._state_dim_warned = True
            robot_state = robot_state[: self.state_dim]

        obs = {"observation.state": robot_state.unsqueeze(0).to(self.device)}

        for camera_name in self.camera_names:
            raw_camera_key = self._resolve_camera_key(raw_obs, camera_name)
            image = torch.from_numpy(raw_obs["rgb_obs"][raw_camera_key]).float() / 255.0
            if image.ndim != 3:
                raise RuntimeError(f"图像 {raw_camera_key} 维度非法: {tuple(image.shape)}")
            if image.shape[0] not in (1, 3):
                image = image.permute(2, 0, 1)

            expected_shape = self.camera_shapes.get(camera_name)
            if expected_shape is not None and tuple(image.shape) != tuple(expected_shape):
                image = F.interpolate(
                    image.unsqueeze(0),
                    size=tuple(expected_shape[-2:]),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)

            obs[f"observation.images.{camera_name}"] = image.unsqueeze(0).to(self.device)

        return obs

    def close(self):
        if self._env is not None:
            self._env.close()


def _predict_action(policy, obs_batch: dict, preprocess_batch: Optional[Callable] = None) -> np.ndarray:
    """调用策略推理，兼容不同 LeRobot 版本的接口。"""
    if preprocess_batch is not None:
        obs_batch = preprocess_batch(obs_batch)

    with torch.no_grad():
        if hasattr(policy, "select_action"):
            action = policy.select_action(obs_batch)
        else:
            logger.warning("策略无 select_action 方法，使用 forward 回退推理。")
            output = policy.forward(obs_batch)
            if isinstance(output, tuple):
                action = output[0]
            elif isinstance(output, dict):
                action = output.get("action", output.get("action_chunk"))
            else:
                action = output

    if action is None:
        raise RuntimeError("策略推理未返回有效动作。")

    if action.ndim == 3:
        action = action[0]
    if action.ndim == 2:
        action = action[0]
    return action.detach().cpu().numpy()


def evaluate_policy_on_calvin(
    policy,
    env_name: str = "D",
    num_episodes: int = 100,
    max_steps: int = 360,
    device: str = "cuda",
    preprocess_batch: Optional[Callable] = None,
    camera_names: Optional[List[str]] = None,
    camera_shapes: Optional[Dict[str, tuple]] = None,
    camera_aliases: Optional[Dict[str, str]] = None,
    state_dim: Optional[int] = None,
) -> Dict[str, float]:
    """在 CALVIN 环境上运行零样本评估。

    仅当环境返回显式 success 字段时，才报告 success_rate。
    """
    env = CalvinEnv(
        env_name=env_name,
        max_steps=max_steps,
        device=device,
        camera_names=camera_names,
        camera_shapes=camera_shapes,
        camera_aliases=camera_aliases,
        state_dim=state_dim,
    )
    policy.eval()

    if hasattr(policy, "reset"):
        policy.reset()

    successes = 0
    total_steps = 0

    for ep in range(num_episodes):
        obs = env.reset()
        if hasattr(policy, "reset"):
            policy.reset()

        done = False
        ep_steps = 0
        info = {}

        while not done:
            obs_batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in obs.items()}
            action = _predict_action(policy, obs_batch, preprocess_batch=preprocess_batch)
            obs, reward, done, info = env.step(action)
            ep_steps += 1

        if "success" not in info:
            env.close()
            raise RuntimeError(
                "当前 CALVIN 环境接口未返回 success 字段，无法安全计算 SuccessRate。"
            )

        total_steps += ep_steps
        if info["success"]:
            successes += 1

        if (ep + 1) % 10 == 0:
            logger.info(
                "Episode %d/%d: steps=%d, success=%s, 累计成功率=%.1f%%",
                ep + 1,
                num_episodes,
                ep_steps,
                info["success"],
                successes / (ep + 1) * 100,
            )

    env.close()

    results = {
        "metric": "success_rate",
        "success_rate": successes / num_episodes,
        "avg_steps": total_steps / num_episodes,
        "num_successes": successes,
        "num_episodes": num_episodes,
    }

    logger.info(
        "评估完成 Env=%s: 成功率=%.2f%% (%d/%d), 平均步数=%.1f",
        env_name,
        results["success_rate"] * 100,
        successes,
        num_episodes,
        results["avg_steps"],
    )

    return results
