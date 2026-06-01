"""
CALVIN Gym 环境封装，用于策略零样本评估。

将 CALVIN 原始观测转为 LeRobot ACTPolicy 期望的格式，
调用 policy.select_action() 执行动作分块推理。
"""

import logging
import os
from typing import Dict, Optional

import numpy as np
import torch

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
    ):
        self.env_name = env_name
        self.max_steps = max_steps
        self.device = device

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
            logger.warning("CALVIN 环境初始化失败 (无 Mujoco 或仅需训练时属正常): %s", e)
            self._env = None

        self._step_count = 0

    @staticmethod
    def _default_task_cfg() -> str:
        import calvin_env
        pkg_dir = os.path.dirname(calvin_env.__file__)
        cfg = os.path.join(pkg_dir, "..", "calvin_models", "conf", "callbacks",
                          "rollout", "tasks", "tasks_ABCD_D.yaml")
        if not os.path.exists(cfg):
            cfg = os.path.join(pkg_dir, "conf", "tasks.yaml")
        return cfg

    def reset(self) -> Dict[str, torch.Tensor]:
        self._step_count = 0
        if self._env is None:
            raise RuntimeError("CALVIN 环境未初始化。")
        return self._process_obs(self._env.reset())

    def step(self, action: np.ndarray):
        if self._env is None:
            raise RuntimeError("CALVIN 环境未初始化。")

        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()
        action = np.squeeze(action)

        raw_obs, reward, done, info = self._env.step(action)
        self._step_count += 1

        if self._step_count >= self.max_steps:
            done = True

        return self._process_obs(raw_obs), reward, done, info

    def _process_obs(self, raw_obs: dict) -> Dict[str, torch.Tensor]:
        """将 CALVIN 原始观测转为 LeRobot 标准 observation dict 格式。

        LeRobot ACTPolicy 期望的键:
          - observation.state: (1, state_dim)
          - observation.images.{camera_name}: (1, C, H, W), float32 [0,1]
        """
        rgb_static = torch.from_numpy(raw_obs["rgb_obs"]["rgb_static"]).float() / 255.0
        rgb_gripper = torch.from_numpy(raw_obs["rgb_obs"]["rgb_gripper"]).float() / 255.0

        rgb_static = rgb_static.permute(2, 0, 1).unsqueeze(0)
        rgb_gripper = rgb_gripper.permute(2, 0, 1).unsqueeze(0)

        robot_state = torch.from_numpy(raw_obs["robot_obs"]).float().unsqueeze(0)

        return {
            "observation.state": robot_state.to(self.device),
            "observation.images.rgb_static": rgb_static.to(self.device),
            "observation.images.rgb_gripper": rgb_gripper.to(self.device),
        }

    def close(self):
        if self._env is not None:
            self._env.close()


def _predict_action(policy, obs_batch: dict) -> np.ndarray:
    """调用策略推理，兼容不同 LeRobot 版本的接口。"""
    # 主流接口: select_action
    if hasattr(policy, "select_action"):
        action = policy.select_action(obs_batch)
    # 旧版回退: 直接调用 forward 再取输出
    else:
        logger.warning("策略无 select_action 方法，使用 forward 回退推理。")
        with torch.no_grad():
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
        action = action[0]  # (1, chunk, dim) → (chunk, dim)
    # 取 chunk 的第一步，兼容不同 LeRobot 版本返回形状
    if action.ndim == 2:
        action = action[0]  # (chunk, dim) → (dim,)
    return action.cpu().numpy()


def evaluate_policy_on_calvin(
    policy,
    env_name: str = "D",
    num_episodes: int = 100,
    max_steps: int = 360,
    device: str = "cuda",
) -> Dict[str, float]:
    """在 CALVIN 环境上运行零样本评估，使用 LeRobot ACTPolicy 接口。

    调用 policy.select_action(obs_dict) 获取动作，
    利用 ACT 的动作分块 + 时序集成机制进行推理。
    """
    env = CalvinEnv(env_name=env_name, max_steps=max_steps, device=device)
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

        while not done:
            obs_batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                         for k, v in obs.items()}

            action = _predict_action(policy, obs_batch)

            obs, reward, done, info = env.step(action)
            ep_steps += 1

        total_steps += ep_steps

        if info.get("success", False):
            successes += 1

        if (ep + 1) % 10 == 0:
            logger.info(
                "Episode %d/%d: steps=%d, success=%s, 累计成功率=%.1f%%",
                ep + 1, num_episodes, ep_steps,
                info.get("success", False),
                successes / (ep + 1) * 100,
            )

    env.close()

    results = {
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
