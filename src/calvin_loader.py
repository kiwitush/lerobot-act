"""
CALVIN 数据加载器，封装 LeRobotDataset 与环境筛选逻辑。

优先从 HuggingFace Hub 加载 lerobot/calvin，也可回退到本地 LeRobot 格式数据。
"""

import logging
from pathlib import Path
from typing import List, Optional

import torch

logger = logging.getLogger(__name__)


def build_calvin_dataloader(
    repo_id: str = "lerobot/calvin",
    split: str = "train",
    envs: Optional[List[str]] = None,
    batch_size: int = 32,
    num_workers: int = 4,
    shuffle: bool = True,
    local_dir: Optional[str] = None,
):
    """构建 CALVIN DataLoader，返回 LeRobot 兼容的 batch dict。

    优先从 HuggingFace Hub 加载 `repo_id`，失败时回退到 `local_dir`。

    Args:
        repo_id: HuggingFace Hub 上的 LeRobot 格式数据集 ID。
        split: 数据集分片，如 "train" 或 "validation"。
        envs: 要保留的环境列表，如 ["A"] 或 ["A","B","C"]。None 表示不过滤。
        batch_size: 批大小。
        num_workers: DataLoader 工作进程数。
        shuffle: 是否打乱数据。
        local_dir: 本地 LeRobot 格式数据目录，作为 Hub 加载失败时的回退。

    Returns:
        torch.utils.data.DataLoader: 每个 batch 为 dict，键包含
            observation.state, observation.images.{cam}, action, action_is_pad 等。
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = _load_dataset(repo_id, split, local_dir)

    if envs is not None and len(envs) > 0:
        dataset = _filter_by_env(dataset, envs, repo_id, split, local_dir)

    logger.info(
        "%s 分片加载完成: %d 帧, %d 片段%s",
        split,
        len(dataset),
        dataset.num_episodes,
        f", 环境筛选: {envs}" if envs else "",
    )

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )


def build_calvin_val_dataloader(
    repo_id: str = "lerobot/calvin",
    batch_size: int = 32,
    num_workers: int = 0,
    local_dir: Optional[str] = None,
):
    """构建验证 DataLoader，使用 validation 分片，不打乱。"""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = _load_dataset(repo_id, "validation", local_dir)

    logger.info("验证集加载完成: %d 帧, %d 片段", len(dataset), dataset.num_episodes)

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )


def get_dataset_stats(repo_id: str = "lerobot/calvin", local_dir: Optional[str] = None):
    """获取数据集归一化统计量，用于配置 ACTPolicy 的输入/输出归一化。"""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = _load_dataset(repo_id, "train", local_dir)
    return dataset.stats


def _load_dataset(repo_id: str, split: str, local_dir: Optional[str]):
    """尝试从 Hub 加载，失败则回退到本地目录。"""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    try:
        dataset = LeRobotDataset(repo_id, split=split)
        logger.info("从 HuggingFace Hub 加载: %s (split=%s)", repo_id, split)
        return dataset
    except Exception as e:
        logger.warning("从 Hub 加载失败: %s", e)

    if local_dir is not None:
        local_path = Path(local_dir)
        if local_path.exists():
            logger.info("回退到本地数据: %s", local_path)
            return LeRobotDataset(str(local_path), split=split)

    raise FileNotFoundError(
        f"无法加载数据集。请确认 {repo_id} 存在于 HuggingFace Hub，"
        f"或通过 --data.local_dir 指定本地 LeRobot 格式数据路径。"
    )


def _filter_by_env(dataset, envs: List[str], repo_id: str, split: str, local_dir: Optional[str] = None):
    """根据环境标签筛选数据集，使用 LeRobotDataset 的 episodes 参数重建。

    LeRobotDataset.__init__ 接受 episodes: list[int] | None 参数，
    直接通过该参数筛选比 Subset 更干净，能正确复用视频读取器等内部状态。
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    env_col = _find_env_column(dataset)
    if env_col is None:
        logger.warning("数据集中未找到环境标签列，无法筛选环境。将使用全部数据。")
        return dataset

    env_set = set(envs)

    episode_env_map = _build_episode_env_map(dataset, env_col)
    if episode_env_map is None:
        logger.warning("无法读取 episode→env 映射，将使用全部数据。")
        return dataset

    matching_episodes = [
        ep_idx for ep_idx, ep_env in episode_env_map.items()
        if ep_env in env_set
    ]

    if len(matching_episodes) == 0:
        available = set(episode_env_map.values())
        raise ValueError(
            f"环境筛选 {envs} 后数据集为空！可用环境: {available}"
        )

    logger.info("环境筛选: %s → 保留 %d/%d 个 episode",
                envs, len(matching_episodes), len(episode_env_map))

    # 优先从 Hub 重建，失败则回退到本地路径
    try:
        return LeRobotDataset(repo_id, split=split, episodes=matching_episodes)
    except Exception:
        if local_dir is not None:
            logger.info("Hub 重建失败，回退到本地路径筛选: %s", local_dir)
            return LeRobotDataset(str(Path(local_dir)), split=split, episodes=matching_episodes)
        raise


def _build_episode_env_map(dataset, env_col: str) -> dict:
    """构建 {episode_index: env_label} 映射表。"""
    # 从 hf_dataset 读取 (episode_index, env_col) 并去重
    if hasattr(dataset, "hf_dataset"):
        hf = dataset.hf_dataset
        if env_col in hf.column_names and "episode_index" in hf.column_names:
            # 取每个 episode 的第一条记录的环境标签
            seen = {}
            for ep_idx, env_val in zip(hf["episode_index"], hf[env_col]):
                ep_idx = int(ep_idx)
                if ep_idx not in seen:
                    seen[ep_idx] = env_val
            return seen

    # 从 episodes 元数据读取
    if hasattr(dataset, "episodes") and hasattr(dataset.episodes, "column_names"):
        eps = dataset.episodes
        if env_col in eps.column_names:
            return {
                i: eps[env_col][i]
                for i in range(len(eps[env_col]))
            }

    return None



def _find_env_column(dataset) -> str:
    """在数据集元数据中查找环境标签列。"""
    # 检查 hf_dataset 的列
    if hasattr(dataset, "hf_dataset"):
        hf = dataset.hf_dataset
        for col in ["env", "environment", "env_name"]:
            if col in hf.column_names:
                return col

    # 检查 episode 元数据
    if hasattr(dataset, "episodes") and hasattr(dataset.episodes, "column_names"):
        for col in ["env", "environment", "env_name"]:
            if col in dataset.episodes.column_names:
                return col

    return None


