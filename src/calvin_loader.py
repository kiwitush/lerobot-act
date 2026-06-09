"""
CALVIN 数据加载器，封装 LeRobotDataset、环境筛选和数据特征推断。
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch

logger = logging.getLogger(__name__)

DEFAULT_TRAIN_REPO_ID = "CollisionCode/calvin_abc_d_lerobot_v2.1"
DEFAULT_EVAL_REPO_ID = "CollisionCode/calvin_d_d_lerobot_v2.1"


def load_calvin_dataset(
    repo_id: Union[str, Dict[str, str]] = DEFAULT_TRAIN_REPO_ID,
    split: str = "train",
    envs: Optional[List[str]] = None,
    local_dir: Optional[Union[str, Dict[str, str]]] = None,
    strict_env_filter: bool = True,
    env_episode_map_path: Optional[str] = None,
):
    """加载 CALVIN 数据集。

    支持两种数据组织方式:
      1. 单个 repo/path，要求数据集中自带环境标签，才能执行 env 过滤。
      2. `{env_name: repo_id}` 映射，按环境分别加载后拼接。
    """
    if isinstance(repo_id, dict):
        if not envs:
            raise ValueError("repo_id 为按环境映射时，必须显式提供 envs。")

        datasets = []
        for env in envs:
            if env not in repo_id:
                raise ValueError(f"未为环境 {env} 提供 repo_id。可用键: {sorted(repo_id)}")
            env_local_dir = None
            if isinstance(local_dir, dict):
                env_local_dir = local_dir.get(env)
            dataset = _load_dataset(repo_id[env], split, env_local_dir)
            datasets.append(dataset)

        if len(datasets) == 1:
            return datasets[0]

        logger.info("按环境独立数据源加载: %s", envs)
        return torch.utils.data.ConcatDataset(datasets)

    dataset = _load_dataset(repo_id, split, local_dir if isinstance(local_dir, str) else None)

    if envs:
        dataset = _filter_by_env(
            dataset=dataset,
            envs=envs,
            repo_id=repo_id,
            split=split,
            local_dir=local_dir if isinstance(local_dir, str) else None,
            strict_env_filter=strict_env_filter,
            env_episode_map_path=env_episode_map_path,
        )

    return dataset


def build_calvin_dataloader(
    repo_id: Union[str, Dict[str, str]] = DEFAULT_TRAIN_REPO_ID,
    split: str = "train",
    envs: Optional[List[str]] = None,
    batch_size: int = 32,
    num_workers: int = 4,
    shuffle: bool = True,
    local_dir: Optional[Union[str, Dict[str, str]]] = None,
    strict_env_filter: bool = True,
    env_episode_map_path: Optional[str] = None,
):
    """构建训练/评估 DataLoader。"""
    dataset = load_calvin_dataset(
        repo_id=repo_id,
        split=split,
        envs=envs,
        local_dir=local_dir,
        strict_env_filter=strict_env_filter,
        env_episode_map_path=env_episode_map_path,
    )

    base_dataset = get_base_dataset(dataset)
    logger.info(
        "%s 分片加载完成: %d 帧, %d 片段%s",
        split,
        len(dataset),
        getattr(base_dataset, "num_episodes", -1),
        f", 环境筛选: {envs}" if envs else "",
    )

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=shuffle,
    )


def build_calvin_val_dataloader(
    repo_id: Union[str, Dict[str, str]] = DEFAULT_TRAIN_REPO_ID,
    batch_size: int = 32,
    num_workers: int = 0,
    local_dir: Optional[Union[str, Dict[str, str]]] = None,
    envs: Optional[List[str]] = None,
    strict_env_filter: bool = True,
    env_episode_map_path: Optional[str] = None,
):
    """构建验证 DataLoader，不打乱。"""
    return build_calvin_dataloader(
        repo_id=repo_id,
        split="validation",
        envs=envs,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        local_dir=local_dir,
        strict_env_filter=strict_env_filter,
        env_episode_map_path=env_episode_map_path,
    )


def get_dataset_stats(dataset_or_repo=DEFAULT_TRAIN_REPO_ID, local_dir: Optional[str] = None):
    """获取数据集归一化统计量。"""
    dataset = dataset_or_repo
    if isinstance(dataset_or_repo, (str, dict)):
        dataset = load_calvin_dataset(dataset_or_repo, split="train", local_dir=local_dir)

    base_dataset = get_base_dataset(dataset)
    if hasattr(base_dataset, "stats"):
        return base_dataset.stats
    if hasattr(base_dataset, "meta") and hasattr(base_dataset.meta, "stats"):
        return base_dataset.meta.stats
    return None


def infer_dataset_spec(dataset, preferred_camera_names: Optional[List[str]] = None) -> Dict[str, Any]:
    """从 LeRobotDataset 样本中推断 ACT 需要的输入/输出特征。"""
    base_dataset = get_base_dataset(dataset)
    if len(base_dataset) == 0:
        raise ValueError("数据集为空，无法推断特征。")

    sample = base_dataset[0]

    discovered_cameras = []
    camera_shapes = {}
    for key, value in sample.items():
        if not key.startswith("observation.images."):
            continue
        camera_name = key.split("observation.images.", 1)[1]
        discovered_cameras.append(camera_name)
        camera_shapes[camera_name] = _normalize_image_shape(_shape_from_value(value))

    if preferred_camera_names:
        ordered_cameras = [cam for cam in preferred_camera_names if cam in camera_shapes]
        ordered_cameras.extend(cam for cam in discovered_cameras if cam not in ordered_cameras)
        camera_names = ordered_cameras
    else:
        camera_names = discovered_cameras

    if not camera_names:
        raise ValueError("数据集未暴露任何 observation.images.* 特征，无法构建视觉 ACT。")

    state_dim = _infer_vector_dim(sample.get("observation.state"))
    action_dim = _infer_vector_dim(sample.get("action"))
    if state_dim is None or action_dim is None:
        raise ValueError("无法从数据集样本中推断 observation.state/action 维度。")

    return {
        "camera_names": camera_names,
        "camera_shapes": camera_shapes,
        "state_dim": state_dim,
        "action_dim": action_dim,
    }


def get_base_dataset(dataset):
    """对 ConcatDataset 取第一个底层数据集，便于读取 stats / meta。"""
    if isinstance(dataset, torch.utils.data.ConcatDataset):
        if not dataset.datasets:
            raise ValueError("ConcatDataset 为空。")
        return get_base_dataset(dataset.datasets[0])
    return dataset


def _load_dataset(repo_id: str, split: str, local_dir: Optional[str]):
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
        f"无法加载数据集。请确认远端仓库 {repo_id} 可访问，"
        f"或通过 data.local_dir 指定本地路径。"
    )


def _filter_by_env(
    dataset,
    envs: List[str],
    repo_id: str,
    split: str,
    local_dir: Optional[str] = None,
    strict_env_filter: bool = True,
    env_episode_map_path: Optional[str] = None,
):
    """按 episode 环境标签筛选数据。筛选失败时默认直接报错。"""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    env_col = _find_env_column(dataset)
    episode_env_map = _build_episode_env_map(dataset, env_col) if env_col is not None else None

    if episode_env_map is None and env_episode_map_path:
        episode_env_map = _load_external_episode_env_map(env_episode_map_path)

    if episode_env_map is None:
        message = (
            f"数据集 {repo_id} (split={split}) 无法提供环境标签，"
            f"不能安全完成 env 过滤 {envs}。请提供带环境元数据的数据集，"
            "或通过 data.env_episode_map_path 提供 episode->env 映射文件。"
        )
        if strict_env_filter:
            raise ValueError(message)
        logger.warning(message)
        return dataset

    env_set = {str(env) for env in envs}
    matching_episodes = [
        ep_idx for ep_idx, ep_env in episode_env_map.items()
        if str(ep_env) in env_set
    ]

    if not matching_episodes:
        available = sorted({str(v) for v in episode_env_map.values()})
        raise ValueError(f"环境筛选 {envs} 后数据集为空，可用环境: {available}")

    logger.info("环境筛选: %s -> 保留 %d/%d 个 episode", envs, len(matching_episodes), len(episode_env_map))

    try:
        return LeRobotDataset(repo_id, split=split, episodes=matching_episodes)
    except Exception:
        if local_dir is not None:
            logger.info("Hub 重建失败，回退到本地路径: %s", local_dir)
            return LeRobotDataset(str(Path(local_dir)), split=split, episodes=matching_episodes)
        raise


def _load_external_episode_env_map(path: str) -> Dict[int, str]:
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"环境映射文件不存在: {path}")

    with open(path_obj, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, dict):
        return {int(k): str(v) for k, v in payload.items()}

    if isinstance(payload, list):
        mapping = {}
        for item in payload:
            mapping[int(item["episode_index"])] = str(item["env"])
        return mapping

    raise ValueError("环境映射文件格式不支持，需为 dict 或 list[{'episode_index','env'}]。")


def _build_episode_env_map(dataset, env_col: Optional[str]) -> Optional[Dict[int, str]]:
    """读取每条 episode 的环境标签，返回 {episode_index: env_label}。"""
    if env_col is None:
        return None

    if hasattr(dataset, "hf_dataset"):
        hf = dataset.hf_dataset
        if env_col in getattr(hf, "column_names", []) and "episode_index" in getattr(hf, "column_names", []):
            seen = {}
            for ep_idx, env_val in zip(hf["episode_index"], hf[env_col]):
                ep_idx = int(ep_idx)
                if ep_idx not in seen:
                    seen[ep_idx] = str(env_val)
            if seen:
                return seen

    episode_tables = []
    if hasattr(dataset, "episodes"):
        episode_tables.append(dataset.episodes)
    if hasattr(dataset, "meta") and hasattr(dataset.meta, "episodes"):
        episode_tables.append(dataset.meta.episodes)

    for eps in episode_tables:
        column_names = getattr(eps, "column_names", [])
        if env_col not in column_names:
            continue

        episode_indices = None
        for candidate in ["episode_index", "index", "episode_id"]:
            if candidate in column_names:
                episode_indices = eps[candidate]
                break

        if episode_indices is None:
            episode_indices = list(range(len(eps[env_col])))

        mapping = {}
        for idx, env_val in zip(episode_indices, eps[env_col]):
            mapping[int(idx)] = str(env_val)
        if mapping:
            return mapping

    return None


def _find_env_column(dataset) -> Optional[str]:
    """在数据集元数据中查找环境标签列名。"""
    candidate_columns = ["env", "environment", "env_name", "scene", "scene_name"]

    collections = []
    if hasattr(dataset, "hf_dataset"):
        collections.append(getattr(dataset.hf_dataset, "column_names", []))
    if hasattr(dataset, "episodes"):
        collections.append(getattr(dataset.episodes, "column_names", []))
    if hasattr(dataset, "meta") and hasattr(dataset.meta, "episodes"):
        collections.append(getattr(dataset.meta.episodes, "column_names", []))

    for column_names in collections:
        for col in candidate_columns:
            if col in column_names:
                return col

    return None


def _shape_from_value(value: Any):
    if value is None:
        return None
    if hasattr(value, "shape"):
        return tuple(int(x) for x in value.shape)
    return None


def _normalize_image_shape(shape):
    if shape is None or len(shape) != 3:
        raise ValueError(f"无法识别图像 shape: {shape}")
    if shape[0] in (1, 3):
        return tuple(shape)
    return (shape[2], shape[0], shape[1])


def _infer_vector_dim(value: Any) -> Optional[int]:
    shape = _shape_from_value(value)
    if shape is None or len(shape) == 0:
        return None
    return int(shape[-1])
