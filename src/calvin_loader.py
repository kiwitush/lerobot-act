"""
CALVIN 数据加载器，封装 LeRobotDataset、环境筛选和数据特征推断。
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch

logger = logging.getLogger(__name__)

DEFAULT_TRAIN_REPO_ID = "xiaoma26/calvin-lerobot"
DEFAULT_EVAL_REPO_ID = "xiaoma26/calvin-lerobot"


def load_calvin_dataset(
    repo_id: Union[str, Dict[str, str]] = DEFAULT_TRAIN_REPO_ID,
    split: str = "train",
    envs: Optional[List[str]] = None,
    local_dir: Optional[Union[str, Dict[str, str]]] = None,
    strict_env_filter: bool = True,
    env_episode_map_path: Optional[str] = None,
    chunk_size: int = 1,
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
            dataset = _load_dataset(repo_id[env], split, env_local_dir, chunk_size=chunk_size)
            datasets.append(dataset)

        if len(datasets) == 1:
            return datasets[0]

        logger.info("按环境独立数据源加载: %s", envs)
        return torch.utils.data.ConcatDataset(datasets)

    dataset = _load_dataset(repo_id, split, local_dir if isinstance(local_dir, str) else None, chunk_size=chunk_size)

    if envs:
        dataset = _filter_by_env(
            dataset=dataset,
            envs=envs,
            repo_id=repo_id,
            split=split,
            local_dir=local_dir if isinstance(local_dir, str) else None,
            strict_env_filter=strict_env_filter,
            env_episode_map_path=env_episode_map_path,
            chunk_size=chunk_size,
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
    """从数据集样本中推断 ACT 需要的输入/输出特征，兼容 LeRobot 和 CALVIN 原生格式。"""
    base_dataset = get_base_dataset(dataset)
    if len(base_dataset) == 0:
        raise ValueError("数据集为空，无法推断特征。")

    sample = base_dataset[0]
    is_calvin = _is_calvin_format(sample)
    discovered_cameras = []
    camera_shapes = {}

    if is_calvin:
        # CALVIN 原生格式: image, wrist_image
        discovered_cameras = ["rgb_static", "rgb_gripper"]
        camera_shapes["rgb_static"] = _normalize_image_shape(_shape_from_value(sample["image"]))
        camera_shapes["rgb_gripper"] = _normalize_image_shape(_shape_from_value(sample["wrist_image"]))
    else:
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
        raise ValueError("数据集未暴露任何图像特征，无法构建视觉 ACT。")

    state_dim = _infer_vector_dim(sample.get("observation.state", sample.get("state")))
    action_dim = _infer_vector_dim(sample.get("action", sample.get("actions")))
    if state_dim is None or action_dim is None:
        raise ValueError("无法从数据集样本中推断 state/action 维度。")

    return {
        "camera_names": camera_names,
        "camera_shapes": camera_shapes,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "is_calvin_format": is_calvin,
    }


def get_base_dataset(dataset):
    """对 ConcatDataset 取第一个底层数据集，便于读取 stats / meta。"""
    if isinstance(dataset, torch.utils.data.ConcatDataset):
        if not dataset.datasets:
            raise ValueError("ConcatDataset 为空。")
        return get_base_dataset(dataset.datasets[0])
    return dataset


def _is_calvin_format(sample: dict) -> bool:
    """检测是否为 CALVIN 原生格式 (image/wrist_image/state/actions)。"""
    has_calvin = {"image", "wrist_image", "state", "actions"}.issubset(sample.keys())
    has_lerobot = any(k.startswith("observation.images.") for k in sample)
    return has_calvin and not has_lerobot


class _CalvinRemapper(torch.utils.data.Dataset):
    """CALVIN key 映射 + 动作 chunk 组装。"""

    _KEY_MAP = {
        "image": "observation.images.rgb_static",
        "wrist_image": "observation.images.rgb_gripper",
        "state": "observation.state",
        "actions": "action",
    }

    def __init__(self, dataset, chunk_size: int = 100):
        self._dataset = dataset
        self._chunk_size = max(chunk_size, 1)
        self._ep_starts = None
        self._ep_ends = None
        self._ep_arr = None
        if self._chunk_size > 1:
            self._build_episode_index()
        for attr in ("hf_dataset", "stats", "meta", "episodes", "num_episodes"):
            if hasattr(dataset, attr):
                setattr(self, attr, getattr(dataset, attr))

    def _build_episode_index(self):
        logger.info("构建 episode 边界索引...")
        eps = self._dataset.meta.episodes
        col_names = getattr(eps, "column_names", [])
        logger.info("episode 元数据列: %s, 记录数: %d", col_names, len(eps))
        if "length" in col_names:
            lengths = np.array(eps["length"], dtype=np.int64)
        elif "index" in col_names:
            starts = np.array(eps["index"], dtype=np.int64)
            lengths = np.diff(starts, append=len(self._dataset))
        else:
            # 回退：逐条读取
            ep_counts = {}
            for item in eps:
                ep = int(item.get("episode_index", item.get("index", item.get("episode_id", 0))))
                length = int(item.get("length", item.get("num_frames", 0)))
                ep_counts[ep] = length
            sorted_eps = sorted(ep_counts.items())
            lengths = np.array([l for _, l in sorted_eps], dtype=np.int64)
        self._ep_ends = np.cumsum(lengths)
        self._ep_starts = np.concatenate([[0], self._ep_ends[:-1]])
        self._num_eps = len(lengths)
        logger.info("episode 索引构建完成: %d episodes, %d frames", self._num_eps, self._ep_ends[-1])

    def _ep_for_idx(self, idx: int) -> int:
        """返回给定 frame index 所属的 episode 边界位置索引。"""
        return int(np.searchsorted(self._ep_ends, idx, side="right"))

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx):
        sample = self._dataset[idx]
        result = {self._KEY_MAP[k]: v for k, v in sample.items() if k in self._KEY_MAP}
        if self._chunk_size > 1 and "action" in result:
            result["action"] = self._build_chunk(idx)
        elif "action" in result and result["action"].ndim == 1:
            result["action"] = result["action"].unsqueeze(0)
        return result

    def _build_chunk(self, idx):
        pos = self._ep_for_idx(idx)
        ep_end = int(self._ep_ends[pos])
        need = self._chunk_size
        actual = min(need, ep_end - idx)

        hf = self._dataset.hf_dataset
        actions = [torch.as_tensor(a) for a in hf["actions"][idx:idx + actual]]
        while len(actions) < need:
            actions.append(actions[-1].clone())
        return torch.stack(actions, dim=0)


def _maybe_wrap(ds, chunk_size: int = 1):
    if len(ds) > 0 and _is_calvin_format(ds[0]):
        logger.info("检测到 CALVIN 原生格式，自动映射 key (chunk_size=%d)。", chunk_size)
        return _CalvinRemapper(ds, chunk_size=chunk_size)
    return ds


def _load_dataset(repo_id: str, split: str, local_dir: Optional[str], chunk_size: int = 1):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    # 若提供了 local_dir，优先从本地路径加载
    if local_dir is not None:
        local_path = Path(local_dir)
        if local_path.exists():
            logger.info("从本地路径加载: %s (repo_id=%s, split=%s)", local_path, repo_id, split)
            try:
                return _maybe_wrap(LeRobotDataset(repo_id, root=str(local_path), split=split), chunk_size=chunk_size)
            except TypeError:
                return _maybe_wrap(LeRobotDataset(repo_id, root=str(local_path)), chunk_size=chunk_size)

    # 尝试 HuggingFace Hub
    try:
        dataset = LeRobotDataset(repo_id, split=split)
        logger.info("从 HuggingFace Hub 加载: %s (split=%s)", repo_id, split)
        return _maybe_wrap(dataset)
    except TypeError:
        # LeRobot 0.5+ 移除了 split 参数，回退到无 split 加载
        dataset = LeRobotDataset(repo_id)
        logger.info("从 HuggingFace Hub 加载: %s (split 参数不被支持，已忽略)", repo_id)
        return _maybe_wrap(dataset)
    except Exception as e:
        logger.warning("从 Hub 加载失败: %s", e)

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
    chunk_size: int = 1,
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

    root_kwarg = {}
    if local_dir is not None:
        root_kwarg["root"] = str(Path(local_dir))

    try:
        filtered = LeRobotDataset(repo_id, episodes=matching_episodes, **root_kwarg)
        return _maybe_wrap(filtered, chunk_size=chunk_size)
    except Exception:
        if local_dir is not None:
            logger.info("Hub 重建失败，回退到本地路径: %s", local_dir)
            filtered = LeRobotDataset(repo_id, root=str(Path(local_dir)), episodes=matching_episodes)
            return _maybe_wrap(filtered, chunk_size=chunk_size)
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
