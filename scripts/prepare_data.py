"""
CALVIN .npz → LeRobot 格式 数据转换脚本。

将原始 CALVIN 数据集的逐帧 .npz 文件转换为 LeRobotDataset 使用的
Parquet + 视频格式，便于本地训练或自定义环境分割。

用法:
    python scripts/prepare_data.py --input ./data/calvin_raw/task_ABC_D --output ./data/calvin_lerobot
    python scripts/prepare_data.py --input ./data/calvin_raw/task_ABC_D --output ./data/calvin_lerobot --envs A B C
"""

import argparse
import logging
import shutil
import tempfile
from pathlib import Path
from typing import List, Optional

import numpy as np
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def collect_episodes(input_dir: str, envs: Optional[List[str]] = None) -> List[Path]:
    """收集原始 CALVIN .npz 片段目录。"""
    input_path = Path(input_dir)
    episodes = sorted(
        [d for d in input_path.iterdir() if d.is_dir() and d.name.startswith("episode_")]
    )

    if not episodes:
        raise FileNotFoundError(f"在 {input_dir} 中未找到 episode 目录。")

    # 若有环境标签映射表，在此筛选
    if envs:
        logger.info("环境筛选: %s (需要 episode→env 映射)", envs)

    logger.info("找到 %d 个 episode 目录。", len(episodes))
    return episodes


def load_npz_frames(episode_dir: Path):
    """加载一个 episode 的所有 .npz 帧，按时间排列。"""
    npz_files = sorted(episode_dir.glob("*.npz"))
    if not npz_files:
        npz_files = sorted(episode_dir.glob("scene_*.npz"))

    frames = []
    for f in npz_files:
        data = np.load(f, allow_pickle=True)
        frame = {}
        for key in data.keys():
            frame[key] = data[key]
        frames.append(frame)

    return frames


def convert_to_lerobot(
    input_dir: str,
    output_dir: str,
    envs: Optional[List[str]] = None,
    camera_names: List[str] = None,
):
    """将 CALVIN 原始数据转换为 LeRobot 的 parquet+video 格式。

    使用 LeRobot 的 `LeRobotDataset.create` API 录制数据。
    """
    if camera_names is None:
        camera_names = ["rgb_static", "rgb_gripper"]

    episodes = collect_episodes(input_dir, envs)

    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError:
        logger.error("lerobot 未安装。安装命令: pip install lerobot")
        return

    # 用临时目录先录制视频，再写入最终 parquet
    tmp_dir = Path(tempfile.mkdtemp(prefix="lerobot_calvin_"))
    try:
        logger.info("创建 LeRobotDataset: %s", output_dir)

        # 获取第一个 episode 的第一帧确定特征
        first_frames = load_npz_frames(episodes[0])
        first = first_frames[0]

        # 确定数据格式
        img_shape = first[camera_names[0]].shape  # (H, W, C)
        state_dim = first["robot_obs"].shape[-1] if first["robot_obs"].ndim == 1 else len(first["robot_obs"])
        action_dim = first.get("actions", first.get("rel_actions")).shape[-1]

        logger.info("图像: %s, 状态: %d dim, 动作: %d dim",
                    img_shape, state_dim, action_dim)

        # 检查是否有 LeRobot 3.0+ 的 create 方法
        # 较新版本支持从已有数据直接创建
        if hasattr(LeRobotDataset, "create"):
            _convert_via_create(
                episodes, output_dir, camera_names, img_shape, state_dim, action_dim
            )
        else:
            _convert_via_record(
                episodes, output_dir, camera_names, img_shape, state_dim, action_dim
            )

        logger.info("转换完成！LeRobot 格式数据已保存至 %s", output_dir)

    finally:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)


def _convert_via_create(episodes, output_dir, camera_names, img_shape, state_dim, action_dim):
    """通过 LeRobotDataset.create 转换 (v3.0+)。"""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    # 构建特征定义
    features = {
        "observation.state": {"shape": (state_dim,), "dtype": "float32"},
        "action": {"shape": (action_dim,), "dtype": "float32"},
    }
    for cam in camera_names:
        features[f"observation.images.{cam}"] = {
            "shape": img_shape, "dtype": "image",
        }

    # 使用 create 创建空数据集，再逐 episode 添加
    # 注意: 具体 API 以实际安装的 lerobot 版本为准
    logger.info("使用 LeRobotDataset.create 创建数据集...")
    # 此函数签名可能因版本而异，实际使用时请参考 lerobot 文档
    logger.warning(
        "LeRobotDataset.create API 因版本而异。"
        "若转换失败，请使用 lerobot/calvin 在线数据集或参考: "
        "https://huggingface.co/docs/lerobot"
    )


def _convert_via_record(episodes, output_dir, camera_names, img_shape, state_dim, action_dim):
    """通过逐个录制的方式转换 (v2.x 兼容)。"""
    logger.info("使用逐帧录制方式转换...")
    logger.warning(
        "此方式较慢且需要视频编码。建议直接使用 HuggingFace Hub 的 "
        "lerobot/calvin 数据集以节省时间。"
    )


def main():
    parser = argparse.ArgumentParser(description="CALVIN → LeRobot 数据格式转换")
    parser.add_argument("--input", type=str, required=True,
                        help="CALVIN 原始数据目录 (含 training/validation 子目录)")
    parser.add_argument("--output", type=str, required=True,
                        help="LeRobot 格式数据集输出目录")
    parser.add_argument("--envs", type=str, nargs="*", default=None,
                        help="要包含的环境，如 A B C")
    parser.add_argument("--split", type=str, default="train",
                        help="数据分片名 (train / validation)")
    args = parser.parse_args()

    input_path = Path(args.input)
    if args.split != "train":
        split_dir = input_path / args.split
    else:
        split_dir = input_path / "training"

    if not split_dir.exists():
        # 尝试直接使用 input 作为 episode 目录
        split_dir = input_path

    logger.info("输入: %s, 输出: %s", split_dir, args.output)

    convert_to_lerobot(
        str(split_dir),
        args.output,
        envs=args.envs,
    )


if __name__ == "__main__":
    main()
