"""
统一日志接口，同时支持 WandB 和 SwanLab。

用法:
    logger = get_logger(project="lerobot-act", backend="wandb")
    logger.log({"train/loss": 0.05}, step=100)
"""

import logging
from typing import Any, Dict, Optional

_logger = logging.getLogger(__name__)


class DummyLogger:
    """空日志器，在未启用任何在线日志时使用。"""

    def log(self, metrics: Dict[str, Any], step: int = 0):
        pass

    def watch(self, model, **kwargs):
        pass

    def finish(self):
        pass

    def save(self, path: str):
        pass


class WandBLogger:
    def __init__(self, project: str, name: str, config: Optional[Dict] = None, **kwargs):
        import wandb
        wandb.init(project=project, name=name, config=config, **kwargs)
        self._wandb = wandb

    def log(self, metrics: Dict[str, Any], step: int = 0):
        self._wandb.log(metrics, step=step)

    def watch(self, model, **kwargs):
        self._wandb.watch(model, **kwargs)

    def finish(self):
        self._wandb.finish()

    def save(self, path: str):
        self._wandb.save(path)


class SwanLabLogger:
    def __init__(self, project: str, name: str, config: Optional[Dict] = None, **kwargs):
        import swanlab
        swanlab.init(project=project, experiment_name=name, config=config, **kwargs)
        self._swanlab = swanlab

    def log(self, metrics: Dict[str, Any], step: int = 0):
        self._swanlab.log(metrics, step=step)

    def watch(self, model, **kwargs):
        pass  # SwanLab 不支持原生模型监控

    def finish(self):
        self._swanlab.finish()

    def save(self, path: str):
        self._swanlab.save(path)


def get_logger(
    project: str = "lerobot-act",
    name: str = "baseline",
    config: Optional[Dict] = None,
    backend: str = "wandb",
    **kwargs,
):
    """创建实验日志器。backend 可选 'wandb'、'swanlab' 或 'none'。"""
    backend = backend.lower()
    if backend == "wandb":
        return WandBLogger(project=project, name=name, config=config, **kwargs)
    elif backend == "swanlab":
        return SwanLabLogger(project=project, name=name, config=config, **kwargs)
    else:
        return DummyLogger()


def get_logger_safe(
    project: str = "lerobot-act",
    name: str = "baseline",
    config: Optional[Dict] = None,
    backend: str = "wandb",
    **kwargs,
):
    """创建日志器，初始化失败时自动降级为 DummyLogger，不中断训练。"""
    try:
        return get_logger(project=project, name=name, config=config, backend=backend, **kwargs)
    except Exception as e:
        _logger.warning("初始化 %s 日志器失败: %s，降级为空日志器。", backend, e)
        return DummyLogger()
