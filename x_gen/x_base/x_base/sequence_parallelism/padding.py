"""
Sequence Parallelism - Padding Manager

Padding 管理，替代全局 PAD_DICT。
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import torch.distributed as dist
from torch.distributed import ProcessGroup

from .errors import PadNotSetError


@dataclass
class PadManager:
    """Padding 管理器"""

    _pads: dict[str, int] = field(default_factory=dict)

    def set(self, name: str, dim_size: int, sp_size: int) -> int:
        """计算并设置 padding"""
        pad = (sp_size - (dim_size % sp_size)) % sp_size
        self._pads[name] = pad
        return pad

    def set_from_group(self, name: str, dim_size: int, group: ProcessGroup) -> int:
        """从 ProcessGroup 计算 padding"""
        return self.set(name, dim_size, dist.get_world_size(group))

    def get(self, name: str) -> int:
        """获取 padding，不存在则抛出 PadNotSetError"""
        if name not in self._pads:
            raise PadNotSetError(name)
        return self._pads[name]

    def get_or_default(self, name: str, default: int = 0) -> int:
        """获取 padding，不存在则返回默认值"""
        return self._pads.get(name, default)

    def is_set(self, name: str) -> bool:
        return name in self._pads

    def clear(self) -> None:
        self._pads.clear()


# 全局实例
_pad_manager: PadManager | None = None


def get_pad_manager() -> PadManager:
    """获取全局 PadManager 实例"""
    global _pad_manager
    if _pad_manager is None:
        _pad_manager = PadManager()
    return _pad_manager


# 向后兼容的全局字典
PAD_DICT: dict[str, int] = {}


def set_pad(name: str, dim_size: int, parallel_group: ProcessGroup) -> int:
    """设置 padding (向后兼容，deprecated)"""
    warnings.warn("set_pad() is deprecated. Use PadManager.set_from_group() instead.", DeprecationWarning, stacklevel=2)
    sp_size = dist.get_world_size(parallel_group)
    pad = (sp_size - (dim_size % sp_size)) % sp_size
    global PAD_DICT
    PAD_DICT[name] = pad
    get_pad_manager().set(name, dim_size, sp_size)
    return pad


def get_pad(name: str) -> int:
    """获取 padding (向后兼容，deprecated)"""
    warnings.warn("get_pad() is deprecated. Use PadManager.get() instead.", DeprecationWarning, stacklevel=2)
    manager = get_pad_manager()
    if manager.is_set(name):
        return manager.get(name)
    global PAD_DICT
    if name not in PAD_DICT:
        raise PadNotSetError(name)
    return PAD_DICT[name]
