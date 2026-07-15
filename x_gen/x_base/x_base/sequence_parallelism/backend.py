"""
Sequence Parallelism - Communication Backend

通信后端抽象和 PyTorch Distributed 实现。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol

import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed import ProcessGroup, Work


class CollectiveBackend(Protocol):
    """通信后端协议"""

    def all_gather(self, tensor: Tensor, group: ProcessGroup, async_op: bool = False) -> tuple[Tensor, Work | None]: ...
    def reduce_scatter(
        self, tensor: Tensor, group: ProcessGroup, async_op: bool = False
    ) -> tuple[Tensor, Work | None]: ...
    def all_to_all(self, tensor: Tensor, group: ProcessGroup, scatter_dim: int, gather_dim: int) -> Tensor: ...


class BaseCommBackend(ABC):
    """通信后端抽象基类"""

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def all_gather(self, tensor: Tensor, group: ProcessGroup, async_op: bool = False) -> tuple[Tensor, Work | None]:
        pass

    @abstractmethod
    def reduce_scatter(self, tensor: Tensor, group: ProcessGroup, async_op: bool = False) -> tuple[Tensor, Work | None]:
        pass

    @abstractmethod
    def all_to_all_4d(
        self, tensor: Tensor, scatter_dim: int, gather_dim: int, group: ProcessGroup, use_sync: bool = False
    ) -> Tensor:
        pass


class TorchDistBackend(BaseCommBackend):
    """PyTorch Distributed 后端"""

    @property
    def name(self) -> str:
        return "torch_dist"

    def all_gather(self, tensor: Tensor, group: ProcessGroup, async_op: bool = False) -> tuple[Tensor, Work | None]:
        world_size = dist.get_world_size(group)
        if world_size == 1:
            return tensor.unsqueeze(0), None
        buffer_shape = (world_size,) + tensor.shape
        output = torch.empty(buffer_shape, dtype=tensor.dtype, device=tensor.device)
        buffer_list = list(torch.chunk(output, world_size, dim=0))
        if async_op:
            handle = dist.all_gather(buffer_list, tensor, group=group, async_op=True)
            return output, handle
        dist.all_gather(buffer_list, tensor, group=group)
        return output, None

    def reduce_scatter(self, tensor: Tensor, group: ProcessGroup, async_op: bool = False) -> tuple[Tensor, Work | None]:
        world_size = dist.get_world_size(group)
        if world_size == 1:
            return tensor.squeeze(0), None
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        output = torch.empty(tensor.shape[1:], dtype=tensor.dtype, device=tensor.device)
        buffer_list = list(torch.chunk(tensor, world_size, dim=0))
        if async_op:
            handle = dist.reduce_scatter(output, buffer_list, group=group, async_op=True)
            return output, handle
        dist.reduce_scatter(output, buffer_list, group=group)
        return output, None

    def all_to_all_4d(
        self, tensor: Tensor, scatter_dim: int, gather_dim: int, group: ProcessGroup, use_sync: bool = False
    ) -> Tensor:
        world_size = dist.get_world_size(group)
        if world_size == 1:
            return tensor

        if scatter_dim == 2 and gather_dim == 1:
            # (bs, seqlen/P, h) -> (bs, seqlen, h/P)
            bs, shard_seqlen, h = tensor.shape
            seqlen = shard_seqlen * world_size
            shard_h = h // world_size
            input_t = tensor.reshape(bs, shard_seqlen, world_size, shard_h).transpose(0, 2).contiguous()
            output = torch.empty_like(input_t)
            dist.all_to_all_single(output, input_t, group=group)
            if use_sync:
                torch.cuda.synchronize()
            return output.reshape(seqlen, bs, shard_h).transpose(0, 1).contiguous().reshape(bs, seqlen, shard_h)

        elif scatter_dim == 1 and gather_dim == 2:
            # (bs, seqlen, h/P) -> (bs, seqlen/P, h)
            bs, seqlen, shard_h = tensor.shape
            h = shard_h * world_size
            shard_seqlen = seqlen // world_size
            input_t = (
                tensor.reshape(bs, world_size, shard_seqlen, shard_h)
                .transpose(0, 3)
                .transpose(0, 1)
                .contiguous()
                .reshape(world_size, shard_h, shard_seqlen, bs)
            )
            output = torch.empty_like(input_t)
            dist.all_to_all_single(output, input_t, group=group)
            if use_sync:
                torch.cuda.synchronize()
            return output.reshape(h, shard_seqlen, bs).transpose(0, 2).contiguous().reshape(bs, shard_seqlen, h)

        raise ValueError(f"Invalid dims: scatter={scatter_dim}, gather={gather_dim}")

    def model_sharding(self, model: torch.nn.Module, group: ProcessGroup | None = None) -> None:
        rank = dist.get_rank(group)
        world_size = dist.get_world_size(group)
        for _, param in model.named_parameters():
            pad = (world_size - param.numel() % world_size) % world_size
            padded = param.data.view(-1) if pad == 0 else torch.nn.functional.pad(param.data.view(-1), [0, pad])
            splits = padded.split(padded.numel() // world_size)
            param.data = splits[rank]


# 全局默认后端
_default_backend: TorchDistBackend | None = None


def get_default_backend() -> TorchDistBackend:
    global _default_backend
    if _default_backend is None:
        _default_backend = TorchDistBackend()
    return _default_backend
