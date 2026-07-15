"""
Sequence Parallelism - Pipeline Enable

Pipeline 集成入口。
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch.distributed as dist


@runtime_checkable
class Parallelizable(Protocol):
    """可并行化模块协议"""

    def enable_parallel(
        self,
        dp_size: int,
        sp_size: int,
        enable_cp: bool,
        ulysses_size: int | None = None,
        ring_size: int | None = None,
        **kwargs,
    ) -> None: ...


@runtime_checkable
class PipelineAdapter(Protocol):
    """Pipeline 适配器协议"""

    @property
    def transformer(self) -> Parallelizable: ...
    @property
    def transformer_2(self) -> Parallelizable | None: ...
    @property
    def transformer_3(self) -> Parallelizable | None: ...


def enable_sp(
    pipeline: PipelineAdapter,
    sp_size: int | None = None,
    ulysses_size: int | None = None,
    ring_size: int | None = None,
    enable_transformer_2: bool = False,
    enable_transformer_3: bool = False,
) -> None:
    """启用序列并行"""
    if sp_size is None:
        sp_size = dist.get_world_size()
        dp_size = 1
    else:
        if ulysses_size is not None and ring_size is not None:
            if ulysses_size * ring_size != sp_size:
                raise ValueError(
                    f"ulysses size {ulysses_size} * ring size {ring_size} must be equal to sp_size {sp_size}"
                )
        elif dist.get_world_size() % sp_size != 0:
            raise ValueError(f"world_size {dist.get_world_size()} must be divisible by sp_size {sp_size}")
        dp_size = dist.get_world_size() // sp_size

    pipeline.transformer.enable_parallel(
        dp_size, sp_size, enable_cp=False, ulysses_size=ulysses_size, ring_size=ring_size
    )

    if enable_transformer_2 and pipeline.transformer_2:
        pipeline.transformer_2.enable_parallel(
            dp_size, sp_size, enable_cp=False, ulysses_size=ulysses_size, ring_size=ring_size
        )

    if enable_transformer_3 and pipeline.transformer_3:
        actual_sp = min(sp_size, 4) if sp_size > 4 else sp_size
        actual_dp = dist.get_world_size() // actual_sp
        pipeline.transformer_3.enable_parallel(
            actual_dp, actual_sp, enable_cp=False, ulysses_size=ulysses_size, ring_size=ring_size
        )
