#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""Patch vllm_ascend.ops.gdn_attn_builder to thread chunk_offsets_idx.

Adds the ``chunk_offsets_idx`` field to ``GDNChunkedPrefillMetadata`` and
``_GDNChunkedPrefillBufferSlot``, adds the ``_fill_chunk_offsets_idx_cpu``
helper and updates the allocate/slice/fill/build helpers so that the
prebuilt chunk meta carries the per-chunk sequence offsets consumed by
cloud_ops_turbo AscendC operators.
"""

from dataclasses import dataclass

import torch
import vllm_ascend.ops.gdn_attn_builder as _builder
import vllm_ascend.ops.triton.gdn_chunk_meta as _chunk_meta

_validate_cu_seqlens = _chunk_meta._validate_cu_seqlens
_build_seq_lens = _chunk_meta._build_seq_lens


@dataclass
class GDNChunkedPrefillMetadata:
    cu_seqlens_cpu: torch.Tensor
    cu_seqlens_host: tuple[int, ...]
    chunk_indices_chunk64_host: tuple[int, ...]
    chunk_indices_chunk64: torch.Tensor
    chunk_offsets_chunk64: torch.Tensor
    update_chunk_offsets_chunk64: torch.Tensor
    final_chunk_indices_chunk64: torch.Tensor
    chunk_indices_large_block: torch.Tensor
    block_indices_cumsum: torch.Tensor
    chunk_offsets_idx: torch.Tensor
    _buffer_slot: object | None = None


@dataclass
class _GDNChunkedPrefillBufferSlot:
    chunk_indices_chunk64: torch.Tensor
    chunk_offsets_chunk64: torch.Tensor
    update_chunk_offsets_chunk64: torch.Tensor
    final_chunk_indices_chunk64: torch.Tensor
    chunk_indices_large_block: torch.Tensor
    block_indices_cumsum: torch.Tensor
    chunk_offsets_idx: torch.Tensor


def _fill_chunk_offsets_idx_cpu(
    out: torch.Tensor,
    cu_seqlens_cpu: torch.Tensor,
    chunk_size: int,
) -> int:
    seq_idx = 0
    last_seqlens = 0
    out[0] = 0
    idx = 1
    for _, seqlens in enumerate(cu_seqlens_cpu[1:].tolist()):
        if seqlens == last_seqlens:
            continue
        else:
            last_seqlens = seqlens
        while seq_idx + chunk_size < seqlens:
            seq_idx += chunk_size
            out[idx] = seq_idx
            idx += 1
        seq_idx = seqlens
        out[idx] = seq_idx
        idx += 1
    return idx


# Reuse the existing helpers that are unchanged.
_fill_chunk_indices_cpu = _builder._fill_chunk_indices_cpu
_fill_chunk_offsets_cpu = _builder._fill_chunk_offsets_cpu
_fill_update_chunk_offsets_cpu = _builder._fill_update_chunk_offsets_cpu
_fill_final_chunk_indices_cpu = _builder._fill_final_chunk_indices_cpu
_build_chunk_indices_host = _builder._build_chunk_indices_host
_build_chunk_meta_shape_info = _builder._build_chunk_meta_shape_info
_build_chunk_meta_size_info = _builder._build_chunk_meta_size_info
_GDNChunkMetaSizeInfo = _builder._GDNChunkMetaSizeInfo
_GDNChunkMetaShapeInfo = _builder._GDNChunkMetaShapeInfo


def _allocate_chunk_meta_cpu_tensors(shape_info: _GDNChunkMetaSizeInfo) -> dict[str, torch.Tensor]:
    return {
        "chunk_indices_chunk64": torch.empty(
            (shape_info.num_chunk_indices_chunk64, 2),
            dtype=torch.int32,
        ),
        "chunk_offsets_chunk64": torch.empty(
            (shape_info.num_seqs + 1,),
            dtype=torch.int32,
        ),
        "update_chunk_offsets_chunk64": torch.empty(
            (shape_info.num_seqs + 1,),
            dtype=torch.int32,
        ),
        "final_chunk_indices_chunk64": torch.empty(
            (shape_info.num_seqs,),
            dtype=torch.int32,
        ),
        "chunk_indices_large_block": torch.empty(
            (shape_info.num_chunk_indices_large_block, 2),
            dtype=torch.int32,
        ),
        "block_indices_cumsum": torch.empty(
            (shape_info.num_block_indices_cumsum, 2),
            dtype=torch.int32,
        ),
        "chunk_offsets_idx": torch.empty(
            (shape_info.num_chunk_indices_chunk64 + 1,),
            dtype=torch.int32,
        ),
    }


def _slice_chunk_meta_slot_tensors(
    slot: _GDNChunkedPrefillBufferSlot,
    shape_info: _GDNChunkMetaSizeInfo,
) -> dict[str, torch.Tensor]:
    return {
        "chunk_indices_chunk64": slot.chunk_indices_chunk64[: shape_info.num_chunk_indices_chunk64],
        "chunk_offsets_chunk64": slot.chunk_offsets_chunk64[: shape_info.num_seqs + 1],
        "update_chunk_offsets_chunk64": slot.update_chunk_offsets_chunk64[: shape_info.num_seqs + 1],
        "final_chunk_indices_chunk64": slot.final_chunk_indices_chunk64[: shape_info.num_seqs],
        "chunk_indices_large_block": slot.chunk_indices_large_block[: shape_info.num_chunk_indices_large_block],
        "block_indices_cumsum": slot.block_indices_cumsum[: shape_info.num_block_indices_cumsum],
        "chunk_offsets_idx": slot.chunk_offsets_idx[: shape_info.num_chunk_indices_chunk64 + 1],
    }


def _fill_chunk_meta_cpu_tensors(
    tensors: dict[str, torch.Tensor],
    shape_info: _GDNChunkMetaShapeInfo,
    cu_seqlens_cpu: torch.Tensor,
    chunk_size: int,
) -> None:
    _fill_chunk_indices_cpu(
        tensors["chunk_indices_chunk64"],
        shape_info.chunk_counts_chunk64,
    )
    _fill_chunk_offsets_cpu(
        tensors["chunk_offsets_chunk64"],
        shape_info.chunk_counts_chunk64,
    )
    _fill_update_chunk_offsets_cpu(
        tensors["update_chunk_offsets_chunk64"],
        shape_info.chunk_counts_chunk64,
    )
    _fill_final_chunk_indices_cpu(
        tensors["final_chunk_indices_chunk64"],
        shape_info.chunk_counts_chunk64,
    )
    _fill_chunk_indices_cpu(
        tensors["chunk_indices_large_block"],
        shape_info.chunk_counts_large_block,
    )
    _fill_chunk_indices_cpu(
        tensors["block_indices_cumsum"],
        shape_info.chunk_counts_cumsum,
    )
    _fill_chunk_offsets_idx_cpu(
        tensors["chunk_offsets_idx"],
        cu_seqlens_cpu,
        chunk_size,
    )


def _fill_chunk_meta_device_tensors(
    builder,
    cu_seqlens: torch.Tensor,
    tensors: dict[str, torch.Tensor],
) -> None:
    # Look up build_chunk_meta_device at call time so the patched version from
    # patch_gdn_chunk_meta is always used regardless of import ordering.
    build_chunk_meta_device = _chunk_meta.build_chunk_meta_device
    seq_lens = None
    validate_inputs = True
    if cu_seqlens.device.type == "npu":
        _validate_cu_seqlens(cu_seqlens, builder._ascend_gdn_chunk_size)
        assert builder._ascend_gdn_large_block_size > 0
        assert builder._ascend_gdn_cumsum_block_size > 0
        seq_lens = _build_seq_lens(cu_seqlens)
        validate_inputs = False
    build_chunk_meta_device(
        cu_seqlens=cu_seqlens,
        chunk_size=builder._ascend_gdn_chunk_size,
        out_chunk_indices=tensors["chunk_indices_chunk64"],
        out_chunk_offsets=tensors["chunk_offsets_chunk64"],
        out_update_chunk_offsets=tensors["update_chunk_offsets_chunk64"],
        out_final_chunk_indices=tensors["final_chunk_indices_chunk64"],
        out_chunk_offsets_idx=tensors["chunk_offsets_idx"],
        seq_lens=seq_lens,
        validate_inputs=validate_inputs,
    )
    build_chunk_meta_device(
        cu_seqlens=cu_seqlens,
        chunk_size=builder._ascend_gdn_large_block_size,
        out_chunk_indices=tensors["chunk_indices_large_block"],
        seq_lens=seq_lens,
        validate_inputs=validate_inputs,
    )
    build_chunk_meta_device(
        cu_seqlens=cu_seqlens,
        chunk_size=builder._ascend_gdn_cumsum_block_size,
        out_chunk_indices=tensors["block_indices_cumsum"],
        seq_lens=seq_lens,
        validate_inputs=validate_inputs,
    )


def _build_chunked_prefill_metadata(
    builder,
    tensors: dict[str, torch.Tensor],
    *,
    cu_seqlens_cpu: torch.Tensor,
    slot: _GDNChunkedPrefillBufferSlot | None = None,
) -> GDNChunkedPrefillMetadata:
    return GDNChunkedPrefillMetadata(
        cu_seqlens_cpu=cu_seqlens_cpu,
        cu_seqlens_host=tuple(cu_seqlens_cpu.to(torch.int64).tolist()),
        chunk_indices_chunk64_host=_build_chunk_indices_host(
            cu_seqlens_cpu,
            builder._ascend_gdn_chunk_size,
        ),
        chunk_indices_chunk64=tensors["chunk_indices_chunk64"],
        chunk_offsets_chunk64=tensors["chunk_offsets_chunk64"],
        update_chunk_offsets_chunk64=tensors["update_chunk_offsets_chunk64"],
        final_chunk_indices_chunk64=tensors["final_chunk_indices_chunk64"],
        chunk_indices_large_block=tensors["chunk_indices_large_block"],
        block_indices_cumsum=tensors["block_indices_cumsum"],
        chunk_offsets_idx=tensors["chunk_offsets_idx"],
        _buffer_slot=slot,
    )


def _allocate_chunked_prefill_slot(builder, device: torch.device):
    max_num_batched_tokens = builder.vllm_config.scheduler_config.max_num_batched_tokens
    max_num_seqs = builder.vllm_config.scheduler_config.max_num_seqs
    return _GDNChunkedPrefillBufferSlot(
        chunk_indices_chunk64=torch.empty(
            (max_num_batched_tokens, 2),
            dtype=torch.int32,
            device=device,
        ),
        chunk_offsets_chunk64=torch.empty(
            (max_num_seqs + 1,),
            dtype=torch.int32,
            device=device,
        ),
        update_chunk_offsets_chunk64=torch.empty(
            (max_num_seqs + 1,),
            dtype=torch.int32,
            device=device,
        ),
        final_chunk_indices_chunk64=torch.empty(
            (max_num_seqs,),
            dtype=torch.int32,
            device=device,
        ),
        chunk_indices_large_block=torch.empty(
            (max_num_batched_tokens, 2),
            dtype=torch.int32,
            device=device,
        ),
        block_indices_cumsum=torch.empty(
            (max_num_batched_tokens, 2),
            dtype=torch.int32,
            device=device,
        ),
        chunk_offsets_idx=torch.empty(
            max_num_batched_tokens + 1,
            dtype=torch.int32,
            device=device,
        ),
    )


def _build_non_spec_chunked_prefill_meta_cpu(builder, cu_seqlens_cpu: torch.Tensor) -> GDNChunkedPrefillMetadata:
    shape_info = _build_chunk_meta_shape_info(builder, cu_seqlens_cpu)
    tensors = _allocate_chunk_meta_cpu_tensors(shape_info)
    _fill_chunk_meta_cpu_tensors(tensors, shape_info, cu_seqlens_cpu, builder._ascend_gdn_chunk_size)
    return _build_chunked_prefill_metadata(builder, tensors, cu_seqlens_cpu=cu_seqlens_cpu)


# Apply replacements onto the vllm_ascend module namespace.
_builder.GDNChunkedPrefillMetadata = GDNChunkedPrefillMetadata
_builder._GDNChunkedPrefillBufferSlot = _GDNChunkedPrefillBufferSlot
_builder._fill_chunk_offsets_idx_cpu = _fill_chunk_offsets_idx_cpu
_builder._allocate_chunk_meta_cpu_tensors = _allocate_chunk_meta_cpu_tensors
_builder._slice_chunk_meta_slot_tensors = _slice_chunk_meta_slot_tensors
_builder._fill_chunk_meta_cpu_tensors = _fill_chunk_meta_cpu_tensors
_builder._fill_chunk_meta_device_tensors = _fill_chunk_meta_device_tensors
_builder._build_chunked_prefill_metadata = _build_chunked_prefill_metadata
_builder._allocate_chunked_prefill_slot = _allocate_chunked_prefill_slot
_builder._build_non_spec_chunked_prefill_meta_cpu = _build_non_spec_chunked_prefill_meta_cpu
