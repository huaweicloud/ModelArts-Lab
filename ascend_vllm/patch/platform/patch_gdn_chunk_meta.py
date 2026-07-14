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

"""Patch vllm_ascend.ops.triton.gdn_chunk_meta to support chunk_offsets_idx.

Adds the ``out_chunk_offsets_idx`` output tensor and the ``cu_seqlens`` input
to ``_build_chunk_meta_device_from_seq_lens`` and ``build_chunk_meta_device``.
The new tensor carries per-chunk sequence offsets consumed by the
cloud_ops_turbo AscendC operators.
"""

import torch
import vllm_ascend.ops.triton.gdn_chunk_meta as _meta

_validate_optional_output = _meta._validate_optional_output
_build_seq_lens = _meta._build_seq_lens
_build_chunk_counts = _meta._build_chunk_counts
_build_chunk_offsets = _meta._build_chunk_offsets
_build_final_chunk_indices = _meta._build_final_chunk_indices
_validate_cu_seqlens = _meta._validate_cu_seqlens


def _fill_chunk_offsets_idx_device(
    out: torch.Tensor,
    cu_seqlens: torch.Tensor,
    chunk_size: int,
) -> int:
    if out is None:
        return 0
    cu_seqlens_cpu = cu_seqlens.cpu() if cu_seqlens.device.type != "cpu" else cu_seqlens
    # Build on CPU first, then copy to device in one shot to avoid
    # per-element NPU kernel launches.
    out_cpu = torch.empty(out.shape[0], dtype=torch.int32)
    seq_idx = 0
    last_seqlens = 0
    out_cpu[0] = 0
    idx = 1
    for _, seqlens in enumerate(cu_seqlens_cpu.tolist()):
        if seqlens == last_seqlens:
            continue
        else:
            last_seqlens = seqlens
        while seq_idx + chunk_size < seqlens:
            seq_idx += chunk_size
            out_cpu[idx] = seq_idx
            idx += 1
        seq_idx = seqlens
        out_cpu[idx] = seq_idx
        idx += 1
    out[:idx].copy_(out_cpu[:idx])
    return idx


def _build_chunk_meta_device_from_seq_lens(
    cu_seqlens: torch.Tensor,
    seq_lens: torch.Tensor,
    chunk_size: int,
    out_chunk_indices: torch.Tensor | None = None,
    out_chunk_offsets: torch.Tensor | None = None,
    out_update_chunk_offsets: torch.Tensor | None = None,
    out_final_chunk_indices: torch.Tensor | None = None,
    out_chunk_offsets_idx: torch.Tensor | None = None,
) -> None:
    if (
        out_chunk_indices is None
        and out_chunk_offsets is None
        and out_update_chunk_offsets is None
        and out_final_chunk_indices is None
        and out_chunk_offsets_idx is None
    ):
        return

    num_seqs = seq_lens.shape[0]
    expected_prefix_shape = (num_seqs + 1,)
    expected_final_shape = (num_seqs,)

    _validate_optional_output(
        "out_chunk_indices",
        out_chunk_indices,
        expected_shape=None,
        expected_device=seq_lens.device,
    )
    if out_chunk_indices is not None and (out_chunk_indices.ndim != 2 or out_chunk_indices.shape[1] != 2):
        raise ValueError(
            f"chunk_gated_delta_rule meta: out_chunk_indices must have shape [num_chunks, 2],"
            f"got {tuple(out_chunk_indices.shape)}"
        )
    _validate_optional_output(
        "out_chunk_offsets",
        out_chunk_offsets,
        expected_shape=expected_prefix_shape,
        expected_device=seq_lens.device,
    )
    _validate_optional_output(
        "out_update_chunk_offsets",
        out_update_chunk_offsets,
        expected_shape=expected_prefix_shape,
        expected_device=seq_lens.device,
    )
    _validate_optional_output(
        "out_final_chunk_indices",
        out_final_chunk_indices,
        expected_shape=expected_final_shape,
        expected_device=seq_lens.device,
    )

    if num_seqs == 0:
        if out_chunk_offsets is not None:
            out_chunk_offsets.zero_()
        if out_update_chunk_offsets is not None:
            out_update_chunk_offsets.zero_()
        if out_final_chunk_indices is not None:
            out_final_chunk_indices.zero_()
        if out_chunk_offsets_idx is not None:
            out_chunk_offsets_idx.zero_()
        return

    chunk_counts = _build_chunk_counts(seq_lens, chunk_size)

    _validate_optional_output(
        "out_chunk_offsets_idx",
        out_chunk_offsets_idx,
        expected_shape=None,
        expected_device=seq_lens.device,
    )

    chunk_offsets = out_chunk_offsets
    if chunk_offsets is None and out_chunk_indices is not None:
        chunk_offsets = torch.empty(
            expected_prefix_shape,
            dtype=seq_lens.dtype,
            device=seq_lens.device,
        )
    update_chunk_offsets = out_update_chunk_offsets
    if update_chunk_offsets is None and out_final_chunk_indices is not None:
        update_chunk_offsets = torch.empty(
            expected_prefix_shape,
            dtype=seq_lens.dtype,
            device=seq_lens.device,
        )

    if chunk_offsets is not None:
        _build_chunk_offsets(chunk_counts, chunk_offsets, add_one=0)

    if update_chunk_offsets is not None:
        _build_chunk_offsets(chunk_counts, update_chunk_offsets, add_one=1)

    _fill_chunk_offsets_idx_device(out_chunk_offsets_idx, cu_seqlens, chunk_size)

    if out_final_chunk_indices is not None:
        _build_final_chunk_indices(
            chunk_counts,
            update_chunk_offsets,
            out_final_chunk_indices,
        )

    if out_chunk_indices is not None:
        total_chunks = out_chunk_indices.shape[0]
        if total_chunks == 0:
            return
        rows = torch.arange(total_chunks, device=seq_lens.device, dtype=chunk_offsets.dtype)
        compact_chunk_offsets = torch.unique_consecutive(chunk_offsets)
        seq_indices = torch.bucketize(rows, compact_chunk_offsets[1:], right=True)
        chunk_starts = compact_chunk_offsets.index_select(0, seq_indices)
        out_chunk_indices[:, 0].copy_(seq_indices.to(dtype=out_chunk_indices.dtype))
        out_chunk_indices[:, 1].copy_((rows - chunk_starts).to(dtype=out_chunk_indices.dtype))


def build_chunk_meta_device(
    cu_seqlens: torch.Tensor,
    chunk_size: int,
    out_chunk_indices: torch.Tensor | None = None,
    out_chunk_offsets: torch.Tensor | None = None,
    out_update_chunk_offsets: torch.Tensor | None = None,
    out_final_chunk_indices: torch.Tensor | None = None,
    out_chunk_offsets_idx: torch.Tensor | None = None,
    *,
    seq_lens: torch.Tensor | None = None,
    validate_inputs: bool = True,
) -> None:
    if validate_inputs:
        _validate_cu_seqlens(cu_seqlens, chunk_size)
    elif chunk_size <= 0:
        raise ValueError(f"chunk_gated_delta_rule meta: chunk_size must be positive, got {chunk_size}")
    _build_chunk_meta_device_from_seq_lens(
        cu_seqlens,
        _build_seq_lens(cu_seqlens) if seq_lens is None else seq_lens,
        chunk_size,
        out_chunk_indices=out_chunk_indices,
        out_chunk_offsets=out_chunk_offsets,
        out_update_chunk_offsets=out_update_chunk_offsets,
        out_final_chunk_indices=out_final_chunk_indices,
        out_chunk_offsets_idx=out_chunk_offsets_idx,
    )


_meta._fill_chunk_offsets_idx_device = _fill_chunk_offsets_idx_device
_meta._build_chunk_meta_device_from_seq_lens = _build_chunk_meta_device_from_seq_lens
_meta.build_chunk_meta_device = build_chunk_meta_device
