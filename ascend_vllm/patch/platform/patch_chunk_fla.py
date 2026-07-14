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

"""Patch vllm_ascend.ops.triton.fla.chunk.chunk_gated_delta_rule_fwd.

Routes the WY-representation kernels (chunk_scaled_dot_kkt / solve_tril /
recompute_w_u) to the cloud_ops_turbo AscendC operators when
VLLM_ASCEND_DISABLE_CLOUD_OPS_TURBO is 0. The cloud_ops path keeps w/u in
head-first [B, H, T, K/V] layout to skip redundant transposes for the
downstream AscendC operators, transposing back to time-first only for the
Triton hupdate kernel on the PCP path.

This patch tracks the upstream chunk_gated_delta_rule_fwd refactor: it
mirrors the _compact_empty_segments / keep_meta / cu_seqlens_kern flow and
the corrected PCP state recursion (s_i = a_i + Phi_i (s_{i-1} - s_0)).
"""

import torch
import vllm_ascend.envs as envs

# cloud_ops_turbo is imported lazily inside the runtime branch. A module-level
# import triggers device property queries before init_device_properties_triton()
# runs during worker setup, causing "Device properties not initialized" errors.
import vllm_ascend.ops.triton.fla.chunk as _chunk
from vllm.distributed import get_pcp_group
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.fla.ops.utils import SUPPRESS_LEVEL
from vllm_ascend.ops.gdn_attn_builder import _compact_empty_segments
from vllm_ascend.ops.triton.fla.chunk_delta_h import chunk_gated_delta_rule_fwd_h  # noqa: F401
from vllm_ascend.ops.triton.fla.chunk_delta_hupdate import chunk_gated_delta_rule_fwd_hupdate
from vllm_ascend.ops.triton.fla.chunk_o import chunk_fwd_o  # noqa: F401
from vllm_ascend.ops.triton.fla.cumsum import chunk_local_cumsum
from vllm_ascend.ops.triton.fla.utils import prepare_final_chunk_indices


def _build_chunk_offsets_idx_from_cu_seqlens(
    cu_seqlens: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    """Build chunk_offsets_idx from cu_seqlens when prebuilt_meta is unavailable.

    This produces the same tensor as _fill_chunk_offsets_idx_cpu/_device in
    gdn_attn_builder.py, but is computed on-the-fly as a fallback.
    """
    cu_seqlens_cpu = cu_seqlens.cpu() if cu_seqlens.device.type != "cpu" else cu_seqlens
    # Compute total number of chunks to determine output size
    seq_lens = cu_seqlens_cpu[1:] - cu_seqlens_cpu[:-1]
    chunk_counts = (seq_lens + chunk_size - 1) // chunk_size
    num_chunks = int(chunk_counts.sum())

    # Build entirely on CPU, then copy to device in one shot.
    out_cpu = torch.empty(num_chunks + 1, dtype=torch.int32)
    seq_idx = 0
    last_seqlens = 0
    out_cpu[0] = 0
    idx = 1
    for _, seqlens in enumerate(cu_seqlens_cpu[1:].tolist()):
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
    out = out_cpu[:idx].to(device=cu_seqlens.device)
    return out


def _patched_chunk_gated_delta_rule_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    output_final_state: bool,
    cu_seqlens: torch.LongTensor | None = None,
    prebuilt_meta=None,
):
    forward_context = get_forward_context()
    num_decodes = 0
    attn_metadata = forward_context.attn_metadata
    if attn_metadata is not None and isinstance(attn_metadata, dict):
        attn_metadata = next(iter(attn_metadata.values()), None)
    if attn_metadata is not None:
        num_decodes = attn_metadata.num_decodes
    chunk_size = 64
    use_cloud_ops = not envs.VLLM_ASCEND_DISABLE_CLOUD_OPS_TURBO
    block_indices_cumsum = None if prebuilt_meta is None else prebuilt_meta.block_indices_cumsum
    cu_seqlens_host = None if prebuilt_meta is None else prebuilt_meta.cu_seqlens_host
    chunk_indices_chunk64 = None if prebuilt_meta is None else prebuilt_meta.chunk_indices_chunk64
    chunk_indices_chunk64_host = None if prebuilt_meta is None else prebuilt_meta.chunk_indices_chunk64_host
    chunk_offsets_chunk64 = None if prebuilt_meta is None else prebuilt_meta.chunk_offsets_chunk64
    update_chunk_offsets_chunk64 = None if prebuilt_meta is None else prebuilt_meta.update_chunk_offsets_chunk64
    final_chunk_indices_chunk64 = None if prebuilt_meta is None else prebuilt_meta.final_chunk_indices_chunk64
    chunk_indices_large_block = None if prebuilt_meta is None else prebuilt_meta.chunk_indices_large_block
    chunk_offsets_idx = None if prebuilt_meta is None else getattr(prebuilt_meta, "chunk_offsets_idx", None)
    # Fallback: build chunk_offsets_idx from cu_seqlens when prebuilt_meta is
    # unavailable (e.g., cudagraph capture path or non-Ascend builder path).
    if chunk_offsets_idx is None and cu_seqlens is not None:
        chunk_offsets_idx = _build_chunk_offsets_idx_from_cu_seqlens(cu_seqlens, chunk_size)
    g = chunk_local_cumsum(
        g,
        chunk_size=chunk_size,
        cu_seqlens=cu_seqlens,
        block_indices=block_indices_cumsum,
    )
    # obtain WY representation. u is actually the new v.
    if use_cloud_ops:
        import cloud_ops_turbo  # noqa: F401

        beta_bht = beta.transpose(1, 2).contiguous()
        g_bht = g.transpose(1, 2).contiguous()

        A = torch.ops.cloud_ops_turbo.cloud_chunk_scaled_dot_kkt(
            k,
            beta_bht,
            g_bht,
            chunk_offsets_idx,
            chunk_size=chunk_size,
        )

        A = torch.ops.cloud_ops_turbo.cloud_solve_tril(A, chunk_offsets_idx)

        w, u = torch.ops.cloud_ops_turbo.cloud_recompute_wu(
            k,
            v,
            A,
            beta_bht,
            g_bht,
            chunk_offsets_idx,
            chunk_size=chunk_size,
        )

        # cloud_recompute_wu returns w: [B, H, T, K], u: [B, H, T, V] (head-first).
        # Keep head-first format for downstream AscendC operators which also
        # expect [B, H, T, K/V].  Only transpose to time-first when the Triton
        # hupdate kernel (PCP path) needs it.
    else:
        from vllm_ascend.ops.triton.fla.chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd
        from vllm_ascend.ops.triton.fla.solve_tril import solve_tril
        from vllm_ascend.ops.triton.fla.wy_fast import recompute_w_u_fwd

        A = chunk_scaled_dot_kkt_fwd(
            k=k,
            beta=beta,
            g_cumsum=g,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices_chunk64,
            output_dtype=torch.float32,
        )
        A = solve_tril(
            A=A,
            cu_seqlens=cu_seqlens,
            chunk_indices_large_block=chunk_indices_large_block,
            chunk_indices_bt=chunk_indices_chunk64,
            output_dtype=k.dtype,
        )
        w, u = recompute_w_u_fwd(
            k=k,
            v=v,
            beta=beta,
            A=A,
            g_cumsum=g,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices_chunk64,
        )

    if use_cloud_ops:
        # cloud_ops path: w/u are already head-first [B, H, T, K/V].
        # AscendC operators expect head-first, so skip the transpose.
        k_ascendc = k.to(torch.bfloat16).transpose(1, 2).contiguous()
        w_ascendc = w.to(torch.bfloat16)  # already [B, H, T, K]
        u_ascendc = u.to(torch.bfloat16)  # already [B, H, T, V]
        g_ascendc = g_bht  # already [B, H, T], reused from cloud_ops_turbo call
        q_ascendc = q.to(torch.bfloat16).transpose(1, 2).contiguous()
    else:
        # Triton path: w/u are time-first [B, T, H, K/V].
        # AscendC operators expect head-first, so transpose.
        k_ascendc = k.to(torch.bfloat16).transpose(1, 2).contiguous()
        w_ascendc = w.to(torch.bfloat16).transpose(1, 2).contiguous()
        u_ascendc = u.to(torch.bfloat16).transpose(1, 2).contiguous()
        g_ascendc = g.transpose(1, 2).contiguous()
        q_ascendc = q.to(torch.bfloat16).transpose(1, 2).contiguous()

    cu_seqlens = None if cu_seqlens is None else cu_seqlens.to(torch.int64)
    chunk_indices = None if chunk_indices_chunk64 is None else chunk_indices_chunk64.to(torch.int64)
    if cu_seqlens_host is None and cu_seqlens is not None:
        cu_seqlens_host = tuple(cu_seqlens.tolist())
    if chunk_indices_chunk64_host is None and chunk_indices is not None:
        chunk_indices_chunk64_host = tuple(chunk_indices.flatten().tolist())
    # Compact zero-length segments for the AscendC kernels (see
    # _compact_empty_segments).  chunk_indices_chunk64 is already compact-
    # ranked and is reused as-is; only cu_seqlens / initial_state need
    # compacting.
    if prebuilt_meta is not None and hasattr(prebuilt_meta, "keep_meta"):
        cu_seqlens_kern = cu_seqlens_host if prebuilt_meta.cu_seqlens_kern is None else prebuilt_meta.cu_seqlens_kern
        keep_meta = prebuilt_meta.keep_meta
        initial_state_kern = (
            initial_state[keep_meta] if initial_state is not None and keep_meta is not None else initial_state
        )
    else:
        cu_seqlens_kern, initial_state_kern, keep_meta = _compact_empty_segments(
            cu_seqlens_host,
            initial_state,
            device=initial_state.device if initial_state is not None else None,
        )
    h, v_new, final_state = torch.ops._C_ascend.chunk_gated_delta_rule_fwd_h(
        k_ascendc,
        w_ascendc,
        u_ascendc,
        g=g_ascendc,
        gk=None,
        initial_state=initial_state_kern,
        output_final_state=True,
        chunk_size=64,
        save_new_value=True,
        cu_seqlens=cu_seqlens_kern,
        chunk_indices=chunk_indices_chunk64_host,
        use_exp2=False,
        transpose_state_layout=False,
    )
    if keep_meta is not None:
        # Scatter the compacted final_state back to the original [N, H, K, V]
        # layout the PCP state recursion expects; empty segments keep their
        # initial state.
        _fs_full = initial_state.clone()
        _fs_full[keep_meta] = final_state
        final_state = _fs_full

    if get_pcp_group().world_size > 1:
        # When integrating mtp, since `mix_qkv` has been split, `num_decode`
        # cannot be directly obtained from the metadata and needs to be recalculated.
        actual_num_decodes = getattr(prebuilt_meta, "num_decodes", None)
        if actual_num_decodes is None:
            actual_num_decodes = num_decodes
        # chunk_gated_delta_rule_fwd_hupdate expects time-first [B, T, H, K/V].
        # In cloud_ops path w/u are head-first, so transpose for hupdate.
        w_tf = w.transpose(1, 2).contiguous() if use_cloud_ops else w
        u_tf = u.transpose(1, 2).contiguous() if use_cloud_ops else u
        h_update = chunk_gated_delta_rule_fwd_hupdate(
            k=k,
            w=w_tf,
            u=u_tf,
            g=g,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices_chunk64,
            chunk_offsets=chunk_offsets_chunk64,
            update_chunk_offsets=update_chunk_offsets_chunk64,
            num_decodes=actual_num_decodes,
        )
        all_final_state = get_pcp_group().all_gather(final_state.unsqueeze(0), 0)
        final_chunk_indices = final_chunk_indices_chunk64
        if final_chunk_indices is None:
            final_chunk_indices = prepare_final_chunk_indices(cu_seqlens, chunk_size)
        final_h_update = h_update[:, final_chunk_indices, :, :, :]
        all_final_h_update = get_pcp_group().all_gather(final_h_update, 0)

        updated_state = final_state.new_empty(get_pcp_group().world_size, *final_state.shape)
        updated_state[0, ...] = all_final_state[0]
        for i in range(1, get_pcp_group().world_size):
            # correct_i = all_final_state[i] + Phi_i * (correct_{i-1} - s0)
            updated_final_state = all_final_state[i] + torch.matmul(
                all_final_h_update[i, ...], updated_state[i - 1, ...] - initial_state
            )
            updated_state[i, ...] = updated_final_state

        final_state = updated_state[-1, ...]

        if get_pcp_group().rank_in_group == 0:
            updated_h_state = torch.zeros_like(final_state)
        else:
            updated_h_state = updated_state[get_pcp_group().rank_in_group - 1, ...]

        if get_pcp_group().rank_in_group > 0:
            rerun_initial_state = initial_state.clone()
            prefill_seq_offset = actual_num_decodes
            prefill_slice = slice(prefill_seq_offset, final_state.shape[0])
            rerun_initial_state[prefill_slice] = updated_h_state[prefill_slice]
            h, v_new, _ = chunk_gated_delta_rule_fwd_h(
                k=k,
                w=w_tf,
                u=u_tf,
                g=g,
                initial_state=rerun_initial_state,
                output_final_state=True,
                cu_seqlens=cu_seqlens,
                chunk_indices=chunk_indices_chunk64,
                chunk_offsets=chunk_offsets_chunk64,
            )
            h = h.transpose(1, 2).contiguous()
            v_new = v_new.transpose(1, 2).contiguous()

    o_ascendc = torch.ops._C_ascend.chunk_fwd_o(
        q_ascendc,
        k_ascendc,
        v_new,
        h,
        scale,
        g=g_ascendc,
        g_gamma=None,
        cu_seqlens=cu_seqlens_host,
        chunk_indices=chunk_indices_chunk64_host,
        chunk_size=64,
        transpose_state_layout=False,
    )

    o = o_ascendc.to(torch.bfloat16).transpose(1, 2).contiguous()
    v_new = v_new.to(torch.bfloat16).transpose(1, 2).contiguous()
    h = h.to(torch.bfloat16).transpose(1, 2).contiguous()

    if SUPPRESS_LEVEL < 3:
        return g, o, A, final_state, None, None, None
    elif SUPPRESS_LEVEL >= 3:
        return g, o, A, final_state, w, h, v_new


_chunk._build_chunk_offsets_idx_from_cu_seqlens = _build_chunk_offsets_idx_from_cu_seqlens
_chunk.chunk_gated_delta_rule_fwd = _patched_chunk_gated_delta_rule_fwd
