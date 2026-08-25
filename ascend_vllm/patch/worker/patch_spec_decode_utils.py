from __future__ import annotations

from vllm.triton_utils import tl, triton

_PATCH_APPLIED = False


@triton.jit
def copy_and_expand_dflash_and_dspark_inputs_kernel_single_grid(
    # Inputs
    next_token_ids_ptr,  # [num_reqs]
    target_positions_ptr,  # [num_context]
    context_slot_mapping_ptr,  # [num_context]
    # Outputs
    out_input_ids_ptr,  # [num_query_total] (output)
    out_context_positions_ptr,  # [num_context] (output)
    out_query_positions_ptr,  # [num_query_total] (output)
    out_context_slot_mapping_ptr,  # [num_context] (output)
    out_query_slot_mapping_ptr,  # [num_query_total] (output)
    out_token_indices_ptr,  # [num_reqs * num_speculative_tokens] (output)
    # Block table
    block_table_ptr,  # [max_reqs, max_blocks]
    block_table_stride,  # stride of block_table dim 0 (in elements)
    # Metadata
    query_start_loc_ptr,  # [num_reqs + 1]
    seq_lens_ptr,  # [num_reqs]
    num_rejected_tokens_ptr,  # [num_reqs] or null (0) when not padded
    # Scalars
    parallel_drafting_token_id,  # tl.int32
    block_size,  # tl.int32
    num_query_per_req,  # tl.int32
    num_speculative_tokens,  # tl.int32
    total_input_tokens,  # tl.int32
    batch_size,  # tl.int32
    max_model_len,  # tl.int32
    HAS_NUM_REJECTED: tl.constexpr = False,
    SAMPLE_FROM_ANCHOR: tl.constexpr = False,
):
    req_idx = tl.program_id(axis=0)
    ctx_start = tl.load(query_start_loc_ptr + req_idx)
    ctx_end = tl.load(query_start_loc_ptr + req_idx + 1)
    num_ctx = ctx_end - ctx_start

    for j in range(0, num_ctx):
        ctx_pos_idx = ctx_start + j
        pos = tl.load(target_positions_ptr + ctx_pos_idx)
        tl.store(out_context_positions_ptr + ctx_pos_idx, pos)

        slot = tl.load(context_slot_mapping_ptr + ctx_pos_idx)
        tl.store(out_context_slot_mapping_ptr + ctx_pos_idx, slot)

    if HAS_NUM_REJECTED:
        num_rejected = tl.load(num_rejected_tokens_ptr + req_idx)
        valid_ctx_end = ctx_end - num_rejected
    else:
        num_rejected = 0
        valid_ctx_end = ctx_end

    # A discarded or padded request may have no valid sampled tokens left.
    # Do not read valid_ctx_end - 1 in that case, as it may belong to another request.
    has_valid_context = (
        (ctx_start >= 0)
        & (valid_ctx_end > ctx_start)
        & (valid_ctx_end <= ctx_end)
        & (valid_ctx_end <= total_input_tokens)
    )

    seq_len = tl.load(seq_lens_ptr + req_idx)
    effective_seq_len = seq_len - num_rejected

    last_pos = tl.load(
        target_positions_ptr + valid_ctx_end - 1,
        mask=has_valid_context,
        other=-1,
    )

    for q_idx in range(0, num_query_per_req):
        query_pos = last_pos + 1 + q_idx
        query_out_idx = req_idx * num_query_per_req + q_idx
        query_cache_pos = effective_seq_len + q_idx
        block_num_q = query_cache_pos // block_size

        position_in_range = (query_pos >= 0) & (query_pos < max_model_len)

        block_in_range = (query_cache_pos >= 0) & (block_num_q >= 0) & (block_num_q < block_table_stride)

        is_valid_query = has_valid_context & position_in_range & block_in_range

        # Clamp query_pos to max_model_len - 1 to avoid out-of-bounds access in
        # position embeddings when draft tokens exceed the model's maximum sequence length.
        # Draft tokens with clamped positions should be ignored in the attention backend.
        # Preserve the original max-length fix: clamp overflow to the last legal
        # position while marking the query invalid for KV-cache writes below.
        clamped_query_pos = tl.minimum(query_pos, max_model_len - 1)
        clamped_query_pos = tl.maximum(clamped_query_pos, 0)
        safe_query_pos = tl.where(has_valid_context, clamped_query_pos, 0)
        tl.store(out_query_positions_ptr + query_out_idx, safe_query_pos)

        # Mask out-of-bounds block_num_q to avoid out-of-bounds access in block_table
        block_id_q = tl.load(
            block_table_ptr + req_idx * block_table_stride + block_num_q,
            mask=is_valid_query,
            other=0,
        ).to(tl.int64)
        slot_q = block_id_q * block_size + (query_cache_pos % block_size)
        # -1 is the padding slot; invalid draft rows must not write KV cache.
        slot_q = tl.where(is_valid_query, slot_q, -1)
        tl.store(out_query_slot_mapping_ptr + query_out_idx, slot_q)

        if q_idx == 0:
            input_id = tl.load(next_token_ids_ptr + req_idx)
        else:
            input_id = parallel_drafting_token_id

        # Keep the original input ID for max-length overflow. Only requests
        # without any valid context are converted to dummy input.
        input_id = tl.where(has_valid_context, input_id, 0)
        tl.store(out_input_ids_ptr + query_out_idx, input_id)

        if SAMPLE_FROM_ANCHOR:
            sample_out_idx = req_idx * num_speculative_tokens + q_idx
            tl.store(out_token_indices_ptr + sample_out_idx, query_out_idx)
        else:
            if q_idx > 0:
                sample_out_idx = req_idx * num_speculative_tokens + (q_idx - 1)
                tl.store(out_token_indices_ptr + sample_out_idx, query_out_idx)


def apply_patch() -> None:
    global _PATCH_APPLIED
    if _PATCH_APPLIED:
        return

    from vllm_ascend.ops.triton.spec_decode import utils as spec_decode_utils

    spec_decode_utils.copy_and_expand_dflash_and_dspark_inputs_kernel_single_grid = (
        copy_and_expand_dflash_and_dspark_inputs_kernel_single_grid
    )

    _PATCH_APPLIED = True


apply_patch()
