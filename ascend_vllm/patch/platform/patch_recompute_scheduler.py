from __future__ import annotations

import functools

import numpy as np

_PATCH_APPLIED = False


def _patch_recompute_scheduler() -> None:
    """Patch RecomputeScheduler for HMA invalid-block handling."""
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm_ascend.core import recompute_scheduler as rs

    def update_requests_with_invalid_blocks(
        self,
        requests,
        invalid_block_ids: set[int],
        num_scheduled_tokens: dict[str, int],
        evict_blocks: bool = True,
    ) -> tuple[set[str], int, set[int]]:
        affected_req_ids: set[str] = set()
        total_affected_tokens = 0
        blocks_to_evict: set[int] = set()
        marked_invalid_block_ids: set[int] = set()

        for request in requests:
            is_affected = False
            marked_invalid_block = False
            req_id = request.request_id

            req_block_id_groups = self.kv_cache_manager.get_block_ids(req_id)
            req_num_computed_tokens = request.num_computed_tokens - num_scheduled_tokens.get(req_id, 0)
            req_num_computed_blocks = (req_num_computed_tokens + self.block_size - 1) // self.block_size

            max_blocks = min(req_num_computed_blocks, max((len(group) for group in req_block_id_groups), default=0))

            for idx in range(max_blocks):
                block_ids_at_idx = [group[idx] for group in req_block_id_groups if idx < len(group)]

                invalid_block_ids_at_idx = [block_id for block_id in block_ids_at_idx if block_id in invalid_block_ids]
                if not invalid_block_ids_at_idx:
                    continue

                is_affected = True

                if all(block_id in marked_invalid_block_ids for block_id in invalid_block_ids_at_idx):
                    continue

                marked_invalid_block_ids.update(invalid_block_ids_at_idx)

                if marked_invalid_block:
                    continue

                marked_invalid_block = True
                request.num_computed_tokens = idx * self.block_size
                total_affected_tokens += req_num_computed_tokens - request.num_computed_tokens

                if evict_blocks:
                    for group in req_block_id_groups:
                        blocks_to_evict.update(group[idx:])

            if is_affected:
                if not marked_invalid_block:
                    total_affected_tokens += request.num_computed_tokens - req_num_computed_tokens
                    request.num_computed_tokens = req_num_computed_tokens

                affected_req_ids.add(request.request_id)

        return affected_req_ids, total_affected_tokens, blocks_to_evict

    def get_routed_experts(self, request):
        """Provide the upstream Scheduler routed-experts helper."""
        if not self.vllm_config.model_config.enable_return_routed_experts:
            return None

        kv_blocks = self.kv_cache_manager.get_blocks(request.request_id)
        block_ids = kv_blocks.get_block_ids()[self.routed_experts_attn_gid]
        num_tokens = request.num_tokens - 1

        block_ids_array = np.array(block_ids, dtype=np.int32)
        num_blocks = len(block_ids)

        attn_group = self.kv_cache_config.kv_cache_groups[self.routed_experts_attn_gid]
        block_size = attn_group.kv_cache_spec.block_size

        block_offsets = np.arange(0, block_size)
        slot_mapping = (
            block_offsets.reshape((1, block_size)) + block_ids_array.reshape((num_blocks, 1)) * block_size
        ).flatten()[:num_tokens]

        return self.routed_experts_reader.get_routed_experts(indices=slot_mapping)

    def patch_invalid_blocks_signature(cls) -> None:
        """Adapt old invalid-block calls to vLLM 0.21.0 scheduler output."""
        token_attr = "_modelarts_current_num_scheduled_tokens"
        missing = object()

        origin_update_from_output = cls.update_from_output
        if not getattr(origin_update_from_output, "_modelarts_kv_failure_update_wrapped", False):

            @functools.wraps(origin_update_from_output)
            def patched_update_from_output(self, scheduler_output, model_runner_output):
                previous_tokens = getattr(self, token_attr, missing)
                setattr(self, token_attr, scheduler_output.num_scheduled_tokens)

                try:
                    return origin_update_from_output(
                        self,
                        scheduler_output,
                        model_runner_output,
                    )
                finally:
                    if previous_tokens is missing:
                        if hasattr(self, token_attr):
                            delattr(self, token_attr)
                    else:
                        setattr(self, token_attr, previous_tokens)

            patched_update_from_output._modelarts_kv_failure_update_wrapped = True
            cls.update_from_output = patched_update_from_output

        origin_handle_invalid_blocks = cls._handle_invalid_blocks
        if not getattr(origin_handle_invalid_blocks, "_modelarts_kv_failure_handle_wrapped", False):

            @functools.wraps(origin_handle_invalid_blocks)
            def patched_handle_invalid_blocks(
                self,
                invalid_block_ids: set[int],
                num_scheduled_tokens: dict[str, int] | None = None,
            ) -> set[str]:
                # vLLM 0.21.0 requires num_scheduled_tokens. Ascend's
                # RecomputeScheduler may still call this method with one arg.
                if num_scheduled_tokens is None:
                    num_scheduled_tokens = getattr(self, token_attr, missing)
                    if num_scheduled_tokens is missing:
                        raise RuntimeError("num_scheduled_tokens is required when handling invalid KV blocks.")

                return origin_handle_invalid_blocks(
                    self,
                    invalid_block_ids,
                    num_scheduled_tokens,
                )

            patched_handle_invalid_blocks._modelarts_kv_failure_handle_wrapped = True
            cls._handle_invalid_blocks = patched_handle_invalid_blocks

    # RecomputeScheduler inherits _handle_invalid_blocks from upstream
    # Scheduler. Patch the base class too, otherwise an old one-argument
    # call can still resolve to Scheduler._handle_invalid_blocks directly.
    patch_invalid_blocks_signature(Scheduler)

    rs.RecomputeScheduler._update_requests_with_invalid_blocks = update_requests_with_invalid_blocks
    rs.RecomputeScheduler._get_routed_experts = get_routed_experts
    patch_invalid_blocks_signature(rs.RecomputeScheduler)

    if hasattr(rs, "AsyncRecomputeScheduler"):
        rs.AsyncRecomputeScheduler._update_requests_with_invalid_blocks = update_requests_with_invalid_blocks
        rs.AsyncRecomputeScheduler._get_routed_experts = get_routed_experts
        patch_invalid_blocks_signature(rs.AsyncRecomputeScheduler)


def apply_patch() -> None:
    global _PATCH_APPLIED

    if _PATCH_APPLIED:
        return

    _patch_recompute_scheduler()
    _PATCH_APPLIED = True


apply_patch()
