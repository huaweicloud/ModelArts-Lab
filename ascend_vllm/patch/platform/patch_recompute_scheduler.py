from __future__ import annotations

import functools
from typing import Any

_PATCH_APPLIED = False
_KV_FAILURE_PATCH_MARKER = "_modelarts_kv_load_failure_recompute_scheduler_patch_applied"
_FAILED_REQUEST_IDS_ATTR = "_modelarts_kv_load_failed_request_ids"
_MISSING = object()


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
        """Reset requests affected by failed KV loads for full recomputation."""
        affected_req_ids: set[str] = set()
        total_affected_tokens = 0
        blocks_to_evict: set[int] = set()

        for request in requests:
            req_id = request.request_id
            req_block_id_groups = self.kv_cache_manager.get_block_ids(req_id)

            # Block IDs are allocated from one global BlockPool, so flattening
            # groups is sufficient for identifying the owning request.
            req_block_ids = {block_id for group in req_block_id_groups for block_id in group if block_id is not None}

            affected_block_ids = req_block_ids & invalid_block_ids
            if not affected_block_ids:
                continue

            # Exclude tokens scheduled in the current step because their
            # outputs have not yet become part of the stable computed prefix.
            req_num_computed_tokens = request.num_computed_tokens - num_scheduled_tokens.get(req_id, 0)
            affected_req_ids.add(req_id)
            total_affected_tokens += req_num_computed_tokens
            request.num_computed_tokens = 0

            # If the caller requests eviction, all request blocks are
            # downstream of the new computed-token boundary (zero).
            if evict_blocks:
                blocks_to_evict.update(req_block_ids)

        return (
            affected_req_ids,
            total_affected_tokens,
            blocks_to_evict,
        )

    Scheduler._update_requests_with_invalid_blocks = update_requests_with_invalid_blocks
    rs.RecomputeScheduler._update_requests_with_invalid_blocks = update_requests_with_invalid_blocks
    rs.AsyncRecomputeScheduler._update_requests_with_invalid_blocks = update_requests_with_invalid_blocks


def _patch_kv_load_failure_outputs() -> None:
    from vllm_ascend.core.recompute_scheduler import RecomputeScheduler

    current_update_from_output = RecomputeScheduler.update_from_output
    if getattr(current_update_from_output, _KV_FAILURE_PATCH_MARKER, False):
        return

    @functools.wraps(current_update_from_output)
    def patched_update_from_output(
        self: RecomputeScheduler,
        *args: Any,
        **kwargs: Any,
    ) -> dict[int, Any]:
        previous_capture = getattr(self, _FAILED_REQUEST_IDS_ATTR, _MISSING)
        current_capture: set[str] = set()
        setattr(self, _FAILED_REQUEST_IDS_ATTR, current_capture)

        try:
            engine_core_outputs = current_update_from_output(self, *args, **kwargs)
            for client_outputs in engine_core_outputs.values():
                for output in client_outputs.outputs:
                    if output.request_id not in current_capture:
                        continue
                    if output.kv_transfer_params is None:
                        output.kv_transfer_params = {"kv_load_failed": True}
                    else:
                        output.kv_transfer_params["kv_load_failed"] = True
            return engine_core_outputs
        finally:
            if previous_capture is _MISSING:
                delattr(self, _FAILED_REQUEST_IDS_ATTR)
            else:
                setattr(self, _FAILED_REQUEST_IDS_ATTR, previous_capture)

    setattr(patched_update_from_output, _KV_FAILURE_PATCH_MARKER, True)
    RecomputeScheduler.update_from_output = patched_update_from_output


def apply_patch() -> None:
    global _PATCH_APPLIED

    if _PATCH_APPLIED:
        return

    _patch_recompute_scheduler()
    _patch_kv_load_failure_outputs()
    _PATCH_APPLIED = True


apply_patch()
