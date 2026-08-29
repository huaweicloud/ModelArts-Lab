# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import functools
from typing import Any

from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import Request

_PATCH_APPLIED = False
_PATCH_MARKER = "_modelarts_cached_tokens_patch_applied"
_KV_FAILURE_PATCH_MARKER = "_modelarts_kv_load_failure_scheduler_patch_applied"
_FAILED_REQUEST_IDS_ATTR = "_modelarts_kv_load_failed_request_ids"
_MISSING = object()


def _patch_scheduler() -> None:
    current_free_request = Scheduler._free_request

    if getattr(current_free_request, _PATCH_MARKER, False):
        return

    @functools.wraps(current_free_request)
    def patched_free_request(
        self: Scheduler,
        request: Request,
        *args: Any,
        **kwargs: Any,
    ) -> dict[str, Any] | None:
        kv_transfer_params = current_free_request(
            self,
            request,
            *args,
            **kwargs,
        )

        # P/D: Pass the prefill node's cached prompt-token count
        # to the decode node through kv_transfer_params.
        if kv_transfer_params is not None and request.prefill_stats is not None:
            kv_transfer_params["num_cached_tokens"] = request.prefill_stats.num_cached_tokens

        return kv_transfer_params

    setattr(patched_free_request, _PATCH_MARKER, True)
    Scheduler._free_request = patched_free_request


def _patch_kv_load_failure_outputs() -> None:
    current_handle_invalid_blocks = Scheduler._handle_invalid_blocks
    if not getattr(current_handle_invalid_blocks, _KV_FAILURE_PATCH_MARKER, False):

        @functools.wraps(current_handle_invalid_blocks)
        def patched_handle_invalid_blocks(
            self: Scheduler,
            *args: Any,
            **kwargs: Any,
        ) -> set[str]:
            failed_request_ids = current_handle_invalid_blocks(self, *args, **kwargs)

            # Under the recompute policy, async requests are rescheduled and
            # must not be surfaced as terminal KV-load failures.
            if failed_request_ids and not self.recompute_kv_load_failures:
                captured_request_ids = getattr(
                    self,
                    _FAILED_REQUEST_IDS_ATTR,
                    None,
                )
                if captured_request_ids is not None:
                    captured_request_ids.update(failed_request_ids)

            return failed_request_ids

        setattr(
            patched_handle_invalid_blocks,
            _KV_FAILURE_PATCH_MARKER,
            True,
        )
        Scheduler._handle_invalid_blocks = patched_handle_invalid_blocks

    current_update_from_output = Scheduler.update_from_output
    if getattr(current_update_from_output, _KV_FAILURE_PATCH_MARKER, False):
        return

    @functools.wraps(current_update_from_output)
    def patched_update_from_output(
        self: Scheduler,
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
    Scheduler.update_from_output = patched_update_from_output


def apply_patch() -> None:
    global _PATCH_APPLIED

    if _PATCH_APPLIED:
        return

    _patch_scheduler()
    _patch_kv_load_failure_outputs()
    _PATCH_APPLIED = True


apply_patch()
