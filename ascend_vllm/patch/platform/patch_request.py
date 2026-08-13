# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from functools import wraps
from typing import Any

from vllm.v1.metrics.stats import PrefillStats
from vllm.v1.request import Request
from vllm_ascend.patch.platform import patch_async_swa_kv_lifetime

_original_request_init = patch_async_swa_kv_lifetime._patched_request_init


@wraps(_original_request_init)
def _patched_reqeust_init(self: Request, *args: Any, **kwargs: Any) -> None:
    _original_request_init(self, *args, **kwargs)
    # P/D: Decode will get num_cached_tokens from kv_transfer_params, passed by prefill
    # adapt begin
    if self.kv_transfer_params is not None and "num_cached_tokens" in self.kv_transfer_params:
        self.num_cached_tokens = self.kv_transfer_params["num_cached_tokens"]
    else:
        self.num_cached_tokens = None
    # adapt end


def take_prefill_stats(self) -> PrefillStats | None:
    if self.prefill_stats is None:
        return None
    # adapt begin
    if self.num_cached_tokens is not None:
        self.prefill_stats.num_cached_tokens = self.num_cached_tokens
    # adapt end
    prefill_stats = self.prefill_stats
    self.prefill_stats = None
    return prefill_stats


Request.__init__ = _patched_reqeust_init
Request.take_prefill_stats = take_prefill_stats
