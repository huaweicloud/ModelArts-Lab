# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time
from collections import deque
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

import torch
from vllm.multimodal.inputs import MultiModalFeatureSpec
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import SamplingParams
from vllm.utils import length_from_prompt_token_ids_or_embeds
from vllm.v1.engine import EngineCoreEvent
from vllm.v1.metrics.stats import PrefillStats
from vllm.v1.request import Request, RequestStatus, StreamingUpdate
from vllm.v1.structured_output.request import StructuredOutputRequest
from vllm.v1.utils import ConstantList

if TYPE_CHECKING:
    from vllm.lora.request import LoRARequest
    from vllm.v1.core.kv_cache_utils import BlockHash


def __request_init__(
    self,
    request_id: str,
    prompt_token_ids: list[int] | None,
    sampling_params: SamplingParams | None,
    pooling_params: PoolingParams | None,
    client_index: int = 0,
    arrival_time: float | None = None,
    prompt_embeds: torch.Tensor | None = None,
    prompt_is_token_ids: list[bool] | None = None,
    mm_features: list[MultiModalFeatureSpec] | None = None,
    lora_request: "LoRARequest | None" = None,
    cache_salt: str | None = None,
    priority: int = 0,
    trace_headers: Mapping[str, str] | None = None,
    block_hasher: Callable[["Request"], list["BlockHash"]] | None = None,
    resumable: bool = False,
    reasoning_ended: bool | None = None,
    reasoning_parser_kwargs: dict[str, Any] | None = None,
    abort_immediately: bool = False,
) -> None:
    self.request_id = request_id
    self.client_index = client_index
    self.priority = priority
    self.sampling_params = sampling_params
    self.pooling_params = pooling_params
    self.lora_request = lora_request
    self.structured_output_request = StructuredOutputRequest.from_sampling_params(sampling_params)
    if self.structured_output_request is not None:
        self.structured_output_request.reasoning_ended = reasoning_ended
        self.structured_output_request.reasoning_parser_kwargs = reasoning_parser_kwargs
    self.arrival_time = arrival_time if arrival_time is not None else time.time()

    self.status = RequestStatus.WAITING
    self.events: list[EngineCoreEvent] = []
    self.stop_reason: int | str | None = None

    # P/D: Connector-specific KV transfer parameters.
    self.kv_transfer_params: dict[str, Any] | None = None

    if pooling_params is not None:
        # Pooling models.
        self.max_tokens = 1
    elif sampling_params is not None:
        # Generative models.
        assert sampling_params.max_tokens is not None
        self.max_tokens = sampling_params.max_tokens
        if self.structured_output_request is not None:
            self.status = RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR

        if sampling_params.extra_args is not None:
            self.kv_transfer_params = sampling_params.extra_args.get("kv_transfer_params")
    else:
        raise ValueError("sampling_params and pooling_params can't both be unset")

    self.prompt_token_ids = prompt_token_ids
    self.prompt_embeds = prompt_embeds
    # Per-position mask used in mixed-mode (chat completion with
    # prompt_embeds). `None` except when both `prompt_token_ids` and
    # `prompt_embeds` are set and their positions are interleaved.
    self.prompt_is_token_ids = prompt_is_token_ids
    # Cache per-block prompt-embed hashes to avoid rehashing the same
    # tensor slices when generating extra keys.
    self._prompt_embeds_per_block_hashes: dict[tuple[int, int], bytes] = {}
    self.num_prompt_tokens = length_from_prompt_token_ids_or_embeds(prompt_token_ids, prompt_embeds)
    self._output_token_ids: list[int] = []
    self._all_token_ids: list[int] = (
        self.prompt_token_ids.copy() if self.prompt_token_ids is not None else [0] * self.num_prompt_tokens
    )

    # Used in async scheduling.
    self.num_output_placeholders = 0
    self.async_tokens_to_discard = 0

    # V2+PP+async: Enforces `pp_size` cadence between same-request decode steps
    # so the worker's broadcast slot ring stays consistent.
    self.next_decode_eligible_step = 0

    self.spec_token_ids: list[int] = []
    self.num_computed_tokens = 0
    self.cache_salt: str | None = cache_salt

    # Multi-modal related
    self.mm_features = mm_features or []

    # Read-only views
    # Prevent directly appending to these lists since
    # they should also be updated simultaneously.
    self.output_token_ids = ConstantList(self._output_token_ids)
    self.all_token_ids = ConstantList(self._all_token_ids)
    # trace_headers
    self.trace_headers = trace_headers

    # True if this request is scheduled as a non-final prefill chunk.
    self.is_prefill_chunk = False

    # The number of NaNs in logits. A value greater than 0
    # indicates that the output is corrupted
    self.num_nans_in_logits = 0

    # The number of times this request has been preempted by the scheduler.
    self.num_preemptions = 0

    self.prefill_stats: PrefillStats | None = PrefillStats()

    # P/D: Decode will get num_cached_tokens from kv_transfer_params, passed by prefill
    # adapt begin
    if self.kv_transfer_params is not None and "num_cached_tokens" in self.kv_transfer_params:
        self.num_cached_tokens = self.kv_transfer_params["num_cached_tokens"]
    else:
        self.num_cached_tokens = None
    # adapt end

    self.block_hashes: list[BlockHash] = []
    # Store the block hasher without binding self to avoid creating a
    # reference cycle (Request -> partial -> Request) that prevents
    # immediate garbage collection via reference counting.
    self._block_hasher: Callable[[Request], list[BlockHash]] | None = block_hasher
    self.update_block_hashes()

    self.skip_reading_prefix_cache = self.get_skip_reading_prefix_cache()

    # Used for streaming
    self.resumable = resumable
    # None entry in the queue means finished.
    self.streaming_queue: deque[StreamingUpdate | None] | None = None

    # If True, request should be aborted immediately after being added to
    # the scheduler so the connector's request_finished hook runs.
    self.abort_immediately = abort_immediately


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


Request.__init__ = __request_init__
Request.take_prefill_stats = take_prefill_stats
