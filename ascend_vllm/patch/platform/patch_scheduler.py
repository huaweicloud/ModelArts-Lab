# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import time
from collections import defaultdict

from vllm.distributed.kv_events import KVEventBatch
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import KVConnectorStats
from vllm.logger import logger
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.sched.utils import remove_all
from vllm.v1.engine import EngineCoreOutput, EngineCoreOutputs
from vllm.v1.metrics.perf import PerfStats
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.spec_decode.metrics import SpecDecodingStats


def update_from_output(
    self,
    scheduler_output: SchedulerOutput,
    model_runner_output: ModelRunnerOutput,
) -> dict[int, EngineCoreOutputs]:
    sampled_token_ids = model_runner_output.sampled_token_ids
    logprobs = model_runner_output.logprobs
    prompt_logprobs_dict = model_runner_output.prompt_logprobs_dict
    num_scheduled_tokens = scheduler_output.num_scheduled_tokens
    pooler_outputs = model_runner_output.pooler_output
    num_nans_in_logits = model_runner_output.num_nans_in_logits
    kv_connector_output = model_runner_output.kv_connector_output
    cudagraph_stats = model_runner_output.cudagraph_stats

    perf_stats: PerfStats | None = None
    if self.perf_metrics and self.perf_metrics.is_enabled():
        perf_stats = self.perf_metrics.get_step_perf_stats_per_gpu(scheduler_output)

    outputs: dict[int, list[EngineCoreOutput]] = defaultdict(list)
    spec_decoding_stats: SpecDecodingStats | None = None
    kv_connector_stats: KVConnectorStats | None = (
        kv_connector_output.kv_connector_stats if kv_connector_output else None
    )
    if kv_connector_stats and self.connector:
        kv_stats = self.connector.get_kv_connector_stats()
        if kv_stats:
            kv_connector_stats = kv_connector_stats.aggregate(kv_stats)

    failed_kv_load_req_ids = None
    if kv_connector_output and kv_connector_output.invalid_block_ids:
        # These blocks contain externally computed tokens that failed to
        # load. Identify affected requests and adjust their computed token
        # count to trigger recomputation of the invalid blocks.
        failed_kv_load_req_ids = self._handle_invalid_blocks(
            kv_connector_output.invalid_block_ids,
            num_scheduled_tokens,
        )

    # Persist per-step routed experts into the scheduler-side slot
    # buffer (CPU->CPU fancy-index assign; ~few MB per step).
    # MUST precede the per-request routing reads below: stopped
    # requests may terminate on tokens generated in this very step,
    # whose routing was just D2H'd into model_runner_output.
    routing_data = None
    routing_offsets: dict[str, int] = {}
    if model_runner_output.routed_experts is not None:
        re = model_runner_output.routed_experts
        self.routed_experts_mgr.store_batch(re.routing_data, re.slot_mapping)
        routing_data = re.routing_data.astype(
            self.routed_experts_mgr.routed_experts_by_slot.dtype,
            copy=False,
        )
        # Build offset map using model runner's request order
        # (input_batch ordering), NOT scheduler dict order.
        offset = 0
        for rid in model_runner_output.req_ids:
            routing_offsets[rid] = offset
            offset += num_scheduled_tokens[rid]

    # NOTE(woosuk): As len(num_scheduled_tokens) can be up to 1K or more,
    # the below loop can be a performance bottleneck. We should do our best
    # to avoid expensive operations inside the loop.
    stopped_running_reqs: set[Request] = set()
    stopped_preempted_reqs: set[Request] = set()
    for req_id, num_tokens_scheduled in num_scheduled_tokens.items():
        assert num_tokens_scheduled > 0
        if failed_kv_load_req_ids and req_id in failed_kv_load_req_ids:
            # skip failed or rescheduled requests from KV load failure
            continue
        request = self.requests.get(req_id)
        if request is None or request.is_finished():
            # The request is already finished. This can happen if the
            # request is aborted while the model is executing it (e.g.,
            # in pipeline parallelism or in async scheduling).
            # NOTE(Kuntai): When delay_free_blocks=True (for async KV
            # cache transfer in KV connector), the aborted request will not
            # be set to None (in order to finish async KV transfer).
            # In this case, we use is_finished() to check.
            continue

        req_index = model_runner_output.req_id_to_index[req_id]
        generated_token_ids = sampled_token_ids[req_index] if sampled_token_ids else []

        scheduled_spec_token_ids = scheduler_output.scheduled_spec_decode_tokens.get(req_id)
        if scheduled_spec_token_ids and generated_token_ids:
            num_draft_tokens = len(scheduled_spec_token_ids)
            num_accepted = len(generated_token_ids) - 1
            num_rejected = num_draft_tokens - num_accepted
            # num_computed_tokens represents the number of tokens
            # processed in the current step, considering scheduled
            # tokens and rejections. If some tokens are rejected,
            # num_computed_tokens is decreased by the number of rejected
            # tokens.
            if request.num_computed_tokens > 0:
                request.num_computed_tokens -= num_rejected
            # If async scheduling, num_output_placeholders also includes
            # the scheduled spec tokens count and so is similarly adjusted.
            if request.num_output_placeholders > 0:
                request.num_output_placeholders -= num_rejected
            spec_decoding_stats = self.make_spec_decoding_stats(
                spec_decoding_stats,
                num_draft_tokens=num_draft_tokens,
                num_accepted_tokens=num_accepted,
                num_invalid_spec_tokens=scheduler_output.num_invalid_spec_tokens,
                request_id=req_id,
            )

        # Free encoder inputs only after the step has actually executed.
        if request.has_encoder_inputs:
            self._free_encoder_inputs(request)

        stopped = False
        new_logprobs = None
        new_token_ids = generated_token_ids
        pooler_output = pooler_outputs[req_index] if pooler_outputs else None
        kv_transfer_params = None
        status_before_stop = request.status
        num_output_tokens_before = len(request._output_token_ids)

        # Check for stop and update request status.
        if new_token_ids:
            new_token_ids, stopped = self._update_request_with_output(request, new_token_ids)
        elif request.pooling_params and pooler_output is not None:
            # Pooling stops as soon as there is output.
            request.status = RequestStatus.FINISHED_STOPPED
            stopped = True

        if new_token_ids and self.structured_output_manager.should_advance(request):
            struct_output_request = request.structured_output_request
            assert struct_output_request is not None
            assert struct_output_request.grammar is not None
            if not struct_output_request.grammar.accept_tokens(  # type: ignore[union-attr]
                req_id, new_token_ids
            ):
                logger.error(
                    "Unexpected: grammar rejected tokens %s for request %s. Terminating request.",
                    new_token_ids,
                    req_id,
                )
                request.status = RequestStatus.FINISHED_ERROR
                request.resumable = False
                stopped = True

        routed_experts = None
        if self.enable_return_routed_experts and routing_data is not None and new_token_ids:
            req_offset = routing_offsets[req_id]
            end = req_offset + num_tokens_scheduled
            block_ids = self._re_block_ids.pop(req_id, [])
            if num_output_tokens_before == 0:
                # Prefill completed: read full prompt routing from
                # slot buffer using the block-ID snapshot taken at
                # schedule time (immune to async preemption).
                if (
                    request.sampling_params is not None
                    and request.sampling_params.routed_experts_prompt_start is not None
                ):
                    prompt_start = request.sampling_params.routed_experts_prompt_start
                    assert prompt_start < request.num_prompt_tokens
                else:
                    prompt_start = 0
                routed_experts = self.routed_experts_mgr.get(
                    block_ids,
                    request.num_prompt_tokens,
                    token_start=prompt_start,
                )
            else:
                if scheduled_spec_token_ids:
                    # Spec decode: accepted tokens at the START of
                    # the scheduled range, rejected at the end.
                    routed_experts = routing_data[req_offset : req_offset + len(new_token_ids)]
                else:
                    # Normal decode / re-prefill: token(s) at the END.
                    routed_experts = routing_data[end - len(new_token_ids) : end]

        finish_reason = None
        if stopped:
            # Capture finish_reason BEFORE _handle_stopped_request, which may
            # reset the status to WAITING for streaming requests that continue.
            finish_reason = request.get_finished_reason()
            finished = self._handle_stopped_request(request)
            if finished:
                kv_transfer_params = self._free_request(request)

            if status_before_stop == RequestStatus.RUNNING:
                stopped_running_reqs.add(request)
            else:
                stopped_preempted_reqs.add(request)

        # Extract sample logprobs if needed.
        if request.sampling_params is not None and request.sampling_params.num_logprobs is not None and logprobs:
            new_logprobs = logprobs.slice_request(req_index, len(new_token_ids))

        if num_nans_in_logits is not None and req_id in num_nans_in_logits:
            request.num_nans_in_logits = num_nans_in_logits[req_id]

        # Get prompt logprobs for this request.
        prompt_logprobs_tensors = prompt_logprobs_dict.get(req_id)
        if new_token_ids or pooler_output is not None or kv_transfer_params or stopped:
            # adapt begin: pass num cached tokens to deocde by kv_transfer_params
            if kv_transfer_params is not None and request.prefill_stats is not None:
                kv_transfer_params["num_cached_tokens"] = request.prefill_stats.num_cached_tokens
            # adapt end
            # Add EngineCoreOutput for this Request.
            outputs[request.client_index].append(
                EngineCoreOutput(
                    request_id=req_id,
                    new_token_ids=new_token_ids,
                    finish_reason=finish_reason,
                    new_logprobs=new_logprobs,
                    new_prompt_logprobs_tensors=prompt_logprobs_tensors,
                    pooling_output=pooler_output,
                    stop_reason=request.stop_reason,
                    events=request.take_events(),
                    prefill_stats=request.take_prefill_stats(),
                    kv_transfer_params=kv_transfer_params,
                    trace_headers=request.trace_headers,
                    routed_experts=routed_experts,
                    num_nans_in_logits=request.num_nans_in_logits,
                )
            )
        else:
            # Invariant: EngineCore returns no partial prefill outputs.
            assert not prompt_logprobs_tensors

    # Remove the stopped requests from the running and waiting queues.
    if stopped_running_reqs:
        self.running = remove_all(self.running, stopped_running_reqs)
    if stopped_preempted_reqs:
        # This is a rare case and unlikely to impact performance.
        self.waiting.remove_requests(stopped_preempted_reqs)

    if failed_kv_load_req_ids and not self.recompute_kv_load_failures:
        requests = [self.requests[req_id] for req_id in failed_kv_load_req_ids]
        self.finish_requests(failed_kv_load_req_ids, RequestStatus.FINISHED_ERROR)
        for request in requests:
            outputs[request.client_index].append(
                EngineCoreOutput(
                    request_id=request.request_id,
                    new_token_ids=[],
                    finish_reason=request.get_finished_reason(),
                    events=request.take_events(),
                    trace_headers=request.trace_headers,
                )
            )

    # KV Connector: update state for finished KV Transfers.
    if kv_connector_output:
        self._update_from_kv_xfer_finished(kv_connector_output)

    # collect KV cache events from KV cache manager
    events = self.kv_cache_manager.take_events()

    # collect KV cache events from connector
    if self.connector is not None:
        connector_events = self.connector.take_events()
        if connector_events:
            if events is None:
                events = list(connector_events)
            else:
                events.extend(connector_events)

    # publish collected KV cache events
    if events:
        batch = KVEventBatch(ts=time.time(), events=events)
        self.kv_event_publisher.publish(batch)

    # Create EngineCoreOutputs for all clients that have requests with
    # outputs in this step.
    engine_core_outputs = {client_index: EngineCoreOutputs(outputs=outs) for client_index, outs in outputs.items()}

    finished_req_ids = self.finished_req_ids_dict
    if finished_req_ids:
        # Include ids of requests that finished since last outputs
        # were sent.
        for client_index, finished_set in finished_req_ids.items():
            # Set finished request set in EngineCoreOutputs for this client.
            if (eco := engine_core_outputs.get(client_index)) is not None:
                eco.finished_requests = finished_set
            else:
                engine_core_outputs[client_index] = EngineCoreOutputs(finished_requests=finished_set)
        finished_req_ids.clear()

    if (stats := self.make_stats(spec_decoding_stats, kv_connector_stats, cudagraph_stats, perf_stats)) is not None:
        # Return stats to only one of the front-ends.
        if (eco := next(iter(engine_core_outputs.values()), None)) is None:
            # We must return the stats even if there are no request
            # outputs this step.
            engine_core_outputs[0] = eco = EngineCoreOutputs()
        eco.scheduler_stats = stats

    return engine_core_outputs


Scheduler.update_from_output = update_from_output
AsyncScheduler.update_from_output = update_from_output
