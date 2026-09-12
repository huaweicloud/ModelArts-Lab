from __future__ import annotations

import functools
import threading
import time
from numbers import Integral
from typing import Any
import math

from vllm.v1.request import RequestStatus
from vllm.logger import init_logger
from vllm_ascend import envs as ascend_envs

logger = init_logger("vllm.ascend_vllm.patch.worker.patch_mooncake_hybrid_connector")

_PATCH_APPLIED = False

# Peer-level circuit breaker for KV pulls on the decode side.
#
# Lifecycle per request (recv-thread transfer):
#   1. pull start  -> cb_is_circuit_open(host)?
#   2a. circuit open  -> SKIP the pull, mark destination blocks invalid, raise
#                        (no ~33s RDMA timeout is burned; the request is not
#                        counted as a new failure).
#   2b. transfer ok   -> cb_record_success(host) resets the breaker.
#   2c. transfer fail -> cb_record_failure(host) increments the count; at
#                        threshold the host's circuit opens for
#                        _CIRCUIT_BREAKER_WINDOW_SECONDS.
#   3. Invalid blocks are drained by post_forward (get_block_ids_with_load_errors)
#      into KVConnectorOutput.invalid_block_ids. The scheduler's invalid-block
#      handling then applies kv_load_failure_policy: on the D node this defaults
#      to "fail", so the request is finished with FINISHED_ERROR (stop_reason
#      "KV cache load failed ..."). With policy "recompute" it recomputes locally.
#
# After _CIRCUIT_BREAKER_FAILURE_THRESHOLD consecutive transfer failures to the
# same P host (remote_host), the host's circuit opens for
# _CIRCUIT_BREAKER_WINDOW_SECONDS: every subsequent pull is skipped and its
# destination blocks are marked invalid.
_CIRCUIT_BREAKER_FAILURE_THRESHOLD = ascend_envs.MODELARTS_KV_CIRCUIT_BREAKER_THRESHOLD
_CIRCUIT_BREAKER_WINDOW_SECONDS = ascend_envs.MODELARTS_KV_CIRCUIT_BREAKER_WINDOW_SECONDS


def _iter_block_ids(block_ids: Any):
    """Flatten BlockIds into individual block ids."""
    if not block_ids:
        return

    for group in block_ids:
        if group is None:
            continue

        if isinstance(group, Integral):
            yield int(group)
            continue

        for block_id in group:
            if block_id is not None:
                yield int(block_id)


def _count_block_ids(block_ids: Any) -> int:
    """Count individual block ids in a (possibly nested) BlockIds structure."""
    return sum(1 for _ in _iter_block_ids(block_ids))


def _patch_mooncake_hybrid_connector() -> None:
    """Patch MooncakeHybridConnector to report KV load failures."""
    from vllm_ascend.distributed.kv_transfer.kv_p2p import mooncake_hybrid_connector as mhc

    recv_cls = mhc.KVCacheRecvingThread

    if not getattr(recv_cls, "_modelarts_mooncake_hybrid_connector_patch_applied", False):
        origin_init = recv_cls.__init__

        @functools.wraps(origin_init)
        def patched_init(self, *args, **kwargs):
            origin_init(self, *args, **kwargs)

            # Store local KV block ids whose remote load failed.
            self.invalid_block_ids = set()

            # The recv thread writes this set while the model runner reads it.
            self.failed_recv_requests_lock = threading.Lock()

            # Peer-level circuit breaker state, keyed by remote_host (P machine).
            self.cb_failure_counts: dict[str, int] = {}
            self.cb_open_until: dict[str, float] = {}
            self.cb_lock = threading.Lock()

        def ensure_failure_state(self) -> None:
            """Initialize failure-tracking fields for existing instances."""
            if not hasattr(self, "invalid_block_ids"):
                self.invalid_block_ids = set()
            if not hasattr(self, "failed_recv_requests_lock"):
                self.failed_recv_requests_lock = threading.Lock()

        def mark_failed_recv_request(self, local_block_ids) -> None:
            """Mark local destination blocks as invalid after KV load failure."""
            ensure_failure_state(self)
            with self.failed_recv_requests_lock:
                self.invalid_block_ids.update(_iter_block_ids(local_block_ids))

        def get_and_clear_invalid_block_ids(self) -> set[int]:
            """Return failed block ids once, then clear the internal set."""
            ensure_failure_state(self)
            with self.failed_recv_requests_lock:
                invalid_block_ids = set(self.invalid_block_ids)
                self.invalid_block_ids.clear()
            return invalid_block_ids

        def ensure_circuit_breaker_state(self) -> None:
            """Initialize circuit-breaker fields for existing instances."""
            if not hasattr(self, "cb_failure_counts"):
                self.cb_failure_counts = {}
            if not hasattr(self, "cb_open_until"):
                self.cb_open_until = {}
            if not hasattr(self, "cb_lock"):
                self.cb_lock = threading.Lock()

        def cb_peer_host(self, req_meta) -> str:
            """Return the P machine (remote_host) a transfer targets."""
            return str(req_meta.get("remote_host", ""))

        def cb_is_circuit_open(self, host: str) -> bool:
            """True if the host's circuit is open (pull should be skipped).

            If the window has already expired, the entry is dropped so the next
            attempt is a real probe (half-open state).
            """
            ensure_circuit_breaker_state(self)
            if not host:
                return False
            with self.cb_lock:
                open_until = self.cb_open_until.get(host)
                if open_until is None:
                    return False
                if time.time() < open_until:
                    return True
                self.cb_open_until.pop(host, None)
                return False

        def cb_record_failure(self, host: str) -> int:
            """Record a real transfer failure and open the circuit at threshold.

            The counter is deliberately NOT reset when the circuit opens: after
            the window expires, a single failed probe re-opens it immediately,
            avoiding repeatedly burning N x ~33s timeouts for a still-down host.
            Only a successful transfer resets the counter. Returns the updated
            consecutive failure count (0 if no host) for caller logging.
            """
            ensure_circuit_breaker_state(self)
            if not host:
                return 0
            with self.cb_lock:
                count = self.cb_failure_counts.get(host, 0) + 1
                self.cb_failure_counts[host] = count
                if count >= _CIRCUIT_BREAKER_FAILURE_THRESHOLD:
                    self.cb_open_until[host] = time.time() + _CIRCUIT_BREAKER_WINDOW_SECONDS
                    logger.warning(
                        "[KV-CB] opening circuit breaker for P host %s after %d "
                        "consecutive KV pull failures (window=%.0fs).",
                        host,
                        count,
                        _CIRCUIT_BREAKER_WINDOW_SECONDS,
                    )
                return count

        def cb_record_success(self, host: str) -> None:
            """Record a successful transfer and reset the host's breaker."""
            ensure_circuit_breaker_state(self)
            if not host:
                return
            with self.cb_lock:
                self.cb_failure_counts.pop(host, None)
                self.cb_open_until.pop(host, None)

        def wrap_transfer(method_name: str) -> None:
            """Wrap a transfer method so failures are reported before re-raising."""
            origin_method = getattr(recv_cls, method_name, None)
            if origin_method is None:
                return
            if getattr(origin_method, "_modelarts_wrapped", False):
                return

            @functools.wraps(origin_method)
            def wrapped(self, req_meta, *args, **kwargs):
                request_id = req_meta.get("request_id", "?")
                host = self.cb_peer_host(req_meta)
                num_blocks = _count_block_ids(req_meta.get("local_block_ids", ()))
                logger.debug(
                    "[KV-CB] pull start req=%s host=%s num_blocks=%d",
                    request_id,
                    host,
                    num_blocks,
                )
                # Circuit open: skip the pull entirely and mark the request's
                # destination blocks invalid. The scheduler's invalid-block
                # handling honors kv_load_failure_policy: with the D-node default
                # "fail" the request is finished with FINISHED_ERROR (stop_reason
                # "KV cache load failed ..."), so no ~33s RDMA timeout is burned.
                # Raised before the try so skipped pulls neither count as new
                # failures nor consume an RDMA timeout.
                if self.cb_is_circuit_open(host):
                    logger.warning(
                        "[KV-CB] pull SKIPPED req=%s host=%s circuit breaker open "
                        "(window=%.0fs); marking %d blocks invalid -> request "
                        "will be failed by the scheduler.",
                        request_id,
                        host,
                        _CIRCUIT_BREAKER_WINDOW_SECONDS,
                        num_blocks,
                    )
                    try:
                        self._mark_failed_recv_request(req_meta.get("local_block_ids", ()))
                    except Exception:
                        logger.exception(
                            "[KV-CB] failed to mark invalid KV blocks req=%s.",
                            request_id,
                        )
                    raise RuntimeError(
                        f"KV pull skipped: circuit breaker open for host {host}"
                    )
                try:
                    result = origin_method(self, req_meta, *args, **kwargs)
                    self.cb_record_success(host)
                    logger.debug(
                        "[KV-CB] pull done req=%s host=%s ok", request_id, host
                    )
                    return result
                except Exception:
                    failure_count = self.cb_record_failure(host)
                    logger.warning(
                        "[KV-CB] pull FAILED req=%s host=%s failure_count=%d; "
                        "marking %d blocks invalid -> request will be failed by "
                        "the scheduler.",
                        request_id,
                        host,
                        failure_count,
                        num_blocks,
                    )
                    try:
                        self._mark_failed_recv_request(req_meta.get("local_block_ids", ()))
                    except Exception:
                        logger.exception(
                            "[KV-CB] failed to mark invalid KV blocks req=%s.",
                            request_id,
                        )
                    raise

            wrapped._modelarts_wrapped = True
            setattr(recv_cls, method_name, wrapped)

        recv_cls.__init__ = patched_init
        recv_cls._mark_failed_recv_request = mark_failed_recv_request
        recv_cls.get_and_clear_invalid_block_ids = get_and_clear_invalid_block_ids
        recv_cls.ensure_circuit_breaker_state = ensure_circuit_breaker_state
        recv_cls.cb_peer_host = cb_peer_host
        recv_cls.cb_is_circuit_open = cb_is_circuit_open
        recv_cls.cb_record_failure = cb_record_failure
        recv_cls.cb_record_success = cb_record_success

        wrap_transfer("_transfer_kv_cache")
        wrap_transfer("_transfer_kv_cache_all_groups")

        recv_cls._modelarts_mooncake_hybrid_connector_patch_applied = True

    # Patch MooncakeConnectorScheduler.request_finished_all_groups to also
    # handle FINISHED_STOPPED status (in addition to FINISHED_LENGTH_CAPPED).

    def patched_request_finished_all_groups(self, request, block_ids):
        params = request.kv_transfer_params
        if (
            params is None
            or not params.get("do_remote_decode")
            # adapt begin : fix  kv_transfer_params is null for the first token is the end flag error.
            or request.status not in (
                RequestStatus.FINISHED_LENGTH_CAPPED,
                RequestStatus.FINISHED_STOPPED,
            )
            # adapt end
        ):
            return False, None

        computed_block_ids = self._compute_transfer_block_ids(block_ids, request.num_prompt_tokens)
        computed_block_ids = self.get_sw_clipped_blocks(computed_block_ids)
        computed_block_lens = [len(block_id_list) for block_id_list in computed_block_ids]
        delay_free_blocks = sum(computed_block_lens) > 0
        if delay_free_blocks:
            logger.info("Delaying free of %d blocks for request %s", sum(computed_block_lens), request.request_id)
            self._reqs_need_send[request.request_id] = time.time()

        num_prompt_blocks = math.ceil(request.num_prompt_tokens / self.block_size)

        return delay_free_blocks, dict(
            do_remote_prefill=True,
            do_remote_decode=False,
            remote_block_ids=computed_block_ids,
            remote_engine_id=self.engine_id,
            remote_request_id=request.request_id,
            remote_host=self.side_channel_host,
            remote_port=self.side_channel_port,
            remote_ptp_size=self.tp_size,
            last_token_id=request.output_token_ids[-1],
            remote_multi_nodes_meta_mapping=self.multi_nodes_meta_mapping,
            num_prompt_blocks=num_prompt_blocks,
        )

    def connector_get_block_ids_with_load_errors(self) -> set[int]:
        """Forward load errors from the connector facade to the worker."""
        if self.connector_worker is None:
            return set()
        return self.connector_worker.get_block_ids_with_load_errors()

    def worker_get_block_ids_with_load_errors(self) -> set[int]:
        """Return invalid local block ids from the decode-side recv thread."""
        if self.kv_role == "kv_consumer" and self.kv_recv_thread is not None:
            invalid_block_ids = self.kv_recv_thread.get_and_clear_invalid_block_ids()
            if invalid_block_ids:
                logger.info(
                    "[KV-CB] handing %d invalid block(s) to post_forward for "
                    "the scheduler: %s",
                    len(invalid_block_ids),
                    sorted(invalid_block_ids),
                )
            return invalid_block_ids
        return set()

    mhc.MooncakeConnector.get_block_ids_with_load_errors = connector_get_block_ids_with_load_errors
    mhc.MooncakeConnectorWorker.get_block_ids_with_load_errors = worker_get_block_ids_with_load_errors
    mhc.MooncakeConnectorScheduler.request_finished_all_groups = patched_request_finished_all_groups


def apply_patch() -> None:
    global _PATCH_APPLIED

    if _PATCH_APPLIED:
        return

    _patch_mooncake_hybrid_connector()
    _PATCH_APPLIED = True


apply_patch()
