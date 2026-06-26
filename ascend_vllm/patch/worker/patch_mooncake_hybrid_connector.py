from __future__ import annotations

import functools
import threading
from numbers import Integral
from typing import Any

from vllm.logger import init_logger

logger = init_logger("vllm.ascend_vllm.patch.worker.patch_mooncake_hybrid_connector")

_PATCH_APPLIED = False


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

        def wrap_transfer(method_name: str) -> None:
            """Wrap a transfer method so failures are reported before re-raising."""
            origin_method = getattr(recv_cls, method_name, None)
            if origin_method is None:
                return
            if getattr(origin_method, "_modelarts_wrapped", False):
                return

            @functools.wraps(origin_method)
            def wrapped(self, req_meta, *args, **kwargs):
                try:
                    return origin_method(self, req_meta, *args, **kwargs)
                except Exception:
                    try:
                        self._mark_failed_recv_request(req_meta.get("local_block_ids", ()))
                    except Exception:
                        logger.exception("Failed to mark invalid KV blocks.")
                    raise

            wrapped._modelarts_wrapped = True
            setattr(recv_cls, method_name, wrapped)

        recv_cls.__init__ = patched_init
        recv_cls._mark_failed_recv_request = mark_failed_recv_request
        recv_cls.get_and_clear_invalid_block_ids = get_and_clear_invalid_block_ids

        wrap_transfer("_transfer_kv_cache")
        wrap_transfer("_transfer_kv_cache_all_groups")

        recv_cls._modelarts_mooncake_hybrid_connector_patch_applied = True

    def connector_get_block_ids_with_load_errors(self) -> set[int]:
        """Forward load errors from the connector facade to the worker."""
        assert self.connector_worker is not None
        return self.connector_worker.get_block_ids_with_load_errors()

    def worker_get_block_ids_with_load_errors(self) -> set[int]:
        """Return invalid local block ids from the decode-side recv thread."""
        if self.kv_role == "kv_consumer" and self.kv_recv_thread is not None:
            return self.kv_recv_thread.get_and_clear_invalid_block_ids()
        return set()

    mhc.MooncakeConnector.get_block_ids_with_load_errors = connector_get_block_ids_with_load_errors
    mhc.MooncakeConnectorWorker.get_block_ids_with_load_errors = worker_get_block_ids_with_load_errors


def apply_patch() -> None:
    global _PATCH_APPLIED

    if _PATCH_APPLIED:
        return

    _patch_mooncake_hybrid_connector()
    _PATCH_APPLIED = True


apply_patch()
