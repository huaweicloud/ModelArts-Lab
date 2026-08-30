# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import functools
from typing import Any

from vllm.entrypoints.generate.base.serving import GenerateBaseServing
from vllm.entrypoints.openai.engine.protocol import GenerationError
from vllm.logger import init_logger

logger = init_logger("vllm.ascend_vllm.patch.platform.patch_generate_base_serving")

_PATCH_APPLIED = False
_PATCH_MARKER = "_modelarts_generate_base_serving_kv_load_failure_patch_applied"


def _patch_raise_if_error() -> None:
    current_raise_if_error = GenerateBaseServing._raise_if_error
    if getattr(current_raise_if_error, _PATCH_MARKER, False):
        return

    @functools.wraps(current_raise_if_error)
    def patched_raise_if_error(
        self: GenerateBaseServing,
        finish_reason: str | None,
        request_id: str,
        stop_reason: str | None = None,
    ) -> None:
        if finish_reason != "error":
            return

        if stop_reason:
            message = stop_reason
        else:
            message = "Internal server error"

        logger.error(
            "Request %s failed with an internal error during generation: %s",
            request_id,
            message,
        )
        raise GenerationError(message)

    setattr(patched_raise_if_error, _PATCH_MARKER, True)
    GenerateBaseServing._raise_if_error = patched_raise_if_error


def apply_patch() -> None:
    global _PATCH_APPLIED

    if _PATCH_APPLIED:
        return

    _patch_raise_if_error()
    _PATCH_APPLIED = True


apply_patch()
