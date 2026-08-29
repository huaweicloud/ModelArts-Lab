# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import functools
from collections.abc import AsyncIterator
from typing import Any

from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat

_PATCH_APPLIED = False
_PATCH_MARKER = "_modelarts_chat_completion_kv_load_failure_patch_applied"


async def _checked_result_generator(
    serving: OpenAIServingChat,
    result_generator: AsyncIterator[Any],
    request_id: str,
) -> AsyncIterator[Any]:
    """Raise before vLLM emits the first logical streaming response chunk."""
    async for result in result_generator:
        kv_transfer_params = getattr(result, "kv_transfer_params", None)
        for output in result.outputs:
            serving._raise_if_error(
                output.finish_reason,
                request_id,
                kv_transfer_params,
            )
        yield result


def _patch_chat_completion_generators() -> None:
    current_stream_generator = OpenAIServingChat.chat_completion_stream_generator
    if not getattr(current_stream_generator, _PATCH_MARKER, False):

        @functools.wraps(current_stream_generator)
        async def patched_stream_generator(
            self: OpenAIServingChat,
            request: Any,
            result_generator: AsyncIterator[Any],
            request_id: str,
            *args: Any,
            **kwargs: Any,
        ) -> AsyncIterator[str]:
            checked_generator = _checked_result_generator(
                self,
                result_generator,
                request_id,
            )
            async for chunk in current_stream_generator(
                self,
                request,
                checked_generator,
                request_id,
                *args,
                **kwargs,
            ):
                yield chunk

        setattr(patched_stream_generator, _PATCH_MARKER, True)
        OpenAIServingChat.chat_completion_stream_generator = patched_stream_generator

    current_full_generator = OpenAIServingChat.chat_completion_full_generator
    if not getattr(current_full_generator, _PATCH_MARKER, False):

        @functools.wraps(current_full_generator)
        async def patched_full_generator(
            self: OpenAIServingChat,
            request: Any,
            result_generator: AsyncIterator[Any],
            request_id: str,
            *args: Any,
            **kwargs: Any,
        ) -> Any:
            checked_generator = _checked_result_generator(
                self,
                result_generator,
                request_id,
            )
            return await current_full_generator(
                self,
                request,
                checked_generator,
                request_id,
                *args,
                **kwargs,
            )

        setattr(patched_full_generator, _PATCH_MARKER, True)
        OpenAIServingChat.chat_completion_full_generator = patched_full_generator


def apply_patch() -> None:
    global _PATCH_APPLIED

    if _PATCH_APPLIED:
        return

    _patch_chat_completion_generators()
    _PATCH_APPLIED = True


apply_patch()
