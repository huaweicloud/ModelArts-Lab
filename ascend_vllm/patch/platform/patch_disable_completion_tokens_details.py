import importlib

from vllm.entrypoints.openai.chat_completion import serving as chat_serving
from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
from vllm.entrypoints.openai.engine import protocol as engine_protocol

importlib.import_module("vllm_ascend.patch.platform.patch_minimax_usage_accounting")


def _make_usage_info_without_completion_details(
    self,
    *,
    prompt_tokens: int,
    completion_tokens: int,
    num_cached_tokens: int | None = None,
    reasoning_tokens: int | None = None,
):
    usage = engine_protocol.UsageInfo(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        num_cached_tokens=num_cached_tokens,
    )

    if self.enable_prompt_tokens_details and num_cached_tokens:
        usage.prompt_token_details = chat_serving.PromptTokenUsageInfo(
            cached_tokens=num_cached_tokens,
        )

    return usage


def _inject_stream_usage_details_noop(data, state):
    return data


OpenAIServingChat._make_usage_info = _make_usage_info_without_completion_details

stream_gen = OpenAIServingChat.chat_completion_stream_generator
if "_inject_stream_usage_details" in stream_gen.__globals__:
    stream_gen.__globals__["_inject_stream_usage_details"] = _inject_stream_usage_details_noop

full_gen = OpenAIServingChat.chat_completion_full_generator
if "_make_full_response_usage" in full_gen.__globals__:
    old_make_full_response_usage = full_gen.__globals__["_make_full_response_usage"]

    def _make_full_response_usage_without_completion_details(self, state):
        if state.final_res is None:
            return None

        return self._make_usage_info(
            prompt_tokens=state.num_prompt_tokens,
            completion_tokens=sum(state.completion_tokens),
            num_cached_tokens=state.num_cached_tokens,
            reasoning_tokens=None,
        )

    full_gen.__globals__["_make_full_response_usage"] = _make_full_response_usage_without_completion_details
