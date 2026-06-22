import importlib

from vllm.entrypoints.openai.chat_completion import protocol as chat_protocol
from vllm.entrypoints.openai.chat_completion import serving as chat_serving
from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
from vllm.entrypoints.openai.engine import protocol as engine_protocol

importlib.import_module("vllm_ascend.patch.platform.patch_minimax_usage_accounting")


class UsageInfo(engine_protocol.OpenAIBaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0
    completion_tokens: int | None = 0
    prompt_tokens_details: engine_protocol.PromptTokenUsageInfo | None = None


UsageInfo.__module__ = engine_protocol.__name__
engine_protocol.UsageInfo = UsageInfo
chat_protocol.UsageInfo = UsageInfo
chat_serving.UsageInfo = UsageInfo


def _rebuild_model_field(model_cls, field_name: str, annotation) -> None:
    if not hasattr(model_cls, "model_fields") or field_name not in model_cls.model_fields:
        return
    model_cls.__annotations__[field_name] = annotation
    model_cls.model_fields[field_name].annotation = annotation
    model_cls.model_rebuild(force=True)


_rebuild_model_field(chat_protocol.ChatCompletionResponse, "usage", UsageInfo)
_rebuild_model_field(chat_protocol.ChatCompletionStreamResponse, "usage", UsageInfo | None)
_rebuild_model_field(engine_protocol.RequestResponseMetadata, "final_usage_info", UsageInfo | None)


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
        total_tokens=prompt_tokens + completion_tokens,
    )

    if self.enable_prompt_tokens_details and num_cached_tokens is not None:
        usage.prompt_tokens_details = chat_serving.PromptTokenUsageInfo(
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
