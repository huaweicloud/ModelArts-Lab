from __future__ import annotations

import importlib.util
import sys
import types
import uuid
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict

ROOT = Path(__file__).resolve().parents[2]
PATCH_PATH = ROOT / "ascend_vllm" / "patch" / "platform" / "patch_disable_completion_tokens_details.py"


def _make_package(name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__path__ = []  # type: ignore[attr-defined]
    return module


def _install_vllm_stubs(monkeypatch: pytest.MonkeyPatch) -> type[Any]:
    vllm = _make_package("vllm")
    entrypoints = _make_package("vllm.entrypoints")
    openai = _make_package("vllm.entrypoints.openai")
    chat_completion = _make_package("vllm.entrypoints.openai.chat_completion")
    chat_protocol = types.ModuleType("vllm.entrypoints.openai.chat_completion.protocol")
    chat_serving = types.ModuleType("vllm.entrypoints.openai.chat_completion.serving")
    engine = _make_package("vllm.entrypoints.openai.engine")
    protocol = types.ModuleType("vllm.entrypoints.openai.engine.protocol")

    class OpenAIBaseModel(BaseModel):
        model_config = ConfigDict(extra="allow")

    class PromptTokenUsageInfo(OpenAIBaseModel):
        cached_tokens: int | None = None

    class UsageInfo(OpenAIBaseModel):
        prompt_tokens: int = 0
        total_tokens: int = 0
        completion_tokens: int | None = 0
        prompt_tokens_details: PromptTokenUsageInfo | None = None

    class ChatCompletionResponse(OpenAIBaseModel):
        usage: UsageInfo | None = None

    class ChatCompletionStreamResponse(OpenAIBaseModel):
        usage: UsageInfo | None = None

    class RequestResponseMetadata(BaseModel):
        final_usage_info: UsageInfo | None = None

    class OriginalUsageInfo:
        def __init__(
            self,
            *,
            prompt_tokens: int,
            completion_tokens: int,
            total_tokens: int,
        ) -> None:
            self.prompt_tokens = prompt_tokens
            self.completion_tokens = completion_tokens
            self.total_tokens = total_tokens

    vars(chat_protocol)["UsageInfo"] = UsageInfo
    vars(chat_protocol)["ChatCompletionResponse"] = ChatCompletionResponse
    vars(chat_protocol)["ChatCompletionStreamResponse"] = ChatCompletionStreamResponse
    vars(chat_serving)["UsageInfo"] = OriginalUsageInfo
    exec(
        """
def _inject_stream_usage_details(data, state):
    return {"injected": True, "data": data, "state": state}


def _make_full_response_usage(self, state):
    return UsageInfo(
        prompt_tokens=-1,
        completion_tokens=-1,
        total_tokens=-2,
    )


class OpenAIServingChat:
    enable_prompt_tokens_details = False

    def chat_completion_stream_generator(self, data, state):
        return _inject_stream_usage_details(data, state)

    def chat_completion_full_generator(self, state):
        return _make_full_response_usage(self, state)
""",
        chat_serving.__dict__,
    )

    vars(chat_serving)["PromptTokenUsageInfo"] = PromptTokenUsageInfo
    vars(protocol)["OpenAIBaseModel"] = OpenAIBaseModel
    vars(protocol)["PromptTokenUsageInfo"] = PromptTokenUsageInfo
    vars(protocol)["UsageInfo"] = UsageInfo
    vars(protocol)["RequestResponseMetadata"] = RequestResponseMetadata

    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.entrypoints", entrypoints)
    monkeypatch.setitem(sys.modules, "vllm.entrypoints.openai", openai)
    monkeypatch.setitem(sys.modules, "vllm.entrypoints.openai.chat_completion", chat_completion)
    monkeypatch.setitem(
        sys.modules,
        "vllm.entrypoints.openai.chat_completion.protocol",
        chat_protocol,
    )
    monkeypatch.setitem(sys.modules, "vllm.entrypoints.openai.chat_completion.serving", chat_serving)
    monkeypatch.setitem(sys.modules, "vllm.entrypoints.openai.engine", engine)
    monkeypatch.setitem(sys.modules, "vllm.entrypoints.openai.engine.protocol", protocol)

    vllm_ascend = _make_package("vllm_ascend")
    vllm_ascend_patch = _make_package("vllm_ascend.patch")
    vllm_ascend_patch_platform = _make_package("vllm_ascend.patch.platform")
    minimax_patch = types.ModuleType("vllm_ascend.patch.platform.patch_minimax_usage_accounting")

    monkeypatch.setitem(sys.modules, "vllm_ascend", vllm_ascend)
    monkeypatch.setitem(sys.modules, "vllm_ascend.patch", vllm_ascend_patch)
    monkeypatch.setitem(sys.modules, "vllm_ascend.patch.platform", vllm_ascend_patch_platform)
    monkeypatch.setitem(
        sys.modules,
        "vllm_ascend.patch.platform.patch_minimax_usage_accounting",
        minimax_patch,
    )

    return chat_serving.__dict__["OpenAIServingChat"]


def _load_patch_module(monkeypatch: pytest.MonkeyPatch) -> Any:
    _install_vllm_stubs(monkeypatch)

    module_name = f"patch_disable_completion_tokens_details_under_test_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, PATCH_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module


def test_make_usage_info_drops_completion_details_and_keeps_prompt_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_patch_module(monkeypatch)
    service = module.OpenAIServingChat()
    service.enable_prompt_tokens_details = True

    usage = service._make_usage_info(
        prompt_tokens=11,
        completion_tokens=7,
        num_cached_tokens=3,
        reasoning_tokens=5,
    )

    assert usage.prompt_tokens == 11
    assert usage.completion_tokens == 7
    assert usage.total_tokens == 18
    assert usage.prompt_tokens_details.cached_tokens == 3
    assert not hasattr(usage, "num_cached_tokens")
    assert not hasattr(usage, "completion_tokens_details")

    payload = module.chat_protocol.ChatCompletionResponse(usage=usage).model_dump()
    assert payload["usage"] == {
        "prompt_tokens": 11,
        "total_tokens": 18,
        "completion_tokens": 7,
        "prompt_tokens_details": {"cached_tokens": 3},
    }


def test_make_usage_info_keeps_zero_cached_prompt_details(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_patch_module(monkeypatch)
    service = module.OpenAIServingChat()
    service.enable_prompt_tokens_details = True

    usage = service._make_usage_info(
        prompt_tokens=11,
        completion_tokens=7,
        num_cached_tokens=0,
    )

    assert usage.total_tokens == 18
    assert usage.prompt_tokens_details.cached_tokens == 0
    assert not hasattr(usage, "completion_tokens_details")


def test_stream_usage_injector_is_replaced_with_noop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_patch_module(monkeypatch)
    service = module.OpenAIServingChat()
    data = {"usage": {"prompt_tokens": 1}}

    result = service.chat_completion_stream_generator(data, types.SimpleNamespace())

    assert result is data


def test_full_response_usage_uses_basic_usage_without_completion_details(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_patch_module(monkeypatch)
    service = module.OpenAIServingChat()
    state = types.SimpleNamespace(
        final_res=object(),
        num_prompt_tokens=13,
        completion_tokens=[2, 3, 5],
        num_cached_tokens=4,
    )

    usage = service.chat_completion_full_generator(state)

    assert usage.prompt_tokens == 13
    assert usage.completion_tokens == 10
    assert usage.total_tokens == 23
    assert usage.prompt_tokens_details is None
    assert not hasattr(usage, "completion_tokens_details")


def test_full_response_usage_returns_none_before_final_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_patch_module(monkeypatch)
    service = module.OpenAIServingChat()
    state = types.SimpleNamespace(
        final_res=None,
        num_prompt_tokens=13,
        completion_tokens=[2, 3, 5],
        num_cached_tokens=4,
    )

    assert service.chat_completion_full_generator(state) is None
