from __future__ import annotations

import importlib.util
import sys
import types
import uuid
from pathlib import Path
from typing import Any, ClassVar

import pytest

ROOT = Path(__file__).resolve().parents[3]
PATCH_PATH = ROOT / "ascend_vllm" / "patch" / "platform" / "patch_deepseek_v4_validation.py"


class _FakeTokenizer:
    def __init__(
        self,
        *,
        vocab_size: int = 100,
        added_vocab: dict[str, int] | None = None,
    ) -> None:
        self.vocab_size = vocab_size
        self._added_vocab = dict(added_vocab or {"<extra>": vocab_size})
        self.encode_calls: list[dict[str, Any]] = []

    def __copy__(self) -> _FakeTokenizer:
        clone = type(self)(vocab_size=self.vocab_size, added_vocab=self._added_vocab)
        clone.encode_calls = self.encode_calls
        return clone

    def get_added_vocab(self) -> dict[str, int]:
        return dict(self._added_vocab)

    def encode(self, prompt_str: str, *, add_special_tokens: bool = True, **kwargs: Any) -> list[Any]:
        self.encode_calls.append(
            {
                "prompt_str": prompt_str,
                "add_special_tokens": add_special_tokens,
                "kwargs": kwargs,
            }
        )
        return ["encoded", prompt_str]


def _make_package(name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__path__ = []  # type: ignore[attr-defined]
    return module


def _install_external_dependency_stubs(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    transformers = types.ModuleType("transformers")

    class PreTrainedTokenizerFast:
        calls: ClassVar[list[dict[str, Any]]] = []
        tokenizer: ClassVar[_FakeTokenizer] = _FakeTokenizer(vocab_size=512, added_vocab={"<tool>": 512})

        @classmethod
        def from_pretrained(cls, *args: Any, **kwargs: Any) -> _FakeTokenizer:
            cls.calls.append({"args": args, "kwargs": kwargs})
            return cls.tokenizer

    vars(transformers)["PreTrainedTokenizerFast"] = PreTrainedTokenizerFast

    vllm = _make_package("vllm")
    entrypoints = _make_package("vllm.entrypoints")
    openai = _make_package("vllm.entrypoints.openai")
    chat_completion = _make_package("vllm.entrypoints.openai.chat_completion")
    chat_protocol = types.ModuleType("vllm.entrypoints.openai.chat_completion.protocol")
    tokenizers = _make_package("vllm.tokenizers")
    deepseek_v4 = types.ModuleType("vllm.tokenizers.deepseek_v4")

    class DeepseekV4Tokenizer:
        @classmethod
        def from_pretrained(cls, *args: Any, **kwargs: Any) -> None:
            raise AssertionError("from_pretrained should be patched")

    encode_messages_calls: list[dict[str, Any]] = []
    cached_tokenizers: list[Any] = []

    def encode_messages(
        messages: list[dict[str, Any]],
        *,
        thinking_mode: str,
        drop_thinking: bool,
        reasoning_effort: str | None,
    ) -> str:
        encode_messages_calls.append(
            {
                "messages": messages,
                "thinking_mode": thinking_mode,
                "drop_thinking": drop_thinking,
                "reasoning_effort": reasoning_effort,
            }
        )
        return f"prompt:{thinking_mode}:{reasoning_effort or 'none'}:{len(messages)}"

    def get_cached_tokenizer(tokenizer: Any) -> Any:
        cached_tokenizers.append(tokenizer)
        return tokenizer

    def original_get_deepseek_v4_tokenizer(tokenizer: Any) -> Any:
        return tokenizer

    vars(deepseek_v4)["HfTokenizer"] = _FakeTokenizer
    vars(deepseek_v4)["DeepseekV4Tokenizer"] = DeepseekV4Tokenizer
    vars(deepseek_v4)["encode_messages"] = encode_messages
    vars(deepseek_v4)["encode_messages_calls"] = encode_messages_calls
    vars(deepseek_v4)["get_cached_tokenizer"] = get_cached_tokenizer
    vars(deepseek_v4)["cached_tokenizers"] = cached_tokenizers
    vars(deepseek_v4)["get_deepseek_v4_tokenizer"] = original_get_deepseek_v4_tokenizer

    vllm_ascend = _make_package("vllm_ascend")
    vllm_ascend_patch = _make_package("vllm_ascend.patch")
    vllm_ascend_patch_platform = _make_package("vllm_ascend.patch.platform")
    ascend_deepseek_patch = types.ModuleType("vllm_ascend.patch.platform.patch_deepseek_v4_thinking")
    vars(ascend_deepseek_patch)["_patched_get_deepseek_v4_tokenizer"] = object()

    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.entrypoints", entrypoints)
    monkeypatch.setitem(sys.modules, "vllm.entrypoints.openai", openai)
    monkeypatch.setitem(sys.modules, "vllm.entrypoints.openai.chat_completion", chat_completion)
    monkeypatch.setitem(
        sys.modules,
        "vllm.entrypoints.openai.chat_completion.protocol",
        chat_protocol,
    )
    monkeypatch.setitem(sys.modules, "vllm.tokenizers", tokenizers)
    monkeypatch.setitem(sys.modules, "vllm.tokenizers.deepseek_v4", deepseek_v4)
    monkeypatch.setitem(sys.modules, "vllm_ascend", vllm_ascend)
    monkeypatch.setitem(sys.modules, "vllm_ascend.patch", vllm_ascend_patch)
    monkeypatch.setitem(sys.modules, "vllm_ascend.patch.platform", vllm_ascend_patch_platform)
    monkeypatch.setitem(
        sys.modules,
        "vllm_ascend.patch.platform.patch_deepseek_v4_thinking",
        ascend_deepseek_patch,
    )

    return {
        "PreTrainedTokenizerFast": PreTrainedTokenizerFast,
        "deepseek_v4": deepseek_v4,
        "ascend_deepseek_patch": ascend_deepseek_patch,
    }


def _load_patch_module(monkeypatch: pytest.MonkeyPatch) -> tuple[Any, dict[str, Any]]:
    stubs = _install_external_dependency_stubs(monkeypatch)

    module_name = f"patch_deepseek_v4_validation_under_test_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, PATCH_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module, stubs


def _patched_tokenizer(monkeypatch: pytest.MonkeyPatch) -> tuple[Any, dict[str, Any]]:
    module, stubs = _load_patch_module(monkeypatch)
    tokenizer = _FakeTokenizer(vocab_size=256, added_vocab={"<a>": 256, "<b>": 257})
    return module.deepseek_v4_tokenizer.get_deepseek_v4_tokenizer(tokenizer), stubs


def test_apply_patch_replaces_vllm_and_ascend_tokenizer_hooks(monkeypatch: pytest.MonkeyPatch) -> None:
    module, stubs = _load_patch_module(monkeypatch)
    deepseek_v4 = stubs["deepseek_v4"]

    assert module._VALIDATION_PATCH_APPLIED is True
    assert deepseek_v4.get_deepseek_v4_tokenizer is module._patched_get_deepseek_v4_tokenizer
    assert (
        stubs["ascend_deepseek_patch"]._patched_get_deepseek_v4_tokenizer is module._patched_get_deepseek_v4_tokenizer
    )


def test_from_pretrained_wraps_fast_tokenizer_and_uses_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    module, stubs = _load_patch_module(monkeypatch)
    deepseek_v4 = stubs["deepseek_v4"]

    tokenizer = deepseek_v4.DeepseekV4Tokenizer.from_pretrained("repo/model", revision="main")

    assert stubs["PreTrainedTokenizerFast"].calls == [{"args": ("repo/model",), "kwargs": {"revision": "main"}}]
    assert deepseek_v4.cached_tokenizers == [tokenizer]
    assert isinstance(tokenizer, _FakeTokenizer)
    assert tokenizer.__class__.__name__ == "DSV4_FakeTokenizer"
    assert module.deepseek_v4_tokenizer.get_deepseek_v4_tokenizer is module._patched_get_deepseek_v4_tokenizer


def test_apply_chat_template_returns_prompt_string_with_tools_and_reasoning_effort(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer, stubs = _patched_tokenizer(monkeypatch)
    conversation = [{"role": "user", "content": "hello"}]
    tools = [{"type": "function", "function": {"name": "lookup"}}]

    prompt = tokenizer.apply_chat_template(
        conversation,
        tools=tools,
        tokenize=False,
        enable_thinking=True,
        drop_thinking=False,
        reasoning_effort="xhigh",
    )

    assert prompt == "prompt:thinking:max:2"
    assert conversation == [{"role": "user", "content": "hello"}]
    assert stubs["deepseek_v4"].encode_messages_calls == [
        {
            "messages": [
                {"role": "system", "tools": tools},
                {"role": "user", "content": "hello"},
            ],
            "thinking_mode": "thinking",
            "drop_thinking": False,
            "reasoning_effort": "max",
        }
    ]


def test_apply_chat_template_tokenizes_prompt_with_supported_tokenizer_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer, stubs = _patched_tokenizer(monkeypatch)

    result = tokenizer.apply_chat_template(
        [{"role": "user", "content": "hello"}],
        tokenize=True,
        thinking=True,
        reasoning_effort="none",
        truncation=True,
        max_length=8,
        padding=True,
    )

    assert result == ["encoded", "prompt:chat:none:1"]
    assert stubs["deepseek_v4"].encode_messages_calls[0]["thinking_mode"] == "chat"
    assert stubs["deepseek_v4"].encode_messages_calls[0]["reasoning_effort"] is None
    assert tokenizer.encode_calls == [
        {
            "prompt_str": "prompt:chat:none:1",
            "add_special_tokens": False,
            "kwargs": {"truncation": True, "max_length": 8},
        }
    ]


@pytest.mark.parametrize(
    ("messages", "expected_error"),
    [
        (
            [{"role": "user"}, {"role": "system"}],
            "Invalid system message at message index 1",
        ),
        (
            [{"role": "assistant"}],
            "Invalid consecutive assistant message at message index 0",
        ),
        (
            [{"role": "unknown"}],
            "Unkown role: unknown",
        ),
        (
            [{"role": "assistant", "tool_calls": []}],
            "Invalid message at message[0].tool_calls: empty error",
        ),
        (
            [{"role": "assistant", "tool_calls": [{"id": "call-1"}]}],
            "must be followed by a tool messages",
        ),
        (
            [{"role": "assistant", "tool_calls": [{"id": "call-1"}]}, {"role": "tool", "tool_call_id": "other"}],
            "must be followed by a tool messages",
        ),
    ],
)
def test_apply_chat_template_rejects_invalid_messages(
    monkeypatch: pytest.MonkeyPatch,
    messages: list[dict[str, Any]],
    expected_error: str,
) -> None:
    tokenizer, _ = _patched_tokenizer(monkeypatch)

    with pytest.raises(ValueError) as exc_info:
        tokenizer.apply_chat_template(messages, tokenize=False)
    assert expected_error in str(exc_info.value)


def test_apply_chat_template_accepts_terminal_assistant_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    tokenizer, stubs = _patched_tokenizer(monkeypatch)

    prompt = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": "finish this"},
            {"role": "assistant", "content": "partial", "prefix": True},
        ],
        tokenize=False,
    )

    assert prompt == "prompt:chat:none:2"
    assert stubs["deepseek_v4"].encode_messages_calls[0]["messages"][-1]["prefix"] is True


def test_apply_chat_template_accepts_matching_tool_responses(monkeypatch: pytest.MonkeyPatch) -> None:
    tokenizer, stubs = _patched_tokenizer(monkeypatch)

    prompt = tokenizer.apply_chat_template(
        [
            {
                "role": "assistant",
                "tool_calls": [{"id": "call-1"}, {"id": "call-2"}],
            },
            {"role": "tool", "tool_call_id": "call-2"},
            {"role": "tool", "tool_call_id": "call-1"},
            {"role": "user", "content": "continue"},
        ],
        tokenize=False,
    )

    assert prompt == "prompt:chat:none:4"
    assert len(stubs["deepseek_v4"].encode_messages_calls) == 1


def test_len_and_added_vocab_are_snapshotted(monkeypatch: pytest.MonkeyPatch) -> None:
    source = _FakeTokenizer(vocab_size=10, added_vocab={"<a>": 10, "<b>": 11})
    module, _ = _load_patch_module(monkeypatch)

    tokenizer = module.deepseek_v4_tokenizer.get_deepseek_v4_tokenizer(source)
    source._added_vocab["<c>"] = 12
    returned_vocab = tokenizer.get_added_vocab()
    returned_vocab["<d>"] = 13

    assert len(tokenizer) == 12
    assert tokenizer.get_added_vocab() == {"<a>": 10, "<b>": 11}
