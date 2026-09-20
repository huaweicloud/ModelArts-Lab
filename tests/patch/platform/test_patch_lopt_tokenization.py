#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-cloud project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import asyncio
import importlib.util
import logging
import sys
import types
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

ROOT = Path(__file__).resolve().parents[3]
PATCH_PATH = ROOT / "ascend_vllm" / "patch" / "platform" / "patch_lopt_tokenization.py"


class _Params:
    return_token_offsets: bool = False

    @staticmethod
    def get_encode_kwargs() -> dict[str, Any]:
        return {"add_special_tokens": False}


class _ParamsWithOffsets:
    return_token_offsets: bool = True

    @staticmethod
    def get_encode_kwargs() -> dict[str, Any]:
        return {"add_special_tokens": False}


class _FakeLopt:
    def __init__(self, usable: bool = True) -> None:
        self.usable = usable
        self.encoded_texts: list[str] = []

    def can_use(self, text: str, encode_kwargs: dict[str, Any]) -> bool:
        return self.usable

    def encode(self, text: str, **encode_kwargs: Any) -> list[int]:
        self.encoded_texts.append(text)
        return [11, 12, 13]


def _make_package(name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__path__ = []  # type: ignore[attr-defined]
    return module


class _StubRenderer:
    """Minimal renderer surface so patch_lopt_tokenization.py can import."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.tokenizer = None
        self.mm_processor = None

    def shutdown(self) -> None:
        pass

    def render_messages(self, messages, params: Any) -> Any:
        raise NotImplementedError

    async def render_messages_async(self, messages, params: Any) -> Any:
        raise NotImplementedError

    def _tokenize_prompt(self, prompt: Any, params: Any) -> Any:
        raise NotImplementedError

    async def _tokenize_prompt_async(self, prompt: Any, params: Any) -> Any:
        raise NotImplementedError


class _StubHfRenderer(_StubRenderer):
    pass


class _StubDeepseekV4Renderer(_StubRenderer):
    pass


class _StubChatParams:
    chat_template_kwargs: dict[str, Any] | None = None


class _StubTokenizeParams:
    return_token_offsets: bool = False

    @staticmethod
    def get_encode_kwargs() -> dict[str, Any]:
        return {"add_special_tokens": False}


def _install_dependency_stubs(
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, Any]:
    """Install stub modules for the vLLM / vllm-ascend imports so the patch
    module can be imported in isolation (CI has neither package installed)."""

    # -- vllm.* ------------------------------------------------------------
    vllm = _make_package("vllm")
    vllm_inputs = types.ModuleType("vllm.inputs")
    vllm_inputs.TextPrompt = dict  # type: ignore[attr-defined]
    vllm_inputs.TokensPrompt = dict  # type: ignore[attr-defined]

    vllm_logger = types.ModuleType("vllm.logger")

    def init_logger(name: str) -> logging.Logger:
        return logging.getLogger(name)

    vllm_logger.init_logger = init_logger  # type: ignore[attr-defined]

    vllm_renderers = _make_package("vllm.renderers")
    vllm_renderers_hf = types.ModuleType("vllm.renderers.hf")
    vllm_renderers_hf.HfRenderer = _StubHfRenderer  # type: ignore[attr-defined]
    vllm_renderers_deepseek = types.ModuleType("vllm.renderers.deepseek_v4")
    vllm_renderers_deepseek.DeepseekV4Renderer = _StubDeepseekV4Renderer  # type: ignore[attr-defined]
    vllm_renderers_params = types.ModuleType("vllm.renderers.params")
    vllm_renderers_params.ChatParams = _StubChatParams  # type: ignore[attr-defined]
    vllm_renderers_params.TokenizeParams = _StubTokenizeParams  # type: ignore[attr-defined]

    vllm_tokenizers = _make_package("vllm.tokenizers")
    vllm_tokenizers_hf = types.ModuleType("vllm.tokenizers.hf")
    vllm_tokenizers_hf.maybe_make_thread_pool = MagicMock()  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.inputs", vllm_inputs)
    monkeypatch.setitem(sys.modules, "vllm.logger", vllm_logger)
    monkeypatch.setitem(sys.modules, "vllm.renderers", vllm_renderers)
    monkeypatch.setitem(sys.modules, "vllm.renderers.hf", vllm_renderers_hf)
    monkeypatch.setitem(sys.modules, "vllm.renderers.deepseek_v4", vllm_renderers_deepseek)
    monkeypatch.setitem(sys.modules, "vllm.renderers.params", vllm_renderers_params)
    monkeypatch.setitem(sys.modules, "vllm.tokenizers", vllm_tokenizers)
    monkeypatch.setitem(sys.modules, "vllm.tokenizers.hf", vllm_tokenizers_hf)

    # -- ascend_vllm.patch.* (avoid triggering the real package __init__) --
    ascend_patch = _make_package("ascend_vllm.patch")
    ascend_patch_platform = _make_package("ascend_vllm.patch.platform")
    patch_envs = types.ModuleType("ascend_vllm.patch.platform.patch_envs")
    ascend_patch_platform.patch_envs = patch_envs  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "ascend_vllm.patch", ascend_patch)
    monkeypatch.setitem(sys.modules, "ascend_vllm.patch.platform", ascend_patch_platform)
    monkeypatch.setitem(
        sys.modules,
        "ascend_vllm.patch.platform.patch_envs",
        patch_envs,
    )

    return {
        "vllm_renderers_hf": vllm_renderers_hf,
        "vllm_renderers_deepseek": vllm_renderers_deepseek,
        "patch_envs": patch_envs,
    }


def _load_patch_module(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Any, dict[str, Any]]:
    """Install stubs and import the patch module via importlib (isolated)."""
    stubs = _install_dependency_stubs(monkeypatch)

    module_name = f"patch_lopt_tokenization_under_test_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, PATCH_PATH)
    assert spec is not None, "Could not build spec for patch_lopt_tokenization.py"
    assert spec.loader is not None, "Spec has no loader"

    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)

    return module, stubs


def test_attach_lopt_to_deepseek_v4_compatible_renderer(monkeypatch: pytest.MonkeyPatch) -> None:
    module, _ = _load_patch_module(monkeypatch)

    tokenizer = SimpleNamespace(is_fast=True)
    renderer = SimpleNamespace(tokenizer=tokenizer, mm_processor=None)
    expected_lopt = object()

    monkeypatch.setattr(
        module,
        "_config_from_env",
        lambda: SimpleNamespace(thread_workers=4),
    )
    monkeypatch.setattr(module, "maybe_make_thread_pool", lambda tokenizer, copies: tokenizer)
    monkeypatch.setattr(
        module,
        "LosslessParallelTokenizer",
        lambda tokenizer, config: expected_lopt,
    )

    module._attach_lopt(renderer)

    assert renderer._ascend_lopt_tokenizer is expected_lopt


@pytest.mark.parametrize(
    ("tokenizer", "mm_processor"),
    [
        (None, None),
        (MagicMock(is_fast=False), None),
        (MagicMock(is_fast=True), object()),
    ],
)
def test_attach_lopt_skips_unsupported_renderer(
    monkeypatch: pytest.MonkeyPatch,
    tokenizer: Any,
    mm_processor: Any,
) -> None:
    module, _ = _load_patch_module(monkeypatch)

    renderer = SimpleNamespace(tokenizer=tokenizer, mm_processor=mm_processor)

    module._attach_lopt(renderer)

    assert not hasattr(renderer, "_ascend_lopt_tokenizer")


def test_deepseek_v4_sync_tokenization_uses_lopt(monkeypatch: pytest.MonkeyPatch) -> None:
    module, _ = _load_patch_module(monkeypatch)

    lopt = _FakeLopt()
    renderer = SimpleNamespace(_ascend_lopt_tokenizer=lopt)

    result = module._patched_deepseek_v4_renderer_tokenize_prompt(
        renderer,
        {"prompt": "a sufficiently long prompt"},
        _Params(),
    )

    assert result["prompt_token_ids"] == [11, 12, 13]
    assert lopt.encoded_texts == ["a sufficiently long prompt"]


def test_deepseek_v4_sync_tokenization_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    module, _ = _load_patch_module(monkeypatch)

    lopt = _FakeLopt(usable=False)
    renderer = SimpleNamespace(_ascend_lopt_tokenizer=lopt)
    fallback_result = {"prompt": "short", "prompt_token_ids": [99]}

    def fallback(renderer, prompt, params):
        return fallback_result

    monkeypatch.setattr(module, "_original_deepseek_v4_renderer_tokenize_prompt", fallback)

    result = module._patched_deepseek_v4_renderer_tokenize_prompt(
        renderer,
        {"prompt": "short"},
        _Params(),
    )

    assert result is fallback_result
    assert lopt.encoded_texts == []


def test_return_token_offsets_defers_to_standard(monkeypatch: pytest.MonkeyPatch) -> None:
    module, _ = _load_patch_module(monkeypatch)

    lopt = _FakeLopt()
    renderer = SimpleNamespace(_ascend_lopt_tokenizer=lopt)
    fallback_result = {"prompt": "text", "prompt_token_ids": [42]}

    def fallback(renderer, prompt, params):
        return fallback_result

    result = module._tokenize_prompt_with_lopt(
        renderer,
        {"prompt": "a sufficiently long prompt"},
        _ParamsWithOffsets(),
        fallback,
    )

    assert result is fallback_result
    assert lopt.encoded_texts == []


def test_deepseek_v4_async_tokenization_uses_lopt(monkeypatch: pytest.MonkeyPatch) -> None:
    module, _ = _load_patch_module(monkeypatch)

    lopt = _FakeLopt()
    renderer = SimpleNamespace(_ascend_lopt_tokenizer=lopt, _executor=None)

    result = asyncio.run(
        module._patched_deepseek_v4_renderer_tokenize_prompt_async(
            renderer,
            {"prompt": "a sufficiently long prompt"},
            _Params(),
        )
    )

    assert result["prompt_token_ids"] == [11, 12, 13]
    assert lopt.encoded_texts == ["a sufficiently long prompt"]


def test_deepseek_v4_async_tokenization_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    module, _ = _load_patch_module(monkeypatch)

    lopt = _FakeLopt(usable=False)
    renderer = SimpleNamespace(_ascend_lopt_tokenizer=lopt, _executor=None)
    fallback_result = {"prompt": "short", "prompt_token_ids": [99]}

    async def fallback(renderer, prompt, params):
        return fallback_result

    monkeypatch.setattr(module, "_original_deepseek_v4_renderer_tokenize_prompt_async", fallback)

    result = asyncio.run(
        module._patched_deepseek_v4_renderer_tokenize_prompt_async(
            renderer,
            {"prompt": "short"},
            _Params(),
        )
    )

    assert result is fallback_result
    assert lopt.encoded_texts == []
