from __future__ import annotations

import importlib.util
import re
import sys
import types
import uuid
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

ROOT = Path(__file__).resolve().parents[3]
PATCH_PATH = ROOT / "ascend_vllm" / "patch" / "worker" / "patch_deepseek_v4_dspark.py"


# ---------------------------------------------------------------------------
# Stub helpers — minimal vllm / vllm_ascend / torch surface that the patch
# module imports at load time. Behaviour is only configured on the symbols
# that the patched ``load_weights`` actually consults at runtime; everything
# else falls back to ``MagicMock`` auto-spec.
# ---------------------------------------------------------------------------


def _make_package(name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__path__ = []  # type: ignore[attr-defined]
    return module


def _install_stubs(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Install stub modules so ``patch_deepseek_v4_dspark.py`` can be
    imported without the real vLLM / vLLM-Ascend runtime being present."""

    torch_mod = MagicMock(name="torch")
    torch_mod.__name__ = "torch"
    torch_mod.__path__ = []
    torch_mod.float8_e8m0fnu = "float8_e8m0fnu"
    torch_mod.uint8 = "uint8"

    def _view(_self, dtype):  # used by ``loaded_weight.view(torch.uint8)``
        return ("viewed", dtype)

    torch_mod.Tensor.view = _view

    npu_mod = MagicMock(name="torch.npu")
    npu_mod.__name__ = "torch.npu"

    torch_npu_mod = MagicMock(name="torch_npu")
    torch_npu_mod.__name__ = "torch_npu"

    # vllm.distributed — TP rank / world_size are read inside load_weights.
    vllm = _make_package("vllm")
    vllm_distributed = types.ModuleType("vllm.distributed")
    vllm_distributed.get_tensor_model_parallel_rank = MagicMock(return_value=0)  # type: ignore[attr-defined]
    vllm_distributed.get_tensor_model_parallel_world_size = MagicMock(return_value=1)  # type: ignore[attr-defined]

    # vllm.logger — info_once / warning_once only need to swallow the call.
    vllm_logger_mod = types.ModuleType("vllm.logger")
    vllm_logger_inst = MagicMock(name="logger")
    vllm_logger_mod.logger = vllm_logger_inst  # type: ignore[attr-defined]

    # vllm.model_executor.model_loader.weight_utils
    vllm_me_pkg = _make_package("vllm.model_executor")
    vllm_me_mod_pkg = _make_package("vllm.model_executor.model_loader")
    vllm_weight_utils = types.ModuleType("vllm.model_executor.model_loader.weight_utils")

    def _default_weight_loader(param, weight):  # identity copy marker
        param._loaded_with = getattr(param, "_loaded_with", []) + [weight]
        return None

    vllm_weight_utils.default_weight_loader = _default_weight_loader  # type: ignore[attr-defined]

    # vllm_ascend.models.deepseek_v4_dspark
    vllm_ascend = _make_package("vllm_ascend")
    vllm_ascend_models = _make_package("vllm_ascend.models")
    dspark_mod = types.ModuleType("vllm_ascend.models.deepseek_v4_dspark")

    class _DSparkDeepseekV4ForCausalLM:
        """Stand-in target class. Tests replace ``load_weights`` after the
        patch binds it via importlib."""

        # Set by ``_build_dspark_self`` before invoking ``load_weights``.
        pass

    # _EXPERT_SCALE_RE matches names like ``*.experts.<id>.<proj>.scale``
    dspark_mod._EXPERT_SCALE_RE = re.compile(r"\.experts\.\d+\.[A-Za-z_]+\.scale$")  # type: ignore[attr-defined]
    dspark_mod.DSparkDeepseekV4ForCausalLM = _DSparkDeepseekV4ForCausalLM  # type: ignore[attr-defined]

    # Register stubs.
    monkeypatch.setitem(sys.modules, "torch", torch_mod)
    monkeypatch.setitem(sys.modules, "torch.npu", npu_mod)
    monkeypatch.setitem(sys.modules, "torch_npu", torch_npu_mod)
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.distributed", vllm_distributed)
    monkeypatch.setitem(sys.modules, "vllm.logger", vllm_logger_mod)
    monkeypatch.setitem(sys.modules, "vllm.model_executor", vllm_me_pkg)
    monkeypatch.setitem(sys.modules, "vllm.model_executor.model_loader", vllm_me_mod_pkg)
    monkeypatch.setitem(
        sys.modules,
        "vllm.model_executor.model_loader.weight_utils",
        vllm_weight_utils,
    )
    monkeypatch.setitem(sys.modules, "vllm_ascend", vllm_ascend)
    monkeypatch.setitem(sys.modules, "vllm_ascend.models", vllm_ascend_models)
    monkeypatch.setitem(sys.modules, "vllm_ascend.models.deepseek_v4_dspark", dspark_mod)

    return {
        "torch_mod": torch_mod,
        "vllm_logger_inst": vllm_logger_inst,
        "default_weight_loader": _default_weight_loader,
        "DSparkDeepseekV4ForCausalLM": _DSparkDeepseekV4ForCausalLM,
        "dspark_mod": dspark_mod,
    }


def _load_patch_module(monkeypatch: pytest.MonkeyPatch):
    """Install stubs and import the patch module via importlib (isolation)."""
    stubs = _install_stubs(monkeypatch)

    module_name = f"patch_deepseek_v4_dspark_under_test_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, PATCH_PATH)
    assert spec is not None, "Could not build spec for patch_deepseek_v4_dspark.py"
    assert spec.loader is not None, "Spec has no loader"

    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)

    return module, stubs


# ---------------------------------------------------------------------------
# Builder helpers — produce minimal ``self`` mocks that drive the patched
# ``load_weights`` down specific code paths.
# ---------------------------------------------------------------------------


def _build_dspark_self(
    stubs: dict[str, Any],
    *,
    expert_dtype: str | None = None,
    num_attention_heads: int = 4,
    has_own_embed_tokens: bool = False,
    has_own_lm_head: bool = False,
    remap: dict[str, str | None] | None = None,
    expert_mapping: list[tuple[str, str, int, int]] | None = None,
    params: dict[str, Any] | None = None,
) -> Any:
    """Assemble a self-mock that exercises one ``load_weights`` invocation."""
    cls = stubs["DSparkDeepseekV4ForCausalLM"]
    self_obj = cls()

    config = MagicMock(name="config")
    # ``getattr(self.config, "expert_dtype", "fp4")`` must work; default fp4
    # if not explicitly provided.
    if expert_dtype is None:
        config.expert_dtype = "fp4"
    else:
        config.expert_dtype = expert_dtype
    config.num_attention_heads = num_attention_heads
    self_obj.config = config

    self_obj.has_own_embed_tokens = has_own_embed_tokens
    self_obj.has_own_lm_head = has_own_lm_head

    self_obj.model = MagicMock(name="model")
    self_obj.model.get_expert_mapping = MagicMock(return_value=expert_mapping or [])

    remap = remap or {}

    def _remap(name: str) -> str | None:
        return remap.get(name, name)

    self_obj._remap_dspark_name = _remap

    self_obj.named_parameters = MagicMock(return_value=params or {})

    return self_obj


def _make_param(
    name: str,
    *,
    weight_loader=None,
) -> Any:
    """Build a stand-in for a module parameter.

    A parameter with a custom ``weight_loader`` (callable) gets it stored so
    the patched method can invoke it.
    """
    param = MagicMock(name=f"param[{name}]")
    param._param_name = name
    if weight_loader is not None:
        param.weight_loader = weight_loader
    return param


def _make_weight(name: str = "tensor") -> Any:
    """Build a stand-in for a checkpoint tensor with a ``dtype`` attribute.

    The patch only reads ``loaded_weight.dtype`` and (for E8M0 scales)
    ``loaded_weight.view(...)``. A plain ``MagicMock`` with a sentinel
    ``dtype`` attribute is sufficient.
    """
    weight = MagicMock(name=name)
    weight.dtype = "fp32"  # not float8_e8m0fnu → E8M0 branch skipped
    return weight


# ---------------------------------------------------------------------------
# Tests — only the behavioural surface introduced by the patch.
# ---------------------------------------------------------------------------


class TestPatchDeepseekV4Dspark:
    """Tests for ``patch_deepseek_v4_dspark.py``.

    The patch replaces ``DSparkDeepseekV4ForCausalLM.load_weights`` with an
    Ascend-aware implementation tailored for the A5 dSpark draft loader.
    The following behaviours are verified:

    1. Module import monkey-patches ``load_weights`` onto the target class.
    2. ``embed.weight`` / ``head.weight`` are remapped to the canonical
       ``model.embed_tokens.weight`` / ``lm_head.weight`` names whenever
       the dSpark draft does not own its own embedding / head weights.
    3. The ``.scale`` suffix of expert weights is rewritten according to
       the dtype branch (``fp4`` → ``.weight_scale``,
       otherwise → ``.weight_scale_inv``); non-expert scales stay on the
       ``.weight_scale`` Ascend convention.
    4. ``.experts.*`` weights are routed through ``expert_mapping`` and
       the corresponding ``weight_loader`` is invoked with the right
       ``expert_id`` and ``shard_id``.
    """

    # ------------------------------------------------------------------ #
    # 1. Installation                                                    #
    # ------------------------------------------------------------------ #
    def test_patch_installs_load_weights_on_dspark_class(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Importing the patch module must rebind ``load_weights`` on
        ``DSparkDeepseekV4ForCausalLM`` to the patched function."""
        module, stubs = _load_patch_module(monkeypatch)
        cls = stubs["DSparkDeepseekV4ForCausalLM"]

        assert cls.load_weights is module.load_weights, (
            "patch_deepseek_v4_dspark.py did not bind load_weights onto DSparkDeepseekV4ForCausalLM"
        )

    # ------------------------------------------------------------------ #
    # 2. Embed / head weight remap                                       #
    # ------------------------------------------------------------------ #
    def test_load_weights_remaps_embed_and_head_when_not_owned(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``embed.weight`` and ``head.weight`` must be translated to the
        canonical model-level parameter names when the dSpark draft does
        NOT own its own embedding / LM head."""
        _module, stubs = _load_patch_module(monkeypatch)

        loaded: list[tuple[str, Any]] = []

        def _capturing_loader(param, weight, **_kw):
            loaded.append((param._param_name, weight))

        params = {
            "model.embed_tokens.weight": _make_param("model.embed_tokens.weight", weight_loader=_capturing_loader),
            "lm_head.weight": _make_param("lm_head.weight", weight_loader=_capturing_loader),
        }
        self_obj = _build_dspark_self(
            stubs,
            expert_dtype="fp4",
            has_own_embed_tokens=False,
            has_own_lm_head=False,
            params=params,
        )

        weights = [
            ("embed.weight", _make_weight("embed_tensor")),
            ("head.weight", _make_weight("head_tensor")),
        ]
        loaded_params = self_obj.load_weights(weights)

        # Both names must have been routed to the canonical model paths.
        assert ("model.embed_tokens.weight", weights[0][1]) in loaded
        assert ("lm_head.weight", weights[1][1]) in loaded
        assert "model.embed_tokens.weight" in loaded_params
        assert "lm_head.weight" in loaded_params

    # ------------------------------------------------------------------ #
    # 3. Expert scale suffix branches on expert_dtype                    #
    # ------------------------------------------------------------------ #
    def test_load_weights_uses_fp4_weight_scale_suffix_only_when_expert_dtype_fp4(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The patch rewrites ``.scale`` → ``.weight_scale`` when
        ``expert_dtype == 'fp4'`` and → ``.weight_scale_inv`` otherwise
        (matched via ``_EXPERT_SCALE_RE``). Non-expert scales keep the
        Ascend ``.weight_scale`` convention regardless of dtype."""
        _module, stubs = _load_patch_module(monkeypatch)

        captured: list[str] = []

        def _capturing_loader(param, _weight, name_mapped=None, **_kw):
            captured.append(param._param_name)
            return True

        # One expert-quantized scale and one non-expert scale. The expert
        # one needs to be reachable through ``expert_mapping`` so the
        # loader actually fires — otherwise the patch's ``continue``
        # silently drops the weight.
        expert_param = _make_param(
            "model.layers.0.mlp.experts.<M>.gate_proj.weight_scale_inv",
            weight_loader=_capturing_loader,
        )
        expert_param_fp4 = _make_param(
            "model.layers.0.mlp.experts.<M>.gate_proj.weight_scale",
            weight_loader=_capturing_loader,
        )
        non_expert_param = _make_param(
            "model.layers.0.self_attn.q_proj.weight_scale",
            weight_loader=_capturing_loader,
        )

        # ---- branch 1: expert_dtype != 'fp4' ---------------------------
        # expert scale must end up as ``.weight_scale_inv``;
        # non-expert scale must keep ``.weight_scale``.
        expert_mapping_non_fp4 = [
            (
                "model.layers.0.mlp.experts.<M>.gate_proj.weight_scale_inv",
                "model.layers.0.mlp.experts.0.gate_proj.weight_scale_inv",
                0,
                0,
            ),
        ]
        params_non_fp4 = {
            expert_param._param_name: expert_param,
            non_expert_param._param_name: non_expert_param,
        }
        self_obj = _build_dspark_self(
            stubs,
            expert_dtype="mx",
            num_attention_heads=1,
            params=params_non_fp4,
            expert_mapping=expert_mapping_non_fp4,
        )

        loaded_params = self_obj.load_weights(
            [
                (
                    "model.layers.0.mlp.experts.0.gate_proj.scale",
                    _make_weight("expert_scale"),
                ),
                (
                    "model.layers.0.self_attn.q_proj.scale",
                    _make_weight("attn_scale"),
                ),
            ]
        )

        assert "model.layers.0.mlp.experts.<M>.gate_proj.weight_scale_inv" in loaded_params
        assert "model.layers.0.self_attn.q_proj.weight_scale" in loaded_params
        assert captured == [
            "model.layers.0.mlp.experts.<M>.gate_proj.weight_scale_inv",
            "model.layers.0.self_attn.q_proj.weight_scale",
        ]

        # ---- branch 2: expert_dtype == 'fp4' ---------------------------
        # expert scale must end up as ``.weight_scale`` (NOT ``_inv``).
        captured.clear()
        expert_mapping_fp4 = [
            (
                "model.layers.0.mlp.experts.<M>.gate_proj.weight_scale",
                "model.layers.0.mlp.experts.0.gate_proj.weight_scale",
                0,
                0,
            ),
        ]
        self_obj_fp4 = _build_dspark_self(
            stubs,
            expert_dtype="fp4",
            num_attention_heads=1,
            params={expert_param_fp4._param_name: expert_param_fp4},
            expert_mapping=expert_mapping_fp4,
        )
        loaded_params_fp4 = self_obj_fp4.load_weights(
            [
                (
                    "model.layers.0.mlp.experts.0.gate_proj.scale",
                    _make_weight("expert_scale"),
                ),
            ]
        )
        assert "model.layers.0.mlp.experts.<M>.gate_proj.weight_scale" in loaded_params_fp4
        assert captured == ["model.layers.0.mlp.experts.<M>.gate_proj.weight_scale"]

    # ------------------------------------------------------------------ #
    # 4. Expert weight routing through expert_mapping                    #
    # ------------------------------------------------------------------ #
    def test_load_weights_routes_experts_through_expert_mapping(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``*.experts.*`` weights must be dispatched via
        ``self.model.get_expert_mapping()``; the destination
        ``weight_loader`` must be called with ``expert_id`` and
        ``shard_id`` from the matching mapping entry, and the resulting
        parameter name must be tracked in ``loaded_params``."""
        _module, stubs = _load_patch_module(monkeypatch)

        captured: list[dict[str, Any]] = []

        def _capturing_loader(param, weight, name_mapped=None, *, shard_id, expert_id, return_success, **_kw):
            captured.append(
                {
                    "param": param._param_name,
                    "weight": weight,
                    "shard_id": shard_id,
                    "expert_id": expert_id,
                    "return_success": return_success,
                }
            )
            return True

        target_param = _make_param("model.layers.0.mlp.experts.<M>.w13", weight_loader=_capturing_loader)

        # ``weight_name`` must be a substring of the checkpoint name, and
        # ``name.replace(weight_name, param_name)`` must yield a key that
        # is present in ``params_dict``.
        expert_mapping = [
            (
                "model.layers.0.mlp.experts.<M>.w13",
                "model.layers.0.mlp.experts.0.w13",
                0,
                0,
            ),  # shard 0 of expert 0
            (
                "model.layers.0.mlp.experts.<M>.w13",
                "model.layers.0.mlp.experts.0.w13",
                0,
                1,
            ),  # shard 1 of expert 0
            (
                "model.layers.0.mlp.experts.<M>.w2",
                "model.layers.0.mlp.experts.0.w2",
                0,
                0,
            ),
        ]

        self_obj = _build_dspark_self(
            stubs,
            expert_dtype="fp4",
            params={target_param._param_name: target_param},
            expert_mapping=expert_mapping,
        )

        weights = [("model.layers.0.mlp.experts.0.w13", _make_weight("expert_w13"))]
        loaded_params = self_obj.load_weights(weights)

        # Exactly one expert call: routed to (expert_id=0, shard_id=0)
        # because that is the first mapping entry whose weight_name is in
        # the checkpoint name and the loader returns success.
        assert len(captured) == 1
        call = captured[0]
        assert call["param"] == "model.layers.0.mlp.experts.<M>.w13"
        assert call["weight"] == weights[0][1]
        assert call["expert_id"] == 0
        assert call["shard_id"] == 0
        assert call["return_success"] is True
        assert "model.layers.0.mlp.experts.<M>.w13" in loaded_params

    # ------------------------------------------------------------------ #
    # 5. Skip params not in model (patch delta)                          #
    # ------------------------------------------------------------------ #
    def test_load_weights_skips_params_not_in_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When a weight name from the checkpoint does not exist in params_dict,
        the patched load_weights should skip it gracefully (via logger.warning_once)
        instead of crashing with KeyError —— the patch's core delta."""
        _module, stubs = _load_patch_module(monkeypatch)

        # Build self with only a few params —— NOT includeing the weight name we'll pass
        params = {
            "model.embed_tokens.weight": _make_param("model.embed_tokens.weight"),
        }
        self_obj = _build_dspark_self(
            stubs,
            expert_dtype="fp4",
            params=params,
        )

        # This weight name won't be in params_dict
        weights = [
            ("model.layers.0.self_attn.q_proj.weight", _make_weight("missing_param")),
        ]

        # Should NOT raise KeyError —— the patch adds a guard
        loaded_params = self_obj.load_weights(weights)

        assert "model.layers.0.self_attn.q_proj.weight" not in loaded_params

        # Verify logger.warning_once was called
        stubs["vllm_logger_inst"].warning_once.assert_called()
