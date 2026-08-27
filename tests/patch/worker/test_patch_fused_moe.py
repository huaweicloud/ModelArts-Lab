from __future__ import annotations

import importlib.util
import sys
import types
import uuid
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

ROOT = Path(__file__).resolve().parents[3]
PATCH_PATH = ROOT / "ascend_vllm" / "patch" / "worker" / "patch_fused_moe.py"


def _make_package(name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__path__ = []  # type: ignore[attr-defined]
    return module


class _Event:
    """Sentinel event with a distinguishable identity for ``wait_event`` tracking."""

    def __init__(self, name: str) -> None:
        self.name = name

    def __repr__(self) -> str:  # pragma: no cover - debug aid only
        return f"_Event({self.name!r})"


class _RecordingStream:
    """Mock NPU stream that records every ``wait_event`` invocation."""

    def __init__(self) -> None:
        self.waited_events: list[Any] = []

    def wait_event(self, evt) -> None:
        self.waited_events.append(evt)

    def wait_stream(self, _stream) -> None:  # pragma: no cover - structural
        pass

    def record_event(self):  # pragma: no cover - structural
        return _Event("recorded")


def _install_stubs(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Install stub modules for ``torch`` / ``torch_npu`` / ``vllm`` /
    ``vllm_ascend`` so that ``patch_fused_moe.py`` can be imported in isolation.

    The patch module imports several symbols at module load time; only the
    surface used during ``_forward_shared_experts`` execution is given
    functioning behaviour. Everything else falls back to ``MagicMock``.
    """
    # --- torch / torch.npu ------------------------------------------------
    torch_mod = MagicMock()
    torch_mod.__name__ = "torch"
    torch_mod.__path__ = []
    torch_mod.float16 = "float16"
    torch_mod.float8_e4m3fn = "float8_e4m3fn"
    # The patched code writes ``output_dtype=torch.int32`` and
    # ``output_dtype=original_dtype``; int32 is read as an attribute, so
    # give it a sentinel value rather than letting MagicMock autogenerate.
    torch_mod.int32 = "int32"

    npu_mod = MagicMock()
    npu_mod.__name__ = "torch.npu"

    shared_stream = _RecordingStream()
    npu_mod.current_stream = MagicMock(return_value=shared_stream)

    # CRITICAL: ``patch_fused_moe.py`` accesses the npu submodule as
    # ``torch.npu.current_stream()`` — not as ``import torch.npu``. Because
    # ``torch_mod`` is a MagicMock, ``torch_mod.npu`` would auto-generate a
    # child mock by default; binding our prepared ``npu_mod`` here ensures
    # attribute lookups land on the same instance that holds the recording
    # stream, the ``current_stream`` return-value, and any op mocks the
    # individual tests register under ``stubs["torch_mod"].ops._C_ascend.*``.
    torch_mod.npu = npu_mod

    # --- torch_npu --------------------------------------------------------
    torch_npu_mod = MagicMock()
    torch_npu_mod.__name__ = "torch_npu"
    torch_npu_mod.npu_dynamic_quant = MagicMock(return_value=(MagicMock(), MagicMock()))
    torch_npu_mod.npu_dynamic_mx_quant = MagicMock(return_value=(MagicMock(), MagicMock()))
    # ``npu_quant_matmul`` is invoked twice on the int-quant path; let it
    # return a fresh MagicMock per call so call-count assertions work.
    torch_npu_mod.npu_quant_matmul = MagicMock(side_effect=lambda *_a, **_kw: MagicMock())

    # --- vllm -------------------------------------------------------------
    vllm = _make_package("vllm")
    vllm_distributed = types.ModuleType("vllm.distributed")
    vllm_distributed.tensor_model_parallel_all_reduce = MagicMock(  # type: ignore[attr-defined]
        side_effect=lambda x: x  # identity passthrough for the reduce path
    )

    # --- vllm_ascend ------------------------------------------------------
    vllm_ascend = _make_package("vllm_ascend")
    vllm_ascend_ops = _make_package("vllm_ascend.ops")
    vllm_ascend_ops_fused_moe = _make_package("vllm_ascend.ops.fused_moe")

    class _AscendMoERunner:
        # The patch installs ``_forward_shared_experts`` as a class attribute.
        # Test code calls it via the class to verify monkey-patching.
        pass

    class _FusedMoEEvents:
        """Stand-in for ``FusedMoEEvents`` — accepts arbitrary kwargs and
        exposes them as attributes. Mirrors the dataclass in ``fused_moe.py``
        just enough to drive the patched method."""

        def __init__(self, **kwargs) -> None:
            for k, v in kwargs.items():
                setattr(self, k, v)

    fused_moe_mod = types.ModuleType("vllm_ascend.ops.fused_moe.fused_moe")
    fused_moe_mod.AscendMoERunner = _AscendMoERunner  # type: ignore[attr-defined]
    fused_moe_mod.FusedMoEEvents = _FusedMoEEvents  # type: ignore[attr-defined]

    # ascend_forward_context
    class _MoECommType:
        ALLTOALL = "ALLTOALL"
        MC2 = "MC2"
        FUSED_MC2 = "FUSED_MC2"
        ALLGATHER = "ALLGATHER"

    class _ExtraCtx:
        # Default to ALLGATHER so the all-reduce path is bypassed by default.
        moe_comm_type = "ALLGATHER"

    ascend_fwd_mod = types.ModuleType("vllm_ascend.ascend_forward_context")
    ascend_fwd_mod._EXTRA_CTX = _ExtraCtx()  # type: ignore[attr-defined]
    ascend_fwd_mod.MoECommType = _MoECommType  # type: ignore[attr-defined]

    # quant_type
    class _QuantType:
        NONE = "NONE"
        W8A8 = "W8A8"
        W4A8 = "W4A8"
        W4A8MXFP = "W4A8MXFP"

    quant_pkg = _make_package("vllm_ascend.quantization")
    quant_mod = types.ModuleType("vllm_ascend.quantization.quant_type")
    quant_mod.QuantType = _QuantType  # type: ignore[attr-defined]

    # utils
    class _NoOpStreamSwitch:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    utils_mod = types.ModuleType("vllm_ascend.utils")
    utils_mod.npu_stream_switch = _NoOpStreamSwitch  # type: ignore[attr-defined]
    utils_mod.shared_expert_dp_enabled = MagicMock(return_value=False)  # type: ignore[attr-defined]
    utils_mod.shared_experts_calculation_stream = MagicMock(return_value=object())  # type: ignore[attr-defined]

    # Register all stubs.
    monkeypatch.setitem(sys.modules, "torch", torch_mod)
    monkeypatch.setitem(sys.modules, "torch.npu", npu_mod)
    monkeypatch.setitem(sys.modules, "torch_npu", torch_npu_mod)
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.distributed", vllm_distributed)
    monkeypatch.setitem(sys.modules, "vllm_ascend", vllm_ascend)
    monkeypatch.setitem(sys.modules, "vllm_ascend.ops", vllm_ascend_ops)
    monkeypatch.setitem(sys.modules, "vllm_ascend.ops.fused_moe", vllm_ascend_ops_fused_moe)
    monkeypatch.setitem(sys.modules, "vllm_ascend.ops.fused_moe.fused_moe", fused_moe_mod)
    monkeypatch.setitem(sys.modules, "vllm_ascend.ascend_forward_context", ascend_fwd_mod)
    monkeypatch.setitem(sys.modules, "vllm_ascend.quantization", quant_pkg)
    monkeypatch.setitem(sys.modules, "vllm_ascend.quantization.quant_type", quant_mod)
    monkeypatch.setitem(sys.modules, "vllm_ascend.utils", utils_mod)

    return {
        "torch_mod": torch_mod,
        "torch_npu_mod": torch_npu_mod,
        "npu_mod": npu_mod,
        "AscendMoERunner": _AscendMoERunner,
        "FusedMoEEvents": _FusedMoEEvents,
        "MoECommType": _MoECommType,
        "QuantType": _QuantType,
        "shared_stream": shared_stream,
        "extra_ctx": _ExtraCtx,
    }


def _load_patch_module(monkeypatch: pytest.MonkeyPatch):
    """Install stubs and import the patch module via importlib (isolation)."""
    stubs = _install_stubs(monkeypatch)

    module_name = f"patch_fused_moe_under_test_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, PATCH_PATH)
    assert spec is not None, "Could not build spec for patch_fused_moe.py"
    assert spec.loader is not None, "Spec has no loader"

    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)

    return module, stubs


def _make_unquantized_shared_experts() -> Any:
    """Build a ``_shared_experts`` mock with NO ``weight_scale`` attributes,
    forcing the patched method into its non-quantized ``else`` branch."""

    class _Proj:
        pass  # bare — no weight_scale, so ``hasattr(...)`` is False

    class _SharedExperts:
        def __init__(self) -> None:
            self.gate_up_proj = _Proj()
            self.down_proj = _Proj()

    return _SharedExperts()


def _make_w4a8mxfp_shared_experts() -> Any:
    """Build a ``_shared_experts`` mock with ``weight_scale`` attributes for
    the W4A8MXFP branch. ``gate_up_proj`` and ``down_proj`` are callables
    that return ``[output_tensor]`` (they are called with a (qx, scale)
    tuple and indexed at ``[0]``)."""

    def _proj_factory(prefix: str) -> MagicMock:
        proj = MagicMock(name=f"{prefix}_proj")
        proj.weight_scale = "mx_scale"
        proj.weight_scale_fp32 = "scale_fp32"  # presence forces has_quantized_shared
        proj.weight = "mx_weight"
        proj.return_value = [MagicMock(name=f"mx_{prefix}_out")]
        return proj

    return type(
        "_SharedExperts",
        (),
        {
            "gate_up_proj": _proj_factory("gate_up"),
            "down_proj": _proj_factory("down"),
        },
    )()


def _make_int_quant_shared_experts() -> Any:
    """Build a ``_shared_experts`` mock with ``weight_scale`` attributes for
    the W8A8/W4A8 int-quant branch.

    The int-quant path accesses ``weight`` / ``weight_scale`` /
    ``weight_scale_fp32`` as ATTRIBUTES only — never invokes the projections
    as callables. Returning sentinel strings is enough.
    """

    def _proj_factory() -> MagicMock:
        proj = MagicMock(name="int_proj")
        proj.weight_scale = "int_scale"
        proj.weight = "int_weight"
        proj.weight_scale_fp32 = "int_scale_fp32"
        return proj

    return type(
        "_SharedExperts",
        (),
        {
            "gate_up_proj": _proj_factory(),
            "down_proj": _proj_factory(),
        },
    )()


class _Hidden:
    """Mock hidden_states tensor with dtype attribute."""

    dtype = "float16"


def _run_w4a8mxfp_forward(stubs: dict[str, Any]) -> tuple[Any, Any, Any, Any]:
    """Set up and execute the W4A8MXFP branch of ``_forward_shared_experts``.

    Returns ``(swiglu_calls, result, shared_experts, evts)`` where:
    - ``swiglu_calls`` captures all kwargs passed to ``npu_swiglu_shared_quant``
    - ``result`` is the return value of ``_forward_shared_experts``
    - ``shared_experts`` is the mock shared experts object (for call assertions)
    - ``evts`` is the ``FusedMoEEvents`` used (for event-wait assertions)
    """
    cls = stubs["AscendMoERunner"]
    QuantType = stubs["QuantType"]
    FusedMoEEvents = stubs["FusedMoEEvents"]

    evts = FusedMoEEvents(
        before_routed_experts=_Event("before_routed_experts"),
        after_routed_experts=_Event("after_routed_experts"),
        before_dispatch=_Event("before_dispatch"),
        before_gmm2=_Event("before_gmm2"),
        before_combine=_Event("before_combine"),
        swiglu_limit=0.5,
        swiglu_alpha=0.7,
        swiglu_beta=0.2,
    )

    swiglu_calls: list[Any] = []

    def _swiglu(*_args, **kwargs):
        swiglu_calls.append(kwargs)
        return (MagicMock(name="qx_after_swiglu"), MagicMock(name="scale_after_swiglu"), None)

    stubs["torch_mod"].ops._C_ascend.npu_swiglu_group_quant = MagicMock(side_effect=_swiglu)

    shared_experts = _make_w4a8mxfp_shared_experts()
    mock_self = MagicMock()
    mock_self._shared_experts = shared_experts
    mock_self.quant_type = QuantType.W4A8MXFP
    mock_self.multistream_overlap_shared_expert = False

    result = cls._forward_shared_experts(mock_self, _Hidden(), evts)
    return swiglu_calls, result, shared_experts, evts


class TestPatchFusedMoe:
    """Tests for ``patch_fused_moe.py`` — the patched
    ``AscendMoERunner._forward_shared_experts`` method.

    The patch introduces ONE behavioural delta:

    * In the ``W4A8MXFP`` branch, the call to
      ``torch.ops._C_ascend.npu_swiglu_group_quant(...)`` no longer
      forwards the ``glu_alpha`` and ``glu_bias`` keyword arguments
      (the values of ``fused_moe_evts.swiglu_alpha`` and
      ``fused_moe_evts.swiglu_beta`` are no longer mapped onto those
      parameter names).
    """

    # ------------------------------------------------------------------ #
    # 1. Installation                                                    #
    # ------------------------------------------------------------------ #
    def test_patch_installs_forward_shared_experts_on_ascend_moe_runner(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Loading ``patch_fused_moe`` must attach ``_forward_shared_experts``
        as a class attribute on ``AscendMoERunner`` (monkey-patch)."""
        module, stubs = _load_patch_module(monkeypatch)

        cls = stubs["AscendMoERunner"]
        assert cls._forward_shared_experts is module._forward_shared_experts, (
            "patch_fused_moe.py did not bind _forward_shared_experts to the AscendMoERunner class"
        )

    # ------------------------------------------------------------------ #
    # 2. PATCH DELTA — W4A8MXFP drops glu_alpha and glu_bias              #
    # ------------------------------------------------------------------ #
    def test_w4a8mxfp_branch_swiglu_group_quant_omits_glu_alpha_and_glu_bias(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """In the ``W4A8MXFP`` branch the ``npu_swiglu_group_quant`` op is
        called WITHOUT the ``glu_alpha`` and ``glu_bias`` kwargs — the
        patch's only behavioural change."""
        _module, stubs = _load_patch_module(monkeypatch)
        swiglu_calls, _result, _shared_experts, evts = _run_w4a8mxfp_forward(stubs)

        # The op must have been invoked exactly once.
        assert len(swiglu_calls) == 1, f"expected exactly one npu_swiglu_group_quant call, got {len(swiglu_calls)}"
        kwargs = swiglu_calls[0]

        # PATCH DELTA: glu_alpha and glu_bias must NOT appear in the kwargs
        # forwarded to npu_swiglu_group_quant. Note that the upstream
        # code's signature also did not accept these arguments (which
        # makes them silently-drops — the patch aligns the call site
        # with the actual op signature).
        assert "glu_alpha" not in kwargs, (
            f"PATCH DELTA: glu_alpha was unexpectedly forwarded to npu_swiglu_group_quant: {kwargs!r}"
        )
        assert "glu_bias" not in kwargs, (
            f"PATCH DELTA: glu_bias was unexpectedly forwarded to npu_swiglu_group_quant: {kwargs!r}"
        )

    # ------------------------------------------------------------------ #
    # 3. Regression — preserved kwargs on npu_swiglu_group_quant         #
    # ------------------------------------------------------------------ #
    def test_w4a8mxfp_branch_swiglu_group_quant_preserves_required_kwargs(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The ``clamp_value``, ``dst_type``, ``quant_mode``, ``topk_weight``
        and ``group_index`` kwargs must still be forwarded to
        ``npu_swiglu_group_quant`` after the patch — only ``glu_alpha``
        and ``glu_bias`` are removed."""
        _module, stubs = _load_patch_module(monkeypatch)
        torch_mod = stubs["torch_mod"]
        swiglu_calls, _result, shared_experts, evts = _run_w4a8mxfp_forward(stubs)

        assert len(swiglu_calls) == 1
        kwargs = swiglu_calls[0]

        # Clamp value must come from FusedMoEEvents.swiglu_limit.
        assert kwargs.get("clamp_value") == 0.5, f"clamp_value was clobbered: {kwargs!r}"
        assert kwargs.get("dst_type") == torch_mod.float8_e4m3fn, f"dst_type was clobbered: {kwargs!r}"
        assert kwargs.get("quant_mode") == 2, f"quant_mode was clobbered: {kwargs!r}"
        assert kwargs.get("topk_weight") is None
        assert kwargs.get("group_index") is None

        # The hidden_states tensor (output of gate_up_proj) is the first
        # positional arg of npu_swiglu_group_quant.
        args = stubs["torch_mod"].ops._C_ascend.npu_swiglu_group_quant.call_args[0]
        assert args[0] is shared_experts.gate_up_proj.return_value[0]

    # ------------------------------------------------------------------ #
    # 4. Regression — W4A8MXFP still awaits before_dispatch etc.         #
    # ------------------------------------------------------------------ #
    def test_w4a8mxfp_branch_waits_for_unmodified_events(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The W4A8MXFP branch's event waits are NOT touched by the patch:
        ``before_routed_experts`` (synchronous), ``before_dispatch``,
        ``before_gmm2``, and ``before_combine`` must all still be awaited."""
        _module, stubs = _load_patch_module(monkeypatch)
        swiglu_calls, result, shared_experts, evts = _run_w4a8mxfp_forward(stubs)

        waited = stubs["shared_stream"].waited_events
        assert evts.before_routed_experts in waited, f"W4A8MXFP path must wait on before_routed_experts, got {waited!r}"
        assert evts.before_dispatch in waited, (
            f"W4A8MXFP path must still wait on before_dispatch for gate_up_proj overlap, got {waited!r}"
        )
        assert evts.before_gmm2 in waited, f"W4A8MXFP path must wait on before_gmm2, got {waited!r}"
        assert evts.before_combine in waited, f"W4A8MXFP path must wait on before_combine, got {waited!r}"

        # Final result is the down_proj's [0] output.
        assert result is shared_experts.down_proj.return_value[0]
        # gate_up_proj was invoked exactly once.
        shared_experts.gate_up_proj.assert_called_once()
        shared_experts.down_proj.assert_called_once()

    # ------------------------------------------------------------------ #
    # 5. Regression — int-quant branch still uses glu_alpha/glu_bias      #
    # ------------------------------------------------------------------ #
    def test_int_quant_branch_still_forwards_glu_alpha_and_glu_bias_to_dequant(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The W8A8/W4A8 int-quant branch is NOT touched by the patch —
        ``npu_dequant_swiglu_quant(...)`` must still receive the
        ``glu_alpha`` and ``glu_bias`` kwargs (forwarded from
        ``fused_moe_evts.swiglu_alpha`` / ``swiglu_beta``)."""
        _module, stubs = _load_patch_module(monkeypatch)
        cls = stubs["AscendMoERunner"]
        QuantType = stubs["QuantType"]
        FusedMoEEvents = stubs["FusedMoEEvents"]

        evts = FusedMoEEvents(
            before_routed_experts=_Event("before_routed_experts"),
            after_routed_experts=_Event("after_routed_experts"),
            before_dispatch=_Event("before_dispatch"),
            before_gmm2=_Event("before_gmm2"),
            before_combine=_Event("before_combine"),
            swiglu_limit=1.5,
            swiglu_alpha=1.1,
            swiglu_beta=0.3,
        )

        dequant_calls: list[Any] = []

        def _dequant(*_args, **kwargs):
            dequant_calls.append(kwargs)
            return (MagicMock(name="qx_after_dequant"), MagicMock(name="scale_after_dequant"))

        stubs["torch_mod"].ops._C_ascend.npu_dequant_swiglu_quant = MagicMock(side_effect=_dequant)

        shared_experts = _make_int_quant_shared_experts()
        mock_self = MagicMock()
        mock_self._shared_experts = shared_experts
        mock_self.quant_type = QuantType.W8A8
        mock_self.multistream_overlap_shared_expert = False

        cls._forward_shared_experts(mock_self, _Hidden(), evts)

        assert len(dequant_calls) == 1, f"expected exactly one npu_dequant_swiglu_quant call, got {len(dequant_calls)}"
        kwargs = dequant_calls[0]

        # Int-quant branch is preserved: clamp_limit, glu_alpha, glu_bias
        # must all still be forwarded from the FusedMoEEvents object.
        assert kwargs.get("clamp_limit") == 1.5
        assert kwargs.get("glu_alpha") == 1.1, (
            f"int-quant branch unexpectedly lost glu_alpha forwarding; got {kwargs!r}"
        )
        assert kwargs.get("glu_bias") == 0.3, f"int-quant branch unexpectedly lost glu_bias forwarding; got {kwargs!r}"
        assert kwargs.get("quant_mode") == 1
        assert kwargs.get("swiglu_mode") == 1
        assert kwargs.get("activate_left") is True

        # Int-quant branch reads down_proj attributes only — never invokes
        # them as callables.
        shared_experts.down_proj.assert_not_called()
        shared_experts.gate_up_proj.assert_not_called()

    # ------------------------------------------------------------------ #
    # 6. Early-exit when ``_shared_experts`` is None                      #
    # ------------------------------------------------------------------ #
    def test_returns_none_when_shared_experts_is_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When ``self._shared_experts`` is None the patched method must
        return ``None`` immediately and must NOT touch any NPU streams.

        The patch preserves this early-exit behaviour unchanged from the
        upstream ``fused_moe.py``.
        """
        _module, stubs = _load_patch_module(monkeypatch)
        cls = stubs["AscendMoERunner"]
        FusedMoEEvents = stubs["FusedMoEEvents"]

        mock_self = MagicMock()
        mock_self._shared_experts = None

        evts = FusedMoEEvents(
            before_routed_experts=_Event("before_routed_experts"),
            after_routed_experts=_Event("after_routed_experts"),
            before_dispatch=_Event("before_dispatch"),
            before_gmm2=_Event("before_gmm2"),
            before_combine=_Event("before_combine"),
        )

        # Force moe_comm_type into a setting that triggers the trailing
        # all-reduce — verify the early-exit still bypasses it.
        stubs["extra_ctx"].moe_comm_type = stubs["MoECommType"].MC2

        result = cls._forward_shared_experts(mock_self, _Hidden(), evts)

        assert result is None
        assert stubs["shared_stream"].waited_events == [], "early-exit path must not invoke wait_event on any stream"

        vllm_distributed_mod = sys.modules["vllm.distributed"]
        vllm_distributed_mod.tensor_model_parallel_all_reduce.assert_not_called()

    # ------------------------------------------------------------------ #
    # 7. Non-quantized ``else`` branch is byte-identical to upstream     #
    # ------------------------------------------------------------------ #
    def test_non_quantized_else_branch_is_byte_identical_to_upstream(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Sanity check: the patch does NOT touch the non-quantized ``else``
        branch. The branch must wait on ``before_routed_experts``,
        ``before_dispatch`` (NOT before_gmm2), and ``before_combine`` —
        exactly as the upstream code does."""
        _module, stubs = _load_patch_module(monkeypatch)
        cls = stubs["AscendMoERunner"]
        QuantType = stubs["QuantType"]
        FusedMoEEvents = stubs["FusedMoEEvents"]

        before_routed = _Event("before_routed_experts")
        before_dispatch = _Event("before_dispatch")
        before_combine = _Event("before_combine")
        before_gmm2 = _Event("before_gmm2")
        evts = FusedMoEEvents(
            before_routed_experts=before_routed,
            after_routed_experts=_Event("after_routed_experts"),
            before_dispatch=before_dispatch,
            before_gmm2=before_gmm2,
            before_combine=before_combine,
        )

        mock_self = MagicMock()
        mock_self._shared_experts = _make_unquantized_shared_experts()
        mock_self.quant_type = QuantType.NONE
        mock_self.multistream_overlap_shared_expert = False
        mock_self._shared_experts_part1 = MagicMock(return_value="p1")
        mock_self._shared_experts_part2 = MagicMock(return_value="p2")

        hidden = _Hidden()
        result = cls._forward_shared_experts(mock_self, hidden, evts)

        waited = stubs["shared_stream"].waited_events
        # The upstream code uses ``before_dispatch`` here — this is a
        # regression guard so any future patch "fix" on this line is
        # immediately caught.
        assert before_routed in waited
        assert before_dispatch in waited, (
            f"non-quantized branch must still wait on before_dispatch (upstream behaviour): {waited!r}"
        )
        assert before_combine in waited
        assert before_gmm2 not in waited

        mock_self._shared_experts_part1.assert_called_once_with(hidden)
        mock_self._shared_experts_part2.assert_called_once_with(hidden, "p1")
        assert result == "p2"

        # ALLGATHER context (default) means the all-reduce is NOT applied.
        vllm_distributed_mod = sys.modules["vllm.distributed"]
        vllm_distributed_mod.tensor_model_parallel_all_reduce.assert_not_called()
