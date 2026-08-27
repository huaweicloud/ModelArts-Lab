from __future__ import annotations

import importlib.util
import pickle
import subprocess
import sys
import types
import uuid
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

ROOT = Path(__file__).resolve().parents[3]
PATCH_PATH = ROOT / "ascend_vllm" / "patch" / "platform" / "patch_vllm_registry.py"


# ---------------------------------------------------------------------------
# Stub helpers — minimal ``vllm.model_executor.models.registry`` surface so
# ``patch_vllm_registry.py`` can be imported in isolation. Only the symbols
# the patch consults at import time are wired up; the rest fall back to plain
# ``ModuleType`` package shells.
# ---------------------------------------------------------------------------


def _make_package(name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__path__ = []  # type: ignore[attr-defined]
    return module


def _install_dependency_stubs(
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, Any]:
    """Install stub modules for ``vllm`` and ``vllm.model_executor.models.registry``
    so the patch module can be imported without the real vLLM runtime.

    A unique sentinel is registered as the original ``_run_in_subprocess`` so
    tests can prove the patch *actually replaced* the binding on import.
    """
    vllm = _make_package("vllm")
    vllm_model_executor = _make_package("vllm.model_executor")
    vllm_model_executor_models = _make_package("vllm.model_executor.models")

    registry_mod = types.ModuleType("vllm.model_executor.models.registry")

    # Sentinel — anything but the patched function. Used to prove the patch
    # rewrote the attribute on import.
    original_sentinel = object()
    registry_mod._run_in_subprocess = original_sentinel  # type: ignore[attr-defined]
    # The patch reads this constant at call time, not at import time, so a
    # throwaway list is fine.
    registry_mod._SUBPROCESS_COMMAND = ["python", "-c", "pass"]  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.model_executor", vllm_model_executor)
    monkeypatch.setitem(sys.modules, "vllm.model_executor.models", vllm_model_executor_models)
    monkeypatch.setitem(sys.modules, "vllm.model_executor.models.registry", registry_mod)

    # ``cloudpickle`` is imported lazily inside ``_run_in_subprocess`` and is
    # not available in CI. Register a thin shim in ``sys.modules`` that
    # delegates ``dumps``/``loads`` to stdlib ``pickle`` — sufficient for the
    # simple callables and result values used by these tests.
    cloudpickle_mod = types.ModuleType("cloudpickle")
    cloudpickle_mod.dumps = pickle.dumps  # type: ignore[attr-defined]
    cloudpickle_mod.loads = pickle.loads  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "cloudpickle", cloudpickle_mod)

    return {
        "registry_mod": registry_mod,
        "original_sentinel": original_sentinel,
        "cloudpickle": cloudpickle_mod,
    }


def _load_patch_module(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Any, dict[str, Any]]:
    """Install stubs and import the patch module via importlib (isolated)."""
    stubs = _install_dependency_stubs(monkeypatch)

    module_name = f"patch_vllm_registry_under_test_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, PATCH_PATH)
    assert spec is not None, "Could not build spec for patch_vllm_registry.py"
    assert spec.loader is not None, "Spec has no loader"

    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)

    return module, stubs


# ---------------------------------------------------------------------------
# subprocess mocks — emulate what ``vllm.model_executor.models.registry._run``
# would have written to the output file inside the real subprocess.
# ---------------------------------------------------------------------------


class _FakeCompletedProcess:
    """Stand-in for ``subprocess.CompletedProcess`` used by the patch."""

    def __init__(self, returncode: int, stderr: bytes = b"") -> None:
        self.returncode = returncode
        self.stderr = stderr

    def check_returncode(self) -> None:
        if self.returncode != 0:
            raise subprocess.CalledProcessError(self.returncode, ["dummy"])


def _make_subprocess_side_effect(
    cloudpickle_mod: Any,
    *,
    returncode: int = 0,
    output: Any = None,
    stderr: bytes = b"",
) -> Any:
    """Build a ``subprocess.run`` replacement that mirrors what the real
    subprocess worker does: unpickle ``(fn, output_filepath)``, write a
    pickled result into ``output_filepath``, and return a CompletedProcess.

    When ``returncode == 0`` and ``output is None``, the ``fn()`` result is
    written (success path). When ``output`` is provided, it is written
    instead —— useful for simulating a failed subprocess that still leaves
    a partial result on disk (the patch's recovery scenario)
    """

    def _side_effect(*args: Any, **kwargs: Any) -> _FakeCompletedProcess:
        input_bytes = kwargs.get("input")
        assert input_bytes is not None, "subprocess.run must receive pickled input"
        fn, output_filepath = cloudpickle_mod.loads(input_bytes)
        result = output if output is not None else fn()
        with open(output_filepath, "wb") as f:
            f.write(pickle.dumps(result))
        return _FakeCompletedProcess(returncode=returncode, stderr=stderr)

    return _side_effect


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Module-level callables — used in place of lambdas so the stdlib ``pickle``
# shim (substituted for ``cloudpickle``) can serialise them across the
# mocked subprocess boundary.
# ---------------------------------------------------------------------------


def _ok_callable() -> Any:
    """Return a fixed dict — module-level so stdlib ``pickle`` can serialise it."""

    return {"status": "ok", "value": 1234}


def _ignored_callable() -> Any:
    """Return a tuple that the recovery-path test must NOT see."""

    return ("this", "value", "ignored")


class TestPatchVllmRegistry:
    """Tests for the ``registry._run_in_subprocess`` override in
    ``patch_vllm_registry.py``. Only the patch's semantic delta versus the
    upstream vLLM ``registry._run_in_subprocess`` is exercised here.
    """

    def test_patch_replaces_run_in_subprocess(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Importing the patch module rewrites ``registry._run_in_subprocess``
        with the patched implementation — this is the patch's only side-effect.
        """
        # ``subprocess`` is referenced by the patch module via the stdlib
        # binding; supply a MagicMock so importing doesn't attempt any work.
        monkeypatch.setattr(subprocess, "run", MagicMock())

        module, stubs = _load_patch_module(monkeypatch)
        registry_mod = stubs["registry_mod"]

        # The sentinel we registered before import must have been replaced
        # with the function defined inside the patch module.
        assert registry_mod._run_in_subprocess is not stubs["original_sentinel"]
        assert registry_mod._run_in_subprocess is module._run_in_subprocess

        # And the new binding must NOT be the original upstream function — the
        # patch's whole point is to substitute a different implementation.
        assert callable(registry_mod._run_in_subprocess)
        assert registry_mod._run_in_subprocess.__module__ == module.__name__
        assert registry_mod._run_in_subprocess.__qualname__ == "_run_in_subprocess"

    def test_returns_fn_result_when_subprocess_succeeds(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Happy path — when the subprocess exits 0, the patched
        ``_run_in_subprocess`` reads the pickled result from the output file
        and returns whatever ``fn()`` produced."""
        module, stubs = _load_patch_module(monkeypatch)
        cloudpickle_mod = stubs["cloudpickle"]

        monkeypatch.setattr(
            subprocess,
            "run",
            _make_subprocess_side_effect(cloudpickle_mod),
        )

        result = module._run_in_subprocess(_ok_callable)

        assert result == {"status": "ok", "value": 1234}
        # The callable was transported through the stdlib pickle shim
        # end-to-end and its return value was round-tripped via the
        # mocked subprocess output file.

    def test_recovers_partial_output_when_subprocess_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Recovery path (the patch's distinguishing behavior) — when the
        subprocess exits non-zero but a pickled result is still on disk, the
        patched function returns that result instead of raising ``RuntimeError``.

        This is the behavior the patch ADDS versus the upstream implementation,
        which unconditionally raises on a non-zero return code.
        """
        module, stubs = _load_patch_module(monkeypatch)
        cloudpickle_mod = stubs["cloudpickle"]

        partial = ["partial", "result", "from", "subprocess"]
        monkeypatch.setattr(
            subprocess,
            "run",
            _make_subprocess_side_effect(cloudpickle_mod, returncode=1, output=partial, stderr=b"simulated failure"),
        )

        # Must NOT raise RuntimeError; must return the partial result.
        result = module._run_in_subprocess(_ignored_callable)

        assert result == partial
        # And the returncode-1 scenario is genuinely exercised: if the patch
        # were to fall through to ``raise RuntimeError`` instead of recovering,
        # this assertion would never run.
        assert result[0] == "partial"
