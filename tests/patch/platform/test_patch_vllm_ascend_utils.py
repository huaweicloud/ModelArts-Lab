from __future__ import annotations

import importlib.util
import json
import os
import sys
import types
import uuid
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

ROOT = Path(__file__).resolve().parents[3]
PATCH_PATH = ROOT / "ascend_vllm" / "patch" / "platform" / "patch_vllm_ascend_utils.py"


def _make_package(name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__path__ = []  # type: ignore[attr-defined]
    return module


def _install_dependency_stubs(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Install stub modules for vllm_ascend.utils, vllm.platforms.interface, and vllm_ascend.cpu_binding."""
    vllm = _make_package("vllm")
    vllm_platforms = _make_package("vllm.platforms")
    interface_mod = types.ModuleType("vllm.platforms.interface")

    original_get_assigned_physical_gpu_ids = object()
    interface_mod.get_assigned_physical_gpu_ids = original_get_assigned_physical_gpu_ids  # type: ignore[attr-defined]
    vars(vllm_platforms)["interface"] = interface_mod

    vllm_ascend = _make_package("vllm_ascend")
    vllm_ascend_utils = types.ModuleType("vllm_ascend.utils")
    original_setup = object()
    vllm_ascend_utils.setup_ascend_local_comm_res = original_setup  # type: ignore[attr-defined]
    vllm_ascend_cpu_binding = types.ModuleType("vllm_ascend.cpu_binding")

    class _DeviceInfo:
        @staticmethod
        def get_npu_map_info() -> list[str]:
            return []

    vllm_ascend_cpu_binding.DeviceInfo = _DeviceInfo  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.platforms", vllm_platforms)
    monkeypatch.setitem(sys.modules, "vllm.platforms.interface", interface_mod)
    monkeypatch.setitem(sys.modules, "vllm_ascend", vllm_ascend)
    monkeypatch.setitem(sys.modules, "vllm_ascend.utils", vllm_ascend_utils)
    monkeypatch.setitem(sys.modules, "vllm_ascend.cpu_binding", vllm_ascend_cpu_binding)

    return {
        "interface": interface_mod,
        "original_get_assigned_physical_gpu_ids": original_get_assigned_physical_gpu_ids,
        "vllm_ascend_utils": vllm_ascend_utils,
        "original_setup": original_setup,
        "_DeviceInfo": _DeviceInfo,
    }


def _load_patch_module(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Any, dict[str, Any]]:
    stubs = _install_dependency_stubs(monkeypatch)

    module_name = f"patch_vllm_ascend_utils_under_test_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, PATCH_PATH)
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)

    return module, stubs


class _FakeKVTransferConfig:
    """Mimics the subset of the kv_transfer_config object used by the patch."""

    def __init__(self, extra_config: dict[str, Any] | None) -> None:
        self.kv_connector_extra_config = extra_config


def _write_endpoint_config(directory: Path, device_id: int, payload: dict[str, Any]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"ub_endpoint_npu_{device_id}.json").write_text(json.dumps(payload), encoding="utf-8")


def _set_assigned_ids(monkeypatch: pytest.MonkeyPatch, module: Any, ids: list[int] | None) -> MagicMock:
    """Replace the local get_assigned_physical_gpu_ids binding in the patch module."""
    mock = MagicMock(return_value=ids)
    monkeypatch.setattr(module, "get_assigned_physical_gpu_ids", mock)
    return mock


class TestPatchVllmAscendUtils:
    """Tests for the setup_ascend_local_comm_res override in patch_vllm_ascend_utils.py."""

    def test_patch_replaces_setup_ascend_local_comm_res(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Importing the patch module replaces setup_ascend_local_comm_res on vllm_ascend.utils."""
        module, stubs = _load_patch_module(monkeypatch)
        vllm_ascend_utils = stubs["vllm_ascend_utils"]

        assert vllm_ascend_utils.setup_ascend_local_comm_res is module.setup_ascend_local_comm_res
        assert vllm_ascend_utils.setup_ascend_local_comm_res is not stubs["original_setup"]

    def test_returns_early_when_kv_transfer_config_is_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When kv_transfer_config is None, no device lookup and no env mutation occur."""
        module, _ = _load_patch_module(monkeypatch)
        monkeypatch.delenv("ASCEND_LOCAL_COMM_RES", raising=False)

        module.setup_ascend_local_comm_res(0, None)

        assert "ASCEND_LOCAL_COMM_RES" not in os.environ

    def test_returns_early_when_local_comm_res_path_is_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When ascend_local_comm_res_path is absent, the function exits without touching the env."""
        module, _ = _load_patch_module(monkeypatch)
        monkeypatch.delenv("ASCEND_LOCAL_COMM_RES", raising=False)
        _set_assigned_ids(monkeypatch, module, [0])
        config = _FakeKVTransferConfig(extra_config={})

        module.setup_ascend_local_comm_res(0, config)

        assert "ASCEND_LOCAL_COMM_RES" not in os.environ

    def test_loads_endpoint_config_and_sets_env_var_with_assigned_devices(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Happy path: reads endpoint JSON from the directory and sets ASCEND_LOCAL_COMM_RES compactly."""
        module, _ = _load_patch_module(monkeypatch)
        monkeypatch.delenv("ASCEND_LOCAL_COMM_RES", raising=False)

        endpoint_dir = tmp_path / "endpoints"
        payload = {
            "device_id": 2,
            "host_ip": "10.0.0.2",
            "rank_table": {"version": "1.0", "items": []},
        }
        _write_endpoint_config(endpoint_dir, device_id=2, payload=payload)
        _set_assigned_ids(monkeypatch, module, [2])
        config = _FakeKVTransferConfig(extra_config={"ascend_local_comm_res_path": str(endpoint_dir)})

        module.setup_ascend_local_comm_res(0, config)

        encoded = os.environ["ASCEND_LOCAL_COMM_RES"]
        assert json.loads(encoded) == payload
        assert ", " not in encoded
        assert ": " not in encoded

    def test_falls_back_to_ascend_rt_visible_devices_when_assigned_ids_missing(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """When get_assigned_physical_gpu_ids returns None, ASCEND_RT_VISIBLE_DEVICES is used."""
        module, _ = _load_patch_module(monkeypatch)
        monkeypatch.delenv("ASCEND_LOCAL_COMM_RES", raising=False)
        monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "1,2,3")

        _set_assigned_ids(monkeypatch, module, None)
        endpoint_dir = tmp_path / "endpoints"
        payload = {"device_id": 1, "host_ip": "10.0.0.1"}
        _write_endpoint_config(endpoint_dir, device_id=1, payload=payload)

        config = _FakeKVTransferConfig(extra_config={"ascend_local_comm_res_path": str(endpoint_dir)})

        module.setup_ascend_local_comm_res(0, config)

        assert json.loads(os.environ["ASCEND_LOCAL_COMM_RES"]) == payload

    def test_falls_back_to_device_info_when_env_and_assigned_ids_absent(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """When neither assigned IDs nor ASCEND_RT_VISIBLE_DEVICES exist, DeviceInfo.get_npu_map_info is used."""
        module, stubs = _load_patch_module(monkeypatch)
        monkeypatch.delenv("ASCEND_LOCAL_COMM_RES", raising=False)
        monkeypatch.delenv("ASCEND_RT_VISIBLE_DEVICES", raising=False)

        _set_assigned_ids(monkeypatch, module, None)
        device_info_cls = stubs["_DeviceInfo"]
        device_info_cls.get_npu_map_info = MagicMock(return_value=["7", "5", "9"])  # type: ignore[assignment]

        endpoint_dir = tmp_path / "endpoints"
        payload = {"device_id": 5, "host_ip": "10.0.0.5"}
        _write_endpoint_config(endpoint_dir, device_id=5, payload=payload)

        config = _FakeKVTransferConfig(extra_config={"ascend_local_comm_res_path": str(endpoint_dir)})

        module.setup_ascend_local_comm_res(0, config)

        device_info_cls.get_npu_map_info.assert_called_once_with()
        assert json.loads(os.environ["ASCEND_LOCAL_COMM_RES"]) == payload

    def test_raises_value_error_when_local_rank_is_out_of_bounds(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A local_rank outside the device list raises ValueError."""
        module, _ = _load_patch_module(monkeypatch)
        monkeypatch.delenv("ASCEND_LOCAL_COMM_RES", raising=False)

        endpoint_dir = tmp_path / "endpoints"
        _write_endpoint_config(endpoint_dir, device_id=0, payload={"device_id": 0})

        _set_assigned_ids(monkeypatch, module, [0, 1])
        config = _FakeKVTransferConfig(extra_config={"ascend_local_comm_res_path": str(endpoint_dir)})

        with pytest.raises(ValueError, match="local_rank 5 is out of bounds"):
            module.setup_ascend_local_comm_res(5, config)

        assert "ASCEND_LOCAL_COMM_RES" not in os.environ

    def test_raises_filenotfounderror_when_endpoint_file_missing(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """If the per-device endpoint JSON file does not exist, FileNotFoundError is raised with a helpful message."""
        module, _ = _load_patch_module(monkeypatch)
        monkeypatch.delenv("ASCEND_LOCAL_COMM_RES", raising=False)

        endpoint_dir = tmp_path / "missing"
        endpoint_dir.mkdir(parents=True, exist_ok=True)
        _set_assigned_ids(monkeypatch, module, [4])
        config = _FakeKVTransferConfig(extra_config={"ascend_local_comm_res_path": str(endpoint_dir)})

        with pytest.raises(FileNotFoundError, match="Endpoint config file not found"):
            module.setup_ascend_local_comm_res(0, config)

        assert "ASCEND_LOCAL_COMM_RES" not in os.environ
