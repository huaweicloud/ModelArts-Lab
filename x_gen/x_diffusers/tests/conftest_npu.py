"""
NPU Mock Fixtures for x_diffusers unit tests.

This module provides comprehensive mocks for:
- torch_npu module and its submodules
- x_base parallel computing functions
- NPU-specific tensor operations

Usage:
    Import this in conftest.py or use pytest_plugins to auto-load.
"""

import sys
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest
import torch

# ============================================================
# Mock Classes
# ============================================================


class MockNPUStream:
    """Mock for torch.npu.Stream"""

    def __init__(self, device=None, priority=0):
        self.device = device
        self.priority = priority

    def wait_stream(self, other_stream):
        pass

    def synchronize(self):
        pass


class MockNPUEvent:
    """Mock for torch.npu.Event"""

    def __init__(self, enable_timing=False, blocking=False, interprocess=False):
        self.enable_timing = enable_timing
        self.blocking = blocking
        self.interprocess = interprocess
        self._recorded = False

    def record(self, stream=None):
        self._recorded = True
        self.stream = stream  # 记录关联的流

    def wait(self, stream=None):
        pass

    def query(self):
        return self._recorded

    def elapsed_time(self, end_event):
        return 0.0

    def synchronize(self):
        pass


class MockNPUModule:
    """Mock for torch_npu module"""

    def __init__(self):
        self._streams = {}
        self._current_stream = MockNPUStream()

    def Stream(self, device=None, priority=0):
        return MockNPUStream(device, priority)

    def Event(self, enable_timing=False, blocking=False, interprocess=False):
        return MockNPUEvent(enable_timing, blocking, interprocess)

    def current_stream(self, device=None):
        if device is None:
            device = self._current_stream.device
        if device not in self._streams:
            self._streams[device] = MockNPUStream(device=device)
        return self._streams[device]

    def synchronize(self, device=None):
        pass

    def is_available(self):
        return True

    def device_count(self):
        return 8  # Simulate 8 NPU devices

    def set_device(self, device):
        pass

    def get_device_name(self, device=None):
        return "Ascend910B"

    def get_device_capability(self, device=None):
        return (9, 10)

    def memory_allocated(self, device=None):
        return 0

    def memory_reserved(self, device=None):
        return 0

    def max_memory_allocated(self, device=None):
        return 0

    def max_memory_reserved(self, device=None):
        return 0

    def empty_cache(self):
        pass


class MockAttentionManager:
    """Mock for x_base.attention_manager"""

    @staticmethod
    def attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, **kwargs):
        """Mock attention computation using standard scaled dot product."""
        # Simple scaled dot product attention for testing
        d_k = query.shape[-1]
        scores = torch.matmul(query, key.transpose(-2, -1)) / (d_k**0.5)
        if attn_mask is not None:
            scores = scores + attn_mask
        attn_weights = torch.softmax(scores, dim=-1)
        if dropout_p > 0:
            attn_weights = torch.nn.functional.dropout(attn_weights, p=dropout_p)
        output = torch.matmul(attn_weights, value)
        return output


class MockRopeManager:
    """Mock for x_base.rope_manager"""

    @staticmethod
    def rope(query, key, cos, sin, *args, **kwargs):
        """Apply rotary position embedding (identity for mock)."""
        # In real implementation, this applies rotation
        # For mock, return unchanged tensors
        return query, key


class MockParallelManager:
    """Mock for x_base.ParallelManager"""

    def __init__(self, sp_size=1, sp_group=None, enable_usp=False):
        self.sp_size = sp_size
        self.sp_group = sp_group
        self.enable_usp = enable_usp
        self.ulysses_degree = 1
        self.ring_degree = 1


class MockDistModule:
    """Mock for torch.distributed"""

    @staticmethod
    def is_initialized():
        return False

    @staticmethod
    def get_world_size(group=None):
        return 1

    @staticmethod
    def get_rank(group=None):
        return 0

    @staticmethod
    def all_reduce(tensor, op=None, group=None, async_op=False):
        return tensor

    @staticmethod
    def all_gather(tensor_list, tensor, group=None, async_op=False):
        # 模拟分布式all_gather操作：将输入tensor添加到tensor_list中
        # 在单进程模式下，tensor_list[0] = tensor
        if len(tensor_list) > 0:
            tensor_list[0].copy_(tensor)
        return None

    @staticmethod
    def broadcast(tensor, src, group=None, async_op=False):
        # 模拟分布式broadcast操作：在单进程模式下，tensor保持不变
        # 因为只有一个进程，所以tensor已经是正确的值
        return tensor

    @staticmethod
    def barrier(group=None):
        pass


# ============================================================
# Global State Mocks
# ============================================================

_phaa_enabled = False
_phaa_split_num = None
_pad_value = 0


def set_phaa_enabled(enabled: bool):
    global _phaa_enabled
    _phaa_enabled = enabled


def set_phaa_split_num(num: int | None):
    global _phaa_split_num
    _phaa_split_num = num


def set_pad_value(value: int):
    global _pad_value
    _pad_value = value


# ============================================================
# x_base Function Mocks
# ============================================================


def mock_gather_sequence(tensor, dim=2, group=None):
    """Mock gather_sequence - identity for single process."""
    return tensor


def mock_split_sequence(tensor, dim=2, group=None):
    """Mock split_sequence - identity for single process."""
    return tensor


def mock_all_to_all_before_attn(tensor, group, scatter_dim=2, gather_dim=1):
    """Mock all_to_all communication - identity for single process."""
    return tensor


def mock_all_to_all_after_attn(tensor, group, scatter_dim=1, gather_dim=2):
    """Mock all_to_all communication - identity for single process."""
    return tensor


def mock_get_pad(key="pad"):
    """Mock get_pad."""
    global _pad_value
    return _pad_value


def mock_set_pad(value, key="pad"):
    """Mock set_pad."""
    global _pad_value
    _pad_value = value


def mock_pad_tensor(tensor, pad_size, dim=2):
    """Mock pad_tensor."""
    if pad_size <= 0:
        return tensor
    pad_shape = list(tensor.shape)
    pad_shape[dim] = pad_size
    padding = torch.zeros(pad_shape, dtype=tensor.dtype, device=tensor.device)
    return torch.cat([tensor, padding], dim=dim)


def mock_is_phaa_enabled():
    """Mock is_phaa_enabled."""
    return _phaa_enabled


def mock_get_phaa_split_num():
    """Mock get_phaa_split_num."""
    return _phaa_split_num


# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def mock_torch_npu():
    """Fixture to mock torch_npu module."""
    with patch.dict(sys.modules, {"torch_npu": MockNPUModule()}):  # noqa: SIM117
        with patch.dict(sys.modules, {"torch_npu.contrib": MagicMock()}):
            with patch.dict(sys.modules, {"torch_npu.contrib.transfer_to_npu": MagicMock()}):
                yield sys.modules["torch_npu"]


@pytest.fixture
def mock_torch_npu_on_tensor():
    """Fixture to mock torch.npu on tensor objects."""
    mock_npu = MockNPUModule()

    # Create a mock that can be accessed via torch.npu
    with patch.object(torch, "npu", mock_npu):
        yield mock_npu


@pytest.fixture
def mock_x_base():
    """Fixture to mock x_base module functions."""
    mock_module = MagicMock()

    # Assign mock functions
    mock_module.gather_sequence = mock_gather_sequence
    mock_module.split_sequence = mock_split_sequence
    mock_module.all_to_all_before_attn = mock_all_to_all_before_attn
    mock_module.all_to_all_after_attn = mock_all_to_all_after_attn
    mock_module.get_pad = mock_get_pad
    mock_module.set_pad = mock_set_pad
    mock_module.pad_tensor = mock_pad_tensor
    mock_module.is_phaa_enabled = mock_is_phaa_enabled
    mock_module.get_phaa_split_num = mock_get_phaa_split_num
    mock_module.ParallelManager = MockParallelManager

    # Create singleton managers
    mock_module.attention_manager = MockAttentionManager()
    mock_module.rope_manager = MockRopeManager()

    with patch.dict(sys.modules, {"x_base": mock_module}):
        yield mock_module


@pytest.fixture
def mock_distributed():
    """Fixture to mock torch.distributed for single process."""
    with patch.object(torch, "distributed", MockDistModule):
        yield MockDistModule


@pytest.fixture
def mock_all_npu(mock_torch_npu, mock_x_base, mock_distributed):
    """Combined fixture that mocks all NPU-related modules."""
    return {"torch_npu": mock_torch_npu, "x_base": mock_x_base, "distributed": mock_distributed}


@pytest.fixture
def enable_phaa():
    """Fixture to enable PHAA mode."""
    set_phaa_enabled(True)
    yield
    set_phaa_enabled(False)


# ============================================================
# Context Managers for Selective Mocking
# ============================================================


@contextmanager
def npu_mock_context():
    """Context manager for NPU mocking in imports."""
    # Save original modules
    original_torch_npu = sys.modules.get("torch_npu")
    original_x_base = sys.modules.get("x_base")

    try:
        # Apply mocks
        sys.modules["torch_npu"] = MockNPUModule()
        mock_x_base_module = MagicMock()
        mock_x_base_module.gather_sequence = mock_gather_sequence
        mock_x_base_module.split_sequence = mock_split_sequence
        mock_x_base_module.all_to_all_before_attn = mock_all_to_all_before_attn
        mock_x_base_module.all_to_all_after_attn = mock_all_to_all_after_attn
        mock_x_base_module.get_pad = mock_get_pad
        mock_x_base_module.set_pad = mock_set_pad
        mock_x_base_module.pad_tensor = mock_pad_tensor
        mock_x_base_module.is_phaa_enabled = mock_is_phaa_enabled
        mock_x_base_module.get_phaa_split_num = mock_get_phaa_split_num
        mock_x_base_module.ParallelManager = MockParallelManager
        mock_x_base_module.attention_manager = MockAttentionManager()
        mock_x_base_module.rope_manager = MockRopeManager()
        sys.modules["x_base"] = mock_x_base_module

        yield {"torch_npu": sys.modules["torch_npu"], "x_base": sys.modules["x_base"]}
    finally:
        # Restore original modules
        if original_torch_npu is not None:
            sys.modules["torch_npu"] = original_torch_npu
        elif "torch_npu" in sys.modules:
            del sys.modules["torch_npu"]

        if original_x_base is not None:
            sys.modules["x_base"] = original_x_base
        elif "x_base" in sys.modules:
            del sys.modules["x_base"]


# ============================================================
# Helper Functions for Tests
# ============================================================


def create_mock_attention_processor(heads=40, head_dim=128, sp_size=1):
    """Create a mock Attention processor for testing."""
    attn = MagicMock()
    attn.heads = heads
    attn.head_dim = head_dim
    attn.parallel_manager = MockParallelManager(sp_size=sp_size) if sp_size > 1 else None

    # Mock projections
    attn.to_q = MagicMock(return_value=torch.randn(1, 100, heads * head_dim))
    attn.to_k = MagicMock(return_value=torch.randn(1, 100, heads * head_dim))
    attn.to_v = MagicMock(return_value=torch.randn(1, 100, heads * head_dim))
    attn.to_out = [
        MagicMock(return_value=torch.randn(1, 100, heads * head_dim)),
        MagicMock(return_value=torch.randn(1, 100, heads * head_dim)),
    ]

    # Mock norms
    attn.norm_q = MagicMock(side_effect=lambda x: x)
    attn.norm_k = MagicMock(side_effect=lambda x: x)

    # Mock added projections
    attn.add_k_proj = None
    attn.add_v_proj = None
    attn.add_q_proj = None
    attn.norm_added_k = MagicMock(side_effect=lambda x: x)

    return attn


def create_mock_transformer_block(hidden_size=5120, heads=40, head_dim=128):
    """Create a mock transformer block for testing."""
    block = MagicMock()
    block.hidden_size = hidden_size
    block.num_heads = heads
    block.head_dim = head_dim

    # Mock layers
    block.norm1 = MagicMock()
    block.norm2 = MagicMock()
    block.attn1 = create_mock_attention_processor(heads, head_dim)
    block.attn2 = create_mock_attention_processor(heads, head_dim)
    block.ff = MagicMock()

    return block
