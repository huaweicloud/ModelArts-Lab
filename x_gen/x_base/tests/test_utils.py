"""
Unit tests for utility functions in x_base.

Tests various utility functions including:
- nearest_interp: nearest neighbor interpolation
- CacheContext: caching context management
- Other helper functions
"""

from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from x_base.turbo.utils import (
    CacheContext,
    are_two_tensors_similar,
    batch_func,
    cache_context,
    create_cache_context,
    get_current_cache_context,
    nearest_interp,
)

# ============================================================
# Pytest markers for dependency management
# ============================================================
# Tests requiring torch are marked with @pytest.mark.requires_torch
# Run with: pytest -m "not requires_torch" to skip torch tests

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ============================================================
# Test data constants
# ============================================================
NEAREST_INTERP_CASES = [
    ([1.0, 2.0, 3.0], 6, 1.0, 3.0),
    ([0.0, 1.0], 5, 0.0, 1.0),
    ([-5.0, 0.0, 5.0], 9, -5.0, 5.0),
]

TENSOR_SIMILARITY_CASES = [
    (0.0, 0.1, True),  # Identical
    (0.05, 0.1, True),  # 5% diff, 10% threshold -> similar
    (0.15, 0.1, False),  # 15% diff, 10% threshold -> not similar
    (0.01, 0.001, False),  # 1% diff, 0.1% threshold -> not similar
]


class TestNearestInterp:
    """Test suite for nearest_interp function."""

    def test_nearest_interp_basic(self):
        """Test basic nearest interpolation."""
        src = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        target_length = 10

        result = nearest_interp(src, target_length)

        assert len(result) == target_length
        assert result[0] == src[0]  # First element preserved
        assert result[-1] == src[-1]  # Last element preserved

    def test_nearest_interp_same_length(self):
        """Test interpolation when target length equals source length."""
        src = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        target_length = 5

        result = nearest_interp(src, target_length)

        assert len(result) == target_length
        np.testing.assert_array_almost_equal(result, src)

    @pytest.mark.parametrize("src_data,target_len,expected_first,expected_last", NEAREST_INTERP_CASES)
    def test_nearest_interp_preserves_endpoints(self, src_data, target_len, expected_first, expected_last):
        """Test that interpolation preserves first and last values."""
        src = np.array(src_data)
        result = nearest_interp(src, target_len)

        assert len(result) == target_len
        assert result[0] == expected_first
        assert result[-1] == expected_last

    def test_nearest_interp_upsampling(self):
        """Test upsampling (target > source)."""
        src = np.array([0.0, 1.0])
        target_length = 5

        result = nearest_interp(src, target_length)

        assert len(result) == target_length
        assert result[0] == 0.0
        assert result[-1] == 1.0

    def test_nearest_interp_downsampling(self):
        """Test downsampling (target < source)."""
        src = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        target_length = 5

        result = nearest_interp(src, target_length)

        assert len(result) == target_length
        assert result[0] == src[0]
        assert result[-1] == src[-1]

    def test_nearest_interp_single_element_source(self):
        """Test interpolation from single element."""
        src = np.array([5.0])
        target_length = 10

        result = nearest_interp(src, target_length)

        assert len(result) == target_length
        np.testing.assert_array_almost_equal(result, np.full(target_length, 5.0))

    def test_nearest_interp_single_target(self):
        """Test interpolation to single element."""
        src = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        target_length = 1

        result = nearest_interp(src, target_length)

        assert len(result) == 1
        assert result[0] == src[-1]

    def test_nearest_interp_preserves_monotonicity(self):
        """Test that interpolation preserves monotonicity."""
        src = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        target_length = 20

        result = nearest_interp(src, target_length)

        # Check monotonicity (allowing equal consecutive values for nearest)
        for i in range(len(result) - 1):
            assert result[i] <= result[i + 1] + 1e-10

    def test_nearest_interp_negative_values(self):
        """Test interpolation with negative values."""
        src = np.array([-5.0, -2.5, 0.0, 2.5, 5.0])
        target_length = 10

        result = nearest_interp(src, target_length)

        assert len(result) == target_length
        assert result[0] == -5.0
        assert result[-1] == 5.0

    @pytest.mark.parametrize("scale_factor", [0.5, 1.0, 2.0, 10.0])
    def test_nearest_interp_various_scales(self, scale_factor):
        """Test interpolation with various scale factors."""
        src = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        target_length = int(len(src) * scale_factor)

        if target_length < 1:
            target_length = 1

        result = nearest_interp(src, target_length)

        assert len(result) == target_length
        assert result[0] == src[0]
        assert result[-1] == src[-1]


class TestCacheContext:
    """Test suite for CacheContext class."""

    def test_cache_context_creation(self):
        """Test CacheContext creation."""
        ctx = CacheContext()

        assert ctx.buffers == {}
        assert len(ctx.incremental_name_counters) == 0

    def test_cache_context_get_incremental_name(self):
        """Test incremental name generation."""
        ctx = CacheContext()

        name1 = ctx.get_incremental_name()
        name2 = ctx.get_incremental_name()
        name3 = ctx.get_incremental_name("custom")

        assert name1 == "default_0"
        assert name2 == "default_1"
        assert name3 == "custom_0"

    @pytest.mark.parametrize(
        "prefix,expected_start",
        [
            (None, "default_"),
            ("custom", "custom_"),
            ("buffer", "buffer_"),
        ],
    )
    def test_cache_context_incremental_name_various_prefixes(self, prefix, expected_start):
        """Test incremental name generation with various prefixes."""
        ctx = CacheContext()
        name = ctx.get_incremental_name(prefix)

        assert name.startswith(expected_start)

    def test_cache_context_reset_incremental_names(self):
        """Test resetting incremental name counters."""
        ctx = CacheContext()

        ctx.get_incremental_name()
        ctx.get_incremental_name()

        assert len(ctx.incremental_name_counters) > 0

        ctx.reset_incremental_names()

        assert len(ctx.incremental_name_counters) == 0

    @pytest.mark.requires_torch
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not available")
    def test_cache_context_buffer_operations(self):
        """Test buffer set/get operations."""
        ctx = CacheContext()
        tensor = torch.randn(10, 10)

        ctx.set_buffer("test_buffer", tensor)
        retrieved = ctx.get_buffer("test_buffer")

        assert torch.equal(retrieved, tensor)

    def test_cache_context_get_nonexistent_buffer(self):
        """Test getting a buffer that doesn't exist."""
        ctx = CacheContext()
        result = ctx.get_buffer("nonexistent")

        assert result is None

    @pytest.mark.requires_torch
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not available")
    def test_cache_context_clear_buffers(self):
        """Test clearing all buffers."""
        ctx = CacheContext()

        ctx.set_buffer("buffer1", torch.randn(5, 5))
        ctx.set_buffer("buffer2", torch.randn(5, 5))

        assert len(ctx.buffers) == 2

        ctx.clear_buffers()

        assert len(ctx.buffers) == 0

    @pytest.mark.requires_torch
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not available")
    def test_cache_context_overwrite_buffer(self):
        """Test overwriting an existing buffer."""
        ctx = CacheContext()

        tensor1 = torch.randn(5, 5)
        tensor2 = torch.randn(5, 5)

        ctx.set_buffer("buffer", tensor1)
        ctx.set_buffer("buffer", tensor2)  # Overwrite

        retrieved = ctx.get_buffer("buffer")
        assert torch.equal(retrieved, tensor2)
        assert not torch.equal(retrieved, tensor1)


class TestBatchFunc:
    """Test suite for batch_func utility."""

    @pytest.mark.requires_torch
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not available")
    def test_batch_func_with_2d_tensors(self):
        """Test batch_func with tensors of shape [2, ...]."""

        def double(x):
            return x * 2

        tensor1 = torch.randn(2, 10)  # Will be processed
        tensor2 = torch.randn(3, 10)  # Won't be processed (shape[0] != 2)
        scalar = 5  # Won't be processed

        result = batch_func(double, tensor1, tensor2, scalar)

        assert torch.equal(result[0], tensor1 * 2)
        assert torch.equal(result[1], tensor2)
        assert result[2] == scalar

    def test_batch_func_with_non_tensors(self):
        """Test batch_func with non-tensor arguments."""

        def transform(x):
            return x * 10

        result = batch_func(transform, 1, 2.0, "string", None)

        # All non-tensor args should pass through
        assert result == [1, 2.0, "string", None]

    @pytest.mark.requires_torch
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not available")
    @pytest.mark.parametrize(
        "batch_size,should_process",
        [
            (2, True),  # batch size 2 should be processed
            (1, False),  # batch size 1 should pass through
            (3, False),  # batch size 3 should pass through
            (4, False),  # batch size 4 should pass through
        ],
    )
    def test_batch_func_batch_size_handling(self, batch_size, should_process):
        """Test batch_func only processes tensors with batch_size=2."""

        def transform(x):
            return x * 10

        tensor = torch.randn(batch_size, 10)
        result = batch_func(transform, tensor)

        if should_process:
            assert torch.equal(result[0], tensor * 10)
        else:
            assert torch.equal(result[0], tensor)


class TestPreForward:
    """Test suite for pre_forward function."""

    def test_pre_forward_no_attention_kwargs(self):
        """Test pre_forward without attention_kwargs."""
        mock_self = MagicMock()
        result = pre_forward(mock_self, None)  # noqa: F821

        assert result == 1.0  # Default lora_scale

    @pytest.mark.parametrize(
        "scale_value,expected",
        [
            (0.5, 0.5),
            (1.0, 1.0),
            (2.0, 2.0),
            (None, 1.0),  # None should default to 1.0
        ],
    )
    def test_pre_forward_with_various_scales(self, scale_value, expected):
        """Test pre_forward with various scale values."""
        mock_self = MagicMock()

        if scale_value is None:
            attention_kwargs = {}
        else:
            attention_kwargs = {"scale": scale_value}

        result = pre_forward(mock_self, attention_kwargs)  # noqa: F821

        assert result == expected

    def test_pre_forward_with_empty_attention_kwargs(self):
        """Test pre_forward with empty attention_kwargs dict."""
        mock_self = MagicMock()
        attention_kwargs = {}

        result = pre_forward(mock_self, attention_kwargs)  # noqa: F821

        assert result == 1.0


class TestCacheContextManager:
    """Test suite for cache context manager."""

    def test_cache_context_manager_basic(self):
        """Test basic cache context manager usage."""
        ctx = create_cache_context()

        with cache_context(ctx):
            current = get_current_cache_context()
            assert current == ctx

    def test_create_cache_context(self):
        """Test cache context creation."""
        ctx = create_cache_context()

        assert ctx is not None
        assert hasattr(ctx, "buffers")
        assert hasattr(ctx, "incremental_name_counters")

    def test_cache_context_manager_nested(self):
        """Test nested cache context managers."""
        ctx1 = create_cache_context()
        ctx2 = create_cache_context()

        with cache_context(ctx1):
            assert get_current_cache_context() == ctx1

            with cache_context(ctx2):
                assert get_current_cache_context() == ctx2

            # Should restore to ctx1 after inner context
            assert get_current_cache_context() == ctx1


class TestTensorSimilarity:
    """Test suite for tensor similarity functions."""

    @pytest.mark.requires_torch
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not available")
    def test_are_two_tensors_similar_identical(self):
        """Test similarity check for identical tensors."""
        t1 = torch.randn(10, 10)
        t2 = t1.clone()

        result = are_two_tensors_similar(t1, t2, threshold=0.1)

        assert result is True

    @pytest.mark.requires_torch
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not available")
    def test_are_two_tensors_similar_different(self):
        """Test similarity check for different tensors."""
        t1 = torch.zeros(10, 10)
        t2 = torch.ones(10, 10) * 100  # Very different

        result = are_two_tensors_similar(t1, t2, threshold=0.1)
        assert result is False

    @pytest.mark.requires_torch
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not available")
    @pytest.mark.parametrize("diff_percent,threshold,expected_similar", TENSOR_SIMILARITY_CASES)
    def test_are_two_tensors_similar_various_cases(self, diff_percent, threshold, expected_similar):
        """Test similarity check with various difference/threshold combinations."""
        t1 = torch.ones(100, 100)
        t2 = torch.ones(100, 100) * (1.0 + diff_percent)

        result = are_two_tensors_similar(t1, t2, threshold=threshold)

        assert result == expected_similar, (
            f"diff={diff_percent}, threshold={threshold}: expected {expected_similar}, got {result}"
        )

    @pytest.mark.requires_torch
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not available")
    def test_are_two_tensors_similar_threshold_sensitivity(self):
        """Test similarity check with different thresholds."""
        t1 = torch.ones(10, 10)
        t2 = torch.ones(10, 10) * 1.05  # 5% difference

        # With high threshold, should be similar
        result_high = are_two_tensors_similar(t1, t2, threshold=0.1)
        assert result_high is True

        # With low threshold, should be different
        result_low = are_two_tensors_similar(t1, t2, threshold=0.01)
        assert result_low is False
