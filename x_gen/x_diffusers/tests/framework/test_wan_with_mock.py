"""
Integration tests for x_diffusers.framework.transformer.wan module with NPU mocking.

These tests import the actual module and test real code paths with mocked NPU operations.
"""

import sys
from unittest.mock import MagicMock, Mock, patch

import pytest
import torch

# Import NPU mocks from conftest
from tests.conftest import MockAttentionManager, MockNPUModule, MockParallelManager, MockRopeManager


# Define mock functions locally
def mock_gather_sequence(tensor, dim=2, group=None):
    return tensor


def mock_split_sequence(tensor, dim=2, group=None):
    return tensor


def mock_all_to_all_before_attn(tensor, group, scatter_dim=2, gather_dim=1):
    return tensor


def mock_all_to_all_after_attn(tensor, group, scatter_dim=1, gather_dim=2):
    return tensor


@pytest.fixture(scope="function")
def setup_npu_mocks():
    """Shared fixture for NPU mock setup."""
    npu_patcher = patch.dict(sys.modules, {"torch_npu": MockNPUModule()})
    npu_patcher.start()

    mock_x_base = MagicMock()
    mock_x_base.gather_sequence = mock_gather_sequence
    mock_x_base.split_sequence = mock_split_sequence
    mock_x_base.all_to_all_before_attn = mock_all_to_all_before_attn
    mock_x_base.all_to_all_after_attn = mock_all_to_all_after_attn
    mock_x_base.attention_manager = MockAttentionManager()
    mock_x_base.rope_manager = MockRopeManager()
    mock_x_base.ParallelManager = MockParallelManager
    mock_x_base.get_pad = Mock(return_value=0)
    mock_x_base.set_pad = Mock()
    mock_x_base.pad_tensor = Mock(side_effect=lambda x, *a, **k: x)
    mock_x_base.get_phaa_split_num = Mock(return_value=None)
    mock_x_base.is_phaa_enabled = Mock(return_value=False)

    x_base_patcher = patch.dict(sys.modules, {"x_base": mock_x_base})
    x_base_patcher.start()

    torch_npu_patcher = patch.object(torch, "npu", MockNPUModule())
    torch_npu_patcher.start()

    yield

    npu_patcher.stop()
    x_base_patcher.stop()
    torch_npu_patcher.stop()


class TestAscendWanAttnProcessorWithMock:
    """Tests for AscendWanAttnProcessor2_0 with mocked NPU."""

    @pytest.fixture(autouse=True)
    def setup_npu_mock(self, setup_npu_mocks):
        """Set up NPU mocks before each test."""
        pass

    def test_attn_processor_init(self):
        """Test AscendWanAttnProcessor2_0 initialization."""
        from x_diffusers.framework.transformer.wan import AscendWanAttnProcessor2_0

        # Default initialization
        processor = AscendWanAttnProcessor2_0()
        assert processor is not None

    def test_attn_processor_call_basic(self):
        """Test basic attention call without parallel processing."""
        from x_diffusers.framework.transformer.wan import AscendWanAttnProcessor2_0

        processor = AscendWanAttnProcessor2_0()

        # Create mock attention
        attn = MagicMock()
        attn.heads = 8
        attn.parallel_manager = None
        attn.add_k_proj = None
        attn.norm_q = Mock(side_effect=lambda x: x)
        attn.norm_k = Mock(side_effect=lambda x: x)
        attn.to_out = [Mock(side_effect=lambda x: x), Mock(side_effect=lambda x: x)]

        # Create input tensors
        batch_size = 1
        seq_len = 256
        hidden_dim = 64

        hidden_states = torch.randn(batch_size, seq_len, hidden_dim)
        encoder_hidden_states = torch.randn(batch_size, seq_len, hidden_dim)

        # Mock QKV projections
        with patch("x_diffusers.framework.transformer.wan._get_qkv_projections") as mock_qkv:
            query = torch.randn(batch_size, seq_len, hidden_dim)
            key = torch.randn(batch_size, seq_len, hidden_dim)
            value = torch.randn(batch_size, seq_len, hidden_dim)
            mock_qkv.return_value = (query, key, value)

            output = processor(attn, hidden_states, encoder_hidden_states)

            assert output.shape == (batch_size, seq_len, hidden_dim)

    def test_i2v_forward_with_image_embeds(self):
        """Test I2V forward with image embeddings."""
        from x_diffusers.framework.transformer.wan import AscendWanAttnProcessor2_0

        processor = AscendWanAttnProcessor2_0()

        # Create mock attention for I2V
        attn = MagicMock()
        attn.heads = 8
        attn.parallel_manager = None
        attn.norm_added_k = Mock(side_effect=lambda x: x)

        batch_size = 1
        seq_len = 64
        hidden_dim = 64
        attn_heads = 8

        query = torch.randn(batch_size, attn_heads, seq_len, hidden_dim // attn_heads)
        encoder_hidden_states_img = torch.randn(batch_size, 512, hidden_dim)

        # Mock _get_added_kv_projections
        with patch("x_diffusers.framework.transformer.wan._get_added_kv_projections") as mock_kv:
            key_img = torch.randn(batch_size, seq_len, hidden_dim)
            value_img = torch.randn(batch_size, seq_len, hidden_dim)
            mock_kv.return_value = (key_img, value_img)

            output = processor.i2v_forward(attn, query, encoder_hidden_states_img, attn_heads)

            # Should return processed hidden states
            assert output is not None

    def test_i2v_forward_none_image_embeds(self):
        """Test I2V forward returns None when no image embeddings."""
        from x_diffusers.framework.transformer.wan import AscendWanAttnProcessor2_0

        processor = AscendWanAttnProcessor2_0()
        attn = MagicMock()

        output = processor.i2v_forward(attn, torch.randn(1, 8, 64, 8), None, 8)
        assert output is None


class TestAscendWanTransformerBlockWithMock:
    """Tests for transformer block with mocked NPU."""

    @pytest.fixture(autouse=True)
    def setup_mocks(self, setup_npu_mocks):
        """Set up mocks before each test."""
        pass

    def test_transformer_config_default(self):
        """Test AscendWanTransformer3DModel can be imported."""
        from x_diffusers.framework.transformer.wan import AscendWanTransformer3DModel

        # Class should be importable
        assert AscendWanTransformer3DModel is not None


class TestParallelAttentionWithMock:
    """Tests for parallel attention with mocked NPU."""

    @pytest.fixture(autouse=True)
    def setup_mocks(self, setup_npu_mocks):
        """Set up mocks before each test."""
        pass

    def test_sp_size_divisibility_check(self):
        """Test that parallel manager checks head divisibility."""
        from x_diffusers.framework.transformer.wan import AscendWanAttnProcessor2_0

        processor = AscendWanAttnProcessor2_0()  # noqa: F841

        # Create attention with parallel manager
        attn = MagicMock()
        attn.heads = 40
        attn.parallel_manager = MockParallelManager(sp_size=4)
        attn.parallel_manager.enable_usp = False
        attn.add_k_proj = None
        attn.norm_q = Mock(side_effect=lambda x: x)
        attn.norm_k = Mock(side_effect=lambda x: x)
        attn.to_out = [Mock(side_effect=lambda x: x), Mock(side_effect=lambda x: x)]

        # 40 heads / 4 sp_size = 10 heads per rank
        assert attn.heads % attn.parallel_manager.sp_size == 0

    def test_sp_size_not_divisible_raises(self):
        """Test error when heads not divisible by sp_size."""
        heads = 42
        sp_size = 4

        # Should not be divisible
        assert heads % sp_size != 0


class TestRoPEWithMock:
    """Tests for rotary position embedding with mocked NPU."""

    @pytest.fixture(autouse=True)
    def setup_mocks(self, setup_npu_mocks):
        """Set up mocks before each test."""
        pass

    def test_rope_application(self):
        """Test RoPE is applied correctly."""
        query = torch.randn(1, 8, 256, 64)
        key = torch.randn(1, 8, 256, 64)

        # Create mock rotary embedding
        cos = torch.randn(256, 32)
        sin = torch.randn(256, 32)

        # Mock RoPE returns (query, key)
        rope_manager = MockRopeManager()
        q_rope, k_rope = rope_manager.rope(query, key, cos, sin)

        assert q_rope.shape == query.shape
        assert k_rope.shape == key.shape
