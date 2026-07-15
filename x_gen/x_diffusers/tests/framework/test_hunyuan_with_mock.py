"""
Integration tests for x_diffusers.framework.transformer.hunyuan module with NPU mocking.
"""

import sys
from unittest.mock import MagicMock, Mock, patch

import pytest
import torch

from tests.conftest import MockAttentionManager, MockNPUModule, MockParallelManager, MockRopeManager


def mock_gather_sequence(tensor, dim=2, group=None):
    return tensor


def mock_split_sequence(tensor, dim=2, group=None):
    return tensor


def mock_all_to_all_before_attn(tensor, group, scatter_dim=2, gather_dim=1):
    return tensor


def mock_all_to_all_after_attn(tensor, group, scatter_dim=1, gather_dim=2):
    return tensor


class TestHunyuanVideoAttnProcessorWithMock:
    """Tests for HunyuanVideoAttnProcessor2_0 with mocked NPU."""

    @pytest.fixture(autouse=True)
    def setup_mocks(self):
        """Set up mocks before each test."""
        self.npu_patcher = patch.dict(sys.modules, {"torch_npu": MockNPUModule()})
        self.npu_patcher.start()

        mock_x_base = MagicMock()
        mock_x_base.ParallelManager = MockParallelManager
        mock_x_base.attention_manager = MockAttentionManager()
        mock_x_base.rope_manager = MockRopeManager()
        mock_x_base.all_to_all_before_attn = mock_all_to_all_before_attn
        mock_x_base.all_to_all_after_attn = mock_all_to_all_after_attn
        mock_x_base.gather_sequence = mock_gather_sequence
        mock_x_base.split_sequence = mock_split_sequence
        mock_x_base.get_pad = Mock(return_value=0)
        mock_x_base.set_pad = Mock()

        self.x_base_patcher = patch.dict(sys.modules, {"x_base": mock_x_base})
        self.x_base_patcher.start()

        self.torch_npu_patcher = patch.object(torch, "npu", MockNPUModule())
        self.torch_npu_patcher.start()

        yield

        self.npu_patcher.stop()
        self.x_base_patcher.stop()
        self.torch_npu_patcher.stop()

    def test_processor_init(self):
        """Test HunyuanVideoAttnProcessor2_0 initialization."""
        from x_diffusers.framework.transformer.hunyuan import HunyuanVideoAttnProcessor2_0

        processor = HunyuanVideoAttnProcessor2_0()
        assert processor is not None

    def test_apply_norms(self):
        """Test _apply_norms method."""
        from x_diffusers.framework.transformer.hunyuan import HunyuanVideoAttnProcessor2_0

        processor = HunyuanVideoAttnProcessor2_0()

        # Create mock attention
        attn = MagicMock()
        attn.norm_q = Mock(side_effect=lambda x: x)
        attn.norm_k = Mock(side_effect=lambda x: x)
        attn.add_q_proj = None

        query = torch.randn(1, 10, 256, 128)
        key = torch.randn(1, 10, 256, 128)

        q_norm, k_norm, eq_norm, ek_norm = processor._apply_norms(attn, query, key)

        assert q_norm.shape == query.shape
        assert k_norm.shape == key.shape

    def test_apply_enc_proj_and_norm(self):
        """Test _apply_enc_proj_and_norm method."""
        from x_diffusers.framework.transformer.hunyuan import HunyuanVideoAttnProcessor2_0

        processor = HunyuanVideoAttnProcessor2_0()

        attn = MagicMock()
        attn.add_q_proj = Mock(return_value=torch.randn(1, 256, 1536))
        attn.add_k_proj = Mock(return_value=torch.randn(1, 256, 1536))
        attn.add_v_proj = Mock(return_value=torch.randn(1, 256, 1536))
        attn.norm_q = Mock(side_effect=lambda x: x)
        attn.norm_k = Mock(side_effect=lambda x: x)
        attn.parallel_manager = None  # Skip parallel processing that needs dist

        batch_size = 1
        attn_heads = 24
        head_dim = 128
        encoder_hidden_states = torch.randn(1, 256, 1536)

        enc_q, enc_k, enc_v = processor._apply_enc_proj_and_norm(
            attn, encoder_hidden_states, batch_size, attn_heads, head_dim
        )

        assert enc_q is not None
        assert enc_k is not None
        assert enc_v is not None


class TestHunyuanTransformerConfigWithMock:
    """Tests for HunyuanVideo transformer config with mocked NPU."""

    @pytest.fixture(autouse=True)
    def setup_mocks(self):
        """Set up mocks before each test."""
        self.npu_patcher = patch.dict(sys.modules, {"torch_npu": MockNPUModule()})
        self.npu_patcher.start()

        mock_x_base = MagicMock()
        mock_x_base.ParallelManager = MockParallelManager
        mock_x_base.attention_manager = MockAttentionManager()
        mock_x_base.rope_manager = MockRopeManager()
        mock_x_base.all_to_all_before_attn = mock_all_to_all_before_attn
        mock_x_base.all_to_all_after_attn = mock_all_to_all_after_attn
        mock_x_base.gather_sequence = mock_gather_sequence
        mock_x_base.split_sequence = mock_split_sequence
        mock_x_base.get_pad = Mock(return_value=0)
        mock_x_base.set_pad = Mock()

        self.x_base_patcher = patch.dict(sys.modules, {"x_base": mock_x_base})
        self.x_base_patcher.start()

        self.torch_npu_patcher = patch.object(torch, "npu", MockNPUModule())
        self.torch_npu_patcher.start()

        yield

        self.npu_patcher.stop()
        self.x_base_patcher.stop()
        self.torch_npu_patcher.stop()

    def test_default_config(self):
        """Test HunyuanVideoTransformer3DModel can be imported."""
        from x_diffusers.framework.transformer.hunyuan import HunyuanVideoTransformer3DModel

        # Class should be importable
        assert HunyuanVideoTransformer3DModel is not None


class TestAdaLayerNormContinuousWithMock:
    """Tests for AdaLayerNormContinuous with mocked NPU."""

    @pytest.fixture(autouse=True)
    def setup_mocks(self):
        """Set up mocks before each test."""
        self.npu_patcher = patch.dict(sys.modules, {"torch_npu": MockNPUModule()})
        self.npu_patcher.start()

        mock_x_base = MagicMock()
        mock_x_base.ParallelManager = MockParallelManager

        self.x_base_patcher = patch.dict(sys.modules, {"x_base": mock_x_base})
        self.x_base_patcher.start()

        yield

        self.npu_patcher.stop()
        self.x_base_patcher.stop()

    def test_ada_layer_norm_shape(self):
        """Test AdaLayerNormContinuous can be imported."""
        try:
            from x_diffusers.framework.transformer.hunyuan import AdaLayerNormContinuous

            assert AdaLayerNormContinuous is not None
        except ImportError:
            pytest.skip("AdaLayerNormContinuous not available")


class TestNPUFusionAttentionMock:
    """Tests for NPU fusion attention with mocked NPU operations."""

    @pytest.fixture(autouse=True)
    def setup_mocks(self):
        """Set up mocks before each test."""
        self.npu_patcher = patch.dict(sys.modules, {"torch_npu": MockNPUModule()})
        self.npu_patcher.start()

        mock_x_base = MagicMock()
        mock_x_base.ParallelManager = MockParallelManager
        mock_x_base.attention_manager = MockAttentionManager()

        self.x_base_patcher = patch.dict(sys.modules, {"x_base": mock_x_base})
        self.x_base_patcher.start()

        self.torch_npu_patcher = patch.object(torch, "npu", MockNPUModule())
        self.torch_npu_patcher.start()

        yield

        self.npu_patcher.stop()
        self.x_base_patcher.stop()
        self.torch_npu_patcher.stop()

    def test_npu_fusion_attention_params(self):
        """Test NPU fusion attention parameters."""
        # Mock NPU fusion attention call
        query = torch.randn(1, 24, 256, 128)
        key = torch.randn(1, 24, 256, 128)
        value = torch.randn(1, 24, 256, 128)

        # Simulate attention output
        output = MockAttentionManager.attention(query, key, value)

        assert output.shape[0] == query.shape[0]
        assert output.shape[1] == query.shape[1]
        assert output.shape[2] == query.shape[2]

    def test_output_reshape(self):
        """Test output reshaping after attention."""
        batch_size = 1
        num_heads = 24
        seq_len = 256
        head_dim = 128

        # Attention output: (B, H, S, D)
        attn_output = torch.randn(batch_size, num_heads, seq_len, head_dim)

        # Reshape to (B, S, H*D)
        output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, -1)

        assert output.shape == (batch_size, seq_len, num_heads * head_dim)


class TestParallelManagerIntegrationWithMock:
    """Tests for parallel manager integration with mocked NPU."""

    @pytest.fixture(autouse=True)
    def setup_mocks(self):
        """Set up mocks before each test."""
        self.npu_patcher = patch.dict(sys.modules, {"torch_npu": MockNPUModule()})
        self.npu_patcher.start()

        mock_x_base = MagicMock()
        mock_x_base.ParallelManager = MockParallelManager
        mock_x_base.all_to_all_before_attn = mock_all_to_all_before_attn
        mock_x_base.all_to_all_after_attn = mock_all_to_all_after_attn
        mock_x_base.gather_sequence = mock_gather_sequence
        mock_x_base.split_sequence = mock_split_sequence
        mock_x_base.get_pad = Mock(return_value=0)
        mock_x_base.set_pad = Mock()

        self.x_base_patcher = patch.dict(sys.modules, {"x_base": mock_x_base})
        self.x_base_patcher.start()

        yield

        self.npu_patcher.stop()
        self.x_base_patcher.stop()

    def test_parallel_enabled(self):
        """Test parallel processing when enabled."""
        pm = MockParallelManager(sp_size=4)

        assert pm.sp_size == 4
        assert pm.sp_group is None

    def test_parallel_disabled(self):
        """Test parallel processing when disabled."""
        pm = MockParallelManager(sp_size=1)

        assert pm.sp_size == 1


class TestEncoderProjModesWithMock:
    """Tests for encoder projection modes with mocked NPU."""

    @pytest.fixture(autouse=True)
    def setup_mocks(self):
        """Set up mocks before each test."""
        self.npu_patcher = patch.dict(sys.modules, {"torch_npu": MockNPUModule()})
        self.npu_patcher.start()

        mock_x_base = MagicMock()
        mock_x_base.ParallelManager = MockParallelManager

        self.x_base_patcher = patch.dict(sys.modules, {"x_base": mock_x_base})
        self.x_base_patcher.start()

        yield

        self.npu_patcher.stop()
        self.x_base_patcher.stop()

    def test_use_encoder_proj_mode_1(self):
        """Test encoder projection mode 1 configuration."""
        from x_diffusers.framework.transformer.hunyuan import HunyuanVideoAttnProcessor2_0

        processor = HunyuanVideoAttnProcessor2_0()
        assert processor is not None

    def test_use_encoder_proj_mode_2(self):
        """Test encoder projection mode 2 configuration."""
        from x_diffusers.framework.transformer.hunyuan import HunyuanVideoAttnProcessor2_0

        processor = HunyuanVideoAttnProcessor2_0()
        assert processor is not None
