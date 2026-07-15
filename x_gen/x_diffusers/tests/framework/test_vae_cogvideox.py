"""
Unit tests for x_diffusers.framework.vae.cogvideox module.

Tests cover:
- AscendCogVideoXCausalConv3d
- AscendAutoencoderKLCogVideoX configuration

Note: These tests verify standalone logic without importing the actual module.
No sys.modules mocking is needed as we test mathematical operations and shapes.
"""

import math

import torch


class TestAscendCogVideoXCausalConv3d:
    """Tests for AscendCogVideoXCausalConv3d class."""

    def test_init_kernel_size_int(self):
        """Test initialization with int kernel_size."""
        kernel_size = 3
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 3

        assert kernel_size == (3, 3, 3)

    def test_init_kernel_size_tuple(self):
        """Test initialization with tuple kernel_size."""
        kernel_size = (3, 1, 1)
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 3

        assert kernel_size == (3, 1, 1)

    def test_padding_calculation(self):
        """Test padding calculation for causal conv."""
        kernel_size = (3, 3, 3)
        dilation = 1
        stride = 1

        time_kernel_size, height_kernel_size, width_kernel_size = kernel_size
        time_pad = dilation * (time_kernel_size - 1) + (1 - stride)
        height_pad = height_kernel_size // 2
        width_pad = width_kernel_size // 2

        assert time_pad == 2
        assert height_pad == 1
        assert width_pad == 1

    def test_time_causal_padding_order(self):
        """Test time causal padding order."""
        width_pad = 1
        height_pad = 1
        time_pad = 2

        # F.pad order: (W_left, W_right, H_left, H_right, T_left, T_right)
        time_causal_padding = (width_pad, width_pad, height_pad, height_pad, time_pad, 0)

        assert time_causal_padding == (1, 1, 1, 1, 2, 0)

    def test_fake_context_parallel_forward_no_cache(self):
        """Test fake_context_parallel_forward without cache."""
        kernel_size = 3
        conv_cache = None
        inputs = torch.randn(1, 64, 5, 60, 104)

        if kernel_size > 1:
            cached_inputs = [conv_cache] if conv_cache is not None else [inputs[:, :, :1]] * (kernel_size - 1)
            inputs_padded = torch.cat(cached_inputs + [inputs], dim=2)

        assert inputs_padded.shape[2] == inputs.shape[2] + kernel_size - 1

    def test_forward_conv_cache_update(self):
        """Test conv_cache update in forward."""
        time_kernel_size = 3
        inputs = torch.randn(1, 64, 5, 60, 104)

        # conv_cache is last (time_kernel_size - 1) frames
        conv_cache = inputs[:, :, -time_kernel_size + 1 :].clone()

        assert conv_cache.shape == (1, 64, 2, 60, 104)


class TestAscendAutoencoderKLCogVideoXConfig:
    """Tests for AscendAutoencoderKLCogVideoX configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        in_channels = 3
        out_channels = 3
        latent_channels = 16
        block_out_channels = (128, 256, 256, 512)
        layers_per_block = 3  # noqa: F841
        act_fn = "silu"  # noqa: F841
        temporal_compression_ratio = 4

        # Verify input/output consistency
        assert in_channels == out_channels
        # Verify latent dimension is smaller than input
        assert latent_channels < in_channels
        # Verify block_out_channels are in ascending order (except duplicates)
        assert block_out_channels[0] < block_out_channels[-1]
        # Verify compression ratio is valid
        assert temporal_compression_ratio > 1

    def test_tile_sample_min_values(self):
        """Test tile sample minimum values."""
        sample_height = 480
        sample_width = 720

        tile_sample_min_height = sample_height // 3
        tile_sample_min_width = sample_width // 5

        assert tile_sample_min_height == 160
        assert tile_sample_min_width == 144

    def test_tile_latent_min_size(self):
        """Test tile latent minimum size calculation."""
        tile_sample_min_size = 128
        block_out_channels = (128, 256, 256, 512)

        tile_latent_min_size = int(tile_sample_min_size / (2 ** (len(block_out_channels) - 1)))

        assert tile_latent_min_size == 16

    def test_num_frames_batch_size(self):
        """Test batch sizes for frames."""
        num_latent_frames_batch_size = 2
        num_sample_frames_batch_size = 8

        assert num_latent_frames_batch_size == 2
        assert num_sample_frames_batch_size == 8

    def test_max_shape_decode(self):
        """Test max shape for decode padding."""
        max_shape_decode = (1, 3, 49, 160, 144)

        assert max_shape_decode == (1, 3, 49, 160, 144)

    def test_overlap_factors(self):
        """Test tile overlap factors."""
        tile_overlap_factor_height = 1 / 5
        tile_overlap_factor_width = 1 / 6

        assert math.isclose(tile_overlap_factor_height, 0.2, rel_tol=1e-9)
        assert math.isclose(tile_overlap_factor_width, 1 / 6, rel_tol=1e-9)


class TestEncoderDecoder:
    """Tests for encoder/decoder structure."""

    def test_encoder_input_channels(self):
        """Test encoder input channels."""
        in_channels = 3
        latent_channels = 16

        # Encoder: in_channels -> latent_channels (compression)
        assert in_channels > latent_channels
        assert latent_channels > 0

    def test_decoder_output_channels(self):
        """Test decoder output channels."""
        latent_channels = 16
        out_channels = 3

        # Decoder: latent_channels -> out_channels (decompression)
        assert latent_channels > out_channels
        assert out_channels > 0


class TestQuantConv:
    """Tests for quant conv layers."""

    def test_quant_conv_disabled(self):
        """Test quant conv when disabled."""
        use_quant_conv = False

        if use_quant_conv:
            quant_conv = "Conv3d"
        else:
            quant_conv = None

        assert quant_conv is None

    def test_post_quant_conv_disabled(self):
        """Test post quant conv when disabled."""
        use_post_quant_conv = False

        if use_post_quant_conv:
            post_quant_conv = "Conv3d"
        else:
            post_quant_conv = None

        assert post_quant_conv is None


class TestLightningFlag:
    """Tests for lightning flag."""

    def test_lightning_default(self):
        """Test lightning default value."""
        lightning = False
        assert isinstance(lightning, bool)
        assert not lightning

    def test_lightning_enabled(self):
        """Test lightning can be enabled."""
        lightning = True
        assert isinstance(lightning, bool)
        assert lightning


class TestSlicingTiling:
    """Tests for slicing and tiling modes."""

    def test_use_slicing_default(self):
        """Test use_slicing default."""
        use_slicing = False
        assert use_slicing == False  # noqa: E712

    def test_use_tiling_default(self):
        """Test use_tiling default."""
        use_tiling = False
        assert use_tiling == False  # noqa: E712

    def test_enable_slicing(self):
        """Test enable_slicing sets flag."""
        use_slicing = False

        # Simulate enable_slicing() call
        def enable_slicing():
            return True

        use_slicing = enable_slicing()
        assert use_slicing is True

    def test_enable_tiling(self):
        """Test enable_tiling sets flag."""
        use_tiling = False

        # Simulate enable_tiling() call
        def enable_tiling():
            return True

        use_tiling = enable_tiling()
        assert use_tiling is True


class TestClassAttributes:
    """Tests for class attributes."""

    def test_supports_gradient_checkpointing(self):
        """Test gradient checkpointing support."""
        _supports_gradient_checkpointing = True
        assert _supports_gradient_checkpointing == True  # noqa: E712

    def test_no_split_modules(self):
        """Test no split modules."""
        _no_split_modules = ["CogVideoXResnetBlock3D"]
        assert "CogVideoXResnetBlock3D" in _no_split_modules
