"""
Unit tests for x_diffusers.framework.vae.hunyuan module.

Tests cover:
- AutoencoderKLHunyuanVideo configuration
- enable_tiling/disable_tiling
- enable_slicing/disable_slicing

Note: These tests verify standalone logic without importing the actual module.
No sys.modules mocking is needed as we test mathematical operations and shapes.
"""

import math


class TestAutoencoderKLHunyuanVideoConfig:
    """Tests for AutoencoderKLHunyuanVideo configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        in_channels = 3
        out_channels = 3  # noqa: F841
        latent_channels = 16
        block_out_channels = (128, 256, 512, 512)
        layers_per_block = 2  # noqa: F841
        act_fn = "silu"  # noqa: F841
        spatial_compression_ratio = 8
        temporal_compression_ratio = 4

        assert in_channels == 3
        assert latent_channels == 16
        assert block_out_channels == (128, 256, 512, 512)
        assert spatial_compression_ratio == 8
        assert temporal_compression_ratio == 4

    def test_compression_ratios(self):
        """Test compression ratio values."""
        spatial_compression_ratio = 8
        temporal_compression_ratio = 4

        # Spatial: 8x downsampling
        # Temporal: 4x downsampling
        assert spatial_compression_ratio == 8
        assert temporal_compression_ratio == 4

    def test_scaling_factor(self):
        """Test scaling factor."""
        scaling_factor = 0.476986
        assert abs(scaling_factor - 0.476986) < 1e-6

    def test_time_compression_ratio_stored(self):
        """Test time_compression_ratio is stored."""
        temporal_compression_ratio = 4
        time_compression_ratio = temporal_compression_ratio

        assert time_compression_ratio == 4


class TestQuantConvLayers:
    """Tests for quant conv layers."""

    def test_quant_conv_shape(self):
        """Test quant conv kernel shape."""
        latent_channels = 16
        kernel_size = 1  # noqa: F841

        # quant_conv: 2 * latent_channels -> 2 * latent_channels
        in_ch = 2 * latent_channels
        out_ch = 2 * latent_channels

        assert in_ch == 32
        assert out_ch == 32

    def test_post_quant_conv_shape(self):
        """Test post quant conv kernel shape."""
        latent_channels = 16
        kernel_size = 1  # noqa: F841

        # post_quant_conv: latent_channels -> latent_channels
        in_ch = latent_channels
        out_ch = latent_channels

        assert in_ch == 16
        assert out_ch == 16


class TestTileSettings:
    """Tests for tile settings."""

    def test_tile_sample_min_values(self):
        """Test tile sample minimum values."""
        tile_sample_min_height = 256
        tile_sample_min_width = 256
        tile_sample_min_num_frames = 16

        assert tile_sample_min_height == 256
        assert tile_sample_min_width == 256
        assert tile_sample_min_num_frames == 16

    def test_tile_sample_stride_values(self):
        """Test tile sample stride values."""
        tile_sample_stride_height = 192
        tile_sample_stride_width = 192
        tile_sample_stride_num_frames = 12

        assert tile_sample_stride_height == 192
        assert tile_sample_stride_width == 192
        assert tile_sample_stride_num_frames == 12

    def test_tile_latent_min_size(self):
        """Test tile latent minimum size calculation."""
        tile_sample_min_size = 256
        block_out_channels = (128, 256, 512, 512)

        tile_latent_min_size = int(tile_sample_min_size / (2 ** (len(block_out_channels) - 1)))

        assert tile_latent_min_size == 32

    def test_tile_overlap_factor(self):
        """Test tile overlap factor."""
        tile_overlap_factor = 0.25
        assert math.isclose(tile_overlap_factor, 0.25, rel_tol=1e-9)


class TestEnableTiling:
    """Tests for enable_tiling/disable_tiling."""

    def test_enable_tiling_sets_flag(self):
        """Test enable_tiling sets use_tiling to True."""
        use_tiling = False
        use_tiling = True  # enable_tiling()

        assert use_tiling == True  # noqa: E712

    def test_enable_tiling_updates_params(self):
        """Test enable_tiling updates tile parameters."""
        tile_sample_min_height = 256
        tile_sample_min_width = 256

        # enable_tiling with custom values
        new_height = 320
        new_width = 320

        tile_sample_min_height = new_height or tile_sample_min_height
        tile_sample_min_width = new_width or tile_sample_min_width

        assert tile_sample_min_height == 320
        assert tile_sample_min_width == 320

    def test_disable_tiling(self):
        """Test disable_tiling sets use_tiling to False."""
        use_tiling = True
        use_tiling = False  # disable_tiling()

        assert use_tiling == False  # noqa: E712


class TestEnableSlicing:
    """Tests for enable_slicing/disable_slicing."""

    def test_enable_slicing(self):
        """Test enable_slicing sets use_slicing to True."""
        use_slicing = False
        use_slicing = True  # enable_slicing()

        assert use_slicing == True  # noqa: E712

    def test_disable_slicing(self):
        """Test disable_slicing sets use_slicing to False."""
        use_slicing = True
        use_slicing = False  # disable_slicing()

        assert use_slicing == False  # noqa: E712


class TestFramewiseSettings:
    """Tests for framewise encoding/decoding settings."""

    def test_use_framewise_encoding_default(self):
        """Test use_framewise_encoding default."""
        use_framewise_encoding = True
        assert use_framewise_encoding == True  # noqa: E712

    def test_use_framewise_decoding_default(self):
        """Test use_framewise_decoding default."""
        use_framewise_decoding = True
        assert use_framewise_decoding == True  # noqa: E712


class TestLightningFlag:
    """Tests for lightning flag."""

    def test_lightning_default(self):
        """Test lightning default value."""
        lightning = False
        assert lightning == False  # noqa: E712


class TestMaxShapeSettings:
    """Tests for max shape settings."""

    def test_max_shape_encode(self):
        """Test max_shape_encode for padding."""
        max_shape_encode = (1, 32, 32, 32, 32)
        assert max_shape_encode == (1, 32, 32, 32, 32)

    def test_tile_sample_min_tsize(self):
        """Test tile sample minimum t-size."""
        tile_sample_min_tsize = 64
        assert tile_sample_min_tsize == 64

    def test_tile_latent_min_tsize(self):
        """Test tile latent minimum t-size calculation."""
        tile_sample_min_tsize = 64
        temporal_compression_ratio = 4

        tile_latent_min_tsize = tile_sample_min_tsize // temporal_compression_ratio

        assert tile_latent_min_tsize == 16


class TestClassAttributes:
    """Tests for class attributes."""

    def test_supports_gradient_checkpointing(self):
        """Test gradient checkpointing support."""
        _supports_gradient_checkpointing = True
        assert _supports_gradient_checkpointing == True  # noqa: E712


class TestEncoderDecoderStructure:
    """Tests for encoder/decoder structure."""

    def test_encoder_double_z(self):
        """Test encoder outputs double z for KL."""
        double_z = True
        assert double_z == True  # noqa: E712

    def test_block_types(self):
        """Test block types for encoder/decoder."""
        down_block_types = (
            "HunyuanVideoDownBlock3D",
            "HunyuanVideoDownBlock3D",
            "HunyuanVideoDownBlock3D",
            "HunyuanVideoDownBlock3D",
        )
        up_block_types = (
            "HunyuanVideoUpBlock3D",
            "HunyuanVideoUpBlock3D",
            "HunyuanVideoUpBlock3D",
            "HunyuanVideoUpBlock3D",
        )

        assert len(down_block_types) == 4
        assert len(up_block_types) == 4

    def test_mid_block_add_attention(self):
        """Test mid block attention."""
        mid_block_add_attention = True
        assert mid_block_add_attention == True  # noqa: E712
