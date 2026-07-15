"""
Unit tests for infer_tools.py functionality.

Tests the InferenceManager class and inference utilities.
"""

import math
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ============================================================
# Test data constants
# ============================================================
GUIDANCE_SCALE_MAP = {
    "Wan2.1-T2V-14B": (5.0, None),
    "Wan2.1-I2V-14B": (5.0, None),
    "Wan2.1-T2V-1.3B": (5.0, None),
    "CogVideoX-5b": (5.0, None),
    "HunyuanVideo-T2V-13B": (5.0, None),
    "Wan2.2-I2V-A14B": (3.5, 3.5),
    "Wan2.2-T2V-A14B": (3.0, 4.0),
}


class TestInferenceManager:
    """Test suite for InferenceManager class."""

    def test_init(self):
        """Test InferenceManager initialization."""
        from x_diffusers.adaptor.infer_tools import InferenceManager

        manager = InferenceManager()

        assert manager.init_args is None
        assert manager.pipe is None
        assert manager.sr_pipe is None
        assert manager.v2v_pipe is None

    def test_need_reload_weight_abilities_list(self):
        """Test that abilities list contains expected items."""
        from x_diffusers.adaptor.infer_tools import InferenceManager

        manager = InferenceManager()
        expected_abilities = [
            "sp",
            "turbo_mode",
            "x",
            "x_model_path",
            "fsdp",
            "matmul_a8w8",
            "inf_vram_blocks_num",
        ]

        for ability in expected_abilities:
            assert ability in manager.need_reload_weight_abilities_list


class TestCheckAbilitiesArgsConsistent:
    """Test suite for check_abilities_args_consistent method."""

    def test_consistent_with_none_init_args(self, mock_args):
        """Test consistency check when init_args is None."""
        from x_diffusers.adaptor.infer_tools import InferenceManager

        manager = InferenceManager()

        result = manager.check_abilities_args_consistent(mock_args)

        assert result is False

    def test_consistent_with_same_args(self, mock_args):
        """Test consistency check with same args."""
        from copy import deepcopy

        from x_diffusers.adaptor.infer_tools import InferenceManager

        manager = InferenceManager()
        manager.init_args = deepcopy(mock_args)

        result = manager.check_abilities_args_consistent(mock_args)

        assert result is True

    def test_inconsistent_with_different_sp(self, mock_args):
        """Test inconsistency when sp changes."""
        from copy import deepcopy

        from x_diffusers.adaptor.infer_tools import InferenceManager

        manager = InferenceManager()
        manager.init_args = deepcopy(mock_args)
        manager.init_args.sp = 1

        new_args = deepcopy(mock_args)
        new_args.sp = 4

        result = manager.check_abilities_args_consistent(new_args)

        assert result is False

    def test_inconsistent_with_different_turbo_mode(self, mock_args):
        """Test inconsistency when turbo_mode changes."""
        from copy import deepcopy

        from x_diffusers.adaptor.infer_tools import InferenceManager

        manager = InferenceManager()
        manager.init_args = deepcopy(mock_args)
        manager.init_args.turbo_mode = "default"

        new_args = deepcopy(mock_args)
        new_args.turbo_mode = "faiz"

        result = manager.check_abilities_args_consistent(new_args)

        assert result is False

    def test_inconsistent_with_fuse_lora_change(self, mock_args):
        """Test inconsistency when fuse_lora changes."""
        from copy import deepcopy

        from x_diffusers.adaptor.infer_tools import InferenceManager

        manager = InferenceManager()
        manager.init_args = deepcopy(mock_args)
        manager.init_args.fuse_lora = True

        new_args = deepcopy(mock_args)
        new_args.fuse_lora = False

        result = manager.check_abilities_args_consistent(new_args)

        assert result is False


class TestWanImagePreprocess:
    """Test suite for wan_image_preprocess method."""

    def test_wan_image_preprocess_basic(self, mock_args):
        """Test basic image preprocessing."""
        from PIL import Image
        from x_diffusers.adaptor.infer_tools import InferenceManager

        manager = InferenceManager()
        manager.pipe = MagicMock()
        manager.pipe.vae_scale_factor_spatial = 8
        manager.pipe.transformer.config.patch_size = (1, 2, 2)

        image = Image.new("RGB", (1920, 1080))

        new_height, new_width = manager.wan_image_preprocess(image, 480, 720)

        # Verify dimensions are positive integers
        assert isinstance(new_height, int)
        assert isinstance(new_width, int)
        assert new_height > 0
        assert new_width > 0

        # Verify dimensions are divisible by vae_scale_factor_spatial * patch_size[1]
        # For vae_scale_factor_spatial=8 and patch_size=(1,2,2), dimensions should be divisible by 16
        vae_scale = manager.pipe.vae_scale_factor_spatial
        patch_h = manager.pipe.transformer.config.patch_size[1]
        patch_w = manager.pipe.transformer.config.patch_size[2]
        assert new_height % (vae_scale * patch_h) == 0, (
            f"Height {new_height} should be divisible by {vae_scale * patch_h}"
        )
        assert new_width % (vae_scale * patch_w) == 0, f"Width {new_width} should be divisible by {vae_scale * patch_w}"


class TestAdaBrightImagePreprocess:
    """Test suite for ada_bright_image_preprocess static method."""

    def test_ada_bright_disabled(self, mock_args):
        """Test when ada_brighten is disabled."""
        from PIL import Image
        from x_diffusers.adaptor.infer_tools import InferenceManager

        mock_args.ada_brighten = False
        image = Image.new("RGB", (100, 100), color=(128, 128, 128))

        result = InferenceManager.ada_bright_image_preprocess(image, mock_args)

        assert result == image

    def test_ada_bright_enabled_bright_image(self, mock_args):
        """Test when ada_brighten is enabled with bright image."""
        from PIL import Image
        from x_diffusers.adaptor.infer_tools import InferenceManager

        mock_args.ada_brighten = True
        image = Image.new("RGB", (100, 100), color=(220, 220, 220))

        with patch("x_diffusers.adaptor.infer_tools.infer_info.update_adabrighten") as mock_update:
            result = InferenceManager.ada_bright_image_preprocess(image, mock_args)  # noqa: F841
            mock_update.assert_called()


class TestRestoreBrightness:
    """Test suite for restore_brightness static method."""

    def test_restore_brightness_basic(self):
        """Test basic brightness restoration."""
        from x_diffusers.adaptor.infer_tools import InferenceManager

        # Create test frames
        frames = np.random.rand(10, 100, 100, 3).astype(np.float32)

        result = InferenceManager.restore_brightness(frames)

        assert len(result) == len(frames)
        # Each frame should be a PIL Image
        from PIL import Image

        for frame in result:
            assert isinstance(frame, Image.Image)


class TestGetTempSavePath:
    """Test suite for _get_temp_save_path function."""

    def test_get_temp_save_path_basic(self):
        """Test basic temp path generation."""
        from x_diffusers.adaptor.infer_tools import _get_temp_save_path

        original_path = "/path/to/output.mp4"
        temp_path = _get_temp_save_path(original_path)

        assert "tmp_" in temp_path
        assert temp_path.endswith("output.mp4")

    def test_get_temp_save_path_with_spaces(self):
        """Test temp path with spaces in path."""
        from x_diffusers.adaptor.infer_tools import _get_temp_save_path

        original_path = "/path with spaces/output.mp4"
        temp_path = _get_temp_save_path(original_path)

        assert "tmp_" in temp_path

    def test_get_temp_save_path_empty_raises(self):
        """Test that empty path raises ValueError."""
        from x_diffusers.adaptor.infer_tools import _get_temp_save_path

        with pytest.raises(ValueError):
            _get_temp_save_path("")


class TestGuidanceScaleMap:
    """Test suite for guidance scale mapping."""

    @pytest.mark.parametrize(
        "model,expected_scale",
        [
            ("Wan2.1-T2V-14B", (5.0, None)),
            ("Wan2.1-I2V-14B", (5.0, None)),
            ("Wan2.1-T2V-1.3B", (5.0, None)),
            ("CogVideoX-5b", (5.0, None)),
            ("HunyuanVideo-T2V-13B", (5.0, None)),
            ("Wan2.2-I2V-A14B", (3.5, 3.5)),
            ("Wan2.2-T2V-A14B", (3.0, 4.0)),
        ],
    )
    def test_guidance_scale_map_values(self, model, expected_scale):
        """Test that guidance scale map has correct values."""
        from x_diffusers.adaptor.infer_tools import GUIDANCE_SCALE_MAP as infer_guidance_scale_map

        assert infer_guidance_scale_map[model] == expected_scale


class TestConfigureVaeLightning:
    """Test suite for configure_vae_lightning function."""

    def test_configure_vae_lightning_480p_not_ten_second(self, mock_args, mock_pipe):
        """Test VAE lightning for 480p non-10s."""
        from x_diffusers.adaptor.infer_tools import configure_vae_lightning

        mock_args.resolution = 480
        mock_args.ten_second = False

        configure_vae_lightning(mock_args, mock_pipe)

        assert mock_args.vae_lightning is True
        mock_pipe.enable_vae_lightning.assert_called_with(return_output=False)

    def test_configure_vae_lightning_720p(self, mock_args, mock_pipe):
        """Test VAE lightning for 720p."""
        from x_diffusers.adaptor.infer_tools import configure_vae_lightning

        mock_args.resolution = 720

        configure_vae_lightning(mock_args, mock_pipe)

        assert mock_args.vae_lightning is False
        mock_pipe.enable_vae_lightning.assert_called_with(return_output=True)

    def test_configure_vae_lightning_1080p(self, mock_args, mock_pipe):
        """Test VAE lightning for 1080p."""
        from x_diffusers.adaptor.infer_tools import configure_vae_lightning

        mock_args.resolution = 1080

        configure_vae_lightning(mock_args, mock_pipe)

        assert mock_args.vae_lightning is False


class TestGetPipeCommonKwargs:
    """Test suite for get_pipe_common_kwargs function."""

    def test_get_pipe_common_kwargs_basic(self, mock_args):
        """Test basic kwargs generation."""
        from x_diffusers.adaptor.infer_tools import get_pipe_common_kwargs

        kwargs = get_pipe_common_kwargs(mock_args)

        assert kwargs["prompt"] == mock_args.prompt
        assert kwargs["negative_prompt"] == mock_args.negative_prompt
        assert kwargs["height"] == mock_args.height
        assert kwargs["width"] == mock_args.width
        assert kwargs["num_frames"] == mock_args.frames
        assert kwargs["num_inference_steps"] == mock_args.num_inference_steps
        assert kwargs["guidance_scale"] == mock_args.guidance_scale

    def test_get_pipe_common_kwargs_with_image(self, mock_args):
        """Test kwargs generation with image for i2v."""
        from PIL import Image
        from x_diffusers.adaptor.infer_tools import get_pipe_common_kwargs

        image = Image.new("RGB", (100, 100))

        kwargs = get_pipe_common_kwargs(mock_args, image=image)

        assert "image" in kwargs
        assert kwargs["image"] == image


class TestV2VConstants:
    """Test V2V configuration constants."""

    def test_v2v_constants(self):
        """Test V2V constants have expected values."""
        from x_diffusers.adaptor.infer_tools import (
            V2V_COND_POS_LIST,
            V2V_HISTORY_FRAMES,
            V2V_INFERENCE_STEPS,
            V2V_NOISE_MULT_LIST,
            V2V_OUTPUT_TYPE,
        )

        assert V2V_HISTORY_FRAMES == 17
        assert V2V_COND_POS_LIST == [0, 1, 2, 3, 4]
        assert V2V_NOISE_MULT_LIST == [0, 0.1, 0.3, 0.5, 0.7]
        assert V2V_INFERENCE_STEPS == 4
        assert V2V_OUTPUT_TYPE == "np"


class TestXGuidanceScale:
    """Test X guidance scale constant."""

    def test_x_guidance_scale(self):
        """Test X guidance scale constant value."""
        from x_diffusers.adaptor.infer_tools import X_GUIDANCE_SCALE

        assert math.isclose(X_GUIDANCE_SCALE, 1.0, rel_tol=1e-9)


class TestLoadAndPreprocessImage:
    """Test suite for load_and_preprocess_image function."""

    def test_load_and_preprocess_image_basic(self, mock_args, tmp_path):
        """Test basic image loading and preprocessing."""
        from PIL import Image
        from x_diffusers.adaptor.infer_tools import load_and_preprocess_image

        # Create a test image
        test_image = Image.new("RGB", (1920, 1080), color=(128, 128, 128))
        image_path = str(tmp_path / "test_image.jpg")
        test_image.save(image_path)

        mock_args.width = 720
        mock_args.height = 480
        mock_args.ada_brighten = False

        with patch("x_diffusers.adaptor.infer_tools.load_image") as mock_load:
            mock_load.return_value = test_image
            result = load_and_preprocess_image(image_path, mock_args)

            assert result.size == (720, 480)


class TestRunV2VExtension:
    """Test suite for run_v2v_extension function."""

    def test_run_v2v_extension_basic(self, mock_args):
        """Test basic V2V extension."""
        from x_diffusers.adaptor.infer_tools import run_v2v_extension

        # Create mock base frames
        base_frames = np.random.rand(100, 480, 720, 3).astype(np.float32)

        mock_v2v_pipe = MagicMock()
        mock_v2v_pipe.return_value = MagicMock()
        mock_v2v_pipe.return_value.frames = [np.random.rand(100, 480, 720, 3).astype(np.float32)]

        mock_args.seed = 42

        with patch("torch.Generator") as mock_generator:
            mock_generator.return_value.manual_seed.return_value = MagicMock()
            result = run_v2v_extension(base_frames, mock_args, mock_v2v_pipe)

            assert isinstance(result, np.ndarray)
            assert result.shape == (100, 480, 720, 3)
            mock_v2v_pipe.assert_called_once()
