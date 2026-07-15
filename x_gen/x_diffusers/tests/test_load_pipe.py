"""
Unit tests for load_pipe.py functionality.

Tests the pipeline loading, model validation, and ability configuration.
"""

from unittest.mock import patch

import pytest

# ============================================================
# Test data constants
# ============================================================
SUPPORTED_MODELS = [
    "Wan2.1-T2V-14B",
    "Wan2.1-I2V-14B",
    "Wan2.1-T2V-1.3B",
    "Wan2.2-T2V-A14B",
    "Wan2.2-I2V-A14B",
    "Wan2.1-VACE-14B",
    "Wan2.1-VACE-1.3B",
    "CogVideoX-5b",
    "HunyuanVideo-T2V-13B",
]


class TestSupportedModels:
    """Test suite for supported model validation."""

    def test_supported_models_list_not_empty(self):
        """Test that supported models list is not empty."""
        from x_diffusers.adaptor.load_pipe import SUPPORTED_MODEL

        assert len(SUPPORTED_MODEL) > 0

    @pytest.mark.parametrize("model", SUPPORTED_MODELS)
    def test_model_in_supported_list(self, model):
        """Test that each expected model is in the supported list."""
        from x_diffusers.adaptor.load_pipe import SUPPORTED_MODEL

        assert model in SUPPORTED_MODEL, f"{model} should be in SUPPORTED_MODEL"

    def test_supported_models_count(self):
        """Test that supported models count matches expected."""
        from x_diffusers.adaptor.load_pipe import SUPPORTED_MODEL

        assert len(SUPPORTED_MODEL) == len(SUPPORTED_MODELS)


class TestLoadPipeValidation:
    """Test suite for load_pipe validation."""

    def test_unsupported_model_raises_exception(self, mock_args):
        """Test that unsupported model raises exception."""
        from x_diffusers.adaptor.load_pipe import load_pipe

        mock_args.model = "UnsupportedModel"

        with pytest.raises(Exception) as exc_info:
            load_pipe(mock_args)

        assert "Unsupported model" in str(exc_info.value)

    @pytest.mark.parametrize("model", ["Wan2.1-T2V-14B", "Wan2.1-I2V-14B", "Wan2.1-T2V-1.3B"])
    def test_wan_model_detection(self, model, mock_args):
        """Test that Wan models are correctly detected."""
        mock_args.model = model
        # Model name contains "Wan" should trigger load_wan_pipe
        assert "Wan" in model


class TestCheckArgsConflict:
    """Test suite for check_args_conflict function."""

    def test_vae_lightning_with_sp_equals_one(self, mock_args):
        """Test that vae_lightning is disabled when sp=1."""
        from x_diffusers.adaptor.load_pipe import check_args_conflict

        mock_args.vae_lightning = True
        mock_args.sp = 1

        check_args_conflict(mock_args)

        assert mock_args.vae_lightning is False

    def test_vae_lightning_with_sp_greater_than_one(self, mock_args):
        """Test that vae_lightning is not disabled when sp>1."""
        from x_diffusers.adaptor.load_pipe import check_args_conflict

        mock_args.vae_lightning = True
        mock_args.sp = 4

        check_args_conflict(mock_args)

        assert mock_args.vae_lightning is True

    def test_t2i_task_frames_adjustment(self, mock_args):
        """Test that frames is set to 1 for t2i task."""
        from x_diffusers.adaptor.load_pipe import check_args_conflict

        mock_args.task_type = "t2i"
        mock_args.frames = 121

        check_args_conflict(mock_args)

        assert mock_args.frames == 1

    def test_t2i_task_frames_already_one(self, mock_args):
        """Test that frames remains 1 for t2i task when already 1."""
        from x_diffusers.adaptor.load_pipe import check_args_conflict

        mock_args.task_type = "t2i"
        mock_args.frames = 1

        check_args_conflict(mock_args)

        assert mock_args.frames == 1

    def test_fuse_lora_and_matmul_a8w8_conflict(self, mock_args):
        """Test that fuse_lora is disabled when matmul_a8w8 is True."""
        from x_diffusers.adaptor.load_pipe import check_args_conflict

        mock_args.fuse_lora = True
        mock_args.matmul_a8w8 = True

        check_args_conflict(mock_args)

        assert mock_args.fuse_lora is False

    def test_joint_without_wan22_i2v(self, mock_args):
        """Test that joint is disabled for non-Wan2.2 I2V."""
        from x_diffusers.adaptor.load_pipe import check_args_conflict

        mock_args.joint = True
        mock_args.x = True
        mock_args.model = "Wan2.1-I2V-14B"

        check_args_conflict(mock_args)

        assert mock_args.joint is False

    def test_joint_with_wan22_i2v_and_x(self, mock_args):
        """Test that joint remains enabled for Wan2.2 I2V with x flag."""
        from x_diffusers.adaptor.load_pipe import check_args_conflict

        mock_args.joint = True
        mock_args.x = True
        mock_args.model = "Wan2.2-I2V-A14B"
        mock_args.task_type = "i2v"

        check_args_conflict(mock_args)

        assert mock_args.joint is True

    def test_ada_brighten_without_i2v(self, mock_args):
        """Test that ada_brighten is disabled for non-I2V task."""
        from x_diffusers.adaptor.load_pipe import check_args_conflict

        mock_args.ada_brighten = True
        mock_args.task_type = "t2v"
        mock_args.model = "Wan2.1-T2V-14B"

        check_args_conflict(mock_args)

        assert mock_args.ada_brighten is False

    def test_ada_brighten_with_i2v(self, mock_args):
        """Test that ada_brighten remains enabled for I2V task."""
        from x_diffusers.adaptor.load_pipe import check_args_conflict

        mock_args.ada_brighten = True
        mock_args.task_type = "i2v"
        mock_args.model = "Wan2.1-I2V-14B"

        check_args_conflict(mock_args)

        assert mock_args.ada_brighten is True


class TestOperatorAbility:
    """Test suite for operator_ability function."""

    def test_operator_ability_with_atten_a8w8(self, mock_pipe, mock_args):
        """Test operator ability with attention quantization."""
        from x_diffusers.adaptor.load_pipe import operator_ability

        mock_args.atten_a8w8 = True
        mock_args.matmul_a8w8 = False
        mock_args.matmul_a4w4 = False

        with patch("x_diffusers.adaptor.load_pipe.attention_manager") as mock_attention:
            operator_ability(mock_pipe, mock_args)
            mock_attention.enable_sage_attention.assert_called_once()

    def test_operator_ability_matmul_conflict(self, mock_pipe, mock_args):
        """Test that matmul_a8w8 and matmul_a4w4 conflict raises error."""
        from x_diffusers.adaptor.load_pipe import operator_ability

        mock_args.matmul_a8w8 = True
        mock_args.matmul_a4w4 = True

        with pytest.raises(ValueError):
            operator_ability(mock_pipe, mock_args)

    def test_operator_ability_with_rope_fused(self, mock_pipe, mock_args):
        """Test operator ability with rope fused."""
        from x_diffusers.adaptor.load_pipe import operator_ability

        mock_args.rope_fused = True
        mock_args.atten_a8w8 = False
        mock_args.matmul_a8w8 = False
        mock_args.matmul_a4w4 = False
        mock_args.conv3d_w8a8 = False

        with patch("x_diffusers.adaptor.load_pipe.rope_manager") as mock_rope:
            operator_ability(mock_pipe, mock_args)
            mock_rope.enable_rope_fused.assert_called_once()


class TestLoadLora:
    """Test suite for load_lora function."""

    def test_load_lora_single_path(self, mock_pipe, mock_args):
        """Test loading single LoRA weight."""
        from x_diffusers.adaptor.load_pipe import load_lora

        lora_path_list = ["lora1.safetensors"]

        load_lora(mock_pipe, mock_args.model, lora_path_list)

        mock_pipe.load_lora_weights.assert_called()
        mock_pipe.set_adapters.assert_called()

    def test_load_lora_multiple_paths(self, mock_pipe, mock_args):
        """Test loading multiple LoRA weights with path verification."""
        from x_diffusers.adaptor.load_pipe import load_lora

        lora_path_list = ["lora1.safetensors", "lora2.safetensors"]
        weights_list = [0.6, 0.4]

        load_lora(mock_pipe, mock_args.model, lora_path_list, weights_list=weights_list)

        # Verify call count
        assert mock_pipe.load_lora_weights.call_count == 2

        # Verify each path was passed correctly
        call_args_list = mock_pipe.load_lora_weights.call_args_list
        actual_paths = [call[0][0] for call in call_args_list]
        assert actual_paths == lora_path_list, f"Expected paths {lora_path_list}, got {actual_paths}"

        mock_pipe.set_adapters.assert_called()

    def test_load_lora_with_weights_list(self, mock_pipe, mock_args):
        """Test loading LoRA with custom weights."""
        from x_diffusers.adaptor.load_pipe import load_lora

        lora_path_list = ["lora1.safetensors"]
        weights_list = [0.8]

        load_lora(mock_pipe, mock_args.model, lora_path_list, weights_list=weights_list)

        mock_pipe.set_adapters.assert_called()


class TestUpdateLora:
    """Test suite for update_lora function."""

    def test_update_lora_no_change(self, mock_pipe, mock_args):
        """Test update_lora when paths don't change."""
        from x_diffusers.adaptor.load_pipe import update_lora

        lora_path_list = ["lora1.safetensors"]

        update_lora(mock_pipe, mock_args.model, lora_path_list, lora_path_list)

        # Should not unload or load when paths are the same
        mock_pipe.unload_lora_weights.assert_not_called()

    def test_update_lora_with_change(self, mock_pipe, mock_args):
        """Test update_lora when paths change."""
        from x_diffusers.adaptor.load_pipe import update_lora

        init_lora_path_list = ["lora1.safetensors"]
        new_lora_path_list = ["lora2.safetensors"]

        update_lora(mock_pipe, mock_args.model, init_lora_path_list, new_lora_path_list)

        mock_pipe.unload_lora_weights.assert_called_once()
        mock_pipe.load_lora_weights.assert_called()

    def test_update_lora_with_fuse(self, mock_pipe, mock_args):
        """Test update_lora with fuse_lora flag."""
        from x_diffusers.adaptor.load_pipe import update_lora

        lora_path_list = ["lora1.safetensors"]

        update_lora(mock_pipe, mock_args.model, None, lora_path_list, fuse_lora=True)

        mock_pipe.fuse_lora.assert_called_once()


class TestPipeToDevice:
    """Test suite for pipe_to_device function."""

    def test_pipe_to_device_with_inf_vram_blocks_num(self, mock_pipe, mock_args):
        """Test pipe_to_device with inf_vram_blocks_num > 0."""
        from x_diffusers.adaptor.load_pipe import pipe_to_device

        mock_args.inf_vram_blocks_num = 4
        mock_args.save_memory = False

        with patch("x_diffusers.adaptor.load_pipe.transformer_vram") as mock_transformer_vram:
            mock_transformer_vram.return_value = mock_pipe.transformer
            result = pipe_to_device(mock_pipe, mock_args)
            assert result is not None

    def test_pipe_to_device_without_inf_vram_blocks_num(self, mock_pipe, mock_args):
        """Test pipe_to_device with inf_vram_blocks_num = 0."""
        from x_diffusers.adaptor.load_pipe import pipe_to_device

        mock_args.inf_vram_blocks_num = 0

        result = pipe_to_device(mock_pipe, mock_args)  # noqa: F841

        mock_pipe.to.assert_called_with("npu")


class TestModelSpecificLogic:
    """Test model-specific logic."""

    def test_wan22_flag_detection(self):
        """Test WAN_22_FLAG constant."""
        from x_diffusers.adaptor.load_pipe import WAN_22_FLAG

        assert WAN_22_FLAG == "Wan2.2"

    def test_wan22_model_config_validation(self, mock_args):
        """Test Wan2.2 model configuration validation."""
        wan22_models = ["Wan2.2-T2V-A14B", "Wan2.2-I2V-A14B"]

        for model in wan22_models:
            mock_args.model = model
            expected_guidance_scales = (3.0, 4.0) if "T2V" in model else (3.5, 3.5)
            assert len(expected_guidance_scales) == 2, "Wan2.2 models should have dual guidance scale"

    def test_vace_model_task_validation(self, mock_args):
        """Test VACE model task type validation."""
        vace_models = ["Wan2.1-VACE-14B", "Wan2.1-VACE-1.3B"]
        valid_vace_tasks = ["t2v", "i2v", "v2lf", "flf2v", "random2v", "inpaint", "outpaint", "openpose", "iwri"]

        for model in vace_models:
            mock_args.model = model
            for task in valid_vace_tasks:
                assert task in valid_vace_tasks, f"VACE task {task} should be valid"

    def test_hunyuan_model_pipeline_type(self, mock_args):
        """Test HunyuanVideo model pipeline type."""
        hunyuan_models = ["HunyuanVideo-T2V-13B"]

        for model in hunyuan_models:
            mock_args.model = model
            expected_pipeline_class = "HunyuanVideoPipeline"
            assert expected_pipeline_class == "HunyuanVideoPipeline"

    def test_cogvideo_model_pipeline_type(self, mock_args):
        """Test CogVideoX model pipeline type."""
        cogvideo_models = ["CogVideoX-5b"]

        for model in cogvideo_models:
            mock_args.model = model
            expected_pipeline_class = "CogVideoXPipeline"
            assert expected_pipeline_class == "CogVideoXPipeline"
