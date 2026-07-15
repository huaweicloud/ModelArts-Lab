"""
Unit tests for config.py argument parsing functionality.

Tests the parse_args and add_*_args functions for x_diffusers.
"""

import argparse
from unittest.mock import patch

import pytest

# ============================================================
# Test data constants
# ============================================================
VALID_TASK_TYPES = ["t2v", "i2v", "t2i"]
VALID_VACE_TASKS = ["t2v", "i2v", "v2lf", "flf2v", "random2v", "inpaint", "outpaint", "openpose", "iwri"]
VALID_TURBO_MODES = ["default", "faiz", "next_faiz"]
VALID_RESOLUTIONS = [1080, 720]


class TestParseArgs:
    """Test suite for parse_args function."""

    def test_parse_args_default_values(self):
        """Test that parse_args returns correct default values."""
        from x_diffusers.adaptor.config import parse_args

        # Use no arguments to get defaults (mock sys.argv)
        with patch("sys.argv", ["test_script"]):
            args = parse_args()

        assert args.model == "Wan2.1-T2V-14B"
        assert args.task_type == "t2v"
        assert args.height == 480
        assert args.width == 720
        assert args.frames == 121
        assert args.num_inference_steps == 50
        assert args.seed == 42
        assert args.save_fps == 16
        assert args.turbo_mode == "default"

    def test_parse_args_with_custom_args(self):
        """Test parse_args with custom arguments."""
        from x_diffusers.adaptor.config import parse_args

        custom_args = ["test_script", "--model", "Wan2.2-T2V-A14B", "--height", "720", "--width", "1280"]
        with patch("sys.argv", custom_args):
            args = parse_args()

        assert args.model == "Wan2.2-T2V-A14B"
        assert args.height == 720
        assert args.width == 1280

    def test_parse_args_benchmark_mode(self):
        """Test parse_args in benchmark mode."""
        from x_diffusers.adaptor.config import parse_args

        with patch("sys.argv", ["test_script", "--config"]):  # noqa: SIM117
            with pytest.raises(SystemExit):
                # benchmark mode requires --config with value
                parse_args(mode="benchmark")

    def test_parse_args_benchmark_mode_with_config(self):
        """Test parse_args in benchmark mode with config."""
        from x_diffusers.adaptor.config import parse_args

        with patch("sys.argv", ["test_script", "--config", "benchmark.yaml"]):
            args = parse_args(mode="benchmark")
        assert args.config == "benchmark.yaml"


class TestNetworkArgs:
    """Test suite for network arguments."""

    def test_add_network_args_default_model(self):
        """Test default model value."""
        from x_diffusers.adaptor.config import add_network_args

        parser = argparse.ArgumentParser()
        parser = add_network_args(parser)
        args = parser.parse_args([])

        assert args.model == "Wan2.1-T2V-14B"
        assert args.pretrained_model_name_or_path == "Wan-AI/Wan2.1-T2V-14B-Diffusers"
        assert args.task_type == "t2v"

    @pytest.mark.parametrize("task_type", VALID_TASK_TYPES)
    def test_add_network_args_valid_task_types(self, task_type):
        """Test that all valid task types are accepted."""
        from x_diffusers.adaptor.config import add_network_args

        parser = argparse.ArgumentParser()
        parser = add_network_args(parser)
        args = parser.parse_args(["--task_type", task_type])

        assert args.task_type == task_type

    def test_add_network_args_invalid_task_type(self):
        """Test that invalid task type raises error."""
        from x_diffusers.adaptor.config import add_network_args

        parser = argparse.ArgumentParser()
        parser = add_network_args(parser)

        with pytest.raises(SystemExit):
            parser.parse_args(["--task_type", "invalid"])


class TestInferenceArgs:
    """Test suite for inference arguments."""

    def test_add_inference_args_defaults(self):
        """Test default inference args."""
        from x_diffusers.adaptor.config import add_inference_args

        parser = argparse.ArgumentParser()
        parser = add_inference_args(parser)
        args = parser.parse_args([])

        assert args.prompt == "A cat and a dog baking a cake together"
        # Note: Source code has "Briqht" (typo), not "Bright"
        assert args.negative_prompt == "Briqht tones, overexposed, static"
        assert args.height == 480
        assert args.width == 720
        assert args.frames == 121
        assert args.num_inference_steps == 50
        assert args.flow_shift == 3.0
        assert args.seed == 42
        assert args.save_fps == 16

    def test_add_inference_args_custom_prompt(self):
        """Test custom prompt values."""
        from x_diffusers.adaptor.config import add_inference_args

        parser = argparse.ArgumentParser()
        parser = add_inference_args(parser)
        args = parser.parse_args(["--prompt", "A beautiful sunset", "--negative_prompt", "blurry"])

        assert args.prompt == "A beautiful sunset"
        assert args.negative_prompt == "blurry"

    def test_add_inference_args_custom_resolution(self):
        """Test custom resolution values."""
        from x_diffusers.adaptor.config import add_inference_args

        parser = argparse.ArgumentParser()
        parser = add_inference_args(parser)
        args = parser.parse_args(["--height", "1080", "--width", "1920", "--frames", "81"])

        assert args.height == 1080
        assert args.width == 1920
        assert args.frames == 81


class TestWanVaceArgs:
    """Test suite for Wan VACE arguments."""

    def test_add_wan_vace_args_defaults(self):
        """Test default VACE args."""
        from x_diffusers.adaptor.config import add_wan_vace_args

        parser = argparse.ArgumentParser()
        parser = add_wan_vace_args(parser)
        args = parser.parse_args([])

        assert args.vace_task == ""
        assert args.image_path == ""
        assert args.video_path is None
        assert args.expand_ratio == 0.25

    @pytest.mark.parametrize("vace_task", VALID_VACE_TASKS)
    def test_add_wan_vace_args_valid_tasks(self, vace_task):
        """Test that all valid VACE tasks are accepted."""
        from x_diffusers.adaptor.config import add_wan_vace_args

        parser = argparse.ArgumentParser()
        parser = add_wan_vace_args(parser)
        args = parser.parse_args(["--vace_task", vace_task])

        assert args.vace_task == vace_task

    def test_add_wan_vace_args_invalid_task(self):
        """Test that invalid VACE task raises error."""
        from x_diffusers.adaptor.config import add_wan_vace_args

        parser = argparse.ArgumentParser()
        parser = add_wan_vace_args(parser)

        with pytest.raises(SystemExit):
            parser.parse_args(["--vace_task", "invalid_task"])


class TestAbilitiesArgs:
    """Test suite for abilities arguments."""

    def test_add_abilities_args_defaults(self):
        """Test default abilities args."""
        from x_diffusers.adaptor.config import add_abilities_args

        parser = argparse.ArgumentParser()
        parser = add_abilities_args(parser)
        args = parser.parse_args([])

        assert args.sp == 1
        assert args.turbo_mode == "default"
        assert args.vae_lightning is False
        assert args.frame_interpolation is False
        assert args.x is False
        assert args.joint is False
        assert args.ten_second is False
        assert args.fsdp is None
        assert args.atten_a8w8 is False
        assert args.matmul_a8w8 is False
        assert args.rope_fused is False

    @pytest.mark.parametrize("turbo_mode", VALID_TURBO_MODES)
    def test_add_abilities_args_valid_turbo_modes(self, turbo_mode):
        """Test that all valid turbo modes are accepted."""
        from x_diffusers.adaptor.config import add_abilities_args

        parser = argparse.ArgumentParser()
        parser = add_abilities_args(parser)
        args = parser.parse_args(["--turbo_mode", turbo_mode])

        assert args.turbo_mode == turbo_mode

    def test_add_abilities_args_invalid_turbo_mode(self):
        """Test that invalid turbo mode raises error."""
        from x_diffusers.adaptor.config import add_abilities_args

        parser = argparse.ArgumentParser()
        parser = add_abilities_args(parser)

        with pytest.raises(SystemExit):
            parser.parse_args(["--turbo_mode", "invalid"])

    def test_add_abilities_args_quantization_flags(self):
        """Test quantization flags."""
        from x_diffusers.adaptor.config import add_abilities_args

        parser = argparse.ArgumentParser()
        parser = add_abilities_args(parser)
        args = parser.parse_args(["--atten_a8w8", "--matmul_a8w8", "--rope_fused"])

        assert args.atten_a8w8 is True
        assert args.matmul_a8w8 is True
        assert args.rope_fused is True

    @pytest.mark.parametrize("resolution", VALID_RESOLUTIONS)
    def test_add_abilities_args_valid_resolutions(self, resolution):
        """Test that valid resolutions are accepted."""
        from x_diffusers.adaptor.config import add_abilities_args

        parser = argparse.ArgumentParser()
        parser = add_abilities_args(parser)
        args = parser.parse_args(["--resolution", str(resolution)])

        assert args.resolution == resolution


class TestBenchmarkArgs:
    """Test suite for benchmark arguments."""

    def test_add_benchmark_args_required(self):
        """Test that config is required for benchmark mode."""
        from x_diffusers.adaptor.config import add_benchmark_args

        parser = argparse.ArgumentParser()
        parser = add_benchmark_args(parser)

        with pytest.raises(SystemExit):
            parser.parse_args([])

    def test_add_benchmark_args_with_config(self):
        """Test benchmark args with config file."""
        from x_diffusers.adaptor.config import add_benchmark_args

        parser = argparse.ArgumentParser()
        parser = add_benchmark_args(parser)
        args = parser.parse_args(["--config", "benchmark.yaml"])

        assert args.config == "benchmark.yaml"


class TestArgsValidation:
    """Test suite for argument validation."""

    def test_model_path_consistency(self):
        """Test model and path consistency."""
        from x_diffusers.adaptor.config import parse_args

        with patch(
            "sys.argv", ["test_script", "--model", "Wan2.1-T2V-1.3B", "--pretrained_model_name_or_path", "custom/path"]
        ):
            args = parse_args()

        assert args.model == "Wan2.1-T2V-1.3B"
        assert args.pretrained_model_name_or_path == "custom/path"

    def test_i2v_task_with_image_path(self):
        """Test i2v task with image path."""
        from x_diffusers.adaptor.config import parse_args

        with patch("sys.argv", ["test_script", "--task_type", "i2v", "--i2v_image_path", "input.jpg"]):
            args = parse_args()

        assert args.task_type == "i2v"
        assert args.i2v_image_path == "input.jpg"

    def test_sp_greater_than_one(self):
        """Test sequence parallel with sp > 1."""
        from x_diffusers.adaptor.config import parse_args

        with patch("sys.argv", ["test_script", "--sp", "4"]):
            args = parse_args()

        assert args.sp == 4

    def test_fsdp_options(self):
        """Test FSDP options."""
        from x_diffusers.adaptor.config import parse_args

        # Test 'all' option
        with patch("sys.argv", ["test_script", "--fsdp", "all"]):
            args = parse_args()
        assert args.fsdp == "all"

        # Test 'text_encoder' option
        with patch("sys.argv", ["test_script", "--fsdp", "text_encoder"]):
            args = parse_args()
        assert args.fsdp == "text_encoder"

        # Test 'transformer' option
        with patch("sys.argv", ["test_script", "--fsdp", "transformer"]):
            args = parse_args()
        assert args.fsdp == "transformer"


class TestArgsEdgeCases:
    """Test edge cases for argument parsing."""

    def test_empty_prompt(self):
        """Test empty prompt."""
        from x_diffusers.adaptor.config import parse_args

        with patch("sys.argv", ["test_script", "--prompt", ""]):
            args = parse_args()

        assert args.prompt == ""

    def test_special_chars_in_path(self):
        """Test special characters in save path."""
        from x_diffusers.adaptor.config import parse_args

        with patch("sys.argv", ["test_script", "--save_path", "/path/with spaces/output.mp4"]):
            args = parse_args()

        assert args.save_path == "/path/with spaces/output.mp4"

    def test_large_frame_count(self):
        """Test large frame count."""
        from x_diffusers.adaptor.config import parse_args

        with patch("sys.argv", ["test_script", "--frames", "1000"]):
            args = parse_args()

        assert args.frames == 1000

    def test_high_resolution(self):
        """Test high resolution values."""
        from x_diffusers.adaptor.config import parse_args

        with patch("sys.argv", ["test_script", "--height", "2160", "--width", "3840"]):
            args = parse_args()

        assert args.height == 2160
        assert args.width == 3840

    def test_multiple_lora_paths(self):
        """Test multiple LoRA paths."""
        from x_diffusers.adaptor.config import parse_args

        with patch(
            "sys.argv",
            [
                "test_script",
                "--lora_path_list",
                "lora1.safetensors",
                "lora2.safetensors",
                "--lora_scale_weight_list",
                "0.5",
                "0.5",
            ],
        ):
            args = parse_args()

        assert args.lora_path_list == ["lora1.safetensors", "lora2.safetensors"]
        assert args.lora_scale_weight_list == ["0.5", "0.5"]
