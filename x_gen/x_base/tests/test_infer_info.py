"""
Unit tests for InferInfo class.

Tests the InferInfo class that manages inference configuration.
"""

from argparse import Namespace

import pytest

# ============================================================
# Test data constants
# ============================================================
HUNYUAN_FRAME_ADJUSTMENT_CASES = [
    (1, 1),  # minimum frames
    (5, 5),  # already satisfies 4*k+1
    (9, 9),  # already satisfies 4*k+1
    (10, 9),  # should adjust down to 9
    (50, 49),  # should adjust to 49
    (100, 97),  # should adjust to 97
    (121, 121),  # already satisfies 4*k+1
    (200, 197),  # should adjust to 197
]

VALID_TASK_TYPES = ["t2v", "i2v"]
VALID_RESOLUTIONS = [(832, 480), (1920, 1080), (1280, 720)]


class TestInferInfo:
    """Test suite for InferInfo class."""

    def test_infer_info_default_values(self):
        """Test that InferInfo has correct default values."""
        from x_base.utils.infer_info import InferInfo

        info = InferInfo()

        assert info.model == ""
        assert info.task_type == "t2v"
        assert info.width == 832
        assert info.height == 480
        assert info.frames == 121
        assert info.fps == 16
        assert info.save_path == "./output.mp4"

    def test_infer_info_update_info(self, mock_args):
        """Test InferInfo.update_info() method."""
        from x_base.utils.infer_info import InferInfo

        info = InferInfo()
        info.update_info(mock_args)

        assert info.model == "Wan2.1-T2V-1.3B"
        assert info.task_type == "t2v"
        assert info.width == 832
        assert info.height == 480
        assert info.frames == 121
        assert info.fps == 16
        assert info.save_path == "./output.mp4"

    @pytest.mark.parametrize("width,height", VALID_RESOLUTIONS)
    def test_infer_info_update_shape(self, width, height):
        """Test InferInfo.update_shape() method with various resolutions."""
        from x_base.utils.infer_info import InferInfo

        info = InferInfo()
        info.update_shape(width, height)

        assert info.width == width
        assert info.height == height

    @pytest.mark.parametrize("ada_brighten", [True, False])
    def test_infer_info_update_adabrighten(self, ada_brighten):
        """Test InferInfo.update_adabrighten() method."""
        from x_base.utils.infer_info import InferInfo

        info = InferInfo()
        info.update_adabrighten(ada_brighten)

        assert info.ada_brighten is ada_brighten

    def test_infer_info_str_representation(self):
        """Test InferInfo.__str__() method."""
        from x_base.utils.infer_info import InferInfo

        info = InferInfo()
        info.model = "Wan2.1-T2V-1.3B"
        info.task_type = "t2v"

        str_repr = str(info)

        assert "Wan2.1-T2V-1.3B" in str_repr
        assert "t2v" in str_repr
        assert "832" in str_repr
        assert "480" in str_repr

    def test_infer_info_frames_adjustment_for_hunyuan(self):
        """Test that frames are adjusted for HunyuanVideo (must satisfy 4*k+1)."""
        from x_base.utils.infer_info import InferInfo

        args = Namespace(
            model="HunyuanVideo-T2V-13B",
            task_type="t2v",
            width=832,
            height=480,
            frames=100,  # 100-1 = 99, not divisible by 4
            save_fps=16,
            save_path="./output.mp4",
            ada_brighten=False,
            frame_interpolation=False,
            frame_model_path="",
            ten_second=False,
            i2v_image_path="",
        )

        info = InferInfo()
        info.update_info(args)

        # Should be adjusted to 97 (4*24+1)
        assert info.frames == 97
        assert (info.frames - 1) % 4 == 0

    def test_infer_info_frames_no_adjustment_for_wan(self, mock_args):
        """Test that frames are not adjusted for Wan models."""
        from x_base.utils.infer_info import InferInfo

        info = InferInfo()
        original_frames = mock_args.frames
        info.update_info(mock_args)

        # Wan models don't need frame adjustment
        assert info.frames == original_frames

    @pytest.mark.parametrize("task_type", VALID_TASK_TYPES)
    def test_infer_info_valid_task_types(self, task_type):
        """Test InferInfo accepts valid task types."""
        from x_base.utils.infer_info import InferInfo

        args = Namespace(
            model="Wan2.1-T2V-1.3B",
            task_type=task_type,
            width=832,
            height=480,
            frames=121,
            save_fps=16,
            save_path="./output.mp4",
            ada_brighten=False,
            frame_interpolation=False,
            frame_model_path="",
            ten_second=False,
            i2v_image_path="",
        )

        info = InferInfo()
        info.update_info(args)

        assert info.task_type == task_type

    def test_infer_info_i2v_attributes(self):
        """Test that i2v-specific attributes are set correctly."""
        from x_base.utils.infer_info import InferInfo

        args = Namespace(
            model="Wan2.1-I2V-14B",
            task_type="i2v",
            width=832,
            height=480,
            frames=121,
            save_fps=16,
            save_path="./output.mp4",
            ada_brighten=True,
            frame_interpolation=True,
            frame_model_path="/path/to/model.pth",
            ten_second=False,
            i2v_image_path="/path/to/image.jpg",
        )

        info = InferInfo()
        info.update_info(args)

        assert info.task_type == "i2v"
        assert info.ada_brighten is True
        assert info.frame_interpolation is True
        assert info.frame_model_path == "/path/to/model.pth"
        assert info.i2v_image_path == "/path/to/image.jpg"

    def test_global_infer_info_instance(self):
        """Test that global infer_info instance exists."""
        from x_base.utils.infer_info import infer_info

        assert infer_info is not None
        assert hasattr(infer_info, "model")
        assert hasattr(infer_info, "task_type")


class TestInferInfoEdgeCases:
    """Test edge cases for InferInfo class."""

    def test_infer_info_missing_attribute(self):
        """Test handling of missing attributes in args."""
        from x_base.utils.infer_info import InferInfo

        # Create args with missing attribute
        args = Namespace(
            model="TestModel",
            task_type="t2v",
            width=640,
            height=360,
            frames=65,
            save_fps=24,
            save_path="test.mp4",
        )

        info = InferInfo()

        # update_info should raise AttributeError for missing attributes
        with pytest.raises(AttributeError):
            info.update_info(args)

    @pytest.mark.parametrize(
        "save_path",
        [
            "/path/with spaces/output.mp4",
            "/path/with/中文/output.mp4",
            "C:\\Windows\\path\\output.mp4",
            "./output (1).mp4",
        ],
    )
    def test_infer_info_string_with_special_chars(self, save_path):
        """Test InferInfo with special characters in save_path."""
        from x_base.utils.infer_info import InferInfo

        args = Namespace(
            model="Wan2.1-T2V-1.3B",
            task_type="t2v",
            width=832,
            height=480,
            frames=121,
            save_fps=16,
            save_path=save_path,
            ada_brighten=False,
            frame_interpolation=False,
            frame_model_path="",
            ten_second=False,
            i2v_image_path="",
        )

        info = InferInfo()
        info.update_info(args)

        assert info.save_path == save_path

    @pytest.mark.parametrize("input_frames,expected_frames", HUNYUAN_FRAME_ADJUSTMENT_CASES)
    def test_infer_info_hunyuan_frame_adjustment(self, input_frames, expected_frames):
        """Test HunyuanVideo frame adjustment with parametrized cases."""
        from x_base.utils.infer_info import InferInfo

        args = Namespace(
            model="HunyuanVideo-T2V-13B",
            task_type="t2v",
            width=832,
            height=480,
            frames=input_frames,
            save_fps=16,
            save_path="./output.mp4",
            ada_brighten=False,
            frame_interpolation=False,
            frame_model_path="",
            ten_second=False,
            i2v_image_path="",
        )

        info = InferInfo()
        info.update_info(args)

        assert info.frames == expected_frames, (
            f"Input {input_frames} should adjust to {expected_frames}, got {info.frames}"
        )

    def test_hunyuan_frames_always_satisfy_constraint(self):
        """Test that HunyuanVideo frames always satisfy 4*k+1 after adjustment."""
        from x_base.utils.infer_info import InferInfo

        for input_frames in range(1, 300, 37):  # Sample various inputs
            args = Namespace(
                model="HunyuanVideo-T2V-13B",
                task_type="t2v",
                width=832,
                height=480,
                frames=input_frames,
                save_fps=16,
                save_path="./output.mp4",
                ada_brighten=False,
                frame_interpolation=False,
                frame_model_path="",
                ten_second=False,
                i2v_image_path="",
            )

            info = InferInfo()
            info.update_info(args)

            # After adjustment, must satisfy 4*k+1
            assert (info.frames - 1) % 4 == 0, f"Frames {info.frames} doesn't satisfy 4*k+1 constraint"


class TestInferInfoModelSpecific:
    """Test model-specific behavior."""

    @pytest.mark.parametrize(
        "model,should_adjust",
        [
            ("HunyuanVideo-T2V-13B", True),
            ("HunyuanVideo-I2V-13B", True),
            ("Wan2.1-T2V-1.3B", False),
            ("Wan2.1-T2V-14B", False),
            ("Wan2.1-I2V-14B", False),
            ("Wan2.2-T2V-A14B", False),
            ("CogVideoX-5b", False),
        ],
    )
    def test_frame_adjustment_by_model(self, model, should_adjust):
        """Test frame adjustment behavior varies by model."""
        from x_base.utils.infer_info import InferInfo

        # Use frame count that doesn't satisfy 4*k+1
        input_frames = 100

        args = Namespace(
            model=model,
            task_type="t2v",
            width=832,
            height=480,
            frames=input_frames,
            save_fps=16,
            save_path="./output.mp4",
            ada_brighten=False,
            frame_interpolation=False,
            frame_model_path="",
            ten_second=False,
            i2v_image_path="",
        )

        info = InferInfo()
        info.update_info(args)

        if should_adjust:
            assert info.frames != input_frames, f"{model} should adjust frames"
            assert (info.frames - 1) % 4 == 0
        else:
            assert info.frames == input_frames, f"{model} should NOT adjust frames"
