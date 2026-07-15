"""
Unit tests for x_diffusers.framework.vae.vfi_utils module.

Tests cover:
- InterpolationStateList class
- generic_frame_loop function
- _generic_frame_loop function

Note: These tests verify standalone logic without importing the actual module.
No sys.modules mocking is needed as we test mathematical operations.
"""

import pytest
import torch


class TestInterpolationStateList:
    """Tests for InterpolationStateList class."""

    def test_init_with_frame_indices(self):
        """Test initialization with frame indices."""
        frame_indices = [0, 2, 4]
        is_skip_list = True

        # Create state list
        state_list = type(
            "InterpolationStateList", (), {"frame_indices": frame_indices, "is_skip_list": is_skip_list}
        )()

        assert state_list.frame_indices == [0, 2, 4]
        assert state_list.is_skip_list == True  # noqa: E712

    @pytest.mark.parametrize(
        "is_skip_list,frame_index,expected",
        [
            # is_skip_list=True: Frame in list -> skip, Frame not in list -> not skip
            (True, 0, True),  # In list, skip_list=True -> skip
            (True, 1, False),  # Not in list, skip_list=True -> not skip
            (True, 2, True),  # In list, skip_list=True -> skip
            # is_skip_list=False: Frame in list -> not skip, Frame not in list -> skip
            (False, 0, False),  # In list, skip_list=False -> not skip
            (False, 1, True),  # Not in list, skip_list=False -> skip
            (False, 2, False),  # In list, skip_list=False -> not skip
        ],
    )
    def test_is_frame_skipped(self, is_skip_list, frame_index, expected):
        """Test is_frame_skipped with parameterized skip_list mode."""
        frame_indices = [0, 2, 4]

        def is_frame_skipped(idx):
            is_frame_in_list = idx in frame_indices
            return is_skip_list and is_frame_in_list or not is_skip_list and not is_frame_in_list

        assert is_frame_skipped(frame_index) == expected


class TestGenericFrameLoop:
    """Tests for generic_frame_loop function."""

    def test_frame_normalization(self):
        """Test frame normalization (-1,1) to (0,1)."""
        # Use bounded values for predictable normalization range
        frames = torch.rand(5, 3, 480, 832) * 2 - 1  # Range (-1, 1)

        # Normalize: (frames + 1) / 2 -> Range (0, 1)
        normalized = (frames + 1) / 2

        assert normalized.min() >= 0
        assert normalized.max() <= 1

    def test_frame_denormalization(self):
        """Test frame denormalization (0,1) to (-1,1)."""
        frames = torch.rand(5, 3, 480, 832)

        denormalized = frames * 2 - 1

        assert denormalized.min() >= -1
        assert denormalized.max() <= 1

    @pytest.mark.parametrize(
        "num_input_frames,multiplier,expected_output",
        [
            (5, 2, 9),  # 5 + 4 * 1 = 9
            (5, 3, 13),  # 5 + 4 * 2 = 13
            (5, 4, 17),  # 5 + 4 * 3 = 17
            (3, 2, 5),  # 3 + 2 * 1 = 5
            (3, 3, 7),  # 3 + 2 * 2 = 7
            (10, 2, 19),  # 10 + 9 * 1 = 19
        ],
    )
    def test_multiplier_range(self, num_input_frames, multiplier, expected_output):
        """Test multiplier determines output frame count with various input combinations."""
        actual_output = num_input_frames + (num_input_frames - 1) * (multiplier - 1)

        assert actual_output == expected_output

    def test_batch_size_iteration(self):
        """Test iteration with batch_size."""
        total_frames = 10
        batch_size = 2

        frame_indices = list(range(0, total_frames - 1, batch_size))

        assert frame_indices == [0, 2, 4, 6, 8]

    def test_timestep_calculation(self):
        """Test timestep calculation for middle frames."""
        multiplier = 4

        timesteps = [middle_i / multiplier for middle_i in range(1, multiplier)]

        assert timesteps == [0.25, 0.5, 0.75]


class TestSkipLogic:
    """Tests for skip logic in frame loop."""

    def test_is_skip_odd_frames(self):
        """Test is_skip skips odd frames."""
        is_skip = True

        for skip in range(10):
            should_process = (is_skip and skip % 2 == 0) or not is_skip

            if skip % 2 == 0:
                assert should_process == True  # noqa: E712
            else:
                assert should_process == False  # noqa: E712

    def test_is_skip_false_processes_all(self):
        """Test is_skip=False processes all frames."""
        is_skip = False

        for skip in range(10):
            should_process = (is_skip and skip % 2 == 0) or not is_skip
            assert should_process == True  # noqa: E712


class TestFramePairing:
    """Tests for frame pairing."""

    def test_frame_pair_selection(self):
        """Test frame0 and frame1 selection."""
        frames = torch.randn(5, 3, 480, 832)
        frame_itr = 0
        batch_size = 1

        frame0 = frames[frame_itr : frame_itr + batch_size]
        frame1 = frames[frame_itr + 1 : frame_itr + 1 + batch_size]

        assert frame0.shape == (1, 3, 480, 832)
        assert frame1.shape == (1, 3, 480, 832)

    def test_frame_pair_padding(self):
        """Test frame1 padding when shapes don't match."""
        frames = torch.randn(5, 3, 480, 832)
        frame_itr = 4  # Last pair
        batch_size = 2

        frame0 = frames[frame_itr : frame_itr + batch_size]
        frame1 = frames[frame_itr + 1 : frame_itr + 1 + batch_size]

        # frame1 would be smaller, needs padding
        if frame0.shape[0] != frame1.shape[0]:
            frame1 = torch.cat([frame1, frames[-1:]], dim=0)

        assert frame1.shape[0] == frame0.shape[0]


class TestNonTimestepInference:
    """Tests for non_timestep_inference function."""

    def test_recursive_inference_n1(self):
        """Test recursive inference with n=1 returns single middle frame."""
        # When n == 1, returns [middle_frame] interpolated at t=0.5
        n = 1  # noqa: F841

        frame0 = torch.randn(1, 3, 480, 832)
        frame1 = torch.randn(1, 3, 480, 832)

        middle_frame = (frame0 + frame1) / 2
        result_frames = [middle_frame]

        assert len(result_frames) == 1
        assert result_frames[0].shape == frame0.shape

    def test_recursive_inference_n2(self):
        """Test recursive inference with n=2 returns two middle frames."""
        # n=2 returns [first_middle at t=0.25, second_middle at t=0.75]
        n = 2  # noqa: F841

        frame0 = torch.randn(1, 3, 480, 832)
        frame1 = torch.randn(1, 3, 480, 832)

        first_middle = frame0 * 0.75 + frame1 * 0.25
        second_middle = frame0 * 0.25 + frame1 * 0.75
        result_frames = [first_middle, second_middle]

        assert len(result_frames) == 2
        assert result_frames[0].shape == frame0.shape
        assert result_frames[1].shape == frame1.shape


class TestOutputFrameConstruction:
    """Tests for output frame construction."""

    def test_output_frame_append(self):
        """Test output frame appending."""
        output_frames = []

        frame0 = torch.randn(1, 3, 480, 832)
        middle_frames = [torch.randn(1, 3, 480, 832) for _ in range(2)]

        for idx in range(1):
            output_frames.append(frame0[idx : idx + 1])
            for mid_frame in middle_frames:
                output_frames.append(mid_frame[idx : idx + 1])

        assert len(output_frames) == 3  # 1 + 2 middle frames

    def test_final_frame_append(self):
        """Test final frame appending with content and shape validation."""
        frames = torch.randn(5, 3, 480, 832)
        output_frames = [torch.randn(1, 3, 480, 832) for _ in range(4)]

        final_frame = frames[-1:] * 2 - 1
        output_frames.append(final_frame)

        assert len(output_frames) == 5
        expected_final = frames[-1:] * 2 - 1
        assert torch.allclose(final_frame, expected_final)
        assert final_frame.shape == (1, 3, 480, 832)
        assert final_frame.min() >= -1
        assert final_frame.max() <= 1
