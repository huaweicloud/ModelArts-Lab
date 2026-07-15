import typing

import torch


class InterpolationStateList:
    def __init__(self, frame_indices: list[int], is_skip_list: bool):
        self.frame_indices = frame_indices
        self.is_skip_list = is_skip_list

    def is_frame_skipped(self, frame_index):
        is_frame_in_list = frame_index in self.frame_indices
        return self.is_skip_list and is_frame_in_list or not self.is_skip_list and not is_frame_in_list


def _generic_frame_loop(
    frames,
    batch_size,
    device,
    clear_cache_after_n_frames,
    multiplier: typing.SupportsInt,
    return_middle_frame_function,
    *return_middle_frame_function_args,
    interpolation_states: InterpolationStateList = None,
    use_timestep=True,
    dtype=torch.float16,
    final_logging=True,
    is_skip=False,
):
    def non_timestep_inference(frame0, frame1, n):
        middle = return_middle_frame_function(frame0, frame1, None, *return_middle_frame_function_args)
        if n == 1:
            return [middle]
        first_half = non_timestep_inference(frame0, middle, n=n // 2)
        second_half = non_timestep_inference(middle, frame1, n=n // 2)
        if n % 2:
            return [*first_half, middle, *second_half]
        else:
            return [*first_half, *second_half]

    output_frames = []
    frames = frames.to(dtype=dtype, device=device)

    frames = (frames + 1) / 2
    skip = 0
    for frame_itr in range(0, len(frames) - 1, batch_size):  # Skip the final frame since there are no frames after it
        frame0 = frames[frame_itr : frame_itr + batch_size]
        frame1 = frames[frame_itr + 1 : frame_itr + 1 + batch_size]

        if frame0.shape[0] != frame1.shape[0]:
            frame1 = torch.cat([frame1, frames[-1:]], dim=0)

        if interpolation_states is not None and interpolation_states.is_frame_skipped(frame_itr):
            continue

        # Generate and append a batch of middle frames
        middle_frame_batches = []
        if (is_skip and skip % 2 == 0) or not is_skip:
            for middle_i in range(1, multiplier):
                timestep = middle_i / multiplier
                middle_frame = return_middle_frame_function(
                    frame0, frame1, timestep, *return_middle_frame_function_args
                )
                middle_frame = middle_frame * 2 - 1
                middle_frame_batches.append(middle_frame.to(dtype=dtype))

        frame0 = frame0 * 2 - 1
        for idx in range(batch_size):
            output_frames.append(frame0[idx : idx + 1])
            if (is_skip and skip % 2 == 0) or not is_skip:
                for mid_multi_frame in middle_frame_batches:
                    output_frames.append(mid_multi_frame[idx : idx + 1])
        skip += 1

    output_frames.append((frames[-1:] * 2 - 1).to(dtype=dtype))
    return torch.cat(output_frames, dim=0).float()


def generic_frame_loop(
    frames,
    batch_size,
    device,
    clear_cache_after_n_frames,
    multiplier: typing.SupportsInt,
    return_middle_frame_function,
    *return_middle_frame_function_args,
    interpolation_states: InterpolationStateList = None,
    use_timestep=True,
    dtype=torch.float32,
    is_skip=False,
):
    return _generic_frame_loop(
        frames,
        batch_size,
        device,
        clear_cache_after_n_frames,
        multiplier,
        return_middle_frame_function,
        *return_middle_frame_function_args,
        interpolation_states=interpolation_states,
        use_timestep=use_timestep,
        dtype=dtype,
        is_skip=is_skip,
    )
