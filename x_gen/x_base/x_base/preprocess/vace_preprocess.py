import PIL.Image
from diffusers.utils import load_image, load_video, logging

FRAME_COLOR = 128
BLACK_COLOR = 0
WHITE_COLOR = 255

logger = logging.get_logger("infer")


def i2v_prepare_video_and_mask(img: PIL.Image.Image, height: int, width: int, num_frames: int):
    img = img.resize((width, height))
    frames = [img]
    # Ideally, this should be 127.5 to match original code, but they perform computation on numpy arrays
    # whereas we are passing PIL images. If you choose to pass numpy arrays, you can set it to 127.5 to
    # match the original code.
    frames.extend([PIL.Image.new("RGB", (width, height), (FRAME_COLOR, FRAME_COLOR, FRAME_COLOR))] * (num_frames - 1))
    mask_black = PIL.Image.new("L", (width, height), BLACK_COLOR)
    mask_white = PIL.Image.new("L", (width, height), WHITE_COLOR)
    mask = [mask_black, *[mask_white] * (num_frames - 1)]
    return frames, mask


def v2lf_prepare_video_and_mask(img: PIL.Image.Image, height: int, width: int, num_frames: int):
    img = img.resize((width, height))
    frames = []
    # Ideally, this should be 127.5 to match original code, but they perform computation on numpy arrays
    # whereas we are passing PIL images. If you choose to pass numpy arrays, you can set it to 127.5 to
    # match the original code.
    frames.extend([PIL.Image.new("RGB", (width, height), (FRAME_COLOR, FRAME_COLOR, FRAME_COLOR))] * (num_frames - 1))
    frames.append(img)
    mask_black = PIL.Image.new("L", (width, height), BLACK_COLOR)
    mask_white = PIL.Image.new("L", (width, height), WHITE_COLOR)
    mask = [*[mask_white] * (num_frames - 1), mask_black]
    return frames, mask


def flf2v_prepare_video_and_mask(
    first_img: PIL.Image.Image, last_img: PIL.Image.Image, height: int, width: int, num_frames: int
):
    first_img = first_img.resize((width, height))
    last_img = last_img.resize((width, height))
    frames = []
    frames.append(first_img)
    # Ideally, this should be 127.5 to match original code, but they perform computation on numpy arrays
    # whereas we are passing PIL images. If you choose to pass numpy arrays, you can set it to 127.5 to
    # match the original code.
    frames.extend([PIL.Image.new("RGB", (width, height), (FRAME_COLOR, FRAME_COLOR, FRAME_COLOR))] * (num_frames - 2))
    frames.append(last_img)
    mask_black = PIL.Image.new("L", (width, height), BLACK_COLOR)
    mask_white = PIL.Image.new("L", (width, height), WHITE_COLOR)
    mask = [mask_black, *[mask_white] * (num_frames - 2), mask_black]
    return frames, mask


def random2v_prepare_video_and_mask(
    images: list[PIL.Image.Image], frame_indices: list[int], height: int, width: int, num_frames: int
):
    images = [img.resize((width, height)) for img in images]
    # Ideally, this should be 127.5 to match original code, but they perform computation on numpy arrays
    # whereas we are passing PIL images. If you choose to pass numpy arrays, you can set it to 127.5 to
    # match the original code.
    frames = [PIL.Image.new("RGB", (width, height), (FRAME_COLOR, FRAME_COLOR, FRAME_COLOR))] * num_frames

    mask_black = PIL.Image.new("L", (width, height), BLACK_COLOR)
    mask_white = PIL.Image.new("L", (width, height), WHITE_COLOR)
    mask = [mask_white] * num_frames

    for img, idx in zip(images, frame_indices):
        frames[idx] = img
        mask[idx] = mask_black

    return frames, mask


def inpaint_prepare_video_and_mask(video: list[PIL.Image.Image], height: int, width: int, num_frames: int):
    frames = [frame.resize((width, height)) for frame in video]
    mask_black = PIL.Image.new("L", (width, height), BLACK_COLOR)
    # Make the mask white between top=0, bottom=height, left=width/2 - d, right=width/2 + d
    d = 80
    mask_white = PIL.Image.new("L", (2 * d, height), WHITE_COLOR)
    mask_black.paste(mask_white, (width // 2 - d, 0))
    mask = [mask_black] * num_frames
    for i in range(num_frames):
        new_frame = PIL.Image.new("RGB", (width, height), (FRAME_COLOR, FRAME_COLOR, FRAME_COLOR))
        mask_inverse = mask[i].point(lambda p: 255 - p)
        new_frame.paste(frames[i], mask=mask_inverse)
        frames[i] = new_frame
    return frames, mask


def outpaint_prepare_video_and_mask(
    img: PIL.Image.Image,
    directions: list[str],
    expand_ratio: float,
    height: int,
    width: int,
    num_frames: int,
    mask_blur: float = 0,
):
    image_width, image_height = img.size
    left = int(expand_ratio * image_width) if "left" in directions else 0
    right = int(expand_ratio * image_width) if "right" in directions else 0
    top = int(expand_ratio * image_height) if "up" in directions else 0
    bottom = int(expand_ratio * image_height) if "down" in directions else 0

    crop_left = left
    crop_right = image_width - right
    crop_top = top
    crop_bottom = image_height - bottom
    crop_box = (crop_left, crop_top, crop_right, crop_bottom)
    cropped_image = img.crop(crop_box)
    new_image = PIL.Image.new("RGB", (image_width, image_height), (FRAME_COLOR, FRAME_COLOR, FRAME_COLOR))
    new_image.paste(cropped_image, (left, top))
    new_image.save("output.png")

    mask = PIL.Image.new("L", (image_width, image_height), WHITE_COLOR)
    draw = PIL.ImageDraw.Draw(mask)
    x0 = left + (mask_blur * 2 if left > 0 else 0)
    y0 = top + (mask_blur * 2 if top > 0 else 0)
    x1 = left + cropped_image.width - (mask_blur * 2 if right > 0 else 0)
    y1 = top + cropped_image.height - (mask_blur * 2 if bottom > 0 else 0)
    draw.rectangle((x0, y0, x1, y1), fill="black")
    mask.save("mask.png")

    frames = [new_image]
    frames.extend(
        [PIL.Image.new("RGB", (image_width, image_height), (FRAME_COLOR, FRAME_COLOR, FRAME_COLOR))] * (num_frames - 1)
    )

    mask_white = PIL.Image.new("L", (image_width, image_height), WHITE_COLOR)
    mask = [mask] + [mask_white] * (num_frames - 1)

    return frames, mask


# Inpaint with reference image
def iwri_prepare_video_and_mask(video: list[PIL.Image.Image], height: int, width: int, num_frames: int):
    frames = [frame.resize((width, height)) for frame in video]
    mask_black = PIL.Image.new("L", (width, height), BLACK_COLOR)
    # Make the mask white between top=0, bottom=height, left=width/2 - d, right=width/2 + d
    d = 80
    mask_white = PIL.Image.new("L", (2 * d, height), WHITE_COLOR)
    mask_black.paste(mask_white, (width // 2 - d, 0))
    mask = [mask_black] * num_frames
    for i in range(num_frames):
        new_frame = PIL.Image.new("RGB", (width, height), (FRAME_COLOR, FRAME_COLOR, FRAME_COLOR))
        mask_inverse = mask[i].point(lambda p: 255 - p)
        new_frame.paste(frames[i], mask=mask_inverse)
        frames[i] = new_frame
    return frames, mask


def prepare_video_and_mask(args):
    logger.info(f"running wan vace {args.vace_task} task...")  # noqa: G004
    video, mask, reference_image = None, None, None
    if args.vace_task == "t2v":
        pass
    elif args.vace_task == "i2v":
        image = load_image(args.image_path)
        video, mask = i2v_prepare_video_and_mask(image, args.height, args.width, args.frames)
    elif args.vace_task == "v2lf":
        image = load_image(args.image_path)
        video, mask = v2lf_prepare_video_and_mask(image, args.height, args.width, args.frames)
    elif args.vace_task == "flf2v":
        first_frame = load_image(args.first_frame_path)
        last_frame = load_image(args.last_frame_path)
        video, mask = flf2v_prepare_video_and_mask(first_frame, last_frame, args.height, args.width, args.frames)
    elif args.vace_task == "random2v":
        image_list = [load_image(image_path) for image_path in args.image_path_list]
        video, mask = random2v_prepare_video_and_mask(
            image_list, args.frame_indices, args.height, args.width, args.frames
        )
    elif args.vace_task == "inpaint":
        # Load the video and take every second frame, limiting to 81 frames
        video = load_video(args.video_path)[::2][: args.frames]
        video, mask = inpaint_prepare_video_and_mask(video, args.height, args.width, args.frames)
    elif args.vace_task == "outpaint":
        image = load_image(args.image_path)
        video, mask = outpaint_prepare_video_and_mask(
            image, args.directions, args.expand_ratio, args.height, args.width, args.frames
        )
    elif args.vace_task == "openpose":
        video = load_video(args.video_path)[: args.frames]
        video = [frame.convert("RGB").resize((args.width, args.height)) for frame in video]
    elif args.vace_task == "iwri":
        reference_image = load_image(args.image_path)
        video = load_video(args.video_path)[::2][: args.frames]
        video, mask = iwri_prepare_video_and_mask(video, args.height, args.width, args.frames)
    return video, mask, reference_image
