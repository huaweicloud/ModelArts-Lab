import argparse
import logging

from controlnet_aux import OpenposeDetector
from diffusers.utils import export_to_video, load_video

logging.basicConfig(level=logging.INFO)


def _parse_args():
    parser = argparse.ArgumentParser(description="Preprocess argument for Wan input")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="lllyasviel/Annotators",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--video_path",
        type=str,
        default=None,
        help="The path of the videos to be processed, separated by commas if there are multiple.",
    )
    parser.add_argument(
        "--save_openpose_path", type=str, default="./openpose.mp4", help="The save path for openpose video."
    )
    parser.add_argument("--frames", type=int, default=81, help="The length of the generated video.")
    parser.add_argument("--height", type=int, default=480, help="The resolution for the generated video height.")
    parser.add_argument("--width", type=int, default=832, help="The resolution for the generated video width.")
    parser.add_argument("--save_fps", type=int, default=16, help="The frame of per sec for export videos.")
    return parser.parse_args()


def vace_openpose_prepare_video(args):
    if args.video_path is None:
        raise ValueError("The video path is required for this function.")
    logging.info("Start preprocess the vace openpose video...")
    logging.info("Please wait for the preprocess task to complete...")
    open_pose = OpenposeDetector.from_pretrained(args.pretrained_model_name_or_path)
    open_pose.to("npu")

    video = load_video(args.video_path)[::3][: args.frames]
    video = [frame.convert("RGB").resize((args.width, args.height)) for frame in video]
    openpose_video = [open_pose(frame) for frame in video]

    export_to_video(openpose_video, args.save_openpose_path, quality=8, fps=args.save_fps)
    logging.info("The preprocessing task has been completed.")


if __name__ == "__main__":
    args = _parse_args()
    vace_openpose_prepare_video(args)
