import os

import cv2
import imageio
import numpy as np
import torch
from diffusers.utils import logging
from PIL import Image, ImageEnhance
from torch.multiprocessing import Process, Queue

from ..utils.infer_info import infer_info

logger = logging.get_logger("save_video")

FFMPEG_PATH = os.getenv("FFMPEG_PATH", None)
if FFMPEG_PATH is not None:
    os.environ["IMAGEIO_FFMPEG_EXE"] = FFMPEG_PATH


def write_video(queue):
    frames_num = 0
    with imageio.get_writer(
        infer_info.save_path, fps=infer_info.fps, codec="libx264", quality=8, macro_block_size=1
    ) as writer:
        while True:
            try:
                frame = queue.get()
                if frame is None:
                    break
                if infer_info.save_width is not None and infer_info.save_height is not None:
                    if infer_info.save_width != frame.shape[1] or infer_info.save_height != frame.shape[0]:
                        frame = cv2.resize(
                            frame, (infer_info.save_width, infer_info.save_height), interpolation=cv2.INTER_AREA
                        )

                if infer_info.ada_brighten:
                    img = Image.fromarray(frame)
                    enhancer = ImageEnhance.Brightness(img)
                    img = enhancer.enhance(1.0 / 0.7)
                    frame = np.array(img)
                writer.append_data(frame)
                frames_num += 1
            except Exception as e:
                logger.error(e)
                if frames_num == infer_info.frames - 1:
                    break


class SaveVideoStream:
    def __init__(self):
        logger.info(f"infer_info:{infer_info}")  # noqa: G004
        self.frame_queue = Queue(maxsize=infer_info.frames)
        self.thread = Process(target=write_video, args=(self.frame_queue,))
        self.thread.start()

    def save(self, video):
        for frame in video:
            frame = frame.to(dtype=torch.float32).cpu().numpy().astype(np.uint8)
            self.frame_queue.put(frame)

    def close(self):
        self.frame_queue.put(None)
        self.thread.join()
