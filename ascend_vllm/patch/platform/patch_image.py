from pathlib import Path
from typing import Any

import aiohttp
import numpy as np
import numpy.typing as npt
import torch
from PIL import Image
from vllm.logger import init_logger
from vllm.multimodal.inputs import VideoItem
from vllm.multimodal.media.base import MediaWithBytes
from vllm.multimodal.media.connector import MediaConnector
from vllm.multimodal.media.image import ImageMediaIO
from vllm.multimodal.media.video import VideoMediaIO
from vllm.multimodal.parse import MultiModalDataParser

logger = init_logger(__name__)


class InvalidMediaInputError(ValueError):
    """用户提供的图片/视频路径、URL 或内容不可用。"""


def load_file(self, filepath: Path) -> MediaWithBytes[Image.Image]:
    try:
        data = filepath.read_bytes()
    except OSError as e:
        raise InvalidMediaInputError(f"Cannot read image file '{filepath}': {e}") from e

    try:
        return self.load_bytes(data)
    except Exception as e:
        raise InvalidMediaInputError(f"Cannot decode image file '{filepath}': {e}") from e


_original_fetch_image = MediaConnector.fetch_image
_original_fetch_image_async = MediaConnector.fetch_image_async
_original_load_from_url = MediaConnector.load_from_url
_original_load_from_url_async = MediaConnector.load_from_url_async


def _validate_image_url(image_url: Any) -> str:
    if not isinstance(image_url, str) or not image_url.strip():
        raise InvalidMediaInputError("image_url.url must be a non-empty string")
    return image_url.strip()


def fetch_image(self, image_url, *, image_mode="RGB"):
    image_url = _validate_image_url(image_url)

    try:
        return _original_fetch_image(self, image_url, image_mode=image_mode)
    except aiohttp.ClientResponseError as e:
        if 400 <= e.status < 500:
            raise InvalidMediaInputError(f"Cannot fetch image URL '{image_url}': HTTP {e.status}") from e
        raise
    except aiohttp.ClientError as e:
        raise InvalidMediaInputError(f"Cannot access image URL '{image_url}': {type(e).__name__}") from e


async def fetch_image_async(self, image_url, *, image_mode="RGB"):
    image_url = _validate_image_url(image_url)

    try:
        return await _original_fetch_image_async(self, image_url, image_mode=image_mode)
    except aiohttp.ClientResponseError as e:
        if 400 <= e.status < 500:
            raise InvalidMediaInputError(f"Cannot fetch image URL '{image_url}': HTTP {e.status}") from e
        raise
    except (TimeoutError, aiohttp.ClientError) as e:
        raise InvalidMediaInputError(f"Cannot access image URL '{image_url}': {type(e).__name__}") from e


def load_from_url(self, *args, **kwargs):
    url = kwargs.get("url", args[0] if args else "<unknown>")

    try:
        return _original_load_from_url(self, *args, **kwargs)
    except aiohttp.ClientResponseError as e:
        if 400 <= e.status < 500:
            raise InvalidMediaInputError(f"Cannot fetch media URL '{url}': HTTP {e.status}") from e
        raise
    except aiohttp.ClientError as e:
        raise InvalidMediaInputError(f"Cannot access media URL '{url}': {type(e).__name__}") from e


async def load_from_url_async(self, *args, **kwargs):
    url = kwargs.get("url", args[0] if args else "<unknown>")

    try:
        return await _original_load_from_url_async(self, *args, **kwargs)
    except aiohttp.ClientResponseError as e:
        if 400 <= e.status < 500:
            raise InvalidMediaInputError(f"Cannot fetch media URL '{url}': HTTP {e.status}") from e
        raise
    except (TimeoutError, aiohttp.ClientError) as e:
        raise InvalidMediaInputError(f"Cannot access media URL '{url}': {type(e).__name__}") from e


def load_file_video(self, filepath: Path) -> tuple[npt.NDArray, dict[str, Any]]:
    try:
        with filepath.open("rb") as f:
            data = f.read()
    except OSError as e:
        raise InvalidMediaInputError(f"Cannot read video file '{filepath}': {e}") from e

    try:
        return self.load_bytes(data)
    except Exception as e:
        raise InvalidMediaInputError(f"Cannot decode video file '{filepath}': {e}") from e


def _get_video_with_metadata(
    self,
    video: VideoItem,
) -> tuple[np.ndarray, dict[str, Any] | None]:
    if isinstance(video, tuple):
        return video

    try:
        if isinstance(video, list):
            return np.array(video), None
        if isinstance(video, np.ndarray):
            return video, None
        if isinstance(video, torch.Tensor):
            return video.numpy(), None
    except Exception as e:
        raise InvalidMediaInputError(f"Cannot convert video input of type '{type(video).__name__}': {e}") from e

    raise InvalidMediaInputError(f"Parameter 'video' has unsupported type: '{type(video).__name__}'")


ImageMediaIO.load_file = load_file
VideoMediaIO.load_file = load_file_video
MultiModalDataParser._get_video_with_metadata = _get_video_with_metadata
MediaConnector.fetch_image = fetch_image
MediaConnector.fetch_image_async = fetch_image_async
MediaConnector.load_from_url = load_from_url
MediaConnector.load_from_url_async = load_from_url_async
