import atexit
import base64
import gc
import json
import logging
import os
import queue
import signal
import subprocess
import tempfile
import threading
import time
import uuid
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from flask import Flask, jsonify, request, send_file

try:
    from flask_cors import CORS

    CORS_AVAILABLE = True
except ImportError:
    CORS_AVAILABLE = False

import sys

from diffusers.utils import export_to_video, load_image
from file_util import file_util
from obs_client import ObsStorageClient
from PIL import Image
from x_base import infer_info
from x_diffusers import init_env, parse_args
from x_diffusers.adaptor.infer_tools import GUIDANCE_SCALE_MAP
from x_diffusers.adaptor.load_pipe import (
    load_pipe,
    load_seedvr2_pipe,
    load_v2v_pipe,
    pipe_to_device,
    update_lora,
    update_pipe,
)
from x_diffusers.framework.vae.IFRNet_S_arch import IRFNet_S
from x_diffusers.framework.vae.wan import vfi

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def decode_base64_image(image_data: str) -> str:
    """Decode base64 image data and save to temporary file.

    Args:
        image_data: Base64 encoded image string, can be:
                   - data:image/jpeg;base64,<data>
                   - data:image/png;base64,<data>
                   - plain base64 string

    Returns:
        Path to temporary file containing decoded image
    """
    if image_data.startswith("data:image/"):
        parts = image_data.split(",", 1)
        if len(parts) != 2:
            raise ValueError("Invalid data URI format")

        header, data = parts
        if "image/jpeg" in header or "image/jpg" in header:
            ext = ".jpg"
        elif "image/png" in header:
            ext = ".png"
        elif "image/webp" in header:
            ext = ".webp"
        else:
            ext = ".jpg"
    else:
        data = image_data
        ext = ".jpg"

    try:
        decoded = base64.b64decode(data)
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as f:
            f.write(decoded)
            logger.info(f"Decoded base64 image to temporary file: {f.name}")  # noqa: G004
            return f.name
    except Exception as e:
        raise ValueError(f"Failed to decode base64 image: {str(e)}")


def get_device_for_comm():
    if not dist.is_initialized():
        return "cpu"
    backend = dist.get_backend()
    if backend == "gloo":
        return "cpu"
    if backend in ["nccl", "hccl"]:
        if hasattr(torch, "npu") and torch.npu.is_available():
            rank = dist.get_rank()
            return f"npu:{rank}"
        elif torch.cuda.is_available():
            return "cuda:0"
    return "cpu"


CMD_HEARTBEAT = 0
CMD_EXECUTE = 1
CMD_SHUTDOWN = 2


def broadcast_dict(data: dict, src: int = 0) -> dict:
    if not dist.is_initialized():
        return data
    rank = dist.get_rank()
    device = get_device_for_comm()
    if rank == src:
        data_str = json.dumps(data)
        length = torch.tensor([len(data_str)], dtype=torch.long, device=device)
    else:
        length = torch.tensor([0], dtype=torch.long, device=device)

    dist.broadcast(length, src=src)

    if rank == src:
        data_tensor = torch.tensor(list(data_str.encode("utf-8")), dtype=torch.uint8, device=device)
    else:
        data_tensor = torch.zeros(length.item(), dtype=torch.uint8, device=device)

    dist.broadcast(data_tensor, src=src)

    if rank != src:
        data_str = bytes(data_tensor.cpu().tolist()).decode("utf-8")
        data = json.loads(data_str)
    return data


def broadcast_int(value: int, src: int = 0) -> int:
    if not dist.is_initialized():
        return value
    device = get_device_for_comm()
    tensor = torch.tensor([value], dtype=torch.int, device=device)
    dist.broadcast(tensor, src=src)
    return tensor.item()


MODEL_REGISTRY = {
    "wan2.2-i2v-14b": {
        "model": "Wan2.2-I2V-A14B",
        "path": "/home/models/Wan-AI/Wan2.2-I2V-A14B-Diffusers",
        "task_type": "i2v",
        "default_guidance": (3.5, 3.5),
    },
    "wan2.2-t2v-14b": {
        "model": "Wan2.2-T2V-A14B",
        "path": "/home/models/Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        "task_type": "t2v",
        "default_guidance": (3.0, 4.0),
    },
}

DEFAULT_TEN_SECOND_CONFIG = {
    "ten_second_model_id_t2v": "/home/models/Wan-AI/Wan2.2-T2V-A14B-Diffusers",
    "ten_second_model_path": "/home/models/ttv_14B_4steps/transformer_14B/model.safetensors",
    "ten_second_model_path_2": "/home/models/ttv_14B_4steps/transformer_14B_2/model.safetensors",
    "pusa_lora": "/home/models/pusa_lora_converted/pytorch_lora_weights_high.safetensors",
    "pusa_lora2": "/home/models/pusa_lora_converted/pytorch_lora_weights_low.safetensors",
    "joint_model_path": "/home/models/Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    "seedvr2_model_dir": "/home/models/SeedVR2/",
    "seedvr2_model_name": "seedvr2_ema_7b_fp16.safetensors",
    "x_model_path": "/home/models/6steps/transformer_i2v_14B_1/model.safetensors",
    "x_model_path_2": "/home/models/6steps/transformer_i2v_14B_2/model.safetensors",
    "frame_model_path": "/home/models/IFRNet_S_Vimeo90K.pth",
}

TEN_SECOND_CONFIG = DEFAULT_TEN_SECOND_CONFIG.copy()


class AsyncTask:
    STATUS_MAP = {
        "pending": "pending",
        "queued": "pending",
        "running": "running",
        "completed": "succeeded",
        "succeeded": "succeeded",
        "failed": "failed",
        "cancelled": "cancelled",
    }

    def __init__(self, task_id: str, task_type: str, params: dict[str, Any], created_at_ms: int | None = None):
        self.task_id = task_id
        self.task_type = task_type
        self.params = params
        self.status = "pending"
        self.progress = 0.0
        self.result = None
        self.output = None
        self.error = None
        if created_at_ms:
            self.created_at = datetime.fromtimestamp(created_at_ms / 1000.0)
        else:
            self.created_at = datetime.now()
        self.created_at_ms = created_at_ms or int(self.created_at.timestamp() * 1000)
        self.started_at = None
        self.completed_at = None
        self.infer_time = None

    def get_platform_status(self) -> str:
        return self.STATUS_MAP.get(self.status, self.status)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "status": self.status,
            "progress": f"{self.progress:.1f}%",
            "result": self.result,
            "error": self.error,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "infer_time": self.infer_time,
        }

    def to_platform_dict(self) -> dict[str, Any]:
        return {
            "id": self.task_id,
            "status": self.get_platform_status(),
            "output": self.output or "",
        }


class VideoInferenceManager:
    def __init__(self):
        self.pipe = None
        self.v2v_pipe = None
        self.interpolation_model = None
        self.sr_pipe = None
        self.init_args = None
        self.current_model = None

    def initialize_pipe(self, args):
        if self.pipe is not None:
            logger.info(
                f"[Rank {dist.get_rank() if dist.is_initialized() else 0}] Using cached pipe for {self.current_model}"  # noqa: G004
            )
            return

        model_name = args.model
        rank = dist.get_rank() if dist.is_initialized() else 0
        logger.info(f"[Rank {rank}] Loading pipe for {model_name}...")  # noqa: G004

        self.pipe = load_pipe(args)
        if args.lora_path_list:
            update_lora(self.pipe, args.model, None, args.lora_path_list, fuse_lora=args.fuse_lora, weights_list=None)
        update_pipe(self.pipe, args)
        self.pipe = pipe_to_device(self.pipe, args)

        logger.info(f"[Rank {rank}] Setting return_output=True for VAE")  # noqa: G004
        self.pipe.vae.return_output = True
        if hasattr(self.pipe, "video_processor"):
            self.pipe.video_processor.return_output = True

        if args.ten_second:
            logger.info(f"[Rank {rank}] Loading v2v_pipe for 10s video generation...")  # noqa: G004
            v2v_args = deepcopy(args)
            self.v2v_pipe = load_v2v_pipe(v2v_args)
            v2v_args.matmul_a8w8 = False
            v2v_args.joint = False
            update_pipe(self.v2v_pipe, v2v_args)
            self.v2v_pipe = pipe_to_device(self.v2v_pipe, args)
            logger.info(f"[Rank {rank}] v2v_pipe loaded successfully")  # noqa: G004
            if dist.is_initialized():
                dist.barrier()
                logger.info(f"[Rank {rank}] v2v_pipe loading synchronized across all ranks")  # noqa: G004

        if getattr(args, "adopt_sr", False):
            logger.info(f"[Rank {rank}] Loading sr_pipe for super resolution...")  # noqa: G004
            self.sr_pipe = load_seedvr2_pipe(args)
            logger.info(f"[Rank {rank}] sr_pipe loaded successfully")  # noqa: G004

        self.init_args = args
        self.current_model = model_name
        logger.info(f"[Rank {rank}] Pipe for {model_name} loaded and cached")  # noqa: G004

    def infer_single(self, args, task_id=None, task_manager=None) -> np.ndarray:
        rank = dist.get_rank() if dist.is_initialized() else 0
        need_super_resolution = (args.width == 1920 and args.height == 1080) or (
            args.width == 1080 and args.height == 1920
        )

        if need_super_resolution:
            logger.info(f"[Rank {rank}] 1080P requested, will generate 720P first then super-resolve")  # noqa: G004
            original_width, original_height = args.width, args.height
            if args.width == 1920:
                args.width, args.height = 1280, 720
            else:
                args.width, args.height = 720, 1280

        infer_params = {
            "prompt": args.prompt,
            "negative_prompt": args.negative_prompt,
            "height": args.height,
            "width": args.width,
            "num_frames": args.frames,
            "num_inference_steps": args.num_inference_steps,
            "guidance_scale": args.guidance_scale,
            "guidance_scale_2": args.guidance_scale_2,
            "generator": torch.Generator().manual_seed(args.seed),
        }

        if args.task_type == "i2v":
            image_path = args.i2v_image_path
            if image_path.startswith("data:image/"):
                image_path = decode_base64_image(image_path)
                infer_info.i2v_image_path = image_path
            infer_info.ada_brighten = True
            image = load_image(image_path)
            image = image.resize((args.width, args.height))
            infer_params["image"] = image

        if task_id and task_manager:
            total_steps = args.num_inference_steps

            def progress_callback(pipeline, step: int, timestep: int, callback_kwargs: dict):
                progress_pct = min(100.0, (step + 1) / total_steps * 100.0)
                with task_manager.tasks_lock:
                    task = task_manager.tasks.get(task_id)
                    if task:
                        task.progress = progress_pct
                        if task.status == "cancelled":
                            raise InterruptedError(f"Task {task_id} cancelled at step {step}")
                return callback_kwargs

            infer_params["callback_on_step_end"] = progress_callback
            infer_params["callback_on_step_end_tensor_inputs"] = ["latents"]

        output = self.pipe(**infer_params).frames[0]

        if args.ten_second and self.v2v_pipe is not None:
            logger.info(f"[Rank {rank}] Running V2V extension for 10s video...")  # noqa: G004

            his = 17
            cond_pos_list = [0, 1, 2, 3, 4]
            noise_mult_list = [0, 0.1, 0.3, 0.5, 0.7]
            steps_v2v = 4
            last_frames = output[-his:]

            g_ext = torch.Generator().manual_seed(args.seed)
            ext = self.v2v_pipe(
                conditioning_video=last_frames,
                prompt=args.prompt,
                negative_prompt=args.negative_prompt,
                conditioning_indices=cond_pos_list,
                conditioning_noise_multipliers=noise_mult_list,
                height=args.height,
                width=args.width,
                num_frames=args.frames,
                guidance_scale=args.guidance_scale,
                num_inference_steps=steps_v2v,
                generator=g_ext,
                output_type="np",
            ).frames[0]

            output = np.concatenate([output[:-his], ext], axis=0)
            logger.info(f"[Rank {rank}] V2V extension complete, total frames: {len(output)}")  # noqa: G004

        if getattr(args, "frame_interpolation", False) and args.save_fps > 16:
            output = self._do_frame_interpolation(output, args)
            logger.info(f"[Rank {rank}] Frame interpolation complete, final frames: {len(output)}")  # noqa: G004

        if need_super_resolution and self.sr_pipe is not None:
            logger.info(f"[Rank {rank}] Running super resolution 720P -> 1080P...")  # noqa: G004
            if self.v2v_pipe is not None:
                del self.v2v_pipe
                self.v2v_pipe = None
                gc.collect()
                torch.cuda.empty_cache()
                logger.info(f"[Rank {rank}] V2V pipe released before SR")  # noqa: G004
            output = self.sr_pipe(output)
            args.width, args.height = original_width, original_height
            logger.info(f"[Rank {rank}] Super resolution complete")  # noqa: G004

        return output

    def _load_interpolation_model(self, model_path: str):
        if self.interpolation_model is not None:
            return
        rank = dist.get_rank() if dist.is_initialized() else 0
        logger.info(f"[Rank {rank}] Loading interpolation model from {model_path}...")  # noqa: G004
        self.interpolation_model = IRFNet_S()
        self.interpolation_model.load_state_dict(torch.load(model_path, weights_only=True))
        logger.info(f"[Rank {rank}] Interpolation model loaded")  # noqa: G004

    def _calculate_fps_config(self, fps: int) -> tuple[int, bool]:
        multiplier = fps // 16
        is_skip = False

        if fps % 16 == 0:
            pass
        elif fps % 8 < 4:
            multiplier += 1
            is_skip = True
        else:
            multiplier += 1

        return multiplier, is_skip

    def _split_residual_frames(
        self, output_tensor: torch.Tensor, rank: int, world_size: int
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        num_frames = output_tensor.shape[0]

        if num_frames % world_size != 0 and rank == 0:
            mod = num_frames % world_size
            input1 = output_tensor[:mod]
            output_tensor = output_tensor[mod:]
        else:
            input1 = None

        return output_tensor, input1

    def _get_input_frames(self, output_tensor: torch.Tensor, rank: int, world_size: int, step: int) -> torch.Tensor:
        if rank != (world_size - 1):
            return output_tensor[step * rank : step * (rank + 1) + 1]
        else:
            return torch.cat([output_tensor[step * rank : step * (rank + 1)], output_tensor[-1:]], dim=0)

    def _process_residual_frames(
        self, input1: torch.Tensor | None, device, multiplier: int, is_skip: bool, out: torch.Tensor
    ) -> torch.Tensor:
        out1 = torch.zeros_like(out)[:0].to(device).contiguous()

        if input1 is None:
            return out1

        if input1.shape[0] >= 2:
            input1_permuted = input1.permute(0, 3, 1, 2)
            out1 = vfi(input1_permuted, device, self.interpolation_model, 1, multiplier=multiplier, is_skip=is_skip)
            out1 = out1.permute(0, 2, 3, 1).contiguous().cpu()
        else:
            out1 = input1.contiguous().cpu()

        return out1

    def _gather_results(self, out: torch.Tensor, world_size: int) -> torch.Tensor:
        if world_size == 1:
            return out.cpu()

        local_shape = torch.tensor(out.shape, device=out.device)
        all_shapes = [torch.zeros_like(local_shape) for _ in range(world_size)]
        dist.all_gather(all_shapes, local_shape)

        max_frames = max(s[0].item() for s in all_shapes)
        padded_out = torch.zeros(
            max_frames, out.shape[1], out.shape[2], out.shape[3], device=out.device, dtype=out.dtype
        )
        padded_out[: out.shape[0]] = out

        gathered = [
            torch.zeros(max_frames, out.shape[1], out.shape[2], out.shape[3], device=out.device, dtype=out.dtype)
            for _ in range(world_size)
        ]
        dist.all_gather(gathered, padded_out)

        output_result = torch.cat([g[: all_shapes[i][0].item()] for i, g in enumerate(gathered)], dim=0).cpu()
        return output_result

    def _do_frame_interpolation(self, output: np.ndarray, args) -> np.ndarray:
        rank = dist.get_rank() if dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_initialized() else 1

        frame_model_path = getattr(args, "frame_model_path", TEN_SECOND_CONFIG["frame_model_path"])
        self._load_interpolation_model(frame_model_path)

        output_tensor = torch.from_numpy(output)
        output_tensor, input1 = self._split_residual_frames(output_tensor, rank, world_size)

        step = output_tensor.shape[0] // world_size
        output_tensor = output_tensor.permute(0, 3, 1, 2)
        input_frames = self._get_input_frames(output_tensor, rank, world_size, step)

        fps = args.save_fps
        multiplier, is_skip = self._calculate_fps_config(fps)
        logger.info(f"[Rank {rank}] Frame interpolation: fps={fps}, multiplier={multiplier}, is_skip={is_skip}")  # noqa: G004

        device = rank if torch.npu.is_available() else "cpu"
        out = vfi(input_frames, device, self.interpolation_model, 1, multiplier=multiplier, is_skip=is_skip)[:-1]
        out = out.permute(0, 2, 3, 1).contiguous()

        out1 = torch.zeros_like(out)[:0].to(device).contiguous()
        if rank == 0:
            out1 = self._process_residual_frames(input1, device, multiplier, is_skip, out)

        output_result = self._gather_results(out, world_size)

        if input1 is not None:
            output_result = torch.cat([out1, output_result], dim=0)

        return output_result.detach().numpy()


class VideoGenerationService:
    def __init__(self, config_path: str | None = None):
        self.app = Flask(__name__)
        if CORS_AVAILABLE:
            CORS(self.app)

        self.inference_manager = None
        self.default_config = self._load_default_config(config_path)
        self.is_initialized = False

        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1

        self.task_queue = queue.Queue(maxsize=10)
        self.result_queue = queue.Queue()

        self.tasks: dict[str, AsyncTask] = {}
        self.tasks_lock = threading.Lock()

        self.max_tasks = 1000
        self.task_timeout = timedelta(hours=1)
        self.keep_completed_hours = 2
        self.cleanup_interval = 300

        self.shutdown = False
        self._setup_signal_handlers()

        self._setup_health_routes()
        self._setup_task_routes()
        self._setup_config_routes()
        self._setup_internal_task_routes()
        self._setup_internal_query_routes()

    def _setup_signal_handlers(self):
        def signal_handler(signum, frame):
            logger.info(f"[Rank {self.rank}] Received signal {signum}, initiating graceful shutdown...")  # noqa: G004
            self.shutdown = True
            if self.world_size > 1 and self.rank == 0:
                try:
                    for _ in range(self.world_size - 1):
                        broadcast_int(CMD_SHUTDOWN, src=0)
                    logger.info("[Rank 0] Sent shutdown command to all workers")
                except Exception as e:
                    logger.error(f"[Rank 0] Failed to send shutdown command: {e}")  # noqa: G004

        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
        atexit.register(self._cleanup_on_exit)

    def _cleanup_on_exit(self):
        logger.info(f"[Rank {self.rank}] Service exiting, cleaning up...")  # noqa: G004
        with self.tasks_lock:
            for task in self.tasks.values():
                if task.status in ["pending", "running"]:
                    task.status = "cancelled"
                    task.error = "Service shutdown"

    def _cleanup_completed_tasks(self):
        now = datetime.now()
        cutoff = now - timedelta(hours=self.keep_completed_hours)
        with self.tasks_lock:
            to_remove = []
            for task_id, task in self.tasks.items():
                if task.status in ["completed", "failed", "cancelled"]:
                    if task.completed_at and task.completed_at < cutoff:
                        to_remove.append(task_id)
            for task_id in to_remove:
                del self.tasks[task_id]
            if to_remove:
                logger.info(f"[Rank 0] Cleaned up {len(to_remove)} completed tasks")  # noqa: G004

    def _load_default_config(self, config_path: str | None) -> dict[str, Any]:
        config = {
            "model": "Wan2.2-T2V-A14B",
            "pretrained_model_name_or_path": "/home/models/Wan-AI/Wan2.2-T2V-A14B-Diffusers",
            "task_type": "t2v",
            "dtype": "bf16",
            "sp": 1,
            "fsdp": "all",
            "vae_lightning": "decoder",
            "turbo_mode": "default",
            "inf_vram_blocks_num": 0,
            "atten_a8w8": False,
            "atten_laser": True,
            "matmul_a8w8": True,
            "rope_fused": True,
            "num_inference_steps": 40,
            "save_fps": 16,
            "ten_second": False,
            "ten_second_model_id_t2v": TEN_SECOND_CONFIG["ten_second_model_id_t2v"],
            "ten_second_model_path": TEN_SECOND_CONFIG["ten_second_model_path"],
            "ten_second_model_path_2": TEN_SECOND_CONFIG["ten_second_model_path_2"],
            "pusa_lora": TEN_SECOND_CONFIG["pusa_lora"],
            "pusa_lora2": TEN_SECOND_CONFIG["pusa_lora2"],
            "joint": False,
            "joint_model_path": TEN_SECOND_CONFIG["joint_model_path"],
            "x": False,
            "x_model_path": TEN_SECOND_CONFIG["x_model_path"],
            "x_model_path_2": TEN_SECOND_CONFIG["x_model_path_2"],
            "adopt_sr": False,
            "resolution": 480,
            "seedvr2_model_dir": TEN_SECOND_CONFIG["seedvr2_model_dir"],
            "seedvr2_model_name": TEN_SECOND_CONFIG["seedvr2_model_name"],
            "frame_interpolation": False,
            "frame_model_path": TEN_SECOND_CONFIG["frame_model_path"],
            "external_base_url": None,
            "lora_path_list": None,
        }

        if config_path and os.path.exists(config_path):
            with open(config_path) as f:
                config.update(json.load(f))

        return config

    def _validate_and_convert_param(self, value, param_name, param_type, default=None, min_val=None, max_val=None):
        if value is None:
            return default, None

        try:
            if param_type == int:  # noqa: E721
                converted = int(value)
            elif param_type == float:  # noqa: E721
                converted = float(value)
            elif param_type == str:  # noqa: E721
                converted = str(value)
            else:
                converted = value

            if min_val is not None and converted < min_val:
                return default, f"{param_name} must be >= {min_val}, got {converted}"
            if max_val is not None and converted > max_val:
                return default, f"{param_name} must be <= {max_val}, got {converted}"

            return converted, None
        except (ValueError, TypeError):
            return default, f"{param_name} must be {param_type.__name__}, got {type(value).__name__}"

    ALLOWED_RESOLUTIONS = {
        "480p": [(832, 480), (480, 832)],
        "720p": [(1280, 720), (720, 1280)],
        "1080p": [(1920, 1080), (1080, 1920)],
    }
    KNOWN_PARAMS = {
        "prompt",
        "negative_prompt",
        "width",
        "height",
        "frames",
        "num_inference_steps",
        "seed",
        "guidance_scale",
        "guidance_scale_2",
        "i2v_image_path",
        "save_path",
        "ten_second",
        "adopt_sr",
        "resolution",
        "frame_interpolation",
        "duration",
        "save_fps",
    }

    def _validate_prompt(self, data: dict[str, Any], errors: list[str]) -> str | None:
        prompt = data.get("prompt")
        if not prompt:
            errors.append("prompt is required and cannot be empty")
        elif not isinstance(prompt, str):
            errors.append("prompt must be a string")
        elif len(prompt.strip()) == 0:
            errors.append("prompt cannot be blank")
        elif len(prompt) > 2000:
            errors.append("prompt length cannot exceed 2000 characters")
        return prompt

    def _get_resolution_key(self, width: int, height: int) -> str | None:
        for key, sizes in self.ALLOWED_RESOLUTIONS.items():
            if (width, height) in sizes:
                return key
        return None

    def _validate_resolution_param(self, data: dict[str, Any], errors: list[str]) -> tuple[int, int]:
        resolution = data.get("resolution")
        if resolution is None:
            return self._validate_width_height(data, errors)

        if isinstance(resolution, str):
            resolution = resolution.lower()
            if resolution not in self.ALLOWED_RESOLUTIONS:
                errors.append(
                    f"resolution must be one of: {', '.join(self.ALLOWED_RESOLUTIONS.keys())}, got {resolution}"
                )
                return 832, 480
            return self.ALLOWED_RESOLUTIONS[resolution][0]

        if isinstance(resolution, int):
            res_map = {480: "480p", 720: "720p", 1080: "1080p"}
            if resolution not in res_map:
                errors.append(f"resolution must be 480, 720, or 1080, got {resolution}")
                return 832, 480
            return self.ALLOWED_RESOLUTIONS[res_map[resolution]][0]

        errors.append("resolution must be string (480p/720p/1080p) or int (480/720/1080)")
        return 832, 480

    def _validate_width_height(self, data: dict[str, Any], errors: list[str]) -> tuple[int, int]:
        width = data.get("width")
        height = data.get("height")

        if width is None and height is None:
            return 832, 480

        width, width_err = self._validate_and_convert_param(width, "width", int, default=832, min_val=64, max_val=2048)
        if width_err:
            errors.append(width_err)
        elif width % 2 != 0:
            errors.append(f"width must be even, got {width}")

        height, height_err = self._validate_and_convert_param(
            height, "height", int, default=480, min_val=64, max_val=2048
        )
        if height_err:
            errors.append(height_err)
        elif height % 2 != 0:
            errors.append(f"height must be even, got {height}")

        res_key = self._get_resolution_key(width, height)
        if res_key is None:
            allowed = [f"{w}x{h} ({k})" for k, sizes in self.ALLOWED_RESOLUTIONS.items() for w, h in sizes]
            errors.append(f"Invalid resolution {width}x{height}. Allowed: {', '.join(allowed)}")

        return width, height

    def _validate_duration(self, data: dict[str, Any], errors: list[str]) -> int:
        duration = data.get("duration", 5)
        if duration is None:
            return 5

        duration = self._convert_to_int(duration, errors, "duration")
        if duration not in (5, 10):
            errors.append(f"duration must be 5 or 10, got {duration}")
            return 5
        return duration

    def _convert_to_int(self, value: Any, errors: list[str], name: str) -> int:
        if isinstance(value, int):
            return value
        if isinstance(value, str):
            try:
                return int(value)
            except ValueError:
                errors.append(f"{name} must be an integer")
                return 5
        if isinstance(value, float):
            return int(value)
        errors.append(f"{name} must be an integer")
        return 5

    def _validate_boolean_param(self, value: Any, errors: list[str], name: str) -> bool:
        if value is None:
            return False
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in ("true", "1", "yes")
        if isinstance(value, int):
            return bool(value)
        errors.append(f"{name} must be a boolean")
        return False

    def _validate_save_fps(self, data: dict[str, Any], errors: list[str]) -> int:
        save_fps = data.get("save_fps", 16)
        if save_fps is None:
            return 16
        save_fps = self._convert_to_int(save_fps, errors, "save_fps")
        if save_fps not in (16, 24, 30):
            errors.append(f"save_fps must be 16, 24, or 30, got {save_fps}")
            return 16
        return save_fps

    def _validate_numeric_params(self, data: dict[str, Any], errors: list[str]) -> dict[str, Any]:
        turbo_mode = self.default_config.get("turbo_mode")
        if turbo_mode == "default":
            num = 1
        else:
            num = 10
        params = {}
        param_specs = [
            ("frames", "frames", int, 81, 1, 301),
            ("num_inference_steps", "num_inference_steps", int, 40, num, 200),
            ("seed", "seed", int, 42, 0, 2**31 - 1),
            ("guidance_scale", "guidance_scale", float, None, 0, 100),
            ("guidance_scale_2", "guidance_scale_2", float, None, 0, 100),
        ]
        for key, name, ptype, default, min_val, max_val in param_specs:
            value, err = self._validate_and_convert_param(
                data.get(key), name, ptype, default=default, min_val=min_val, max_val=max_val
            )
            if err:
                errors.append(err)
            params[key] = value
        return params

    def _validate_negative_prompt(self, data: dict[str, Any], errors: list[str]) -> str:
        negative_prompt = data.get("negative_prompt")
        if negative_prompt is None:
            return ""
        if not isinstance(negative_prompt, str):
            errors.append("negative_prompt must be a string")
            return ""
        if len(negative_prompt) > 2000:
            errors.append("negative_prompt length cannot exceed 2000 characters")
            return ""
        return negative_prompt

    def _validate_i2v_params(self, data: dict[str, Any], errors: list[str], warnings: list[str]) -> str | None:
        task_type = self.default_config.get("task_type", "t2v")
        i2v_image_path = data.get("i2v_image_path")
        if task_type == "i2v":
            if not i2v_image_path:
                errors.append("i2v_image_path is required for I2V model")
            elif not isinstance(i2v_image_path, str):
                errors.append("i2v_image_path must be a string")
        elif i2v_image_path is not None:
            warnings.append("i2v_image_path is ignored for T2V model")
        return i2v_image_path

    def _generate_param_warnings(
        self,
        frames: int,
        unknown_params: set,
        ten_second: bool,
        adopt_sr: bool,
        resolution: int,
        save_fps: int,
        frame_interpolation: bool,
        duration: int,
        data: dict,
        warnings: list[str],
    ) -> bool:
        if frames and frames > 81:
            warnings.append(f"Large frame count ({frames}) will significantly increase generation time")
        if unknown_params:
            warnings.append(f"Unknown parameters ignored: {', '.join(unknown_params)}")
        if data.get("ten_second") is not None:
            explicit_ten_second = self._validate_boolean_param(data.get("ten_second"), [], "ten_second")
            if explicit_ten_second != (duration == 10):
                warnings.append(f"ten_second={explicit_ten_second} is overridden by duration={duration}")
        if save_fps > 16 and not frame_interpolation:
            frame_interpolation = True
            warnings.append(f"save_fps={save_fps} > 16, auto-enabling frame_interpolation")
        if ten_second and adopt_sr and resolution > 720:
            warnings.append(f"High resolution ({resolution}p) with 10s + SR may require significant memory")
        return frame_interpolation

    def _validate_negative_prompt(self, data: dict[str, Any], errors: list[str]) -> str:
        negative_prompt = data.get("negative_prompt")
        if negative_prompt is None:
            return ""
        if not isinstance(negative_prompt, str):
            errors.append("negative_prompt must be a string")
            return ""
        if len(negative_prompt) > 2000:
            errors.append("negative_prompt length cannot exceed 2000 characters")
            return ""
        return negative_prompt

    def _validate_i2v_params(self, data: dict[str, Any], errors: list[str], warnings: list[str]) -> str | None:
        task_type = self.default_config.get("task_type", "t2v")
        i2v_image_path = data.get("i2v_image_path")
        if task_type == "i2v":
            if not i2v_image_path:
                errors.append("i2v_image_path is required for I2V model")
            elif not isinstance(i2v_image_path, str):
                errors.append("i2v_image_path must be a string")
        elif i2v_image_path is not None:
            warnings.append("i2v_image_path is ignored for T2V model")
        return i2v_image_path

    def _generate_param_warnings(
        self,
        frames: int,
        unknown_params: set,
        ten_second: bool,
        adopt_sr: bool,
        resolution: int,
        save_fps: int,
        frame_interpolation: bool,
        duration: int,
        data: dict,
        warnings: list[str],
    ) -> bool:
        if frames and frames > 81:
            warnings.append(f"Large frame count ({frames}) will significantly increase generation time")
        if unknown_params:
            warnings.append(f"Unknown parameters ignored: {', '.join(unknown_params)}")
        if data.get("ten_second") is not None:
            explicit_ten_second = self._validate_boolean_param(data.get("ten_second"), [], "ten_second")
            if explicit_ten_second != (duration == 10):
                warnings.append(f"ten_second={explicit_ten_second} is overridden by duration={duration}")
        if save_fps > 16 and not frame_interpolation:
            frame_interpolation = True
            warnings.append(f"save_fps={save_fps} > 16, auto-enabling frame_interpolation")
        if ten_second and adopt_sr and resolution > 720:
            warnings.append(f"High resolution ({resolution}p) with 10s + SR may require significant memory")
        return frame_interpolation

    def _validate_generate_params(self, data: dict[str, Any]) -> tuple:
        errors = []
        warnings = []

        prompt = self._validate_prompt(data, errors)
        width, height = self._validate_resolution_param(data, errors)
        numeric_params = self._validate_numeric_params(data, errors)
        negative_prompt = self._validate_negative_prompt(data, errors)
        i2v_image_path = self._validate_i2v_params(data, errors, warnings)

        unknown_params = set(data.keys()) - self.KNOWN_PARAMS
        duration = self._validate_duration(data, errors)
        ten_second = duration == 10
        adopt_sr = self._validate_boolean_param(data.get("adopt_sr"), errors, "adopt_sr")

        resolution = 480
        if width == 1280 and height == 720:
            resolution = 720
        elif width == 1920 and height == 1080:
            resolution = 1080

        frame_interpolation = self._validate_boolean_param(
            data.get("frame_interpolation"), errors, "frame_interpolation"
        )
        save_fps = self._validate_save_fps(data, errors)

        frame_interpolation = self._generate_param_warnings(
            numeric_params["frames"],
            unknown_params,
            ten_second,
            adopt_sr,
            resolution,
            save_fps,
            frame_interpolation,
            duration,
            data,
            warnings,
        )

        if errors:
            return False, {"errors": errors, "warnings": warnings}

        validated_data = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "width": width,
            "height": height,
            "frames": numeric_params["frames"],
            "num_inference_steps": numeric_params["num_inference_steps"],
            "seed": numeric_params["seed"],
            "guidance_scale": numeric_params["guidance_scale"],
            "guidance_scale_2": numeric_params["guidance_scale_2"],
            "ten_second": ten_second,
            "adopt_sr": adopt_sr,
            "resolution": resolution,
            "frame_interpolation": frame_interpolation,
            "duration": duration,
            "save_fps": save_fps,
        }
        if i2v_image_path:
            validated_data["i2v_image_path"] = i2v_image_path
        if data.get("save_path"):
            validated_data["save_path"] = data["save_path"]

        return True, {"data": validated_data, "warnings": warnings}

    def _setup_health_routes(self):
        @self.app.route("/health", methods=["GET"])
        def health_check():
            current_model = None
            if self.inference_manager and hasattr(self.inference_manager, "current_model"):
                current_model = self.inference_manager.current_model

            return jsonify(
                {
                    "status": "healthy",
                    "initialized": self.is_initialized,
                    "current_model": current_model,
                    "rank": self.rank,
                    "world_size": self.world_size,
                    "supported_models": list(MODEL_REGISTRY.keys()),
                    "timestamp": datetime.now().isoformat(),
                }
            )

        @self.app.route("/models", methods=["GET"])
        def list_models():
            current_model = None
            if self.inference_manager and hasattr(self.inference_manager, "current_model"):
                current_model = self.inference_manager.current_model

            return jsonify(
                {
                    "current_model": current_model,
                    "supported_models": list(MODEL_REGISTRY.keys()),
                    "model_info": MODEL_REGISTRY,
                    "note": "To switch model, restart the service with different config",
                }
            )

        @self.app.route("/init", methods=["POST"])
        def init_service():
            try:
                data = request.get_json() or {}  # noqa: F841

                init_env()
                self.inference_manager = VideoInferenceManager()
                self.is_initialized = True

                logger.info(f"[Rank {self.rank}] VideoGenerationService initialized successfully")  # noqa: G004
                return jsonify({"status": "success", "message": "Service initialized", "rank": self.rank})
            except Exception as e:
                logger.error(f"[Rank {self.rank}] Failed to initialize service: {str(e)}", exc_info=True)  # noqa: G004, G201
                return jsonify({"status": "error", "message": str(e)}), 500

    def _setup_task_routes(self):
        @self.app.route("/generate", methods=["POST"])
        def submit_task():
            try:
                if not self.is_initialized:
                    return jsonify({"status": "error", "message": "Service not initialized. Call /init first"}), 400

                if self.shutdown:
                    return jsonify({"status": "error", "message": "Service is shutting down"}), 503

                self._cleanup_completed_tasks()

                with self.tasks_lock:
                    if len(self.tasks) >= self.max_tasks:
                        return jsonify(
                            {
                                "status": "error",
                                "message": f"Too many tasks (max: {self.max_tasks}). Please wait for tasks to complete.",  # noqa: E501
                            }
                        ), 429

                data = request.get_json()
                if not data:
                    return jsonify({"status": "error", "message": "No JSON data provided"}), 400

                is_valid, result = self._validate_generate_params(data)
                if not is_valid:
                    return jsonify(
                        {"status": "error", "message": "Parameter validation failed", "details": result}
                    ), 400

                validated_data = result["data"]
                warnings = result.get("warnings", [])

                task_id = str(uuid.uuid4())
                task = AsyncTask(task_id=task_id, task_type="video_generation", params=validated_data)

                with self.tasks_lock:
                    self.tasks[task_id] = task

                save_path = validated_data.get("save_path")
                if not save_path:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    save_path = f"/tmp/generated_videos/video_{task_id[:8]}_{timestamp}.mp4"

                try:
                    self.task_queue.put(
                        {"task_id": task_id, "data": validated_data, "save_path": save_path}, block=False
                    )
                except queue.Full:
                    with self.tasks_lock:
                        del self.tasks[task_id]
                    return jsonify({"status": "error", "message": "Task queue is full. Please retry later."}), 503

                logger.info(f"[Rank 0] Task {task_id} submitted")  # noqa: G004
                response = {"status": "accepted", "task_id": task_id, "message": "Task submitted successfully"}
                if warnings:
                    response["warnings"] = warnings
                return jsonify(response)

            except Exception as e:
                logger.error(f"[Rank {self.rank}] Failed to submit task: {str(e)}", exc_info=True)  # noqa: G004, G201
                return jsonify({"status": "error", "message": str(e)}), 500

        @self.app.route("/status/<task_id>", methods=["GET"])
        def get_task_status(task_id):
            with self.tasks_lock:
                task = self.tasks.get(task_id)

            if not task:
                return jsonify({"status": "error", "message": f"Task {task_id} not found"}), 404

            return jsonify(task.to_dict())

        @self.app.route("/result/<task_id>", methods=["GET"])
        def get_task_result(task_id):
            with self.tasks_lock:
                task = self.tasks.get(task_id)

            if not task:
                return jsonify({"status": "error", "message": f"Task {task_id} not found"}), 404

            if task.status != "completed":
                return jsonify(
                    {"status": "error", "message": f"Task is not completed. Current status: {task.status}"}
                ), 400

            result = task.to_dict()
            result["video_path"] = task.result.get("video_path")
            return jsonify(result)

        @self.app.route("/download/<task_id>", methods=["GET"])
        def download_task_result(task_id):
            with self.tasks_lock:
                task = self.tasks.get(task_id)

            if not task:
                return jsonify({"status": "error", "message": f"Task {task_id} not found"}), 404

            if task.status != "completed":
                return jsonify(
                    {"status": "error", "message": f"Task is not completed. Current status: {task.status}"}
                ), 400

            video_path = task.result.get("video_path")
            if not video_path or not os.path.exists(video_path):
                return jsonify({"status": "error", "message": f"Video file not found: {video_path}"}), 404

            filename = os.path.basename(video_path)
            return send_file(video_path, as_attachment=True, download_name=filename, mimetype="video/mp4")

        @self.app.route("/tasks", methods=["GET"])
        def list_tasks():
            with self.tasks_lock:
                tasks_list = [task.to_dict() for task in self.tasks.values()]
            return jsonify({"tasks": tasks_list, "count": len(tasks_list)})

        @self.app.route("/cancel/<task_id>", methods=["POST"])
        def cancel_task(task_id):
            with self.tasks_lock:
                task = self.tasks.get(task_id)

                if not task:
                    return jsonify({"status": "error", "message": f"Task {task_id} not found"}), 404

                if task.status in ["completed", "failed", "cancelled"]:
                    return jsonify(
                        {"status": "error", "message": f"Cannot cancel task with status: {task.status}"}
                    ), 400

                task.status = "cancelled"
                task.completed_at = datetime.now()
                logger.info(f"[Rank 0] Task {task_id} cancelled")  # noqa: G004
                return jsonify({"status": "success", "message": f"Task {task_id} cancelled"})

    def _setup_config_routes(self):
        @self.app.route("/params", methods=["GET"])
        def get_param_spec():
            return jsonify(
                {
                    "parameters": {
                        "prompt": {
                            "type": "string",
                            "required": True,
                            "description": "Text prompt for video generation",
                            "constraints": {"min_length": 1, "max_length": 2000},
                        },
                        "negative_prompt": {
                            "type": "string",
                            "required": False,
                            "default": "",
                            "description": "Negative prompt to avoid certain features",
                            "constraints": {"max_length": 2000},
                        },
                        "width": {
                            "type": "integer",
                            "required": False,
                            "default": 832,
                            "description": "Video width in pixels",
                            "constraints": {"min": 64, "max": 2048},
                        },
                        "height": {
                            "type": "integer",
                            "required": False,
                            "default": 480,
                            "description": "Video height in pixels",
                            "constraints": {"min": 64, "max": 2048},
                        },
                        "frames": {
                            "type": "integer",
                            "required": False,
                            "default": 81,
                            "description": "Number of frames to generate",
                            "constraints": {"min": 1, "max": 200},
                        },
                        "num_inference_steps": {
                            "type": "integer",
                            "required": False,
                            "default": 40,
                            "description": "Number of denoising steps",
                            "constraints": {"min": 1, "max": 200},
                        },
                        "seed": {
                            "type": "integer",
                            "required": False,
                            "default": 42,
                            "description": "Random seed for reproducibility",
                            "constraints": {"min": 0},
                        },
                        "guidance_scale": {
                            "type": "number",
                            "required": False,
                            "default": None,
                            "description": "First guidance scale (model dependent)",
                            "constraints": {"min": 0},
                        },
                        "guidance_scale_2": {
                            "type": "number",
                            "required": False,
                            "default": None,
                            "description": "Second guidance scale (model dependent)",
                            "constraints": {"min": 0},
                        },
                        "save_path": {
                            "type": "string",
                            "required": False,
                            "default": "auto-generated",
                            "description": "Path to save the generated video",
                        },
                        "i2v_image_path": {
                            "type": "string",
                            "required": False,
                            "description": "Input image path for i2v tasks",
                        },
                        "ten_second": {
                            "type": "boolean",
                            "required": False,
                            "default": False,
                            "description": "Enable 10-second video generation (two-stage: 5s base + 5s extension)",
                        },
                        "adopt_sr": {
                            "type": "boolean",
                            "required": False,
                            "default": False,
                            "description": "Enable super-resolution (SeedVR2)",
                        },
                        "resolution": {
                            "type": "integer",
                            "required": False,
                            "default": 480,
                            "description": "Target resolution for super-resolution (480/720/1080)",
                            "constraints": {"min": 480, "max": 2160},
                        },
                        "frame_interpolation": {
                            "type": "boolean",
                            "required": False,
                            "default": False,
                            "description": "Enable frame interpolation for smoother video",
                        },
                        "duration": {
                            "type": "integer",
                            "required": False,
                            "default": 5,
                            "description": "Video duration in seconds (5 or 10). duration=10 enables two-stage generation.",  # noqa: E501
                            "constraints": {"allowed_values": [5, 10]},
                        },
                    },
                    "example": {
                        "prompt": "A cat playing with a ball",
                        "width": 832,
                        "height": 480,
                        "frames": 81,
                        "num_inference_steps": 40,
                        "seed": 42,
                        "duration": 5,
                    },
                    "example_10s": {
                        "prompt": "A cat playing with a ball",
                        "width": 832,
                        "height": 480,
                        "frames": 81,
                        "num_inference_steps": 6,
                        "seed": 42,
                        "duration": 10,
                        "adopt_sr": True,
                        "resolution": 1080,
                    },
                }
            )

        @self.app.route("/config", methods=["GET", "POST"])
        def manage_config():
            if request.method == "GET":
                return jsonify(self.default_config)
            else:
                data = request.get_json()
                self.default_config.update(data)
                return jsonify({"status": "success", "config": self.default_config})

        @self.app.route("/metrics", methods=["GET"])
        def get_metrics():
            with self.tasks_lock:
                completed = [t for t in self.tasks.values() if t.status == "completed" and t.infer_time]
                if completed:
                    infer_times = [t.infer_time for t in completed]
                    avg_infer_time = sum(infer_times) / len(infer_times)
                    min_infer_time = min(infer_times)
                    max_infer_time = max(infer_times)
                else:
                    avg_infer_time = min_infer_time = max_infer_time = 0

                total_tasks = len(self.tasks)
                running_tasks = sum(1 for t in self.tasks.values() if t.status == "running")
                pending_tasks = sum(1 for t in self.tasks.values() if t.status == "pending")
                failed_tasks = sum(1 for t in self.tasks.values() if t.status == "failed")

            return jsonify(
                {
                    "model": self.default_config.get("model"),
                    "world_size": self.world_size,
                    "tasks": {
                        "total": total_tasks,
                        "completed": len(completed),
                        "running": running_tasks,
                        "pending": pending_tasks,
                        "failed": failed_tasks,
                    },
                    "performance": {
                        "avg_infer_time": round(avg_infer_time, 2),
                        "min_infer_time": round(min_infer_time, 2),
                        "max_infer_time": round(max_infer_time, 2),
                    },
                }
            )

    def _parse_string_input(self, external_input: str, parsed: dict[str, Any]) -> bool:
        external_input = external_input.strip()
        if not external_input:
            raise ValueError("input string cannot be empty")

        if external_input.startswith("{") or external_input.startswith("["):
            try:
                return False
            except json.JSONDecodeError:
                parsed["prompt"] = external_input
                return True
        else:
            parsed["prompt"] = external_input
            return True

    def _parse_image_data(self, image_data: Any, parsed: dict[str, Any]) -> None:
        if isinstance(image_data, list) and len(image_data) > 0:
            parsed["i2v_image_path"] = image_data[0]
        elif isinstance(image_data, str):
            parsed["i2v_image_path"] = image_data

    def _extract_basic_params(self, input_dict: dict[str, Any], parsed: dict[str, Any]) -> None:
        for key in ["prompt", "negative_prompt"]:
            if key in input_dict:
                parsed[key] = input_dict[key]

        if "image" in input_dict:
            self._parse_image_data(input_dict["image"], parsed)

        for key in ["width", "height", "frames", "seed", "num_inference_steps", "i2v_image_path", "save_fps"]:
            if key in input_dict:
                parsed[key] = input_dict[key]

    def _parse_size_param(self, size: str, parsed: dict[str, Any]) -> None:
        if isinstance(size, str) and "*" in size:
            try:
                w, h = map(int, size.split("*"))
                parsed["width"] = w
                parsed["height"] = h
            except ValueError:
                pass
        elif isinstance(size, str):
            size_lower = size.lower()
            size_map = {"480": (832, 480), "720": (1280, 720), "1080": (1920, 1080), "2k": (1920, 1080)}
            for key, (w, h) in size_map.items():
                if key in size_lower:
                    parsed["width"], parsed["height"] = w, h
                    break

    def _parse_parameters(self, params: dict[str, Any], parsed: dict[str, Any]) -> None:
        if "size" in params:
            self._parse_size_param(params["size"], parsed)

        for key in ["duration", "seed", "num_inference_steps"]:
            if key in params:
                parsed[key] = params[key]

        if "fps" in params:
            parsed["save_fps"] = params["fps"]

    def _detect_task_type(self, external_input: dict[str, Any], parsed: dict[str, Any]) -> None:
        if "model" not in external_input:
            return
        model_name = external_input["model"]
        for alias, info in MODEL_REGISTRY.items():
            if model_name in [alias, info["model"], info.get("model", "")]:
                parsed["task_type"] = info["task_type"]
                break

    def _parse_dict_input(self, external_input: dict[str, Any], parsed: dict[str, Any]) -> None:
        if "input" in external_input and isinstance(external_input["input"], dict):
            inner_input = external_input["input"]
            self._extract_basic_params(inner_input, parsed)
        else:
            self._extract_basic_params(external_input, parsed)

        if "parameters" in external_input and isinstance(external_input["parameters"], dict):
            self._parse_parameters(external_input["parameters"], parsed)

        self._detect_task_type(external_input, parsed)

    def _parse_external_input(self, external_input: Any) -> dict[str, Any]:
        parsed = {}

        if external_input is None:
            raise ValueError("input cannot be None")

        if isinstance(external_input, str):
            if self._parse_string_input(external_input, parsed):
                return parsed
            try:
                external_input = json.loads(external_input)
            except json.JSONDecodeError:
                return parsed

        if isinstance(external_input, dict):
            self._parse_dict_input(external_input, parsed)

        if "prompt" not in parsed:
            raise ValueError("Could not extract prompt from input")

        return parsed

    def _setup_internal_task_routes(self):
        @self.app.route("/v1/tasks", methods=["POST"])
        def internal_create_task():
            try:
                if not self.is_initialized:
                    return jsonify({"error": "Service not initialized"}), 503

                if self.shutdown:
                    return jsonify({"error": "Service is shutting down"}), 503

                data = request.get_data()
                if not data:
                    return jsonify({"error": "No JSON data provided"}), 400
                data = json.loads(data)
                task_id = data.get("id")
                if not task_id:
                    return jsonify({"error": "Missing required field: id"}), 400

                external_input = data.get("input")
                created_at_ms = data.get("created_at")

                try:
                    parsed_params = self._parse_external_input(external_input)
                except ValueError as e:
                    logger.error(f"[Rank 0] Failed to parse input for task {task_id}: {e}")  # noqa: G004
                    return jsonify({"id": task_id, "status": "failed", "output": "", "error": str(e)}), 400

                with self.tasks_lock:
                    if task_id in self.tasks:
                        existing = self.tasks[task_id]
                        return jsonify(existing.to_platform_dict()), 200

                is_valid, result = self._validate_generate_params(parsed_params)
                if not is_valid:
                    error_msg = "; ".join(result.get("errors", []))
                    logger.error(f"[Rank 0] Validation failed for task {task_id}: {error_msg}")  # noqa: G004
                    validated_data = None
                    task = AsyncTask(
                        task_id=task_id,
                        task_type="video_generation",
                        params=validated_data,
                        created_at_ms=created_at_ms,
                    )
                    task.output = error_msg
                    task.status = "failed"
                    self.tasks[task_id] = task
                    return jsonify({"id": task_id, "status": "failed", "output": "", "error": error_msg}), 400

                validated_data = result["data"]

                task = AsyncTask(
                    task_id=task_id, task_type="video_generation", params=validated_data, created_at_ms=created_at_ms
                )

                with self.tasks_lock:
                    self.tasks[task_id] = task

                save_path = validated_data.get("save_path")
                if not save_path:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    save_path = f"/tmp/generated_videos/video_{task_id[:8]}_{timestamp}.mp4"

                try:
                    self.task_queue.put(
                        {"task_id": task_id, "data": validated_data, "save_path": save_path}, block=False
                    )
                except queue.Full:
                    with self.tasks_lock:
                        del self.tasks[task_id]
                    return jsonify(
                        {"id": task_id, "status": "failed", "output": "", "error": "Task queue is full"}
                    ), 503

                logger.info(f"[Rank 0] Internal task {task_id} created and queued")  # noqa: G004
                return jsonify(task.to_platform_dict()), 200

            except Exception as e:
                logger.error(f"[Rank 0] Failed to create internal task: {str(e)}", exc_info=True)  # noqa: G004, G201
                return jsonify({"error": str(e)}), 500

    def _setup_internal_query_routes(self):
        @self.app.route("/v1/tasks", methods=["GET"])
        def internal_list_tasks():
            try:
                with self.tasks_lock:
                    tasks_list = [task.to_platform_dict() for task in self.tasks.values()]
                return jsonify({"tasks": tasks_list}), 200
            except Exception as e:
                logger.error(f"[Rank 0] Failed to list tasks: {str(e)}", exc_info=True)  # noqa: G004, G201
                return jsonify({"error": str(e)}), 500

        @self.app.route("/v1/tasks/<task_id>", methods=["GET"])
        def internal_get_task(task_id):
            try:
                with self.tasks_lock:
                    task = self.tasks.get(task_id)

                if not task:
                    return jsonify({"error": f"Task {task_id} not found"}), 404

                return jsonify(task.to_platform_dict()), 200
            except Exception as e:
                logger.error(f"[Rank 0] Failed to get task: {str(e)}", exc_info=True)  # noqa: G004, G201
                return jsonify({"error": str(e)}), 500

        @self.app.route("/v1/tasks/<task_id>", methods=["DELETE"])
        def internal_delete_task(task_id):
            try:
                with self.tasks_lock:
                    task = self.tasks.get(task_id)

                    if not task:
                        return jsonify({"error": f"Task {task_id} not found"}), 404

                    if task.status in ["running"]:
                        task.status = "cancelled"
                        task.completed_at = datetime.now()
                        task.error = "Task cancelled by user"
                    elif task.status in ["pending", "queued"]:
                        task.status = "cancelled"
                        task.completed_at = datetime.now()
                        task.error = "Task cancelled before execution"
                    else:
                        del self.tasks[task_id]
                        logger.info(f"[Rank 0] Task {task_id} deleted")  # noqa: G004
                        return jsonify({"id": task_id}), 200

                logger.info(f"[Rank 0] Task {task_id} cancelled")  # noqa: G004
                return jsonify({"id": task_id}), 200

            except Exception as e:
                logger.error(f"[Rank 0] Failed to delete task: {str(e)}", exc_info=True)  # noqa: G004, G201
                return jsonify({"error": str(e)}), 500

    def _update_basic_config(self, config: dict[str, Any], data: dict[str, Any], prompt: str | None) -> None:
        update_config = {
            "negative_prompt": data.get("negative_prompt", ""),
            "height": data.get("height", 480),
            "width": data.get("width", 832),
            "frames": data.get("frames", 81),
            "num_inference_steps": data.get("num_inference_steps", 40),
            "seed": data.get("seed", 42),
        }
        if prompt:
            update_config["prompt"] = prompt
        config.update(update_config)

    def _update_guidance_config(self, config: dict[str, Any], data: dict[str, Any]) -> None:
        if data.get("guidance_scale") is not None:
            config["guidance_scale"] = data["guidance_scale"]
        if data.get("guidance_scale_2") is not None:
            config["guidance_scale_2"] = data["guidance_scale_2"]

    def _update_path_config(self, config: dict[str, Any], data: dict[str, Any], save_path: str | None) -> None:
        if save_path:
            config["save_path"] = save_path
        elif data.get("save_path"):
            config["save_path"] = data["save_path"]
        if data.get("i2v_image_path"):
            config["i2v_image_path"] = data["i2v_image_path"]

    def _apply_ten_second_config(self, config: dict[str, Any]) -> None:
        config["ten_second"] = True
        config["fsdp"] = "text_encoder"
        config["inf_vram_blocks_num"] = 1
        config["save_memory"] = True
        config["vae_lightning"] = "decoder"
        config["atten_laser"] = True
        config["matmul_a8w8"] = True
        config["rope_fused"] = True

    def _update_ten_second_config(self, config: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
        duration = data.get("duration", 5)
        if duration == 10:
            data = dict(data)
            data["ten_second"] = True
        ten_second = data.get("ten_second", False)
        if not ten_second:
            return data
        self._apply_ten_second_config(config)
        task_type = config.get("task_type", "t2v")
        sp = config.get("sp", 1)
        logger.info(f"[Rank {self.rank}] 10s mode with {sp} GPU(s)")  # noqa: G004
        if task_type == "i2v":
            config["joint"] = False
        else:
            config["x_model_path"] = TEN_SECOND_CONFIG["ten_second_model_path"]
            config["x_model_path_2"] = TEN_SECOND_CONFIG["ten_second_model_path_2"]
        return data

    def _update_resolution_config(self, config: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
        width = data.get("width")
        height = data.get("height")
        if width and height and ((width == 1920 and height == 1080) or (width == 1080 and height == 1920)):
            data = dict(data)
            data["resolution"] = 1080
            logger.info(f"[Rank {self.rank}] Detected 1080p from {width}x{height}")  # noqa: G004
        if data.get("adopt_sr", False):
            config["adopt_sr"] = True
            config["resolution"] = data.get("resolution", 1080)
        resolution = data.get("resolution", 480)
        if resolution == 1080:
            config["adopt_sr"] = True
            config["resolution"] = 1080
            config["seedvr2_model_dir"] = data.get("seedvr2_model_dir", TEN_SECOND_CONFIG["seedvr2_model_dir"])
            config["seedvr2_model_name"] = data.get("seedvr2_model_name", TEN_SECOND_CONFIG["seedvr2_model_name"])
        return data

    def _update_fps_config(self, config: dict[str, Any], data: dict[str, Any]) -> None:
        save_fps = data.get("save_fps", 16)
        config["save_fps"] = save_fps
        if save_fps > 16 or data.get("frame_interpolation", False):
            config["frame_interpolation"] = True
            config["frame_model_path"] = data.get("frame_model_path", TEN_SECOND_CONFIG["frame_model_path"])

    def _config_to_arg_list(self, config: dict[str, Any]) -> list[str]:
        excluded_keys = {"external_base_url"}
        arg_list = []
        for key, value in config.items():
            if key in excluded_keys or value is None or value == "":
                continue
            if isinstance(value, bool):
                if value:
                    arg_list.append(f"--{key}")
            elif isinstance(value, list):
                arg_list.append(f"--{key}")
                arg_list.extend([str(v) for v in value])
            else:
                arg_list.extend([f"--{key}", str(value)])
        return arg_list

    def _build_args(self, data: dict[str, Any], save_path: str | None = None, require_prompt: bool = True):
        config = self.default_config.copy()
        prompt = data.get("prompt")
        if require_prompt and not prompt:
            raise ValueError("prompt is required in request")

        self._update_basic_config(config, data, prompt)
        self._update_guidance_config(config, data)
        self._update_path_config(config, data, save_path)
        data = self._update_ten_second_config(config, data)
        data = self._update_resolution_config(config, data)
        self._update_fps_config(config, data)

        if not config.get("adopt_sr", False):
            config.pop("resolution", None)

        arg_list = self._config_to_arg_list(config)
        old_argv = sys.argv
        sys.argv = ["video_server.py"] + arg_list
        try:
            args = parse_args()
        finally:
            sys.argv = old_argv
        return args

    def _get_video_resolution(self, video_path: str) -> tuple[int, int]:
        try:
            cmd = [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height",
                "-of",
                "csv=s=x:p=0",
                video_path,
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if result.returncode == 0 and result.stdout.strip():
                w, h = map(int, result.stdout.strip().split("x"))
                return w, h
        except Exception as e:
            logger.warning(f"Failed to get video resolution: {e}")  # noqa: G004
        return 0, 0

    def _resize_video_to_target(self, video_path: str, target_width: int, target_height: int) -> bool:
        w, h = self._get_video_resolution(video_path)
        if w == target_width and h == target_height:
            return True
        if w == 0 or h == 0:
            logger.error(f"Cannot get video resolution for {video_path}")  # noqa: G004
            return False

        temp_path = video_path + ".temp.mp4"
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            video_path,
            "-vf",
            f"scale={target_width}:{target_height}:force_original_aspect_ratio=decrease,pad={target_width}:{target_height}:(ow-iw)/2:(oh-ih)/2",
            "-q:v",
            "2",
            temp_path,
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode == 0 and os.path.exists(temp_path):
                os.replace(temp_path, video_path)
                logger.info(f"Resized video from {w}x{h} to {target_width}x{target_height}")  # noqa: G004
                return True
            else:
                logger.error(f"ffmpeg failed: {result.stderr}")  # noqa: G004
                if os.path.exists(temp_path):
                    os.remove(temp_path)
        except Exception as e:
            logger.error(f"Failed to resize video: {e}")  # noqa: G004
        return False

    def _prepare_task(self, task_id: str) -> bool:
        with self.tasks_lock:
            task = self.tasks.get(task_id)
            if not task:
                logger.warning(f"[Rank 0] Task {task_id} not found, skipping")  # noqa: G004
                return False
            if task.status == "cancelled":
                logger.info(f"[Rank 0] Task {task_id} was cancelled, skipping")  # noqa: G004
                return False
            task.status = "running"
            task.started_at = datetime.now()
            return True

    def _set_guidance_scale(self, args):
        if args.guidance_scale is None or args.guidance_scale_2 is None:
            args.guidance_scale, args.guidance_scale_2 = GUIDANCE_SCALE_MAP[args.model]
            logger.info(f"Guidance scale set to: {args.guidance_scale}, {args.guidance_scale_2}")  # noqa: G004

    def _save_video(self, output, save_path: str, args):
        if not getattr(args, "adopt_sr", False):
            export_to_video(output, save_path, quality=8, fps=args.save_fps)
        else:
            resolution = getattr(args, "resolution", 1080)
            if resolution == 1080:
                self._resize_video_to_target(save_path, 1920, 1080)

    def _complete_task(self, task_id: str, save_path: str, infer_time: float):
        with self.tasks_lock:
            task = self.tasks.get(task_id)
            if not task or task.status != "running":
                return

            task.status = "completed"
            task.progress = 100.0
            result_data = {"video_path": save_path, "infer_time": round(infer_time, 2)}

            external_base_url = self.default_config.get("external_base_url")
            if external_base_url:
                download_url = f"{external_base_url.rstrip('/')}/download/{task_id}"
                result_data["download_url"] = download_url

            obs_local_path = os.environ.get("OBS_LOCAL_PATH")
            if obs_local_path:
                file_util.copy_file(save_path, obs_local_path)
                obs_client = ObsStorageClient()
                filename = Path(save_path).name
                download_url = obs_client.generate_url(filename)
                obs_client.close()
                result_data["download_url"] = download_url

            task.result = result_data
            task.output = json.dumps(result_data)
            task.completed_at = datetime.now()
            task.infer_time = infer_time

    def _handle_task_interruption(self, task_id: str, error: str):
        logger.info(f"[Rank 0] Task {task_id} interrupted: {error}")  # noqa: G004
        with self.tasks_lock:
            task = self.tasks.get(task_id)
            if task:
                task.status = "cancelled"
                task.error = error
                task.completed_at = datetime.now()

        self._clear_cache()
        logger.info(f"[Rank 0] NPU resources released for cancelled task {task_id}")  # noqa: G004

    def _handle_task_failure(self, task_id: str, error: str):
        logger.error(f"[Rank 0] Task {task_id} failed: {error}", exc_info=True)  # noqa: G004
        with self.tasks_lock:
            task = self.tasks.get(task_id)
            if task:
                task.status = "failed"
                task.error = error
                task.completed_at = datetime.now()

    def _clear_cache(self):
        gc.collect()
        if hasattr(torch, "npu") and torch.npu.is_available():
            torch.npu.empty_cache()
        else:
            torch.cuda.empty_cache()

    def _process_task(self, task_id: str, data: dict[str, Any], save_path: str):
        try:
            if not self._prepare_task(task_id):
                return

            args = self._build_args(data, save_path)
            self._set_guidance_scale(args)

            self.inference_manager.initialize_pipe(args)

            infer_info.update_info(args)
            infer_info.save_path = save_path

            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            torch.cuda.synchronize()
            start_time = time.time()

            output = self.inference_manager.infer_single(args, task_id=task_id, task_manager=self)

            torch.cuda.synchronize()
            infer_time = time.time() - start_time

            self._save_video(output, save_path, args)
            self._complete_task(task_id, save_path, infer_time)

            logger.info(f"[Rank 0] Task {task_id} completed in {infer_time:.2f}s")  # noqa: G004

        except InterruptedError as e:
            self._handle_task_interruption(task_id, str(e))
        except Exception as e:
            self._handle_task_failure(task_id, str(e))

    def _worker_loop(self):
        logger.info(f"[Rank {self.rank}] Worker ready, entering request loop...")  # noqa: G004

        while True:
            try:
                command = broadcast_int(CMD_HEARTBEAT, src=0)

                if command == CMD_HEARTBEAT:
                    continue

                if command == CMD_SHUTDOWN:
                    logger.info(f"[Rank {self.rank}] Received shutdown command, exiting worker loop")  # noqa: G004
                    break

                if command == CMD_EXECUTE:
                    broadcast_data = broadcast_dict({}, src=0)
                    data = broadcast_data["data"]
                    save_path = broadcast_data["save_path"]

                    dist.barrier()
                    logger.info(f"[Rank {self.rank}] All ranks ready, starting inference")  # noqa: G004

                    args = self._build_args(data, save_path)

                    if args.guidance_scale is None or args.guidance_scale_2 is None:
                        args.guidance_scale, args.guidance_scale_2 = GUIDANCE_SCALE_MAP[args.model]

                    self.inference_manager.initialize_pipe(args)
                    infer_info.update_info(args)
                    infer_info.save_path = save_path
                    self.inference_manager.infer_single(args)

            except Exception as e:
                logger.error(f"[Rank {self.rank}] Worker loop error: {str(e)}", exc_info=True)  # noqa: G004, G201
                time.sleep(1)

    def _task_processor_thread(self):
        heartbeat_interval = 30
        last_heartbeat_time = time.time()

        while not self.shutdown:
            try:
                try:
                    task_info = self.task_queue.get(timeout=1.0)
                except queue.Empty:
                    current_time = time.time()
                    if self.world_size > 1 and (current_time - last_heartbeat_time) >= heartbeat_interval:
                        broadcast_int(CMD_HEARTBEAT, src=0)
                        last_heartbeat_time = current_time
                    continue

                if self.world_size > 1:
                    broadcast_int(CMD_EXECUTE, src=0)
                    broadcast_dict({"data": task_info["data"], "save_path": task_info["save_path"]}, src=0)
                    dist.barrier()
                    logger.info("[Rank 0] All workers ready, starting task execution")

                self._process_task(task_info["task_id"], task_info["data"], task_info["save_path"])

                last_heartbeat_time = time.time()

                gc.collect()
                if hasattr(torch, "npu") and torch.npu.is_available():
                    torch.npu.empty_cache()
                else:
                    torch.cuda.empty_cache()

            except Exception as e:
                logger.error(f"[Rank 0] Task processor error: {str(e)}", exc_info=True)  # noqa: G004, G201
                time.sleep(1)

    def _cleanup_thread(self):
        while not self.shutdown:
            try:
                time.sleep(self.cleanup_interval)
                if not self.shutdown:
                    self._cleanup_completed_tasks()
            except Exception as e:
                logger.error(f"[Rank 0] Cleanup thread error: {str(e)}", exc_info=True)  # noqa: G004, G201

    def run(self, host: str = "0.0.0.0", port: int = 5001, debug: bool = False):
        if self.world_size > 1 and self.rank != 0:
            self._worker_loop()
            return

        processor_thread = threading.Thread(target=self._task_processor_thread, daemon=True)
        processor_thread.start()
        logger.info("[Rank 0] Task processor thread started")

        cleanup_thread = threading.Thread(target=self._cleanup_thread, daemon=True)
        cleanup_thread.start()
        logger.info(f"[Rank 0] Cleanup thread started (interval: {self.cleanup_interval}s)")  # noqa: G004

        logger.info(f"[Rank 0] Starting VideoGenerationService on {host}:{port}")  # noqa: G004
        self.app.run(host=host, port=port, debug=debug, threaded=True)

    def _initialize_model(self):
        if not self.is_initialized or self.inference_manager is None:
            logger.warning(f"[Rank {self.rank}] Service not initialized, skip model loading")  # noqa: G004
            return

        if self.inference_manager.pipe is not None:
            logger.info(f"[Rank {self.rank}] Model already loaded")  # noqa: G004
            return

        init_args = self._build_args({}, require_prompt=False)

        if init_args.guidance_scale is None or init_args.guidance_scale_2 is None:
            init_args.guidance_scale, init_args.guidance_scale_2 = GUIDANCE_SCALE_MAP[init_args.model]

        logger.info(f"[Rank {self.rank}] Loading model...")  # noqa: G004
        self.inference_manager.initialize_pipe(init_args)
        logger.info(f"[Rank {self.rank}] Model loaded successfully")  # noqa: G004

    def warmup(self, warmup_steps: int = 10):
        if not self.is_initialized or self.inference_manager is None:
            logger.warning("[Rank 0] Service not initialized, skip warmup")
            return

        if self.rank != 0:
            return

        task_type = self.default_config.get("task_type", "t2v")
        ten_second = self.default_config.get("ten_second", False)
        duration = 10 if ten_second else 5

        logger.info(
            f"[Rank 0] Submitting warmup task (duration={duration}s, resolution=1080P, task_type={task_type})..."  # noqa: G004
        )

        warmup_params = {
            "prompt": "warmup test",
            "width": 1920,
            "height": 1080,
            "frames": 16 * duration + 1,
            "num_inference_steps": warmup_steps,
            "seed": 0,
            "duration": duration,
            "guidance_scale": 3.5,
            "guidance_scale_2": 3.5,
        }

        if task_type == "i2v":
            warmup_image_path = "/tmp/warmup_input.png"
            warmup_image = Image.new("RGB", (1920, 1080), color=(128, 128, 128))
            warmup_image.save(warmup_image_path)
            warmup_params["i2v_image_path"] = warmup_image_path
            logger.info(f"[Rank 0] Created warmup image: {warmup_image_path}")  # noqa: G004

        task_id = str(uuid.uuid4())
        save_path = "/tmp/warmup_video.mp4"
        task = AsyncTask(task_id=task_id, task_type="warmup", params=warmup_params)

        with self.tasks_lock:
            self.tasks[task_id] = task

        try:
            self.task_queue.put({"task_id": task_id, "data": warmup_params, "save_path": save_path}, block=False)
            logger.info(f"[Rank 0] Warmup task {task_id} submitted")  # noqa: G004
        except queue.Full:
            with self.tasks_lock:
                del self.tasks[task_id]
            logger.warning("[Rank 0] Task queue full, skip warmup")


def _apply_ten_second_config_args(args) -> None:
    """Apply command line arguments to TEN_SECOND_CONFIG."""

    arg_to_config = [
        "ten_second_model_id_t2v",
        "ten_second_model_path",
        "ten_second_model_path_2",
        "pusa_lora",
        "pusa_lora2",
        "joint_model_path",
        "seedvr2_model_dir",
        "seedvr2_model_name",
        "x_model_path",
        "x_model_path_2",
        "frame_model_path",
    ]

    for arg_name in arg_to_config:
        arg_value = getattr(args, arg_name, None)
        if arg_value:
            TEN_SECOND_CONFIG[arg_name] = arg_value


def _apply_service_config(service: VideoGenerationService, args, rank: int) -> None:
    """Apply command line arguments to service configuration."""
    if args.model:
        model_info = MODEL_REGISTRY[args.model]
        service.default_config["model"] = model_info["model"]
        service.default_config["pretrained_model_name_or_path"] = model_info["path"]
        service.default_config["task_type"] = model_info["task_type"]
        logger.info(f"[Rank {rank}] Model set to: {args.model}")  # noqa: G004

    if args.model_path:
        service.default_config["pretrained_model_name_or_path"] = args.model_path
        logger.info(f"[Rank {rank}] Model path set to: {args.model_path}")  # noqa: G004

    if args.dtype:
        service.default_config["dtype"] = args.dtype
        logger.info(f"[Rank {rank}] Dtype set to: {args.dtype}")  # noqa: G004

    if args.sp:
        service.default_config["sp"] = args.sp
        logger.info(f"[Rank {rank}] Sequence parallel set to: {args.sp}")  # noqa: G004

    if args.no_fsdp:
        service.default_config["fsdp"] = False
        logger.info(f"[Rank {rank}] FSDP disabled")  # noqa: G004

    if args.ten_second:
        service.default_config["ten_second"] = True
        service.default_config["fsdp"] = "all"
        service.default_config["inf_vram_blocks_num"] = 0
        logger.info(f"[Rank {rank}] 10-second mode enabled (fsdp=all, inf_vram_blocks_num=0)")  # noqa: G004
    if args.x:
        service.default_config["x"] = True
        x_model_path = args.x_model_path
        x_model_path_2 = args.x_model_path_2
        if x_model_path:
            service.default_config["x_model_path"] = x_model_path
            logger.info(f"[Rank {rank}] X model path set to: {x_model_path}")  # noqa: G004
        if x_model_path_2:
            service.default_config["x_model_path_2"] = x_model_path_2
            logger.info(f"[Rank {rank}] X model path 2 set to: {x_model_path_2}")  # noqa: G004
        logger.info(f"[Rank {rank}] Distilled model mode enabled")  # noqa: G004
    if args.turbo_mode != "default":
        service.default_config["turbo_mode"] = args.turbo_mode
        logger.info(f"[Rank {rank}] turbo_mode set to: {args.turbo_mode}")  # noqa: G004
    if args.lora_path_list:
        service.default_config["lora_path_list"] = args.lora_path_list
        logger.info(f"[Rank {rank}] lora_path_list set to: {args.lora_path_list}")  # noqa: G004
    external_url = args.external_base_url or os.environ.get("EXTERNAL_BASE_URL")
    if external_url:
        service.default_config["external_base_url"] = external_url
        logger.info(f"[Rank {rank}] External base URL set to: {external_url}")  # noqa: G004


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Video Generation Service")

    parser.add_argument("--host", type=str, default="0.0.0.0", help="Service host")
    parser.add_argument("--port", type=int, default=5001, help="Service port")
    parser.add_argument("--config", type=str, default=None, help="Config file path")
    parser.add_argument("--init", action="store_true", help="Initialize service on startup")
    parser.add_argument("--warmup", action="store_true", help="Run warmup inference on startup")
    parser.add_argument("--warmup-steps", type=int, default=1, help="Warmup inference steps")

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        choices=list(MODEL_REGISTRY.keys()),
        help="Model name (wan2.2-i2v-14b or wan2.2-t2v-14b)",
    )
    parser.add_argument("--model-path", type=str, default=None, help="Model path (overrides default)")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["fp16", "bf16", "fp32"], help="Data type")
    parser.add_argument("--sp", type=int, default=None, help="Sequence parallel number (number of GPUs)")
    parser.add_argument("--fsdp", action="store_true", default=True, help="Enable FSDP")
    parser.add_argument("--no-fsdp", action="store_true", help="Disable FSDP")
    parser.add_argument("--ten-second", action="store_true", help="Enable 10-second video generation mode")
    parser.add_argument("--external-base-url", type=str, default=None, help="External base URL for download links")
    parser.add_argument("--x", action="store_true", default=False, help="Enable distilled model")
    parser.add_argument("--ten-second-model-id-t2v", type=str, default=None, help="10-second T2V model ID path")
    parser.add_argument(
        "--ten-second-model-path", type=str, default=None, help="10-second model path (transformer_14B)"
    )
    parser.add_argument(
        "--ten-second-model-path-2", type=str, default=None, help="10-second model path 2 (transformer_14B_2)"
    )
    parser.add_argument("--pusa-lora", type=str, default=None, help="Pusa LoRA high path")
    parser.add_argument("--pusa-lora2", type=str, default=None, help="Pusa LoRA low path")
    parser.add_argument("--joint-model-path", type=str, default=None, help="Joint model path (1.3B)")
    parser.add_argument("--seedvr2-model-dir", type=str, default=None, help="SeedVR2 model directory")
    parser.add_argument("--seedvr2-model-name", type=str, default=None, help="SeedVR2 model name")
    parser.add_argument("--x-model-path", type=str, default=None, help="X model path (6steps i2v 14B_1)")
    parser.add_argument("--x-model-path-2", type=str, default=None, help="X model path 2 (6steps i2v 14B_2)")
    parser.add_argument("--frame-model-path", type=str, default=None, help="Frame interpolation model path (IFRNet)")
    parser.add_argument(
        "--lora-path-list",
        nargs="*",
        type=str,
        default=None,
        help="List of LoRA weight paths (multiple paths supported)",
    )
    parser.add_argument(
        "--turbo_mode",
        type=str,
        default="default",
        choices=["default", "faiz", "next_faiz"],
        help="Use different turbo mode inference.",
    )

    args = parser.parse_args()

    _apply_ten_second_config_args(args)

    if args.init:
        init_env()

    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    service = VideoGenerationService(config_path=args.config)

    _apply_service_config(service, args, rank)

    if args.init:
        service.inference_manager = VideoInferenceManager()
        service.is_initialized = True
        logger.info(f"[Rank {rank}] Service pre-initialized (lazy loading mode)")  # noqa: G004

    if args.warmup and rank == 0:
        logger.info(f"[Rank {rank}] Running warmup with {args.warmup_steps} steps...")  # noqa: G004
        service.warmup(warmup_steps=args.warmup_steps)

    if world_size > 1 and rank != 0:
        service._worker_loop()
        return

    service.run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
