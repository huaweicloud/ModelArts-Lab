import base64
import inspect
import io
import json
import logging
import os
import queue
import threading
import time
from datetime import datetime
from queue import Queue
from typing import Any

import torch
import torch.distributed as dist
from flask import Flask, jsonify, request
from PIL import Image

try:
    from flask_cors import CORS

    CORS_AVAILABLE = True
except ImportError:
    CORS_AVAILABLE = False

import sys

from x_diffusers import init_cfg_env, parse_args
from x_diffusers.adaptor.load_pipe import load_pipe, pipe_to_device, update_lora, update_pipe, update_scheduler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

IMAGE_OUTPUT_DIR = "/home/ma-user/generated_images"
CLEANUP_INTERVAL_HOURS = 6
CLEANUP_RETAIN_DAYS = 7
CLEANUP_ENABLED = True
SERVICE_BASE_URL = os.environ.get("SERVICE_BASE_URL", "")

HEARTBEAT_INTERVAL_SECONDS = 30
HEARTBEAT_ENABLED = True

ALLOWED_OUTPUT_DIRS = [
    "/home/ma-user/generated_images",
]


def sanitize_save_path(save_path: str, default_dir: str = IMAGE_OUTPUT_DIR) -> tuple[str, str]:
    """验证并净化保存路径，防止路径遍历攻击"""
    if not save_path:
        os.makedirs(default_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"image_{timestamp}.png"
        safe_path = os.path.join(default_dir, filename)
        return safe_path, filename

    try:
        real_path = os.path.realpath(save_path)
        real_dir = os.path.dirname(real_path)

        is_allowed = any(real_path.startswith(os.path.realpath(allowed_dir)) for allowed_dir in ALLOWED_OUTPUT_DIRS)

        if not is_allowed:
            logger.warning(f"Path traversal attempt blocked: {save_path} -> {real_path}")  # noqa: G004
            raise ValueError(f"save_path must be within allowed directories: {ALLOWED_OUTPUT_DIRS}")

        filename = os.path.basename(real_path)
        os.makedirs(real_dir, exist_ok=True)

        return real_path, filename

    except (OSError, ValueError) as e:
        raise ValueError(f"Invalid save_path: {str(e)}")


def get_base_url(request_host_url: str) -> str:
    base_url = SERVICE_BASE_URL if SERVICE_BASE_URL else request_host_url
    return base_url if base_url.endswith("/") else base_url + "/"


def decode_base64_image(image_base64: str, output_dir: str = "/tmp") -> str:
    try:
        if image_base64.startswith("data:"):
            image_base64 = image_base64.split(",", 1)[1]

        image_data = base64.b64decode(image_base64)
        with Image.open(io.BytesIO(image_data)) as img:
            image = img.convert("RGB")

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            output_path = os.path.join(output_dir, f"input_image_{timestamp}.png")
            os.makedirs(output_dir, exist_ok=True)
            image.save(output_path)

        logger.info(f"Decoded base64 image to: {output_path}")  # noqa: G004
        return output_path
    except Exception as e:
        raise ValueError(f"Failed to decode base64 image: {str(e)}")


def cleanup_old_files():
    """清理超过保留天数的图像文件"""
    if not CLEANUP_ENABLED:
        return

    if not os.path.exists(IMAGE_OUTPUT_DIR):
        return

    try:
        now = time.time()
        cutoff_time = now - (CLEANUP_RETAIN_DAYS * 24 * 3600)
        deleted_count = 0
        deleted_size = 0

        for filename in os.listdir(IMAGE_OUTPUT_DIR):
            filepath = os.path.join(IMAGE_OUTPUT_DIR, filename)
            if os.path.isfile(filepath):
                file_mtime = os.path.getmtime(filepath)
                if file_mtime < cutoff_time:
                    file_size = os.path.getsize(filepath)
                    os.remove(filepath)
                    deleted_count += 1
                    deleted_size += file_size
                    logger.info(f"[Cleanup] Deleted: {filename} ({file_size / 1024:.1f} KB)")  # noqa: G004

        if deleted_count > 0:
            logger.info(f"[Cleanup] Total deleted: {deleted_count} files, {deleted_size / 1024 / 1024:.2f} MB")  # noqa: G004
        else:
            logger.info(f"[Cleanup] No files to delete (retain {CLEANUP_RETAIN_DAYS} days)")  # noqa: G004
    except Exception as e:
        logger.error(f"[Cleanup] Error: {str(e)}")  # noqa: G004


def cleanup_thread_func():
    """定时清理线程"""
    while True:
        time.sleep(CLEANUP_INTERVAL_HOURS * 3600)
        logger.info("[Cleanup] Starting scheduled cleanup...")
        cleanup_old_files()


def generate_warmup_input_image(output_path: str, width: int = 512, height: int = 512):
    """生成 warmup 用的输入图像（纯色渐变图）"""
    try:
        import numpy as np

        img_array = np.zeros((height, width, 3), dtype=np.uint8)
        for i in range(height):
            for j in range(width):
                img_array[i, j, 0] = int(255 * i / height)
                img_array[i, j, 1] = int(255 * j / width)
                img_array[i, j, 2] = int(255 * (i + j) / (height + width))

        img = Image.fromarray(img_array, "RGB")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        img.save(output_path)
        logger.info(f"[Warmup] Generated input image: {output_path}")  # noqa: G004
        return output_path
    except Exception as e:
        logger.error(f"[Warmup] Failed to generate input image: {str(e)}")  # noqa: G004
        return None


def get_device_for_comm():
    """获取分布式通信使用的设备

    注意：torchrun 默认使用 Gloo 后端初始化分布式，Gloo 只支持 CPU tensor 通信。
    即使 transfer_to_npu 会替换后端为 hccl，但在 broadcast 时仍需要确保 tensor 在支持的设备上。
    """
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


def broadcast_dict(data: dict, src: int = 0) -> dict:
    """广播字典数据从src rank到所有rank"""
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
    """广播整数"""
    if not dist.is_initialized():
        return value

    device = get_device_for_comm()
    backend = dist.get_backend()
    rank = dist.get_rank()
    logger.info(f"[Rank {rank}] broadcast_int: backend={backend}, device={device}, value={value}")  # noqa: G004

    tensor = torch.tensor([value], dtype=torch.int, device=device)
    dist.broadcast(tensor, src=src)
    return tensor.item()


MODEL_REGISTRY = {
    "Qwen-Image": "/home/models/Qwen-Image",
    "Qwen-Image-Edit": "/home/models/Qwen-Image-Edit",
    "qwen-image-edit-2509": "/home/models/qwen-image-edit-2509",
    "qwen-image-edit-2511": "/home/models/qwen-image-edit-2511",
    "Qwen-Image-2512": "/home/models/Qwen-Image-2512",
    "Qwen-Image-Edit-2511": "/home/models/Qwen-Image-Edit-2511",
    "longcat-image": "/home/models/longcat-image",
    "longcat-image-edit": "/home/models/longcat-image-edit",
    "z-image": "/home/models/z-image",
}


class CachedImageInferenceManager:
    """单模型推理管理器，切换模型需重启服务"""

    def __init__(self):
        self.pipe = None
        self.init_args = None
        self.current_model = None
        self.loaded = False
        self.inference_lock = threading.Lock()

    @staticmethod
    def load_image(image_path: str):
        if not os.path.exists(image_path):
            raise RuntimeError(f"image path {image_path} is not exists")
        return Image.open(image_path).convert("RGB")

    def initialize_pipe(self, args):
        """初始化pipe，只执行一次"""
        if self.pipe is not None:
            logger.info(
                f"[Rank {dist.get_rank() if dist.is_initialized() else 0}] Using cached pipe for {self.current_model}"  # noqa: G004
            )
            return

        model_name = args.model
        model_path = MODEL_REGISTRY.get(model_name)

        if model_path is None:
            raise ValueError(f"Unsupported model: {model_name}. Supported models: {list(MODEL_REGISTRY.keys())}")

        logger.info(f"[Rank {dist.get_rank() if dist.is_initialized() else 0}] Loading pipe for {model_name}...")  # noqa: G004

        if args.lora_path_list:
            args.fuse_lora = True

        self.pipe = load_pipe(args)

        update_scheduler(self.pipe, args.model, args.lora_path_list)
        update_lora(
            self.pipe,
            args.model,
            init_lora_path_list=None,
            lora_path_list=args.lora_path_list,
            fuse_lora=args.fuse_lora,
            weights_list=args.lora_scale_weight_list,
        )
        update_pipe(self.pipe, args)
        self.pipe = pipe_to_device(self.pipe, args)

        self.init_args = args
        self.current_model = model_name
        self.loaded = True
        logger.info(f"[Rank {dist.get_rank() if dist.is_initialized() else 0}] Pipe for {model_name} loaded and cached")  # noqa: G004

    def set_infer_params(self, args):
        infer_params = dict(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            num_inference_steps=args.num_inference_steps,
            width=args.width,
            height=args.height,
            generator=torch.Generator().manual_seed(args.seed),
            cfg_parallel_size=args.cfg_parallel_size,
            true_cfg_scale=args.true_cfg_scale,
        )

        if args.guidance_scale is not None:
            infer_params["guidance_scale"] = args.guidance_scale

        if args.image_path is not None and args.image_path != "":
            image = self.load_image(args.image_path)
            infer_params.update(image=image)
        elif args.image_path_list:
            image = [self.load_image(image_path) for image_path in args.image_path_list]
            infer_params.update(image=image)

        return infer_params

    def infer(self, args) -> float:
        """执行推理"""
        with self.inference_lock:
            if self.pipe is None:
                self.initialize_pipe(args)

            try:
                os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
            except OSError:
                logger.error(f"Failed to create output directory: {args.save_path}")  # noqa: G004
                return 0.0

            infer_params = self.set_infer_params(args)
            signature = inspect.signature(self.pipe)
            filtered_params = {k: v for k, v in infer_params.items() if k in signature.parameters}

            torch.npu.synchronize()
            start_time = time.time()

            out = self.pipe(**filtered_params)

            torch.npu.synchronize()
            end_time = time.time()
            infer_cost = end_time - start_time
            logger.info(f"time cost:{infer_cost}")  # noqa: G004

            images = out["images"] if isinstance(out, dict) else getattr(out, "images", out)
            image = images[0]

            if torch.distributed.is_initialized():
                if torch.distributed.get_rank() == 0:
                    image.save(args.save_path)
                    logger.info(f"save image in {args.save_path}")  # noqa: G004
            else:
                image.save(args.save_path)
                logger.info(f"save image in {args.save_path}")  # noqa: G004

            return infer_cost


class ImageGenerationService:
    def __init__(self, config_path: str | None = None):
        self.app = Flask(__name__)
        if CORS_AVAILABLE:
            CORS(self.app)

        self.inference_manager = None
        self.default_config = self._load_default_config(config_path)
        self.is_initialized = False

        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1

        self.task_queue = queue.Queue()
        self.result_queue = queue.Queue()

        self.heartbeat_running = False
        self.last_heartbeat_time = time.time()
        self.inference_in_progress = False
        self.inference_lock = threading.Lock()
        self._infer_queue = Queue()
        self._start_infer_worker()
        self._setup_routes()
        self._start_cleanup_thread()

    def _start_cleanup_thread(self):
        """启动定时清理线程（仅在rank 0）"""
        if self.rank == 0 and CLEANUP_ENABLED:
            cleanup_thread = threading.Thread(target=cleanup_thread_func, daemon=True)
            cleanup_thread.start()
            logger.info(
                f"[Cleanup] Thread started (interval: {CLEANUP_INTERVAL_HOURS}h, retain: {CLEANUP_RETAIN_DAYS} days)"  # noqa: G004
            )

    def _start_heartbeat_thread(self):
        """启动心跳线程（仅在rank 0且多卡模式）"""
        if self.rank == 0 and self.world_size > 1 and HEARTBEAT_ENABLED:
            self.heartbeat_running = True
            heartbeat_thread = threading.Thread(target=self._heartbeat_thread_func, daemon=True)
            heartbeat_thread.start()
            logger.info(f"[Heartbeat] Thread started (interval: {HEARTBEAT_INTERVAL_SECONDS}s)")  # noqa: G004

    def _heartbeat_thread_func(self):
        """心跳线程函数，定期广播心跳信号保持连接活跃"""
        while self.heartbeat_running:
            try:
                time.sleep(HEARTBEAT_INTERVAL_SECONDS)

                if (
                    not self.inference_manager
                    or not hasattr(self.inference_manager, "loaded")
                    or not self.inference_manager.loaded
                ):
                    logger.debug("[Heartbeat] Model not loaded yet, skipping heartbeat")
                    continue

                if not dist.is_initialized():
                    continue

                with self.inference_lock:
                    if self.inference_in_progress:
                        logger.debug("[Heartbeat] Inference in progress, skipping heartbeat")
                        continue
                broadcast_int(2, src=0)
                self.last_heartbeat_time = time.time()
                logger.debug("[Heartbeat] Sent heartbeat signal")

            except Exception as e:
                logger.error(f"[Heartbeat] Error: {str(e)}")  # noqa: G004

    def _validate_string(self, value, name: str, max_len: int, required: bool = False) -> str | None:
        if value is None:
            return f"{name} is required" if required else None
        if not isinstance(value, str):
            return f"{name} must be a string"
        if len(value.strip()) == 0 and required:
            return f"{name} cannot be blank"
        if len(value) > max_len:
            return f"{name} length cannot exceed {max_len} characters"
        return None

    def _validate_integer(
        self, value, name: str, min_val: int, max_val: int, default: int, divisible_by: int = None
    ) -> str | None:
        if value is None:
            return None
        if not isinstance(value, int):
            return f"{name} must be an integer"
        if value < min_val or value > max_val:
            return f"{name} must be between {min_val} and {max_val}"
        if divisible_by and value % divisible_by != 0:
            return f"{name} must be divisible by {divisible_by}"
        return None

    def _validate_number(
        self, value, name: str, min_val: float = None, max_val: float = None, allow_none: bool = True
    ) -> str | None:
        if value is None:
            return None if allow_none else f"{name} is required"
        if not isinstance(value, (int, float)):  # noqa: UP038
            return f"{name} must be a number"
        if min_val is not None and value < min_val:
            return f"{name} must be >= {min_val}"
        if max_val is not None and value > max_val:
            return f"{name} must be <= {max_val}"
        return None

    def _validate_prompt(self, data: dict[str, Any], errors: list):
        err = self._validate_string(data.get("prompt"), "prompt", 1000, required=True)
        if err:
            errors.append(err)

    def _validate_dimensions(self, data: dict[str, Any], errors: list):
        for name in ["width", "height"]:
            value = data.get(name, 512)
            err = self._validate_integer(value, name, 64, 1536, 512, divisible_by=8)
            if err and value != 512:
                errors.append(err)

    def _validate_steps(self, data: dict[str, Any], errors: list):
        steps = data.get("num_inference_steps", 20)
        err = self._validate_integer(steps, "num_inference_steps", 1, 200, 20)
        if err and steps != 20:
            errors.append(err)

    def _validate_cfg_scale(self, data: dict[str, Any], errors: list):
        cfg_scale = data.get("true_cfg_scale", 4.0)
        if cfg_scale != 4.0:
            err = self._validate_number(cfg_scale, "true_cfg_scale", 0, 20, allow_none=False)
            if err:
                errors.append(err)

    def _validate_image_base64(self, data: dict[str, Any], errors: list):
        image_base64 = data.get("image_base64")
        if image_base64 is None:
            return
        if not isinstance(image_base64, str):
            errors.append("image_base64 must be a string")
        elif len(image_base64) == 0:
            errors.append("image_base64 cannot be empty")

    def _validate_model(self, data: dict[str, Any], errors: list):
        save_path = data.get("save_path", IMAGE_OUTPUT_DIR)
        args = self._build_args(data, save_path)
        if "Edit" in args.model and not (args.image_path or args.image_path_list):
            errors.append("Edit model must be used with an input image.")

    def _validate_image_inputs(self, data: dict[str, Any], errors: list):
        image_count = 0
        has_single_image = False
        has_image_list = False

        if data.get("image_path") or data.get("image_base64"):
            image_count = 1
            has_single_image = True

        image_path_list = data.get("image_path_list")
        if image_path_list:
            if not isinstance(image_path_list, list):
                errors.append("image_path_list must be a list")
            else:
                has_image_list = True
                image_count = len(image_path_list)

        image_base64_list = data.get("image_base64_list")
        if image_base64_list:
            if not isinstance(image_base64_list, list):
                errors.append("image_base64_list must be a list")
            else:
                if has_image_list:
                    errors.append("Cannot use both image_path_list and image_base64_list")
                else:
                    for i, img_b64 in enumerate(image_base64_list):
                        if not isinstance(img_b64, str):
                            errors.append(f"image_base64_list[{i}] must be a string")
                        elif len(img_b64) == 0:
                            errors.append(f"image_base64_list[{i}] cannot be empty")
                    image_count = len(image_base64_list)

        if has_single_image and has_image_list:
            errors.append(
                "Cannot use both single image (image_path/image_base64) and image list (image_path_list/image_base64_list)"  # noqa: E501
            )

        if image_count > 3:
            errors.append(f"Maximum 3 images supported, got {image_count}")

    def _validate_generate_params(self, data: dict[str, Any]) -> tuple[bool, str]:
        errors = []

        self._validate_prompt(data, errors)
        self._validate_dimensions(data, errors)
        self._validate_steps(data, errors)

        err = self._validate_integer(data.get("seed"), "seed", 0, 2**31 - 1, None)
        if err:
            errors.append(err)

        self._validate_cfg_scale(data, errors)

        err = self._validate_number(data.get("guidance_scale"), "guidance_scale", 0, None)
        if err:
            errors.append(err)

        err = self._validate_string(data.get("negative_prompt"), "negative_prompt", 1000)
        if err:
            errors.append(err)

        if "return_base64" in data and not isinstance(data["return_base64"], bool):
            errors.append("return_base64 must be a boolean")

        err = self._validate_string(data.get("save_path"), "save_path", 500)
        if err:
            errors.append(err)

        self._validate_image_base64(data, errors)

        err = self._validate_string(data.get("image_path"), "image_path", 500)
        if err:
            errors.append(err)

        image_path_list = data.get("image_path_list")
        if image_path_list:
            for i, path in enumerate(image_path_list):
                err = self._validate_string(path, f"image_path_list[{i}]", 500)
                if err:
                    errors.append(err)
        self._validate_image_inputs(data, errors)
        self._validate_model(data, errors)

        if errors:
            return False, "; ".join(errors)
        return True, ""

    def _load_default_config(self, config_path: str | None) -> dict[str, Any]:
        config = {
            "model": "Qwen-Image",
            "pretrained_model_name_or_path": "/home/models/Qwen-Image",
            "dtype": "bf16",
            "cfg_parallel_size": 1,
            "matmul_a8w8": True,
            "atten_laser": True,
            "cache_dit": True,
        }

        if config_path and os.path.exists(config_path):
            with open(config_path) as f:
                config.update(json.load(f))

        return config

    def _route_download(self):
        from flask import send_file

        @self.app.route("/download/<filename>", methods=["GET"])
        def download_image(filename):
            try:
                file_path = os.path.join(IMAGE_OUTPUT_DIR, filename)
                if not os.path.exists(file_path):
                    return jsonify({"status": "error", "message": "File not found"}), 404

                with open(file_path, "rb") as f:
                    data = f.read()
                return send_file(io.BytesIO(data), mimetype="image/png", as_attachment=True, download_name=filename)
            except Exception as e:
                logger.error(f"Download failed: {str(e)}")  # noqa: G004
                return jsonify({"status": "error", "message": str(e)}), 500

    def _route_health(self):
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
                    "cleanup_config": {
                        "enabled": CLEANUP_ENABLED,
                        "interval_hours": CLEANUP_INTERVAL_HOURS,
                        "retain_days": CLEANUP_RETAIN_DAYS,
                        "output_dir": IMAGE_OUTPUT_DIR,
                    },
                    "heartbeat_config": {
                        "enabled": HEARTBEAT_ENABLED,
                        "interval_seconds": HEARTBEAT_INTERVAL_SECONDS,
                        "last_heartbeat": datetime.fromtimestamp(self.last_heartbeat_time).isoformat()
                        if self.world_size > 1
                        else None,
                    },
                }
            )

    def _route_heartbeat(self):
        @self.app.route("/heartbeat", methods=["POST"])
        def trigger_heartbeat():
            try:
                if self.world_size <= 1:
                    return jsonify({"status": "success", "message": "Single card mode, no heartbeat needed"})

                if self.rank != 0:
                    return jsonify({"status": "error", "message": "Heartbeat only sent from rank 0"}), 400

                if not dist.is_initialized():
                    return jsonify({"status": "error", "message": "Distributed not initialized"}), 400

                broadcast_int(2, src=0)
                self.last_heartbeat_time = time.time()

                return jsonify(
                    {"status": "success", "message": "Heartbeat sent", "timestamp": datetime.now().isoformat()}
                )
            except Exception as e:
                logger.error(f"Manual heartbeat failed: {str(e)}")  # noqa: G004
                return jsonify({"status": "error", "message": str(e)}), 500

    def _route_cleanup(self):
        @self.app.route("/cleanup", methods=["POST"])
        def manual_cleanup():
            try:
                if self.rank != 0:
                    return jsonify({"status": "error", "message": "Cleanup only runs on rank 0"}), 400

                cleanup_old_files()

                file_count = 0
                total_size = 0
                if os.path.exists(IMAGE_OUTPUT_DIR):
                    for f in os.listdir(IMAGE_OUTPUT_DIR):
                        fp = os.path.join(IMAGE_OUTPUT_DIR, f)
                        if os.path.isfile(fp):
                            file_count += 1
                            total_size += os.path.getsize(fp)

                return jsonify(
                    {
                        "status": "success",
                        "message": "Cleanup completed",
                        "current_files": file_count,
                        "total_size_mb": round(total_size / 1024 / 1024, 2),
                    }
                )
            except Exception as e:
                logger.error(f"Manual cleanup failed: {str(e)}")  # noqa: G004
                return jsonify({"status": "error", "message": str(e)}), 500

    def _route_init(self):
        @self.app.route("/init", methods=["POST"])
        def init_service():
            try:
                data = request.get_json() or {}
                cfg_parallel_size = data.get("cfg_parallel_size", self.default_config["cfg_parallel_size"])

                if cfg_parallel_size == 2:
                    init_cfg_env(cfg_parallel_size)
                    logger.info(
                        f"[Rank {self.rank}] Initialized CFG parallel environment with size={cfg_parallel_size}"  # noqa: G004
                    )

                self.inference_manager = CachedImageInferenceManager()
                self.is_initialized = True

                logger.info(f"[Rank {self.rank}] ImageGenerationService initialized successfully")  # noqa: G004
                return jsonify({"status": "success", "message": "Service initialized", "rank": self.rank})
            except Exception as e:
                logger.error(f"[Rank {self.rank}] Failed to initialize service: {str(e)}", exc_info=True)  # noqa: G004, G201
                return jsonify({"status": "error", "message": str(e)}), 500

    def _route_generate_single(self, data, save_path, filename, prompt, return_base64, host_url):
        args = self._build_args(data, save_path)
        logger.info(args)
        with self.inference_lock:
            self.inference_in_progress = True
        try:
            infer_time = self.inference_manager.infer(args)
        finally:
            with self.inference_lock:
                self.inference_in_progress = False

        result = {
            "status": "success",
            "infer_time": infer_time,
            "download_url": f"{get_base_url(host_url)}download/{filename}",
        }

        if return_base64 and save_path and os.path.exists(save_path):
            with open(save_path, "rb") as f:
                image_base64 = base64.b64encode(f.read()).decode("utf-8")
            result["image_base64"] = image_base64

        if save_path and os.path.exists(save_path):
            result["save_path"] = save_path

        return result

    def _route_generate_start(self, data, host_url):
        try:
            if not self.is_initialized:
                return {"status": "error", "message": "Service not initialized. Call /init first"}, 400

            if not data:
                return {"status": "error", "message": "No JSON data provided"}, 400

            is_valid, error_msg = self._validate_generate_params(data)
            if not is_valid:
                return {"status": "error", "message": f"Parameter validation failed: {error_msg}"}, 400

            prompt = data.get("prompt")
            return_base64 = data.get("return_base64", True)
            save_path_input = data.get("save_path")

            try:
                save_path, filename = sanitize_save_path(save_path_input)
            except ValueError as e:
                return {"status": "error", "message": str(e)}, 400

            logger.info(f"[Rank 0] Generating image for prompt: {prompt[:100]}...")  # noqa: G004

            if self.world_size > 1:
                self.task_queue.put(
                    {
                        "type": "generate",
                        "data": data,
                        "save_path": save_path,
                        "filename": filename,
                        "host_url": host_url,
                    }
                )
                result = self.result_queue.get()
                return (result, 500) if result.get("status") == "error" else (result, 200)

            else:
                result = self._route_generate_single(data, save_path, filename, prompt, return_base64, host_url)
                return result, 200

        except Exception as e:
            logger.error(f"[Rank {self.rank}] Image generation failed: {str(e)}", exc_info=True)  # noqa: G004, G201
            return {"status": "error", "message": str(e)}, 500

    def _start_infer_worker(self):
        def worker():
            while True:
                task = self._infer_queue.get()
                data = task["data"]
                host_url = task["host_url"]
                response_queue = task["response_queue"]

                try:
                    result = self._route_generate_start(data, host_url)
                    response_queue.put(result)
                except Exception as e:
                    response_queue.put(({"status": "error", "message": str(e)}, 500))

                self._infer_queue.task_done()

        t = threading.Thread(target=worker, daemon=True)
        t.start()

    def _route_generate(self):
        @self.app.route("/generate", methods=["POST"])
        def generate_image():
            data = request.get_json()
            host_url = request.host_url

            response_queue = Queue()

            self._infer_queue.put({"data": data, "host_url": host_url, "response_queue": response_queue})

            result, code = response_queue.get()
            return jsonify(result), code

        return generate_image

    def _route_config(self):
        @self.app.route("/config", methods=["GET", "POST"])
        def manage_config():
            if request.method == "GET":
                return jsonify(self.default_config)
            else:
                data = request.get_json()
                self.default_config.update(data)
                return jsonify({"status": "success", "config": self.default_config})

    def _route_models(self):
        @self.app.route("/models", methods=["GET"])
        def list_models():
            current_model = None
            if self.inference_manager and hasattr(self.inference_manager, "current_model"):
                current_model = self.inference_manager.current_model

            return jsonify(
                {
                    "current_model": current_model,
                    "supported_models": list(MODEL_REGISTRY.keys()),
                    "model_paths": MODEL_REGISTRY,
                    "note": "To switch model, restart the service with different config",
                }
            )

    def _route_params(self):
        @self.app.route("/params", methods=["GET"])
        def get_param_spec():
            return jsonify(
                {
                    "parameters": {
                        "prompt": {
                            "type": "string",
                            "required": True,
                            "description": "Text prompt for image generation",
                            "constraints": {"min_length": 1, "max_length": 1000},
                        },
                        "negative_prompt": {
                            "type": "string",
                            "required": False,
                            "default": " ",
                            "description": "Negative prompt to avoid certain features",
                            "constraints": {"max_length": 1000},
                        },
                        "width": {
                            "type": "integer",
                            "required": False,
                            "default": 512,
                            "description": "Image width in pixels",
                            "constraints": {"min": 64, "max": 1536, "divisible_by": 8},
                        },
                        "height": {
                            "type": "integer",
                            "required": False,
                            "default": 512,
                            "description": "Image height in pixels",
                            "constraints": {"min": 64, "max": 1536, "divisible_by": 8},
                        },
                        "num_inference_steps": {
                            "type": "integer",
                            "required": False,
                            "default": 20,
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
                        "true_cfg_scale": {
                            "type": "number",
                            "required": False,
                            "default": 4.0,
                            "description": "Classifier-free guidance scale",
                            "constraints": {"min": 0, "max": 20},
                        },
                        "guidance_scale": {
                            "type": "number",
                            "required": False,
                            "default": None,
                            "description": "Alternative guidance scale",
                            "constraints": {"min": 0},
                        },
                        "return_base64": {
                            "type": "boolean",
                            "required": False,
                            "default": True,
                            "description": "Return image as base64 string",
                        },
                        "save_path": {
                            "type": "string",
                            "required": False,
                            "default": "auto-generated",
                            "description": "Path to save the generated image",
                            "constraints": {"max_length": 500},
                        },
                        "image_path": {
                            "type": "string",
                            "required": False,
                            "description": "Single input image path (for image edit models)",
                        },
                        "image_base64": {
                            "type": "string",
                            "required": False,
                            "description": "Single input image as base64 encoded string",
                        },
                        "image_path_list": {
                            "type": "array",
                            "required": False,
                            "description": "List of input image paths (1-3 images for qwen-image-edit-2511)",
                            "constraints": {"min_items": 1, "max_items": 3},
                        },
                        "image_base64_list": {
                            "type": "array",
                            "required": False,
                            "description": "List of input images as base64 encoded strings (1-3 images)",
                            "constraints": {"min_items": 1, "max_items": 3},
                        },
                    },
                    "example_single_image": {
                        "prompt": "add a red hat on the person",
                        "image_base64": "<base64_encoded_image>",
                        "width": 1024,
                        "height": 1024,
                    },
                    "example_multi_image": {
                        "prompt": "combine these images into one",
                        "image_base64_list": ["<base64_img1>", "<base64_img2>", "<base64_img3>"],
                        "width": 1024,
                        "height": 1024,
                    },
                    "note": "qwen-image-edit-2511 supports 1-3 input images. Use image_path/image_base64 for single image, or image_path_list/image_base64_list for multiple images.",  # noqa: E501
                }
            )

    def _setup_routes(self):
        self._route_download()
        self._route_health()
        self._route_heartbeat()
        self._route_cleanup()
        self._route_init()
        self._route_generate()
        self._route_config()
        self._route_models()
        self._route_params()

    def _build_args(self, data: dict[str, Any], save_path: str | None = None):
        config = self.default_config.copy()

        prompt = data.get("prompt")
        if not prompt:
            raise ValueError("prompt is required in request")

        negative_prompt = data.get("negative_prompt", " ")
        height = data.get("height", 512)
        width = data.get("width", 512)
        num_inference_steps = data.get("num_inference_steps", 20)
        true_cfg_scale = data.get("true_cfg_scale", 4.0)
        seed = data.get("seed", 42)
        guidance_scale = data.get("guidance_scale")

        config.update(
            {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "height": height,
                "width": width,
                "num_inference_steps": num_inference_steps,
                "true_cfg_scale": true_cfg_scale,
                "seed": seed,
                "guidance_scale": guidance_scale,
            }
        )

        if save_path:
            config["save_path"] = save_path
        elif data.get("save_path"):
            config["save_path"] = data["save_path"]

        image_path = data.get("image_path")
        image_base64 = data.get("image_base64")
        image_base64_list = data.get("image_base64_list")

        if image_base64:
            image_path = decode_base64_image(image_base64, "/tmp/input_images")

        config["image_path"] = image_path

        if image_base64_list:
            decoded_paths = []
            for img_b64 in image_base64_list:
                decoded_path = decode_base64_image(img_b64, "/tmp/input_images")
                decoded_paths.append(decoded_path)
            config["image_path_list"] = decoded_paths
        else:
            config["image_path_list"] = data.get("image_path_list")

        for key in ["lora_path_list", "lora_scale_weight_list", "fuse_lora"]:
            if key in data:
                config[key] = data[key]

        arg_list = []
        for key, value in config.items():
            if value is None or value == "":
                continue
            if isinstance(value, bool):
                if value:
                    arg_list.append(f"--{key}")
            elif isinstance(value, list):
                arg_list.append(f"--{key}")
                arg_list.extend([str(v) for v in value])
            else:
                arg_list.extend([f"--{key}", str(value)])

        old_argv = sys.argv
        sys.argv = ["image_server.py"] + arg_list
        try:
            args = parse_args()
        finally:
            sys.argv = old_argv

        return args

    def _master_comm_thread(self):
        """Rank 0的通信线程，处理与worker的同步"""
        while True:
            try:
                task = self.task_queue.get()
                if task["type"] == "generate":
                    broadcast_int(1, src=0)

                    broadcast_data = {"data": task["data"], "save_path": task["save_path"]}
                    broadcast_dict(broadcast_data, src=0)

                    args = self._build_args(task["data"], task["save_path"])

                    with self.inference_lock:
                        self.inference_in_progress = True
                    try:
                        logger.info(args)
                        infer_time = self.inference_manager.infer(args)
                    finally:
                        with self.inference_lock:
                            self.inference_in_progress = False

                    return_base64 = task["data"].get("return_base64", True)
                    save_path = task["save_path"]
                    filename = task.get("filename", os.path.basename(save_path))
                    host_url = task.get("host_url", "")

                    result = {
                        "status": "success",
                        "infer_time": infer_time,
                        "download_url": f"{get_base_url(host_url)}download/{filename}",
                    }

                    if return_base64 and save_path and os.path.exists(save_path):
                        with open(save_path, "rb") as f:
                            image_base64 = base64.b64encode(f.read()).decode("utf-8")
                        result["image_base64"] = image_base64

                    if save_path and os.path.exists(save_path):
                        result["save_path"] = save_path

                    self.result_queue.put(result)

            except Exception as e:
                logger.error(f"[Rank 0] Communication thread error: {str(e)}", exc_info=True)  # noqa: G004, G201
                self.result_queue.put({"status": "error", "message": str(e)})

    def _worker_loop(self):
        """Worker进程的主循环，等待并参与分布式推理"""
        logger.info(f"[Rank {self.rank}] Worker ready, entering request loop...")  # noqa: G004

        while True:
            try:
                command = broadcast_int(0, src=0)

                if command == 1:
                    broadcast_data = broadcast_dict({}, src=0)

                    data = broadcast_data["data"]
                    save_path = broadcast_data["save_path"]

                    args = self._build_args(data, save_path)

                    self.inference_manager.infer(args)

                elif command == 2:
                    logger.info(f"[Rank {self.rank}] Received heartbeat signal")  # noqa: G004
                    continue

            except Exception as e:
                logger.error(f"[Rank {self.rank}] Worker loop error: {str(e)}", exc_info=True)  # noqa: G004, G201

    def run(self, host: str = "0.0.0.0", port: int = 5000, debug: bool = False):
        if self.rank == 0:
            if self.world_size > 1:
                comm_thread = threading.Thread(target=self._master_comm_thread, daemon=True)
                comm_thread.start()
                logger.info("[Rank 0] Communication thread started")

            self._start_heartbeat_thread()

            logger.info(f"[Rank 0] Starting ImageGenerationService on {host}:{port}")  # noqa: G004
            self.app.run(host=host, port=port, debug=debug, threaded=True)
        else:
            self._worker_loop()


def _create_arg_parser():
    import argparse

    parser = argparse.ArgumentParser(description="Image Generation Service")

    parser.add_argument("--host", type=str, default="0.0.0.0", help="Service host")
    parser.add_argument("--port", type=int, default=5000, help="Service port")
    parser.add_argument("--config", type=str, default=None, help="Config file path")
    parser.add_argument("--init", action="store_true", help="Initialize service on startup")
    parser.add_argument(
        "--cfg-parallel-size", type=int, default=None, choices=[1, 2], help="CFG parallel size (1 or 2)"
    )

    parser.add_argument("--model", type=str, default=None, help="Model name (e.g., Qwen-Image, Qwen-Image-Edit)")
    parser.add_argument(
        "--model-path", type=str, default=None, help="Model path (overrides pretrained_model_name_or_path)"
    )
    parser.add_argument(
        "--dtype", type=str, default=None, choices=["fp16", "bf16", "fp32"], help="Data type (fp16, bf16, fp32)"
    )
    parser.add_argument("--matmul-a8w8", action="store_true", default=None, help="Enable matmul a8w8 quantization")
    parser.add_argument("--no-matmul-a8w8", action="store_true", help="Disable matmul a8w8 quantization")
    parser.add_argument("--atten-laser", action="store_true", default=None, help="Enable attention laser")
    parser.add_argument("--no-atten-laser", action="store_true", help="Disable attention laser")
    parser.add_argument("--cache-dit", action="store_true", default=None, help="Enable cache dit")
    parser.add_argument("--no-cache-dit", action="store_true", help="Disable cache dit")

    parser.add_argument(
        "--lora-path", type=str, nargs="+", default=None, help="LoRA path list (multiple paths supported)"
    )
    parser.add_argument(
        "--lora-weight", type=float, nargs="+", default=None, help="LoRA scale weight list (corresponding to lora-path)"
    )
    parser.add_argument("--fuse-lora", action="store_true", default=False, help="Fuse LoRA weights into model")

    parser.add_argument("--warmup", action="store_true", help="Run warmup inference on startup")
    parser.add_argument("--warmup-width", type=int, default=512, help="Warmup image width (default: 512)")
    parser.add_argument("--warmup-height", type=int, default=512, help="Warmup image height (default: 512)")
    parser.add_argument("--warmup-steps", type=int, default=10, help="Warmup inference steps (default: 10)")
    parser.add_argument("--warmup-prompt", type=str, default="A beautiful landscape", help="Warmup prompt text")

    return parser


def _init_distributed_env(cfg_parallel_size):
    if cfg_parallel_size and cfg_parallel_size == 2:
        if dist.is_initialized():
            logger.info(f"Distributed already initialized with backend: {dist.get_backend()}")  # noqa: G004
            logger.info("Destroying existing process group and re-initializing with hccl...")
            dist.destroy_process_group()
        init_cfg_env(cfg_parallel_size)


def _apply_service_config(service, args):
    if args.model:
        if args.model not in MODEL_REGISTRY:
            raise ValueError(f"Unsupported model: {args.model}. Supported: {list(MODEL_REGISTRY.keys())}")
        service.default_config["model"] = args.model
        service.default_config["pretrained_model_name_or_path"] = MODEL_REGISTRY[args.model]
        logger.info(f"Model set to: {args.model}")  # noqa: G004

    if args.model_path:
        service.default_config["pretrained_model_name_or_path"] = args.model_path
        logger.info(f"Model path set to: {args.model_path}")  # noqa: G004

    if args.dtype:
        service.default_config["dtype"] = args.dtype
        logger.info(f"Dtype set to: {args.dtype}")  # noqa: G004

    if args.cfg_parallel_size:
        service.default_config["cfg_parallel_size"] = args.cfg_parallel_size
        logger.info(f"CFG parallel size set to: {args.cfg_parallel_size}")  # noqa: G004

    bool_flags = [
        ("matmul_a8w8", "no_matmul_a8w8", "matmul_a8w8", "Matmul a8w8"),
        ("atten_laser", "no_atten_laser", "atten_laser", "Attention laser"),
        ("cache_dit", "no_cache_dit", "cache_dit", "Cache dit"),
    ]
    for enable_flag, disable_flag, config_key, name in bool_flags:
        if getattr(args, enable_flag):
            service.default_config[config_key] = True
            logger.info(f"{name} enabled")  # noqa: G004
        elif getattr(args, disable_flag):
            service.default_config[config_key] = False
            logger.info(f"{name} disabled")  # noqa: G004

    if args.lora_path:
        service.default_config["lora_path_list"] = args.lora_path
        logger.info(f"LoRA paths set to: {args.lora_path}")  # noqa: G004
    if args.lora_weight:
        service.default_config["lora_scale_weight_list"] = args.lora_weight
        logger.info(f"LoRA weights set to: {args.lora_weight}")  # noqa: G004
    if args.fuse_lora:
        service.default_config["fuse_lora"] = True
        logger.info("LoRA fusion enabled")

    if args.init:
        service.inference_manager = CachedImageInferenceManager()
        service.is_initialized = True
        logger.info(f"[Rank {service.rank}] Service pre-initialized")  # noqa: G004


def _run_warmup_inference(service, args):
    if not args.warmup:
        return

    warmup_data = {
        "prompt": args.warmup_prompt,
        "width": args.warmup_width,
        "height": args.warmup_height,
        "num_inference_steps": args.warmup_steps,
        "seed": 42,
        "return_base64": False,
    }

    model_name = service.default_config.get("model", "Qwen-Image")
    if "Edit" in model_name:
        warmup_input_path = "/home/ma-user/generated_images/warmup_input_image.png"
        generated_path = generate_warmup_input_image(
            warmup_input_path, width=args.warmup_width, height=args.warmup_height
        )
        if generated_path:
            warmup_data["image_path"] = generated_path
        else:
            logger.warning("[Warmup] Failed to generate input image, skipping warmup")
            return

    mode_str = "image edit" if "Edit" in model_name else "text to image"
    logger.info("[Rank 0] Running warmup inference...")
    logger.info(f"[Rank 0] Warmup mode: {mode_str}")  # noqa: G004

    if service.world_size > 1:
        _run_warmup_distributed(service, warmup_data)
    else:
        _run_warmup_single(service, warmup_data)


def _run_warmup_distributed(service, warmup_data):
    comm_thread = threading.Thread(target=service._master_comm_thread, daemon=True)
    comm_thread.start()
    logger.info("[Rank 0] Communication thread started for warmup")

    service.task_queue.put(
        {"type": "generate", "data": warmup_data, "save_path": "/home/ma-user/generated_images/warmup_output_image.png"}
    )

    result = service.result_queue.get()
    if result.get("status") == "success":
        logger.info(f"[Rank 0] Warmup completed in {result.get('infer_time', 0):.2f}s")  # noqa: G004
    else:
        logger.error(f"[Rank 0] Warmup failed: {result.get('message', 'unknown error')}")  # noqa: G004


def _run_warmup_single(service, warmup_data):
    try:
        warmup_args = service._build_args(warmup_data, "/home/ma-user/generated_images/warmup_output_image.png")
        warmup_time = service.inference_manager.infer(warmup_args)
        logger.info(f"[Rank 0] Warmup completed in {warmup_time:.2f}s")  # noqa: G004
    except Exception as e:
        logger.error(f"[Rank 0] Warmup failed: {str(e)}")  # noqa: G004


def main():
    parser = _create_arg_parser()
    args = parser.parse_args()

    _init_distributed_env(args.cfg_parallel_size)

    service = ImageGenerationService(config_path=args.config)

    _apply_service_config(service, args)

    if service.world_size > 1 and service.rank != 0:
        service._worker_loop()
        return

    _run_warmup_inference(service, args)

    service.run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
