import base64
import hashlib
import io
import multiprocessing as mp
import os
import platform

import torch
from cryptography.fernet import Fernet
from diffusers import WanTransformer3DModel
from diffusers.utils import logging
from safetensors.torch import load_file

# 通过环境变量 ENCRYPTION_WEIGHT 控制是否加密权重
ENCRYPTION_WEIGHT = os.getenv("ENCRYPTION_WEIGHT", "true") != "false"
logger = logging.get_logger("x")


def is_arm():
    machine = platform.machine().lower()
    return any(arch in machine for arch in ("arm", "aarch64"))


def has_npu_device():
    return True


def generate_hardware_key():
    info = {
        "is_arm": is_arm(),
        "has_npu": has_npu_device(),
    }
    key_str = str(sorted(info.items())).encode("utf-8")
    raw_key = hashlib.sha256(key_str).digest()[:32]
    return base64.urlsafe_b64encode(raw_key)


def encrypt_model_weights(model, key, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    state_dict = model.state_dict()
    for name, param in state_dict.items():
        b = io.BytesIO()
        torch.save(param.data, b)
        b.seek(0)
        pth_bytes = b.read()
        encrypted_data = Fernet(key).encrypt(pth_bytes)

        with open(os.path.join(output_dir, f"{name}.bin"), "wb") as f:
            f.write(encrypted_data)


def decrypt_worker(part_path, key):
    with open(part_path, "rb") as f:
        encrypted_data = f.read()
    base_name = os.path.basename(part_path)
    name = base_name.split(".bin")[0]
    decrypted_data = Fernet(key).decrypt(encrypted_data)
    b = io.BytesIO(decrypted_data)
    b.seek(0)
    return name, b


def load_encryption_distillation_weights(x_model_path, base_model_params, process_num=12):
    transformer = WanTransformer3DModel.from_pretrained(**base_model_params)
    device = transformer.device
    transformer = transformer.to("npu")

    files = [os.path.join(x_model_path, f"{name}.bin") for name in transformer.state_dict().keys()]  # noqa: SIM118
    key = generate_hardware_key()
    with mp.Pool(processes=process_num) as pool:
        result = pool.starmap(decrypt_worker, [(f, key) for f in files])

    transformer = transformer.to(device)
    state_dict = transformer.state_dict()
    for item in result:
        name, b = item
        state_dict[name].data.copy_(torch.load(b, weights_only=True))
    return transformer


def load_non_encryption_distillation_weights(model, x_model_path, base_model_params):
    """
    加载非加密权重。

    :param model: 模型名
    :param x_model_path: 蒸馏权重路径
    :param base_model_params: 基础模型构造参数
    :return: 加载蒸馏权重的模型
    """
    if "Wan" not in model:
        logger.error(f"model :{model} is not supported distillation mode")  # noqa: G004
        return None
    if os.path.isfile(x_model_path):
        transformer = WanTransformer3DModel.from_pretrained(**base_model_params)
        transformer.load_state_dict(load_file(x_model_path))
        return transformer
    elif os.path.isdir(x_model_path):
        return WanTransformer3DModel.from_pretrained(
            x_model_path,
            torch_dtype=torch.bfloat16,
        )
    else:
        logger.error(f"x_model_path :{x_model_path} is error")  # noqa: G004
        return None


def load_distillation_weights(model, x_model_path, base_model_params=None):
    if ENCRYPTION_WEIGHT:
        return load_encryption_distillation_weights(x_model_path, base_model_params=base_model_params)
    else:
        return load_non_encryption_distillation_weights(model, x_model_path, base_model_params=base_model_params)


def load_distillation_model(orig_model_path, subfolder, model_name, distillation_weights_path):
    base_model_params = {
        "pretrained_model_name_or_path": orig_model_path,
        "subfolder": subfolder,
        "torch_dtype": torch.bfloat16,
    }
    transformer = load_distillation_weights(model_name, distillation_weights_path, base_model_params)
    return transformer


def update_wan_distillation_pipe_params(pipe_params, args):
    if not args.x:
        return pipe_params

    if "VACE" in args.model:
        raise ValueError(f"The {args.model} model does not yet support x mode. ")

    transformer = load_distillation_model(
        args.pretrained_model_name_or_path, "transformer", args.model, args.x_model_path
    )
    if transformer is not None:
        pipe_params["transformer"] = transformer
    if "Wan2.2" in args.model:
        transformer_2 = load_distillation_model(
            args.pretrained_model_name_or_path, "transformer_2", args.model, args.x_model_path_2
        )
        if transformer_2 is not None:
            pipe_params["transformer_2"] = transformer_2
    return pipe_params
