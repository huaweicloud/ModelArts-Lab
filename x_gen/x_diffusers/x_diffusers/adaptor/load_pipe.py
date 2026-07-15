import os

import cache_dit
import torch
import torch.distributed as dist
from cache_dit import DBCacheConfig
from diffusers import (
    AutoencoderKLWan,
    CogVideoXPipeline,
    FlowMatchEulerDiscreteScheduler,
    HunyuanVideoImageToVideoPipeline,
    HunyuanVideoPipeline,
    WanImageToVideoPipeline,
    WanPipeline,
    WanTransformer3DModel,
    WanVACEPipeline,
)
from diffusers.models.attention_dispatch import AttentionBackendName
from diffusers.models.transformers.transformer_hunyuan_video import HunyuanVideoTransformer3DModel
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from diffusers.utils import logging
from x_base import (
    OffloadManager,
    OffloadManager_For_Save_Memory,
    attention_manager,
    fsdp_init,
    matmul_manager,
    offload_config_manager,
    phaa_on_pipe,
    rope_manager,
    turbo_on_pipe,
)
from x_base.config import CACHE_DIT_CONFIG, QWEN_LORA_SCHEDULER_CONFIG

from ..framework.pipeline import WanImageToVideoPipelineJoint, WanVideoToVideoPipeline
from ..framework.pipeline.registry import get_pipeline_cls_by_hf_class_name, is_registered_mappings
from ..framework.schedulers.pusa_schedulers import FlowMatchEulerDiscreteSchedulerPusa
from .distillation import load_distillation_model, update_wan_distillation_pipe_params

WAN_22_FLAG = "Wan2.2"
VAE_TILING = os.getenv("VAE_TILING", "False") == "True"
logger = logging.get_logger("infer")

try:
    from seedvr.inference import SeedVR2_Pipe

    logger.info("Plugin import seedvr successful!")
except ImportError as e1:
    # 捕获所有导入相关的异常（模块不存在、类不存在等）
    logger.info(f"Plugin import seedvr2 failed: {str(e1)}")  # noqa: G004

SUPPORTED_MODEL = [
    "Wan2.1-T2V-14B",
    "Wan2.1-I2V-14B",
    "Wan2.1-T2V-1.3B",
    "Wan2.2-T2V-A14B",
    "Wan2.2-I2V-A14B",
    "Wan2.1-VACE-14B",
    "Wan2.1-VACE-1.3B",
    "CogVideoX-5b",
    "HunyuanVideo-T2V-13B",
]


def load_wan_pipe(args):
    vae = AutoencoderKLWan.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", torch_dtype=torch.float32
    )
    pipe_params = {
        "pretrained_model_name_or_path": args.pretrained_model_name_or_path,
        "vae": vae,
        "torch_dtype": torch.bfloat16,
    }
    pipe_params = update_wan_distillation_pipe_params(pipe_params, args)

    logger.info(f"building {args.model}'s pipeline...")  # noqa: G004
    pipe = None

    if args.joint:
        subfolder_name = "transformer_2" if "Wan2.2" in args.joint_model_path else "transformer"
        print(subfolder_name, flush=True)
        transformer_3 = WanTransformer3DModel.from_pretrained(args.joint_model_path, subfolder=subfolder_name)
        transformer_3.to(dtype=torch.bfloat16)
        pipe_params["transformer_3"] = transformer_3
        pipe = WanImageToVideoPipelineJoint.from_pretrained(**pipe_params)
        pipe.scheduler.config.flow_shift = args.flow_shift
    elif "VACE" in args.model:
        pipe = WanVACEPipeline.from_pretrained(**pipe_params)
    elif "T2V" in args.model or args.task_type == "t2v" or "T2I" in args.model or args.task_type == "t2i":
        pipe = WanPipeline.from_pretrained(**pipe_params)
    elif "I2V" in args.model or args.task_type == "i2v":
        pipe = WanImageToVideoPipeline.from_pretrained(**pipe_params)
    else:
        logger.error(f"Unknown task_type: {args.task_type}!")  # noqa: G004

    if "Wan2.1" in args.model and not args.joint:
        pipe.scheduler = UniPCMultistepScheduler(
            prediction_type="flow_prediction",
            use_flow_sigmas=True,
            num_train_timesteps=1000,
            flow_shift=args.flow_shift,
        )

    # if not args.ten_second:
    pipe.to(dtype=torch.bfloat16)
    return pipe


def load_v2v_pipe(args):
    logger.info("building v2v pipeline...")
    vae = AutoencoderKLWan.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", torch_dtype=torch.float32
    )

    v2v_transformer = load_distillation_model(
        args.ten_second_model_id_t2v, "transformer", args.model, args.ten_second_model_path
    )
    v2v_transformer_2 = load_distillation_model(
        args.ten_second_model_id_t2v, "transformer_2", args.model, args.ten_second_model_path_2
    )
    v2v = WanVideoToVideoPipeline.from_pretrained(
        args.ten_second_model_id_t2v,
        vae=vae,  # shared VAE
        transformer=v2v_transformer,  # isolated
        transformer_2=v2v_transformer_2,
        torch_dtype=torch.bfloat16,
    )

    v2v.load_lora_weights(
        args.pusa_lora,
        adapter_name="lightning",
    )
    v2v.load_lora_weights(
        args.pusa_lora2,
        adapter_name="lightning_2",
        load_into_transformer_2=True,
    )
    v2v.set_adapters(["lightning", "lightning_2"], adapter_weights=[1.5, 1.4])
    noise_scheduler = FlowMatchEulerDiscreteSchedulerPusa(shift=3.0, sigma_min=0.0, extra_one_step=True)
    v2v.scheduler = noise_scheduler

    if args.conv3d_w8a8:
        v2v.vae = matmul_manager.enable_vae_conv3d_quant(v2v.vae)

    return v2v


def load_seedvr2_pipe(args):
    logger.info("building seedvr2 pipeline...")
    pipe = SeedVR2_Pipe(args)
    return pipe


def load_cogvideo_pipe(args):
    logger.info("building cogvideo t2v pipeline...")
    if "T2V" in args.model or args.task_type == "t2v":
        pipe = CogVideoXPipeline.from_pretrained(args.pretrained_model_name_or_path, torch_dtype=torch.bfloat16)
    else:
        raise Exception(f"Unsupported model, supported models are: {SUPPORTED_MODEL}")
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
    return pipe


def load_hunyuanvideo_pipe(args):
    transformer = HunyuanVideoTransformer3DModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
    )
    if "T2V" in args.model or args.task_type == "t2v":
        logger.info("building hunyuan t2v pipeline...")
        pipe = HunyuanVideoPipeline.from_pretrained(
            args.pretrained_model_name_or_path, transformer=transformer, torch_dtype=torch.float16
        )
    else:
        logger.info("building hunyuan i2v pipeline...")
        pipe = HunyuanVideoImageToVideoPipeline.from_pretrained(
            args.pretrained_model_name_or_path, transformer=transformer, torch_dtype=torch.float16
        )
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
    return pipe


def to_dtype(name: str):
    return {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }[name]


def load_pipe(args):
    # 查看是否在自定义pipeline注册
    if is_registered_mappings(args.pretrained_model_name_or_path):
        return get_pipeline_cls_by_hf_class_name(args.pretrained_model_name_or_path, to_dtype(args.dtype))

    # 是否是支持的其他模型
    if args.model not in SUPPORTED_MODEL:
        raise Exception(f"Unsupported model, supported models are: {SUPPORTED_MODEL}")
    if "Wan" in args.model:
        pipe = load_wan_pipe(args)
    elif "CogVideoX" in args.model:
        pipe = load_cogvideo_pipe(args)
    elif "HunyuanVideo" in args.model:
        pipe = load_hunyuanvideo_pipe(args)
    else:
        pipe = None
    return pipe


def apply_npu_attention(pipe):
    """
    设置自定义 npu attention backend
    """
    for attr_name in dir(pipe):
        if attr_name.startswith("transformer"):
            attr_value = getattr(pipe, attr_name)
            if isinstance(attr_value, torch.nn.Module):
                attr_value.set_attention_backend(AttentionBackendName._NATIVE_NPU)


def operator_ability(pipe, args):
    # 设置自定义attention backend
    apply_npu_attention(pipe)

    if args.atten_a8w8:
        attention_manager.enable_sage_attention()
    if args.atten_laser:
        attention_manager.enable_laser_attention()
    if args.atten_rainfusion:
        attention_manager.enable_rainfusion_attention()
    if args.atten_ada_sparse:
        attention_manager.enable_sparse_attention()

    if args.matmul_a8w8 and args.matmul_a4w4:
        raise ValueError("Parameter conflict: matmul_a8w8 and matmul_a4w4 cannot be specified at the same time")
    if args.matmul_a8w8:
        pipe.transformer = matmul_manager.enable_online_dynamic_quant(pipe.transformer)
        if WAN_22_FLAG in args.model:
            pipe.transformer_2 = matmul_manager.enable_online_dynamic_quant(pipe.transformer_2)
        if args.joint:
            pipe.transformer_3 = matmul_manager.enable_online_dynamic_quant(pipe.transformer_3)
    elif args.matmul_a4w4:
        pipe.transformer = matmul_manager.enable_online_dynamic_quant(pipe.transformer, dtype="w4a4")
        if WAN_22_FLAG in args.model:
            pipe.transformer_2 = matmul_manager.enable_online_dynamic_quant(pipe.transformer_2, dtype="w4a4")
        if args.joint:
            pipe.transformer_3 = matmul_manager.enable_online_dynamic_quant(pipe.transformer_3, dtype="w4a4")

    if args.conv3d_w8a8:
        pipe.vae = matmul_manager.enable_vae_conv3d_quant(pipe.vae)
    if args.rope_fused:
        rope_manager.enable_rope_fused()


def parallelism_ability(pipe, args):
    is_contain_transformer_2 = True if WAN_22_FLAG in args.model else False  # noqa: SIM210
    if args.sp > 1:
        pipe.enable_sp(
            args.sp,
            args.ulysses_degree,
            args.ring_degree,
            enable_transformer_2=is_contain_transformer_2,
            enable_transformer_3=args.joint,
        )
        if args.phaa_num > 0:
            phaa_on_pipe(pipe, is_contain_transformer_2, transformer_3=args.joint, phaa_split_num=args.phaa_num)
    if args.vae_lightning != "none":
        pipe.enable_vae_lightning(return_output=args.adopt_sr or args.ten_second or args.vae_lightning == "encoder")
    elif VAE_TILING:
        pipe.vae.enable_tiling()
    if args.fsdp is not None:
        if args.fsdp not in ["all", "text_encoder", "transformer"]:
            logger.warning(f"Unsupported fsdp: {args.fsdp},fsdp should be 'all' or 'text_encoder' or 'transformer'!")  # noqa: G004
        else:
            te_model_type = "llama" if "HunyuanVideo" in args.model else "t5"
            is_need_dtype = not args.matmul_a8w8
            for p in pipe.transformer.parameters():
                p.requires_grad = False
            if is_contain_transformer_2:
                for p in pipe.transformer_2.parameters():
                    p.requires_grad = False
            fsdp_init(pipe, args, is_need_dtype, te_model_type, transformer_2=is_contain_transformer_2)


def general_ability(pipe, args):
    if args.turbo_mode != "default":
        turbo_on_pipe(pipe, args)
    elif args.cache_dit:
        # Check if pipeline is already cached to avoid re-enabling cache
        if not hasattr(pipe, "_context_manager"):
            cache_dit.enable_cache(
                pipe,
                cache_config=DBCacheConfig(**CACHE_DIT_CONFIG),
            )
            logger.info("use cache dit")
        else:
            logger.info("cache dit already enabled, skip re-enabling")


def check_args_conflict(args):
    if args.vae_lightning != "none" and args.sp == 1:
        args.vae_lightning = "none"
        logger.warning("VAE lightning need sp>1!")
    if args.task_type == "t2i" and args.frames != 1:
        args.frames = 1
        logger.warning("task t2i only support 1 frames!")
    if args.fuse_lora and args.matmul_a8w8:
        args.fuse_lora = False
        logger.warning("Conflict between fuse lora and matmul quant! Disable fuse_lora!")
    if args.joint:
        if not (args.x and ("Wan2.2" in args.model) and ("I2V" in args.model or args.task_type == "i2v")):
            args.joint = False
            logger.warning("Jointinfer only support Wan2.2_distill_i2v! Disable jointinfer!")
    if args.ada_brighten:
        if not ("I2V" in args.model or args.task_type == "i2v"):
            args.ada_brighten = False
            logger.warning("Ada_brighten only support i2v! Disable ada_brighten!")


def update_pipe(pipe, args):
    check_args_conflict(args)
    operator_ability(pipe, args)
    general_ability(pipe, args)
    parallelism_ability(pipe, args)
    return pipe


def load_lora(pipe, model, lora_path_list, weights_list=None, lora_transformer_list=None):
    if weights_list is None:
        weights_list = [1.0 / len(lora_path_list) for _ in lora_path_list]
    else:
        try:
            weights_list = [float(weights) for weights in weights_list]
        except Exception:
            logger.error("Invalid lora_scale_weight_list, it must be a float list!")
            weights_list = [1.0 / len(lora_path_list) for _ in lora_path_list]

    if lora_transformer_list is None:
        lora_transformer_list = [0 for _ in lora_path_list]

    adapter_name_list = [f"lora_{i}" for i in range(len(lora_path_list))]
    success_adapter_name_list = []
    success_weights_list = []
    for i, lora_path in enumerate(lora_path_list):
        try:
            if lora_transformer_list[i] == 0 or lora_transformer_list[i] == 1:
                pipe.load_lora_weights(lora_path, adapter_name=adapter_name_list[i])
            if (lora_transformer_list[i] == 0 or lora_transformer_list[i] == 2) and WAN_22_FLAG in model:
                pipe.load_lora_weights(lora_path, adapter_name=adapter_name_list[i], load_into_transformer_2=True)
            success_adapter_name_list.append(adapter_name_list[i])
            success_weights_list.append(weights_list[i])
        except Exception as e:
            logger.error(f"Load lora_weights {lora_path}. Error: {type(e).__name__}:{e}")  # noqa: G004

    pipe.set_adapters(success_adapter_name_list, success_weights_list)


def update_lora(
    pipe, model, init_lora_path_list, lora_path_list, fuse_lora=False, weights_list=None, lora_transformer_list=None
):
    if init_lora_path_list != lora_path_list:
        if init_lora_path_list:
            logger.info("unload lora weights")
            pipe.unload_lora_weights()
        if lora_path_list is not None:
            logger.info(f"load lora weights from {lora_path_list}")  # noqa: G004
            load_lora(
                pipe, model, lora_path_list, weights_list=weights_list, lora_transformer_list=lora_transformer_list
            )
    if lora_path_list is not None and fuse_lora:
        logger.info("fuse lora...")
        pipe.fuse_lora()


def update_scheduler(pipe, model, lora_path_list):
    if lora_path_list and "qwen" in model.lower():
        scheduler = FlowMatchEulerDiscreteScheduler.from_config(QWEN_LORA_SCHEDULER_CONFIG)
        logger.info(f"update scheduler use config:{QWEN_LORA_SCHEDULER_CONFIG}")  # noqa: G004
        pipe.scheduler = scheduler


def transformer_vram(transformer, inf_vram_blocks_num=0, save_memory=False):
    group = dist.group.WORLD

    # 根据transformer实例自动读取配置或者block_name
    block_names = offload_config_manager.get_block_name(transformer)
    block_module = getattr(transformer, block_names)
    logger.info("apply offload for transformer")

    for name, module in transformer.named_children():
        if name != block_names:
            module.to("npu")

    for name, param in transformer.named_parameters(recurse=False):
        if name != block_names:
            param.data = param.data.to("npu", non_blocking=True)
    if save_memory:
        offloader = OffloadManager_For_Save_Memory(
            transformer,
            module_groups={block_names: block_module},
            keep_n={block_names: int(inf_vram_blocks_num)},
            device=torch.cuda.current_device(),
            dist_group=group,
            sync_at_layer=True if dist.is_initialized() else False,  # noqa: SIM210
        )
    else:
        offloader = OffloadManager(
            transformer,
            module_groups={block_names: block_module},
            keep_n={block_names: int(inf_vram_blocks_num)},
            device=torch.cuda.current_device(),
            dist_group=group,
            sync_at_layer=True if dist.is_initialized() else False,  # noqa: SIM210
        )
    offloader.enable()
    return transformer


def pipe_to_device(pipe, args):
    is_contain_transformer_2 = True if WAN_22_FLAG in args.model else False  # noqa: SIM210

    if int(args.inf_vram_blocks_num) > 0:
        npu = torch.device("npu")
        cpu = torch.device("cpu")  # noqa: F841
        # 2) 所有需要在npu常驻的，pipeline组件，先搬运到npu

        if hasattr(pipe, "vae") and isinstance(pipe.vae, torch.nn.Module):
            pipe.vae.to(npu)

        if hasattr(pipe, "text_encoder") and isinstance(pipe.text_encoder, torch.nn.Module):
            pipe.text_encoder.to(npu)

        if hasattr(pipe, "image_encoder") and isinstance(pipe.image_encoder, torch.nn.Module):
            pipe.image_encoder.to(npu)

        if hasattr(pipe, "image_processor") and isinstance(pipe.image_processor, torch.nn.Module):
            pipe.image_processor.to(npu)

        pipe.transformer = transformer_vram(
            pipe.transformer, inf_vram_blocks_num=args.inf_vram_blocks_num, save_memory=args.save_memory
        )
        if is_contain_transformer_2:
            pipe.transformer_2 = transformer_vram(
                pipe.transformer_2, inf_vram_blocks_num=args.inf_vram_blocks_num, save_memory=args.save_memory
            )
        if hasattr(pipe, "transformer_3"):
            pipe.transformer_3 = transformer_vram(
                pipe.transformer_3, inf_vram_blocks_num=args.inf_vram_blocks_num, save_memory=args.save_memory
            )
    else:
        pipe.to("npu")

    return pipe
