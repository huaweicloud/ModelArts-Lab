import functools

import torch
import torch.distributed as dist
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.wrap import (
    lambda_auto_wrap_policy,
    size_based_auto_wrap_policy,
    transformer_auto_wrap_policy,
)
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.models.umt5.modeling_umt5 import UMT5Block

from ..operator.matmul import WeightQuantLinearModule


def transformer_fsdp(pipe, is_need_dtype, dtype=torch.bfloat16, transformer_2=False):
    transformer_size_wrap_policy = functools.partial(size_based_auto_wrap_policy, min_num_params=20000)
    mp_policy = MixedPrecision(
        param_dtype=dtype,
        reduce_dtype=dtype,
        buffer_dtype=dtype,
    )
    if is_need_dtype:
        pipe.transformer = pipe.transformer.to(torch.bfloat16)

    pipe.transformer = FSDP(
        pipe.transformer,
        auto_wrap_policy=transformer_size_wrap_policy,
        mixed_precision=mp_policy,
        device_id=torch.npu.current_device(),
    )

    if transformer_2:
        if is_need_dtype:
            pipe.transformer_2 = pipe.transformer_2.to(torch.bfloat16)
        pipe.transformer_2 = FSDP(
            pipe.transformer_2,
            auto_wrap_policy=transformer_size_wrap_policy,
            mixed_precision=mp_policy,
            device_id=torch.npu.current_device(),
        )


def transformer_lambda_fsdp(
    model,
    device_id,
    is_need_dtype,
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.bfloat16,
    buffer_dtype=torch.bfloat16,
    process_group=None,
    sharding_strategy=ShardingStrategy.FULL_SHARD,
    sync_module_states=True,
):
    if is_need_dtype:
        model = model.to(param_dtype)

    weight_quant_layers = []

    def _recursive_search(module, prefix=""):
        for name, child in module.named_children():
            child_name = f"{prefix}.{name}" if prefix else name
            if isinstance(child, WeightQuantLinearModule):
                weight_quant_layers.append(child)
            _recursive_search(child, child_name)

    _recursive_search(model)

    model = FSDP(
        module=model,
        process_group=process_group,
        sharding_strategy=sharding_strategy,
        auto_wrap_policy=functools.partial(lambda_auto_wrap_policy, lambda_fn=lambda m: m in model.blocks),
        mixed_precision=MixedPrecision(param_dtype=param_dtype, reduce_dtype=reduce_dtype, buffer_dtype=buffer_dtype),
        device_id=device_id,
        ignored_modules=weight_quant_layers,
        sync_module_states=sync_module_states,
    )
    return model


def text_encoder_fsdp(pipe, model_type: str, dtype=torch.bfloat16):
    if model_type == "t5":
        transformer_cls = {UMT5Block}
        pipe.text_encoder = pipe.text_encoder.to(dtype)
    elif model_type == "llama":
        transformer_cls = {LlamaDecoderLayer}
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls=transformer_cls,
    )

    mp_policy = MixedPrecision(
        param_dtype=dtype,
        reduce_dtype=dtype,
        buffer_dtype=dtype,
    )

    pipe.text_encoder = FSDP(
        pipe.text_encoder,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=mp_policy,
        device_id=torch.npu.current_device(),
    )


def fsdp_init(pipe, args, is_need_dtype, te_model_type: str, dtype=torch.bfloat16, transformer_2=False):
    # Check if distributed computing is available
    if not dist.is_available():
        raise RuntimeError(
            "Distributed computing is not available. Please ensure that PyTorch is installed with distributed support."
        )

    # Initialize FSDP for transformer
    if args.fsdp in ["transformer", "all"]:
        if "Wan" in args.model:
            pipe.transformer = transformer_lambda_fsdp(pipe.transformer, torch.npu.current_device(), is_need_dtype)
            if transformer_2:
                pipe.transformer_2 = transformer_lambda_fsdp(
                    pipe.transformer_2, torch.npu.current_device(), is_need_dtype
                )
            if args.joint:
                pipe.transformer_3 = transformer_lambda_fsdp(
                    pipe.transformer_3, torch.npu.current_device(), is_need_dtype
                )
        else:
            transformer_fsdp(pipe, is_need_dtype, dtype=dtype, transformer_2=transformer_2)

    # Initialize FSDP for text encoder
    if args.fsdp in ["text_encoder", "all"]:
        text_encoder_fsdp(pipe, te_model_type, dtype=dtype)
