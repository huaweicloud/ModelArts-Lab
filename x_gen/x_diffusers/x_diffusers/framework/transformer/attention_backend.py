from typing import TYPE_CHECKING, Union

import torch
from diffusers.models.attention_dispatch import (
    AttentionBackendName,
    _AttentionBackendRegistry,
    _check_device,
    _check_qkv_dtype_bf16_or_fp16,
    _check_shape,
    _maybe_modify_attn_mask_npu,
    _npu_attention_backward_op,
    _npu_attention_forward_op,
    _templated_context_parallel_attention,
)

if TYPE_CHECKING:
    from diffusers.models._modeling_parallel import ParallelConfig

from x_base import attention_manager


@_AttentionBackendRegistry.register(
    AttentionBackendName._NATIVE_NPU,
    constraints=[_check_device, _check_qkv_dtype_bf16_or_fp16, _check_shape],
    supports_context_parallel=True,
)
def _native_npu_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    scale: float | None = None,
    return_lse: bool = False,
    _parallel_config: Union["ParallelConfig", None] = None,
) -> torch.Tensor:
    if return_lse:
        raise ValueError("NPU attention backend does not support setting `return_lse=True`.")
    if _parallel_config is None:
        attn_mask = _maybe_modify_attn_mask_npu(query, key, attn_mask)

        query, key, value = (x.permute(0, 2, 1, 3).contiguous() for x in (query, key, value))
        out = attention_manager.attention(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=False,
        ).type_as(query)

        out = out.permute(0, 2, 1, 3)

    else:
        out = _templated_context_parallel_attention(
            query,
            key,
            value,
            attn_mask,
            dropout_p,
            None,
            scale,
            None,
            return_lse,
            forward_op=_npu_attention_forward_op,
            backward_op=_npu_attention_backward_op,
            _parallel_config=_parallel_config,
        )
    return out
