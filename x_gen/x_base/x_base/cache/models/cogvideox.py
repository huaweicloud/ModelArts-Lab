"""
CogVideoX 模型 Cache 加速实现

包含 TeaCache 加速策略的前向传播实现。
"""

import os
from typing import Any

import numpy as np
import torch
from diffusers.utils import is_torch_version

from ..utils import cogvideox_post_forward, cogvideox_pre_forward

# 环境变量配置
REL_L1_THRESH = float(os.getenv("REL_L1_THRESH", 0.12))


# ============ TeaCache 前向传播 ============
def teacache_cogvideox_forward(
    self,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    timestep: int | float | torch.LongTensor,
    timestep_cond: torch.Tensor | None = None,
    ofs: int | float | torch.LongTensor | None = None,
    image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
    attention_kwargs: dict[str, Any] | None = None,
    return_dict: bool = True,
):
    """CogVideoX TeaCache 前向传播"""
    lora_scale, hidden_states, encoder_hidden_states, emb, batch_size, num_frames, height, width, text_seq_length = (
        cogvideox_pre_forward(
            self, hidden_states, encoder_hidden_states, timestep, timestep_cond, ofs, attention_kwargs
        )
    )

    # TeaCache 核心逻辑
    should_calc = _teacache_should_calc(self, emb)

    # 执行 transformer blocks
    hidden_states, encoder_hidden_states = _teacache_imple(
        self, hidden_states, encoder_hidden_states, should_calc, emb, image_rotary_emb
    )

    # 后置处理
    output = cogvideox_post_forward(
        self,
        hidden_states,
        encoder_hidden_states,
        emb,
        batch_size,
        num_frames,
        height,
        width,
        text_seq_length,
        lora_scale,
        return_dict,
    )

    return output


# ============ 内部辅助函数 ============
def _teacache_should_calc(self, emb):
    """判断 CogVideoX TeaCache 是否跳过"""
    if self.enable_teacache:
        if self.cnt == 0 or self.cnt == self.num_steps - 1:
            should_calc = True
            self.accumulated_rel_l1_distance = 0
        else:
            rescale_func = np.poly1d(self.coefficients)

            rel_diff = (
                ((emb - self.previous_modulated_input).abs().mean() / self.previous_modulated_input.abs().mean())
                .cpu()
                .item()
            )
            self.accumulated_rel_l1_distance += rescale_func(rel_diff)

            if self.accumulated_rel_l1_distance < self.rel_l1_thresh:
                should_calc = False
            else:
                should_calc = True
                self.accumulated_rel_l1_distance = 0

        self.previous_modulated_input = emb
        self.cnt += 1
        if self.cnt == self.num_steps:
            self.cnt = 0
    else:
        should_calc = True

    return should_calc


def _teacache_imple(self, hidden_states, encoder_hidden_states, should_calc, emb, image_rotary_emb):
    """执行 CogVideoX transformer blocks"""
    if self.enable_teacache:
        if not should_calc:
            hidden_states += self.previous_residual
            encoder_hidden_states += self.previous_residual_encoder
        else:
            ori_hidden_states = hidden_states.clone()
            ori_encoder_hidden_states = encoder_hidden_states.clone()

            # Transformer blocks with gradient checkpointing
            for i, block in enumerate(self.transformer_blocks):
                if torch.is_grad_enabled() and self.gradient_checkpointing:
                    hidden_states, encoder_hidden_states = _checkpoint_block(
                        block, hidden_states, encoder_hidden_states, emb, image_rotary_emb
                    )
                else:
                    hidden_states, encoder_hidden_states = block(
                        hidden_states=hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        temb=emb,
                        image_rotary_emb=image_rotary_emb,
                    )

            self.previous_residual = hidden_states - ori_hidden_states
            self.previous_residual_encoder = encoder_hidden_states - ori_encoder_hidden_states
    else:
        # Transformer blocks without cache
        for i, block in enumerate(self.transformer_blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states, encoder_hidden_states = _checkpoint_block(
                    block, hidden_states, encoder_hidden_states, emb, image_rotary_emb
                )
            else:
                hidden_states, encoder_hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=emb,
                    image_rotary_emb=image_rotary_emb,
                )

    return hidden_states, encoder_hidden_states


def _checkpoint_block(block, hidden_states, encoder_hidden_states, emb, image_rotary_emb):
    """Gradient checkpointing wrapper for transformer block"""

    def create_custom_forward(module):
        def custom_forward(*inputs):
            return module(*inputs)

        return custom_forward

    ckpt_kwargs: dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
    return torch.utils.checkpoint.checkpoint(
        create_custom_forward(block),
        hidden_states,
        encoder_hidden_states,
        emb,
        image_rotary_emb,
        **ckpt_kwargs,
    )


# ============ 初始化函数 ============
def teacache_init(pipe, args):
    """初始化 CogVideoX TeaCache"""
    from ..base import get_teacache_config

    # 推断模型配置键
    model_key = _infer_cogvideox_model_key(args)

    # 获取配置
    teacache_cfg = get_teacache_config("cogvideox", model_key, use_ref_steps=False)

    pipe.transformer.__class__.enable_teacache = True
    pipe.transformer.__class__.forward = teacache_cogvideox_forward
    pipe.transformer.__class__.cnt = 0
    pipe.transformer.__class__.num_steps = args.num_inference_steps
    pipe.transformer.__class__.rel_l1_thresh = REL_L1_THRESH
    pipe.transformer.__class__.accumulated_rel_l1_distance = 0
    pipe.transformer.__class__.previous_modulated_input = None
    pipe.transformer.__class__.previous_residual = None
    pipe.transformer.__class__.previous_residual_encoder = None

    # 从配置设置 coefficients
    pipe.transformer.__class__.coefficients = teacache_cfg.get("coefficients", [])


def _infer_cogvideox_model_key(args) -> str:
    """从 args 推断 CogVideoX 模型配置键

    Args:
        args: 包含 model 属性的参数对象

    Returns:
        配置文件中的模型键，如 "CogVideoX-2b", "CogVideoX1.5-5B-I2V"
    """
    model = args.model.lower() if hasattr(args, "model") else ""

    # CogVideoX 1.5 系列
    if "1.5" in model:
        if "i2v" in model:
            return "CogVideoX1.5-5B-I2V"
        return "CogVideoX1.5-5B"

    # CogVideoX 原系列
    if "5b" in model:
        if "i2v" in model:
            return "CogVideoX-5b-I2V"
        return "CogVideoX-5b"

    # 默认返回 2b 配置
    return "CogVideoX-2b"
