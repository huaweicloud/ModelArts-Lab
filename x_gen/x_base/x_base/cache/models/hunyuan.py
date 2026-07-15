"""
Hunyuan 模型 Cache 加速实现

包含 TeaCache 加速策略的前向传播实现。
"""

import os
from typing import Any

import numpy as np
import torch

from ..utils import hunyuan_post_forward, hunyuan_pre_forward

# 环境变量配置
REL_L1_THRESH = float(os.getenv("REL_L1_THRESH", 0.12))


# ============ TeaCache 前向传播 ============
def teacache_hunyuan_forward(
    self,
    hidden_states: torch.Tensor,
    timestep: torch.LongTensor,
    encoder_hidden_states: torch.Tensor,
    encoder_attention_mask: torch.Tensor,
    pooled_projections: torch.Tensor,
    guidance: torch.Tensor = None,
    attention_kwargs: dict[str, Any] | None = None,
    return_dict: bool = True,
) -> torch.Tensor | dict[str, torch.Tensor]:
    """Hunyuan TeaCache 前向传播"""
    (
        lora_scale,
        hidden_states,
        encoder_hidden_states,
        temb,
        attention_mask,
        image_rotary_emb,
        batch_size,
        post_patch_num_frames,
        post_patch_height,
        post_patch_width,
        p,
        p_t,
    ) = hunyuan_pre_forward(
        self,
        hidden_states,
        timestep,
        encoder_hidden_states,
        encoder_attention_mask,
        pooled_projections,
        guidance,
        attention_kwargs,
    )

    # TeaCache 核心逻辑
    should_calc = _teacache_should_calc(self, hidden_states, temb)

    # 执行 transformer blocks
    hidden_states, encoder_hidden_states = _teacache_imple(
        self, hidden_states, encoder_hidden_states, temb, attention_mask, image_rotary_emb, should_calc
    )

    # 后置处理
    output = hunyuan_post_forward(
        self,
        hidden_states,
        temb,
        batch_size,
        post_patch_num_frames,
        post_patch_height,
        post_patch_width,
        p,
        p_t,
        lora_scale,
        return_dict,
    )

    return output


# ============ 内部辅助函数 ============
def _teacache_should_calc(self, hidden_states, temb):
    """判断 Hunyuan TeaCache 是否跳过"""
    if self.enable_teacache:
        temb = temb.clone()
        (norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp) = self.transformer_blocks[0].norm1(
            hidden_states, emb=temb
        )

        if self.cnt == 0 or self.cnt == self.num_steps - 1:
            should_calc = True
            self.accumulated_rel_l1_distance = 0
        else:
            # 使用初始化时设置的 coefficients
            rescale_func = np.poly1d(self.coefficients)

            rel_diff = (
                (
                    (norm_hidden_states - self.previous_modulated_input).abs().mean()
                    / self.previous_modulated_input.abs().mean()
                )
                .cpu()
                .item()
            )
            self.accumulated_rel_l1_distance += rescale_func(rel_diff)

            if self.accumulated_rel_l1_distance < self.rel_l1_thresh:
                should_calc = False
            else:
                should_calc = True
                self.accumulated_rel_l1_distance = 0

        self.previous_modulated_input = norm_hidden_states
        self.cnt += 1
        if self.cnt == self.num_steps:
            self.cnt = 0
    else:
        should_calc = True

    return should_calc


def _teacache_imple(self, hidden_states, encoder_hidden_states, temb, attention_mask, image_rotary_emb, should_calc):
    """执行 Hunyuan transformer blocks"""
    if self.enable_teacache:
        if not should_calc:
            hidden_states += self.previous_residual
        else:
            origin = hidden_states.clone()
            # Pass through DiT blocks
            for block in self.transformer_blocks:
                hidden_states, encoder_hidden_states = block(
                    hidden_states, encoder_hidden_states, temb, attention_mask, image_rotary_emb
                )
            for block in self.single_transformer_blocks:
                hidden_states, encoder_hidden_states = block(
                    hidden_states, encoder_hidden_states, temb, attention_mask, image_rotary_emb
                )
            self.previous_residual = hidden_states - origin
    else:
        # Pass through DiT blocks
        for block in self.transformer_blocks:
            hidden_states, encoder_hidden_states = block(
                hidden_states, encoder_hidden_states, temb, attention_mask, image_rotary_emb
            )
        for block in self.single_transformer_blocks:
            hidden_states, encoder_hidden_states = block(
                hidden_states, encoder_hidden_states, temb, attention_mask, image_rotary_emb
            )

    return hidden_states, encoder_hidden_states


# ============ 初始化函数 ============
def teacache_init(pipe, args):
    """初始化 Hunyuan TeaCache"""
    from ..base import get_teacache_config

    # 获取配置
    teacache_cfg = get_teacache_config("hunyuan", "HunyuanVideo-13B", use_ref_steps=False)

    pipe.transformer.__class__.enable_teacache = True
    pipe.transformer.__class__.forward = teacache_hunyuan_forward
    pipe.transformer.__class__.cnt = 0
    pipe.transformer.__class__.num_steps = args.num_inference_steps
    pipe.transformer.__class__.rel_l1_thresh = REL_L1_THRESH
    pipe.transformer.__class__.accumulated_rel_l1_distance = 0
    pipe.transformer.__class__.previous_modulated_input = None
    pipe.transformer.__class__.previous_residual = None

    # 从配置设置 coefficients
    pipe.transformer.__class__.coefficients = teacache_cfg.get("coefficients", [])
