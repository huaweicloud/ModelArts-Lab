"""
Cache 加速模块工具函数

包含各模型的前向传播前置/后置处理、LoRA 处理等通用函数。
"""

from typing import Any

import torch
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.utils import USE_PEFT_BACKEND, logging, scale_lora_layers, unscale_lora_layers

logger = logging.get_logger(__name__)


# ============ LoRA 处理 ============
def pre_forward(self, attention_kwargs: dict[str, Any] | None = None) -> float:
    """前向传播前的 LoRA 缩放处理

    Args:
        self: Transformer 模型实例
        attention_kwargs: 注意力层参数

    Returns:
        lora_scale: LoRA 缩放因子
    """
    if attention_kwargs is not None:
        attention_kwargs = attention_kwargs.copy()
        lora_scale = attention_kwargs.pop("scale", 1.0)
    else:
        lora_scale = 1.0

    if USE_PEFT_BACKEND:
        scale_lora_layers(self, lora_scale)
    else:
        if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
            logger.warning("Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective.")
    return lora_scale


def post_forward_lora(self, lora_scale: float):
    """前向传播后的 LoRA 恢复处理"""
    if USE_PEFT_BACKEND:
        unscale_lora_layers(self, lora_scale)


# ============ Wan 模型前向传播组件 ============
def wan_pre_forward(
    self,
    hidden_states: torch.Tensor,
    timestep: torch.LongTensor,
    encoder_hidden_states: torch.Tensor,
    encoder_hidden_states_image: torch.Tensor | None = None,
    attention_kwargs: dict[str, Any] | None = None,
) -> tuple:
    """Wan 模型前向传播前置处理

    返回处理后的中间结果，供后续 forward_block 使用。

    Returns:
        (lora_scale, hidden_states, encoder_hidden_states, post_patch_num_frames,
         post_patch_height, post_patch_width, rotary_emb, temb, timestep_proj,
         batch_size, p_t, p_h, p_w)
    """
    # 导入分布式相关模块
    from x_base.operator import attention_manager

    # 1. 初始化阶段
    attention_manager.update_t_idx()
    lora_scale = pre_forward(self, attention_kwargs)

    # 2. 输入变换
    batch_size, num_channels, num_frames, height, width = hidden_states.shape
    p_t, p_h, p_w = self.config.patch_size

    post_patch_num_frames = num_frames // p_t
    post_patch_height = height // p_h
    post_patch_width = width // p_w

    # 生成旋转位置编码
    rotary_emb = self.rope(hidden_states)

    # Patch embedding
    hidden_states = self.patch_embedding(hidden_states)
    hidden_states = hidden_states.flatten(2).transpose(1, 2)

    # 处理时间步序列（Wan 2.2 ti2v）
    if timestep.ndim == 2:
        ts_seq_len = timestep.shape[1]
        timestep = timestep.flatten()
    else:
        ts_seq_len = None

    # 3. 条件处理
    temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = self.condition_embedder(
        timestep, encoder_hidden_states, encoder_hidden_states_image, timestep_seq_len=ts_seq_len
    )

    if ts_seq_len is not None:
        timestep_proj = timestep_proj.unflatten(2, (6, -1))
    else:
        timestep_proj = timestep_proj.unflatten(1, (6, -1))

    # 融合图像和文本条件
    if encoder_hidden_states_image is not None:
        encoder_hidden_states = torch.concat([encoder_hidden_states_image, encoder_hidden_states], dim=1)

    return (
        lora_scale,
        hidden_states,
        encoder_hidden_states,
        post_patch_num_frames,
        post_patch_height,
        post_patch_width,
        rotary_emb,
        temb,
        timestep_proj,
        batch_size,
        p_t,
        p_h,
        p_w,
    )


def wan_post_forward(
    self,
    hidden_states: torch.Tensor,
    temb: torch.Tensor,
    batch_size: int,
    post_patch_num_frames: int,
    post_patch_height: int,
    post_patch_width: int,
    p_t: int,
    p_h: int,
    p_w: int,
    lora_scale: float,
) -> torch.Tensor:
    """Wan 模型前向传播后置处理

    Args:
        hidden_states: 经过 transformer blocks 处理的状态
        temb: 时间嵌入
        batch_size, post_patch_*: 维度参数
        p_t, p_h, p_w: patch 尺寸
        lora_scale: LoRA 缩放因子

    Returns:
        output: 最终输出张量
    """
    from x_base.sequence_parallelism import gather_sequence

    # 1. 输出归一化和投影
    if temb.ndim == 3:
        # Wan 2.2 ti2v: batch_size, seq_len, inner_dim
        shift, scale = (self.scale_shift_table.unsqueeze(0) + temb.unsqueeze(2)).chunk(2, dim=2)
        shift = shift.squeeze(2)
        scale = scale.squeeze(2)
    else:
        # 标准: batch_size, inner_dim
        shift, scale = (self.scale_shift_table + temb.unsqueeze(1)).chunk(2, dim=1)

    shift = shift.to(hidden_states.device)
    scale = scale.to(hidden_states.device)

    hidden_states = (self.norm_out(hidden_states.float()) * (1 + scale) + shift).type_as(hidden_states)
    hidden_states = self.proj_out(hidden_states)

    # 2. 维度重构
    hidden_states = hidden_states.reshape(
        batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w, -1
    )
    hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
    output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

    # 3. 上下文并行收集
    if self.parallel_manager is not None and self.parallel_manager.cp_size > 1:
        output = gather_sequence(output, self.parallel_manager.cp_group, dim=0)

    # 4. LoRA 恢复
    post_forward_lora(self, lora_scale)

    return output


# ============ Hunyuan 模型前向传播组件 ============
def hunyuan_pre_forward(
    self,
    hidden_states: torch.Tensor,
    timestep: torch.LongTensor,
    encoder_hidden_states: torch.Tensor,
    encoder_attention_mask: torch.Tensor,
    pooled_projections: torch.Tensor,
    guidance: torch.Tensor = None,
    attention_kwargs: dict[str, Any] | None = None,
) -> tuple:
    """Hunyuan 模型前向传播前置处理"""
    lora_scale = pre_forward(self, attention_kwargs)

    if self.parallel_manager is not None and self.parallel_manager.cp_size > 1:
        raise NotImplementedError("cp_size > 1 is not supported for now.")

    batch_size, num_channels, num_frames, height, width = hidden_states.shape
    p, p_t = self.config.patch_size, self.config.patch_size_t
    post_patch_num_frames = num_frames // p_t
    post_patch_height = height // p
    post_patch_width = width // p
    first_frame_num_tokens = 1 * post_patch_height * post_patch_width

    # RoPE
    image_rotary_emb = self.rope(hidden_states)

    # 条件嵌入
    temb, token_replace_emb = self.time_text_embed(timestep, pooled_projections, guidance)

    hidden_states = self.x_embedder(hidden_states)
    encoder_hidden_states = self.context_embedder(encoder_hidden_states, timestep, encoder_attention_mask)

    # 注意力掩码准备
    latent_sequence_length = hidden_states.shape[1]
    condition_sequence_length = encoder_hidden_states.shape[1]
    sequence_length = latent_sequence_length + condition_sequence_length
    attention_mask = torch.zeros(batch_size, sequence_length, device=hidden_states.device, dtype=torch.bool)

    effective_condition_sequence_length = encoder_attention_mask.sum(dim=1, dtype=torch.int)
    effective_sequence_length = latent_sequence_length + effective_condition_sequence_length

    for i in range(batch_size):
        attention_mask[i, : effective_sequence_length[i]] = True
    attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)

    hidden_states, first_frame_num_tokens = self._split_sequence_before_blocks(hidden_states, first_frame_num_tokens)

    return (
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
    )


def hunyuan_post_forward(
    self,
    hidden_states: torch.Tensor,
    temb: torch.Tensor,
    batch_size: int,
    post_patch_num_frames: int,
    post_patch_height: int,
    post_patch_width: int,
    p: int,
    p_t: int,
    lora_scale: float,
    return_dict: bool = True,
) -> torch.Tensor | Transformer2DModelOutput:
    """Hunyuan 模型前向传播后置处理"""
    hidden_states = self._gather_sequence_after_blocks(hidden_states)

    # 输出投影
    hidden_states = self.norm_out(hidden_states, temb)
    hidden_states = self.proj_out(hidden_states)

    hidden_states = hidden_states.reshape(
        batch_size, post_patch_num_frames, post_patch_height, post_patch_width, -1, p_t, p, p
    )
    hidden_states = hidden_states.permute(0, 4, 1, 5, 2, 6, 3, 7)
    hidden_states = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

    post_forward_lora(self, lora_scale)

    if not return_dict:
        return (hidden_states,)
    return Transformer2DModelOutput(sample=hidden_states)


# ============ CogVideoX 模型前向传播组件 ============
def cogvideox_pre_forward(
    self,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    timestep: int | float | torch.LongTensor,
    timestep_cond: torch.Tensor | None = None,
    ofs: int | float | torch.LongTensor | None = None,
    attention_kwargs: dict[str, Any] | None = None,
) -> tuple:
    """CogVideoX 模型前向传播前置处理"""

    from x_base.sequence_parallelism import set_pad, split_sequence

    lora_scale = pre_forward(self, attention_kwargs)

    batch_size, num_frames, channels, height, width = hidden_states.shape

    # 时间嵌入
    timesteps = timestep
    t_emb = self.time_proj(timesteps)
    t_emb = t_emb.to(dtype=hidden_states.dtype)
    emb = self.time_embedding(t_emb, timestep_cond)

    if self.ofs_embedding is not None:
        ofs_emb = self.ofs_proj(ofs)
        ofs_emb = ofs_emb.to(dtype=hidden_states.dtype)
        ofs_emb = self.ofs_embedding(ofs_emb)
        emb = emb + ofs_emb

    # Patch embedding
    hidden_states = self.patch_embed(encoder_hidden_states, hidden_states)
    hidden_states = self.embedding_dropout(hidden_states)

    text_seq_length = encoder_hidden_states.shape[1]
    encoder_hidden_states = hidden_states[:, :text_seq_length]
    hidden_states = hidden_states[:, text_seq_length:]

    # 序列并行
    if self.parallel_manager.sp_size > 1:
        set_pad("pad", hidden_states.shape[1], self.parallel_manager.sp_group)
        hidden_states = split_sequence(hidden_states, self.parallel_manager.sp_group, dim=1, pad=get_pad("pad"))  # noqa: F821

    return (
        lora_scale,
        hidden_states,
        encoder_hidden_states,
        emb,
        batch_size,
        num_frames,
        height,
        width,
        text_seq_length,
    )


def cogvideox_post_forward(
    self,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    emb: torch.Tensor,
    batch_size: int,
    num_frames: int,
    height: int,
    width: int,
    text_seq_length: int,
    lora_scale: float,
    return_dict: bool = True,
) -> torch.Tensor | Transformer2DModelOutput:
    """CogVideoX 模型前向传播后置处理"""
    from x_base.sequence_parallelism import gather_sequence, get_pad

    if self.parallel_manager.sp_size > 1:
        hidden_states = gather_sequence(hidden_states, self.parallel_manager.sp_group, dim=1, pad=get_pad("pad"))

    hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
    hidden_states = self.norm_final(hidden_states)
    hidden_states = hidden_states[:, text_seq_length:]

    # Final block
    hidden_states = self.norm_out(hidden_states, temb=emb)
    hidden_states = self.proj_out(hidden_states)

    # Unpatchify
    p = self.config.patch_size
    p_t = self.config.patch_size_t

    if p_t is None:
        output = hidden_states.reshape(batch_size, num_frames, height // p, width // p, -1, p, p)
        output = output.permute(0, 1, 4, 2, 5, 3, 6).flatten(5, 6).flatten(3, 4)
    else:
        output = hidden_states.reshape(
            batch_size, (num_frames + p_t - 1) // p_t, height // p, width // p, -1, p_t, p, p
        )
        output = output.permute(0, 1, 5, 4, 2, 6, 3, 7).flatten(6, 7).flatten(4, 5).flatten(1, 2)

    if self.parallel_manager.cp_size > 1:
        output = gather_sequence(output, self.parallel_manager.cp_group, dim=0)

    post_forward_lora(self, lora_scale)

    if not return_dict:
        return (output,)
    return Transformer2DModelOutput(sample=output)
