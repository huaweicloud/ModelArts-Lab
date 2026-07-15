import math
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch_npu
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FromOriginalModelMixin, PeftAdapterMixin
from diffusers.models.attention import FeedForward
from diffusers.models.attention_processor import Attention, AttentionProcessor
from diffusers.models.cache_utils import CacheMixin
from diffusers.models.embeddings import (
    apply_rotary_emb,
    get_1d_rotary_pos_embed,
)
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import AdaLayerNormContinuous, AdaLayerNormZero, AdaLayerNormZeroSingle
from diffusers.models.transformers import transformer_hunyuan_video
from diffusers.models.transformers.transformer_hunyuan_video import (
    HunyuanVideoConditionEmbedding,
    HunyuanVideoPatchEmbed,
    HunyuanVideoSingleTransformerBlock,
    HunyuanVideoTokenRefiner,
    HunyuanVideoTokenReplaceAdaLayerNormZero,
    HunyuanVideoTokenReplaceAdaLayerNormZeroSingle,
    HunyuanVideoTokenReplaceSingleTransformerBlock,
    HunyuanVideoTokenReplaceTransformerBlock,
    HunyuanVideoTransformerBlock,
)
from diffusers.utils import USE_PEFT_BACKEND, scale_lora_layers, unscale_lora_layers
from x_base import (
    ParallelManager,
    all_to_all_after_attn,
    all_to_all_before_attn,
    gather_sequence,
    get_pad,
    set_pad,
    split_sequence,
)


class HunyuanVideoAttnProcessor2_0:
    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "HunyuanVideoAttnProcessor2_0 requires PyTorch 2.0. To use it, please upgrade PyTorch to 2.0."
            )

    def _apply_norms(
        self,
        attn: Attention,
        query: torch.Tensor,
        key: torch.Tensor,
        encoder_query: torch.Tensor | None = None,
        encoder_key: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        if attn.norm_q is not None:
            query = attn.norm_q(query)
            if attn.add_q_proj is None and encoder_query is not None:
                encoder_query = attn.norm_q(encoder_query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)
            if attn.add_q_proj is None and encoder_key is not None:
                encoder_key = attn.norm_k(encoder_key)
        return query, key, encoder_query, encoder_key

    def _apply_enc_proj_and_norm(
        self,
        attn: Attention,
        encoder_hidden_states: torch.Tensor,
        batch_size: int,
        attn_heads: int,
        head_dim: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        encoder_query = attn.add_q_proj(encoder_hidden_states)
        encoder_key = attn.add_k_proj(encoder_hidden_states)
        encoder_value = attn.add_v_proj(encoder_hidden_states)

        if attn.parallel_manager is not None:
            encoder_query, encoder_key, encoder_value = map(
                lambda x: split_sequence(x, attn.parallel_manager.sp_group, dim=2),
                [encoder_query, encoder_key, encoder_value],
            )

        encoder_query = encoder_query.view(batch_size, -1, attn_heads, head_dim).transpose(1, 2)
        encoder_key = encoder_key.view(batch_size, -1, attn_heads, head_dim).transpose(1, 2)
        encoder_value = encoder_value.view(batch_size, -1, attn_heads, head_dim).transpose(1, 2)

        if attn.norm_added_q is not None:
            encoder_query = attn.norm_added_q(encoder_query)
        if attn.norm_added_k is not None:
            encoder_key = attn.norm_added_k(encoder_key)

        return encoder_query, encoder_key, encoder_value

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        image_rotary_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        text_seq_length = encoder_hidden_states.size(1)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = ~attention_mask[0, 0]
            # npu_fusion_attention kernel requires Sq == Skv
            attention_mask = attention_mask.repeat(1, 1, attention_mask.shape[-1], 1)

        # 1. QKV projections
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        use_encoder_proj = attn.add_q_proj is None and encoder_hidden_states is not None  # 2 (single)
        if use_encoder_proj:
            encoder_query = attn.to_q(encoder_hidden_states)
            encoder_key = attn.to_k(encoder_hidden_states)
            encoder_value = attn.to_v(encoder_hidden_states)
        else:
            encoder_query = encoder_key = encoder_value = None

        if attn.parallel_manager is not None and attn.parallel_manager.sp_size > 1:
            if attn.heads % attn.parallel_manager.sp_size != 0:
                raise ValueError(
                    f"Number of heads {attn.heads} must be divisible by sequence parallel size {attn.parallel_manager.sp_size}"  # noqa: E501
                )
            attn_heads = attn.heads // attn.parallel_manager.sp_size
            # normally we operate pad for every all2all. but for more convient implementation
            # we move pad operation to encoder add and remove in hunyuan
            query, key, value = map(
                lambda x: all_to_all_before_attn(x, attn.parallel_manager.sp_group, scatter_dim=2, gather_dim=1),
                [query, key, value],
            )
            if use_encoder_proj:
                encoder_list = [encoder_query, encoder_key, encoder_value]
                encoder_query, encoder_key, encoder_value = map(
                    lambda x: split_sequence(x, attn.parallel_manager.sp_group, dim=2),
                    encoder_list,
                )
        else:
            attn_heads = attn.heads

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn_heads

        query, key, value = [x.view(batch_size, -1, attn_heads, head_dim).transpose(1, 2) for x in [query, key, value]]

        if use_encoder_proj:
            encoder_query, encoder_key, encoder_value = [
                x.view(batch_size, -1, attn_heads, head_dim).transpose(1, 2)
                for x in [encoder_query, encoder_key, encoder_value]
            ]

        # 2. QK normalization
        query, key, encoder_query, encoder_key = self._apply_norms(attn, query, key, encoder_query, encoder_key)

        # 3. Rotational positional embeddings applied to latent stream
        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)

        # 4. Encoder condition QKV projection and normalization
        if attn.add_q_proj is not None and encoder_hidden_states is not None:  # 1 (dual)
            encoder_query, encoder_key, encoder_value = self._apply_enc_proj_and_norm(
                attn, encoder_hidden_states, batch_size, attn_heads, head_dim
            )

        query = torch.cat([query, encoder_query], dim=2)
        key = torch.cat([key, encoder_key], dim=2)
        value = torch.cat([value, encoder_value], dim=2)

        # 5. Attention
        B, N, S, D = query.shape
        hidden_states = torch_npu.npu_fusion_attention(
            query,
            key,
            value,
            head_num=N,
            input_layout="BNSD",
            keep_prob=1.0,
            scale=(1 / math.sqrt(D)),
            atten_mask=attention_mask,
        )[0]

        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3)
        hidden_states, encoder_hidden_states = hidden_states.split(
            [hidden_states.size(1) - text_seq_length, text_seq_length], dim=1
        )

        if attn.parallel_manager is not None and attn.parallel_manager.sp_size > 1:
            hidden_states = all_to_all_after_attn(
                hidden_states, process_group=attn.parallel_manager.sp_group, scatter_dim=1, gather_dim=2
            )
            encoder_hidden_states = gather_sequence(encoder_hidden_states, attn.parallel_manager.sp_group, dim=2)

        # 6. Output projection
        if encoder_hidden_states is not None:
            if getattr(attn, "to_out", None) is not None:
                hidden_states = attn.to_out[0](hidden_states)
                hidden_states = attn.to_out[1](hidden_states)

            if getattr(attn, "to_add_out", None) is not None:
                encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        return hidden_states, encoder_hidden_states


class HunyuanVideoRotaryPosEmbed(nn.Module):
    def __init__(self, patch_size: int, patch_size_t: int, rope_dim: list[int], theta: float = 256.0) -> None:
        super().__init__()

        self.patch_size = patch_size
        self.patch_size_t = patch_size_t
        self.rope_dim = rope_dim
        self.theta = theta

        self.cache_emb: tuple[torch.Tensor, torch.Tensor] | None = None
        self.cache_rope_sizes: tuple[int, int, int] | None = None  # T, H, W

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        rope_sizes = (
            num_frames // self.patch_size_t,
            height // self.patch_size,
            width // self.patch_size,
        )

        if self.cache_emb is not None and self.cache_rope_sizes == rope_sizes:
            return self.cache_emb[0].to(hidden_states.device), self.cache_emb[1].to(hidden_states.device)

        axes_grids = []
        for i in range(3):
            # Note: The following line diverges from original behaviour. We create the grid on the device, whereas
            # original implementation creates it on CPU and then moves it to device. This results in numerical
            # differences in layerwise debugging outputs, but visually it is the same.
            grid = torch.arange(0, rope_sizes[i], device=hidden_states.device, dtype=torch.float32)
            axes_grids.append(grid)
        grid = torch.meshgrid(*axes_grids, indexing="ij")  # [W, H, T]
        grid = torch.stack(grid, dim=0)  # [3, W, H, T]

        freqs = []
        for i in range(3):
            freq = get_1d_rotary_pos_embed(self.rope_dim[i], grid[i].reshape(-1), self.theta, use_real=True)
            freqs.append(freq)

        freqs_cos = torch.cat([f[0] for f in freqs], dim=1)  # (W * H * T, D / 2)
        freqs_sin = torch.cat([f[1] for f in freqs], dim=1)  # (W * H * T, D / 2)

        self.cache_emb = (freqs_cos, freqs_sin)
        self.cache_rope_sizes = rope_sizes

        return self.cache_emb


class HunyuanVideoSingleTransformerBlock(nn.Module):  # noqa: F811
    def __init__(
        self,
        num_attention_heads: int,
        attention_head_dim: int,
        mlp_ratio: float = 4.0,
        qk_norm: str = "rms_norm",
    ) -> None:
        super().__init__()

        hidden_size = num_attention_heads * attention_head_dim
        mlp_dim = int(hidden_size * mlp_ratio)

        self.attn = Attention(
            query_dim=hidden_size,
            cross_attention_dim=None,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=hidden_size,
            bias=True,
            processor=HunyuanVideoAttnProcessor2_0(),
            qk_norm=qk_norm,
            eps=1e-6,
            pre_only=True,
        )

        self.attn.parallel_manager = None
        self.act_mlp = nn.GELU(approximate="tanh")
        self.proj_mlp = nn.Linear(hidden_size, mlp_dim)
        self.proj_out = nn.Linear(hidden_size + mlp_dim, hidden_size)
        self.norm = AdaLayerNormZeroSingle(hidden_size, norm_type="layer_norm")

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        text_seq_length = encoder_hidden_states.shape[1]
        hidden_states = torch.cat([hidden_states, encoder_hidden_states], dim=1)

        residual = hidden_states

        # 1. Input normalization
        norm_hidden_states, gate = self.norm(hidden_states, emb=temb)
        mlp_hidden_states = self.act_mlp(self.proj_mlp(norm_hidden_states))

        norm_hidden_states, norm_encoder_hidden_states = (
            norm_hidden_states[:, :-text_seq_length, :],
            norm_hidden_states[:, -text_seq_length:, :],
        )

        # 2. Attention
        attn_output, context_attn_output = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            attention_mask=attention_mask,
            image_rotary_emb=image_rotary_emb,
        )
        attn_output = torch.cat([attn_output, context_attn_output], dim=1)

        # 3. Modulation and residual connection
        hidden_states = torch.cat([attn_output, mlp_hidden_states], dim=2)
        hidden_states = gate.unsqueeze(1) * self.proj_out(hidden_states)
        hidden_states = hidden_states + residual

        hidden_states, encoder_hidden_states = (
            hidden_states[:, :-text_seq_length, :],
            hidden_states[:, -text_seq_length:, :],
        )
        return hidden_states, encoder_hidden_states


class HunyuanVideoTransformerBlock(nn.Module):  # noqa: F811
    def __init__(
        self,
        num_attention_heads: int,
        attention_head_dim: int,
        mlp_ratio: float,
        qk_norm: str = "rms_norm",
    ) -> None:
        super().__init__()

        hidden_size = num_attention_heads * attention_head_dim

        self.norm1 = AdaLayerNormZero(hidden_size, norm_type="layer_norm")
        self.norm1_context = AdaLayerNormZero(hidden_size, norm_type="layer_norm")

        self.attn = Attention(
            query_dim=hidden_size,
            cross_attention_dim=None,
            added_kv_proj_dim=hidden_size,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=hidden_size,
            context_pre_only=False,
            bias=True,
            processor=HunyuanVideoAttnProcessor2_0(),
            qk_norm=qk_norm,
            eps=1e-6,
        )

        self.attn.parallel_manager = None
        self.ff = FeedForward(hidden_size, mult=mlp_ratio, activation_fn="gelu-approximate")
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.ff_context = FeedForward(hidden_size, mult=mlp_ratio, activation_fn="gelu-approximate")
        self.norm2_context = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        freqs_cis: tuple[torch.Tensor, torch.Tensor] | None = None,
        *args,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # 1. Input normalization
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb)
        norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(
            encoder_hidden_states, emb=temb
        )

        # 2. Joint attention
        attn_output, context_attn_output = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            attention_mask=attention_mask,
            image_rotary_emb=freqs_cis,
        )

        # 3. Modulation and residual connection
        hidden_states = hidden_states + attn_output * gate_msa.unsqueeze(1)
        encoder_hidden_states = encoder_hidden_states + context_attn_output * c_gate_msa.unsqueeze(1)

        norm_hidden_states = self.norm2(hidden_states)
        norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)

        norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]

        # 4. Feed-forward
        ff_output = self.ff(norm_hidden_states)
        context_ff_output = self.ff_context(norm_encoder_hidden_states)

        hidden_states = hidden_states + gate_mlp.unsqueeze(1) * ff_output
        encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output

        return hidden_states, encoder_hidden_states


class HunyuanVideoTokenReplaceSingleTransformerBlock(nn.Module):  # noqa: F811
    def __init__(
        self,
        num_attention_heads: int,
        attention_head_dim: int,
        mlp_ratio: float = 4.0,
        qk_norm: str = "rms_norm",
    ) -> None:
        super().__init__()

        hidden_size = num_attention_heads * attention_head_dim
        mlp_dim = int(hidden_size * mlp_ratio)

        self.attn = Attention(
            query_dim=hidden_size,
            cross_attention_dim=None,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=hidden_size,
            bias=True,
            processor=HunyuanVideoAttnProcessor2_0(),
            qk_norm=qk_norm,
            eps=1e-6,
            pre_only=True,
        )

        self.attn.parallel_manager = None

        self.norm = HunyuanVideoTokenReplaceAdaLayerNormZeroSingle(hidden_size, norm_type="layer_norm")
        self.proj_mlp = nn.Linear(hidden_size, mlp_dim)
        self.act_mlp = nn.GELU(approximate="tanh")
        self.proj_out = nn.Linear(hidden_size + mlp_dim, hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        token_replace_emb: torch.Tensor = None,
        num_tokens: int = None,
    ) -> torch.Tensor:
        text_seq_length = encoder_hidden_states.shape[1]
        hidden_states = torch.cat([hidden_states, encoder_hidden_states], dim=1)

        residual = hidden_states

        # 1. Input normalization
        norm_hidden_states, gate, tr_gate = self.norm(hidden_states, temb, token_replace_emb, num_tokens)
        mlp_hidden_states = self.act_mlp(self.proj_mlp(norm_hidden_states))

        norm_hidden_states, norm_encoder_hidden_states = (
            norm_hidden_states[:, :-text_seq_length, :],
            norm_hidden_states[:, -text_seq_length:, :],
        )

        # 2. Attention
        attn_output, context_attn_output = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            attention_mask=attention_mask,
            image_rotary_emb=image_rotary_emb,
        )
        attn_output = torch.cat([attn_output, context_attn_output], dim=1)

        # 3. Modulation and residual connection
        hidden_states = torch.cat([attn_output, mlp_hidden_states], dim=2)

        proj_output = self.proj_out(hidden_states)
        hidden_states_zero = proj_output[:, :num_tokens] * tr_gate.unsqueeze(1)
        hidden_states_orig = proj_output[:, num_tokens:] * gate.unsqueeze(1)
        hidden_states = torch.cat([hidden_states_zero, hidden_states_orig], dim=1)
        hidden_states = hidden_states + residual

        hidden_states, encoder_hidden_states = (
            hidden_states[:, :-text_seq_length, :],
            hidden_states[:, -text_seq_length:, :],
        )
        return hidden_states, encoder_hidden_states


class HunyuanVideoTokenReplaceTransformerBlock(nn.Module):  # noqa: F811
    def __init__(
        self,
        num_attention_heads: int,
        attention_head_dim: int,
        mlp_ratio: float,
        qk_norm: str = "rms_norm",
    ) -> None:
        super().__init__()

        hidden_size = num_attention_heads * attention_head_dim

        self.norm1 = HunyuanVideoTokenReplaceAdaLayerNormZero(hidden_size, norm_type="layer_norm")
        self.norm1_context = AdaLayerNormZero(hidden_size, norm_type="layer_norm")

        self.attn = Attention(
            query_dim=hidden_size,
            cross_attention_dim=None,
            added_kv_proj_dim=hidden_size,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=hidden_size,
            context_pre_only=False,
            bias=True,
            processor=HunyuanVideoAttnProcessor2_0(),
            qk_norm=qk_norm,
            eps=1e-6,
        )

        self.attn.parallel_manager = None

        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.ff = FeedForward(hidden_size, mult=mlp_ratio, activation_fn="gelu-approximate")

        self.norm2_context = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.ff_context = FeedForward(hidden_size, mult=mlp_ratio, activation_fn="gelu-approximate")

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        freqs_cis: tuple[torch.Tensor, torch.Tensor] | None = None,
        token_replace_emb: torch.Tensor = None,
        num_tokens: int = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # 1. Input normalization
        (
            norm_hidden_states,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
            tr_gate_msa,
            tr_shift_mlp,
            tr_scale_mlp,
            tr_gate_mlp,
        ) = self.norm1(hidden_states, temb, token_replace_emb, num_tokens)
        norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(
            encoder_hidden_states, emb=temb
        )

        # 2. Joint attention
        attn_output, context_attn_output = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            attention_mask=attention_mask,
            image_rotary_emb=freqs_cis,
        )

        # 3. Modulation and residual connection
        hidden_states_zero = hidden_states[:, :num_tokens] + attn_output[:, :num_tokens] * tr_gate_msa.unsqueeze(1)
        hidden_states_orig = hidden_states[:, num_tokens:] + attn_output[:, num_tokens:] * gate_msa.unsqueeze(1)
        hidden_states = torch.cat([hidden_states_zero, hidden_states_orig], dim=1)
        encoder_hidden_states = encoder_hidden_states + context_attn_output * c_gate_msa.unsqueeze(1)

        norm_hidden_states = self.norm2(hidden_states)
        norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)

        hidden_states_zero = norm_hidden_states[:, :num_tokens] * (1 + tr_scale_mlp[:, None]) + tr_shift_mlp[:, None]
        hidden_states_orig = norm_hidden_states[:, num_tokens:] * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        norm_hidden_states = torch.cat([hidden_states_zero, hidden_states_orig], dim=1)
        norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]

        # 4. Feed-forward
        ff_output = self.ff(norm_hidden_states)
        context_ff_output = self.ff_context(norm_encoder_hidden_states)

        hidden_states_zero = hidden_states[:, :num_tokens] + ff_output[:, :num_tokens] * tr_gate_mlp.unsqueeze(1)
        hidden_states_orig = hidden_states[:, num_tokens:] + ff_output[:, num_tokens:] * gate_mlp.unsqueeze(1)
        hidden_states = torch.cat([hidden_states_zero, hidden_states_orig], dim=1)
        encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output

        return hidden_states, encoder_hidden_states


class HunyuanVideoTransformer3DModel(ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin, CacheMixin):
    r"""
    A Transformer model for video-like data used in [HunyuanVideo](https://huggingface.co/tencent/HunyuanVideo).

    Args:
        in_channels (`int`, defaults to `16`):
            The number of channels in the input.
        out_channels (`int`, defaults to `16`):
            The number of channels in the output.
        num_attention_heads (`int`, defaults to `24`):
            The number of heads to use for multi-head attention.
        attention_head_dim (`int`, defaults to `128`):
            The number of channels in each head.
        num_layers (`int`, defaults to `20`):
            The number of layers of dual-stream blocks to use.
        num_single_layers (`int`, defaults to `40`):
            The number of layers of single-stream blocks to use.
        num_refiner_layers (`int`, defaults to `2`):
            The number of layers of refiner blocks to use.
        mlp_ratio (`float`, defaults to `4.0`):
            The ratio of the hidden layer size to the input size in the feedforward network.
        patch_size (`int`, defaults to `2`):
            The size of the spatial patches to use in the patch embedding layer.
        patch_size_t (`int`, defaults to `1`):
            The size of the tmeporal patches to use in the patch embedding layer.
        qk_norm (`str`, defaults to `rms_norm`):
            The normalization to use for the query and key projections in the attention layers.
        guidance_embeds (`bool`, defaults to `True`):
            Whether to use guidance embeddings in the model.
        text_embed_dim (`int`, defaults to `4096`):
            Input dimension of text embeddings from the text encoder.
        pooled_projection_dim (`int`, defaults to `768`):
            The dimension of the pooled projection of the text embeddings.
        rope_theta (`float`, defaults to `256.0`):
            The value of theta to use in the RoPE layer.
        rope_axes_dim (`Tuple[int]`, defaults to `(16, 56, 56)`):
            The dimensions of the axes to use in the RoPE layer.
        image_condition_type (`str`, *optional*, defaults to `None`):
            The type of image conditioning to use. If `None`, no image conditioning is used. If `latent_concat`, the
            image is concatenated to the latent stream. If `token_replace`, the image is used to replace first-frame
            tokens in the latent stream and apply conditioning.
    """

    _supports_gradient_checkpointing = True
    _skip_layerwise_casting_patterns = ["x_embedder", "context_embedder", "norm"]
    _no_split_modules = [
        "HunyuanVideoTransformerBlock",
        "HunyuanVideoSingleTransformerBlock",
        "HunyuanVideoPatchEmbed",
        "HunyuanVideoTokenRefiner",
    ]

    @register_to_config
    def __init__(
        self,
        in_channels: int = 16,
        out_channels: int = 16,
        num_attention_heads: int = 24,
        attention_head_dim: int = 128,
        num_layers: int = 20,
        num_single_layers: int = 40,
        num_refiner_layers: int = 2,
        mlp_ratio: float = 4.0,
        patch_size: int = 2,
        patch_size_t: int = 1,
        qk_norm: str = "rms_norm",
        guidance_embeds: bool = True,
        text_embed_dim: int = 4096,
        pooled_projection_dim: int = 768,
        rope_theta: float = 256.0,
        rope_axes_dim: tuple[int] = (16, 56, 56),
        image_condition_type: str | None = None,
    ) -> None:
        super().__init__()

        supported_image_condition_types = ["latent_concat", "token_replace"]
        if image_condition_type is not None and image_condition_type not in supported_image_condition_types:
            raise ValueError(
                f"Invalid `image_condition_type` ({image_condition_type}). Supported ones are: {supported_image_condition_types}"  # noqa: E501
            )

        inner_dim = num_attention_heads * attention_head_dim
        out_channels = out_channels or in_channels

        # 1. Latent and condition embedders
        self.x_embedder = HunyuanVideoPatchEmbed((patch_size_t, patch_size, patch_size), in_channels, inner_dim)
        self.context_embedder = HunyuanVideoTokenRefiner(
            text_embed_dim, num_attention_heads, attention_head_dim, num_layers=num_refiner_layers
        )

        self.time_text_embed = HunyuanVideoConditionEmbedding(
            inner_dim, pooled_projection_dim, guidance_embeds, image_condition_type
        )

        # 2. RoPE
        self.rope = HunyuanVideoRotaryPosEmbed(patch_size, patch_size_t, rope_axes_dim, rope_theta)

        # 3. Dual stream transformer blocks
        if image_condition_type == "token_replace":
            self.transformer_blocks = nn.ModuleList(
                [
                    HunyuanVideoTokenReplaceTransformerBlock(
                        num_attention_heads, attention_head_dim, mlp_ratio=mlp_ratio, qk_norm=qk_norm
                    )
                    for _ in range(num_layers)
                ]
            )
        else:
            self.transformer_blocks = nn.ModuleList(
                [
                    HunyuanVideoTransformerBlock(
                        num_attention_heads, attention_head_dim, mlp_ratio=mlp_ratio, qk_norm=qk_norm
                    )
                    for _ in range(num_layers)
                ]
            )

        # 4. Single stream transformer blocks
        if image_condition_type == "token_replace":
            self.single_transformer_blocks = nn.ModuleList(
                [
                    HunyuanVideoTokenReplaceSingleTransformerBlock(
                        num_attention_heads, attention_head_dim, mlp_ratio=mlp_ratio, qk_norm=qk_norm
                    )
                    for _ in range(num_single_layers)
                ]
            )
        else:
            self.single_transformer_blocks = nn.ModuleList(
                [
                    HunyuanVideoSingleTransformerBlock(
                        num_attention_heads, attention_head_dim, mlp_ratio=mlp_ratio, qk_norm=qk_norm
                    )
                    for _ in range(num_single_layers)
                ]
            )

        # 5. Output projection
        self.norm_out = AdaLayerNormContinuous(inner_dim, inner_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out = nn.Linear(inner_dim, patch_size_t * patch_size * patch_size * out_channels)
        # parallel
        self.parallel_manager = None
        self.gradient_checkpointing = False

    def enable_parallel(self, dp_size, sp_size, enable_cp, ulysses_size, ring_size):
        # update cfg parallel
        cp_size = 1
        if enable_cp and sp_size % 2 == 0:
            sp_size = sp_size // 2
            cp_size = 2

        self.parallel_manager: ParallelManager = ParallelManager(dp_size, cp_size, sp_size, ulysses_size, ring_size)

        for _, module in self.named_modules():
            if hasattr(module, "parallel_manager"):
                module.parallel_manager = self.parallel_manager
        pass

    @property
    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.attn_processors
    def attn_processors(self) -> dict[str, AttentionProcessor]:
        r"""
        Returns:
            `dict` of attention processors: A dictionary containing all attention processors used in the model with
            indexed by its weight name.
        """
        processors = {}

        def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors: dict[str, AttentionProcessor]):
            if hasattr(module, "get_processor"):
                processors[f"{name}.processor"] = module.get_processor()

            for sub_name_temp, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name_temp}", child, processors)

            return processors

        for name_temp, module in self.named_children():
            fn_recursive_add_processors(name_temp, module, processors)

        return processors

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.set_attn_processor
    def set_attn_processor(self, processor: AttentionProcessor | dict[str, AttentionProcessor]):
        r"""
        Sets the attention processor to use to compute attention.

        Parameters:
            processor (`dict` of `AttentionProcessor` or only `AttentionProcessor`):
                The instantiated processor class or a dictionary of processor classes that will be set as the processor
                for **all** `Attention` layers.
                If `processor` is a dict, and the key needs to define the path to the corresponding cross attention
                processor. This is strongly recommended when setting trainable attention processors.

        """
        count = len(self.attn_processors.keys())

        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of Processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of Attention layers: {count}. Please make sure to pass {count} processor classes."
            )

        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))

            for sub_name_temp, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name_temp}", child, processor)

        for name_temp, module in self.named_children():
            fn_recursive_attn_processor(name_temp, module, processor)

    def _split_sequence_before_blocks(self, hidden_states: torch.Tensor, first_frame_num_tokens: int):
        if self.parallel_manager is not None and self.parallel_manager.sp_size > 1:
            set_pad("pad", hidden_states.shape[1], self.parallel_manager.sp_group)
            hidden_states = split_sequence(hidden_states, self.parallel_manager.sp_group, dim=1, pad=get_pad("pad"))

            rank = dist.get_rank()

            # Compute how many full "buckets" of tokens of size equal to hidden_states.size(1) fit into first_frame_num_tokens  # noqa: E501
            bucket = max((first_frame_num_tokens - 1) // hidden_states.size(1), 0)

            # Compute how many tokens are left over after distributing full buckets across ranks
            leftover = first_frame_num_tokens - rank * hidden_states.size(1)

            # Determine how many tokens this rank should process:
            # - If the rank is less than the number of full buckets, it gets a full chunk (hidden_states.size(1) tokens)
            # - If the rank is exactly the bucket index (i.e. handles leftover), it gets the remaining tokens
            # - If the rank is greater than the bucket index, it gets 0 tokens
            first_frame_num_tokens = hidden_states.size(1) if rank < bucket else (leftover if rank == bucket else 0)

        return hidden_states, first_frame_num_tokens

    def _gather_sequence_after_blocks(self, hidden_states: torch.Tensor):
        if self.parallel_manager is not None and self.parallel_manager.sp_size > 1:
            hidden_states = gather_sequence(hidden_states, self.parallel_manager.sp_group, dim=1, pad=get_pad("pad"))
        return hidden_states

    def forward(
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
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight of lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
                print("Passing a `scale` via `attention_kwargs` when not using the PEFT backend is ineffective.")

        if self.parallel_manager is not None and self.parallel_manager.cp_size > 1:
            raise NotImplementedError("cp_size > 1 is not supported for now.")

        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p, p_t = self.config.patch_size, self.config.patch_size_t
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p
        post_patch_width = width // p
        first_frame_num_tokens = 1 * post_patch_height * post_patch_width

        # 1. RoPE
        image_rotary_emb = self.rope(hidden_states)

        # 2. Conditional embeddings
        temb, token_replace_emb = self.time_text_embed(timestep, pooled_projections, guidance)

        hidden_states = self.x_embedder(hidden_states)
        encoder_hidden_states = self.context_embedder(encoder_hidden_states, timestep, encoder_attention_mask)

        # 3. Attention mask preparation
        latent_sequence_length = hidden_states.shape[1]
        condition_sequence_length = encoder_hidden_states.shape[1]
        sequence_length = latent_sequence_length + condition_sequence_length
        attention_mask = torch.zeros(
            batch_size, sequence_length, device=hidden_states.device, dtype=torch.bool
        )  # [B, N]

        effective_condition_sequence_length = encoder_attention_mask.sum(dim=1, dtype=torch.int)  # [B,]
        effective_sequence_length = latent_sequence_length + effective_condition_sequence_length

        for i in range(batch_size):
            attention_mask[i, : effective_sequence_length[i]] = True
        # [B, 1, 1, N], for broadcasting across attention heads
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)

        hidden_states, first_frame_num_tokens = self._split_sequence_before_blocks(
            hidden_states, first_frame_num_tokens
        )

        # 4. Transformer blocks
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for block in self.transformer_blocks:
                hidden_states, encoder_hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    attention_mask,
                    image_rotary_emb,
                    token_replace_emb,
                    first_frame_num_tokens,
                )

            for block in self.single_transformer_blocks:
                hidden_states, encoder_hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    attention_mask,
                    image_rotary_emb,
                    token_replace_emb,
                    first_frame_num_tokens,
                )

        else:
            for block in self.transformer_blocks:
                hidden_states, encoder_hidden_states = block(
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    attention_mask,
                    image_rotary_emb,
                    token_replace_emb,
                    first_frame_num_tokens,
                )

            for block in self.single_transformer_blocks:
                hidden_states, encoder_hidden_states = block(
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    attention_mask,
                    image_rotary_emb,
                    token_replace_emb,
                    first_frame_num_tokens,
                )

        hidden_states = self._gather_sequence_after_blocks(hidden_states)

        # 5. Output projection
        hidden_states = self.norm_out(hidden_states, temb)
        hidden_states = self.proj_out(hidden_states)

        hidden_states = hidden_states.reshape(
            batch_size, post_patch_num_frames, post_patch_height, post_patch_width, -1, p_t, p, p
        )
        hidden_states = hidden_states.permute(0, 4, 1, 5, 2, 6, 3, 7)
        hidden_states = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        if USE_PEFT_BACKEND:
            # remove the `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (hidden_states,)

        return Transformer2DModelOutput(sample=hidden_states)


transformer_hunyuan_video.HunyuanVideoRotaryPosEmbed = HunyuanVideoRotaryPosEmbed
transformer_hunyuan_video.HunyuanVideoSingleTransformerBlock = HunyuanVideoSingleTransformerBlock
transformer_hunyuan_video.HunyuanVideoTransformerBlock = HunyuanVideoTransformerBlock
transformer_hunyuan_video.HunyuanVideoTokenReplaceSingleTransformerBlock = (
    HunyuanVideoTokenReplaceSingleTransformerBlock
)
transformer_hunyuan_video.HunyuanVideoTokenReplaceTransformerBlock = HunyuanVideoTokenReplaceTransformerBlock
transformer_hunyuan_video.HunyuanVideoTransformer3DModel = HunyuanVideoTransformer3DModel
