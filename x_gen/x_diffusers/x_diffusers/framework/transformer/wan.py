import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FromOriginalModelMixin, PeftAdapterMixin
from diffusers.models.attention import FeedForward
from diffusers.models.attention_processor import Attention
from diffusers.models.cache_utils import CacheMixin
from diffusers.models.embeddings import PixArtAlphaTextProjection, TimestepEmbedding, Timesteps, get_1d_rotary_pos_embed
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import FP32LayerNorm
from diffusers.models.transformers import transformer_wan
from diffusers.models.transformers.transformer_wan import (
    WanImageEmbedding,
    _get_added_kv_projections,
    _get_qkv_projections,
)
from diffusers.utils import USE_PEFT_BACKEND, logging, scale_lora_layers, unscale_lora_layers
from diffusers.utils.torch_utils import maybe_allow_in_graph

LENGTH_CONTEXT = 512  # wan2.1 is longer than 512, wan2.2 is 512

logger = logging.get_logger("transformer_wan")  # pylint: disable=invalid-name

from x_base import (  # noqa: E402
    ParallelManager,
    all_to_all_after_attn,
    all_to_all_before_attn,
    attention_manager,
    gather_sequence,
    get_pad,
    get_phaa_split_num,
    is_phaa_enabled,
    rope_manager,
    set_pad,
    split_sequence,
)


class AscendWanAttnProcessor2_0:
    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("WanAttnProcessor2_0 requires PyTorch 2.0. To use it, please upgrade PyTorch to 2.0.")

    def i2v_forward(
        self,
        attn: Attention,
        query: torch.Tensor | None = None,
        encoder_hidden_states_img: torch.Tensor | None = None,
        attn_heads: int = None,
    ):
        # I2V task
        hidden_states_img = None
        if encoder_hidden_states_img is not None:
            key_img, value_img = _get_added_kv_projections(attn, encoder_hidden_states_img)
            key_img = attn.norm_added_k(key_img)

            if attn.parallel_manager is not None and attn.parallel_manager.sp_size > 1:
                world_size = torch.distributed.get_world_size()
                rank = torch.distributed.get_rank()
                key_img = torch.tensor_split(key_img, world_size, 2)[rank]
                value_img = torch.tensor_split(value_img, world_size, 2)[rank]

            key_img = key_img.unflatten(2, (attn_heads, -1)).transpose(1, 2)
            value_img = value_img.unflatten(2, (attn_heads, -1)).transpose(1, 2)

            hidden_states_img = attention_manager.attention(
                query, key_img, value_img, attn_mask=None, dropout_p=0.0, is_causal=False
            )

            hidden_states_img = hidden_states_img.transpose(1, 2).flatten(2, 3)
            hidden_states_img = hidden_states_img.type_as(query)

            if attn.parallel_manager is not None and attn.parallel_manager.sp_size > 1:
                hidden_states_img = all_to_all_after_attn(
                    hidden_states_img, attn.parallel_manager.sp_group, scatter_dim=1, gather_dim=2
                )
        return hidden_states_img

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        rotary_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        encoder_hidden_states_img = None
        if attn.add_k_proj is not None:
            # 512 is the context length of the text encoder, hardcoded for now
            image_context_length = encoder_hidden_states.shape[1] - LENGTH_CONTEXT
            encoder_hidden_states_img = encoder_hidden_states[:, :image_context_length]
            encoder_hidden_states = encoder_hidden_states[:, image_context_length:]
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        query, key, value = _get_qkv_projections(attn, hidden_states, encoder_hidden_states)

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        if (
            attn.parallel_manager is not None
            and attn.parallel_manager.sp_size > 1
            and not attn.parallel_manager.enable_usp
        ):
            if attn.heads % attn.parallel_manager.sp_size != 0:
                raise ValueError(
                    f"Number of heads {attn.heads} must be divisible by sequence parallel size {attn.parallel_manager.sp_size}"  # noqa: E501
                )

            if is_phaa_enabled():
                return self._phaa_parallel(
                    attn=attn,
                    query=query,
                    key=key,
                    value=value,
                    encoder_hidden_states_img=encoder_hidden_states_img,
                    attention_mask=attention_mask,
                    rotary_emb=rotary_emb,
                )

            attn_heads = attn.heads // attn.parallel_manager.sp_size
            query, key, value = map(
                lambda x: all_to_all_before_attn(x, attn.parallel_manager.sp_group, scatter_dim=2, gather_dim=1),
                [query, key, value],
            )
        else:
            attn_heads = attn.heads

        query = query.unflatten(2, (attn_heads, -1)).transpose(1, 2)
        key = key.unflatten(2, (attn_heads, -1)).transpose(1, 2)
        value = value.unflatten(2, (attn_heads, -1)).transpose(1, 2)

        if rotary_emb is not None:
            query, key = rope_manager.rope(query, key, *rotary_emb)

        hidden_states_img = self.i2v_forward(attn, query, encoder_hidden_states_img, attn_heads)

        if attn.parallel_manager is not None and attn.parallel_manager.enable_usp:
            query = query.transpose(1, 2).contiguous()
            key = key.transpose(1, 2).contiguous()
            value = value.transpose(1, 2).contiguous()

            window_size = (-1, -1)
            alibi_slopes, attn_bias = None, None  # noqa: F841
            dropout_mask = None  # noqa: F841
            deterministic = False

            hidden_states = self.usp_attn(
                query,
                key,
                value,
                dropout_p=0.0,
                causal=False,
                window_size=window_size,
                softcap=0.0,
                alibi_slopes=alibi_slopes,
                deterministic=deterministic,
                return_attn_probs=True,
                layout="BSND",
            )
            hidden_states = hidden_states.flatten(2, 3)
            hidden_states = hidden_states.type_as(query)
        else:
            hidden_states = attention_manager.attention(
                query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )
            hidden_states = hidden_states.transpose(1, 2).flatten(2, 3)
            hidden_states = hidden_states.type_as(query)

            if attn.parallel_manager is not None and attn.parallel_manager.sp_size > 1:
                hidden_states = all_to_all_after_attn(
                    hidden_states, attn.parallel_manager.sp_group, scatter_dim=1, gather_dim=2
                )

        if hidden_states_img is not None:
            hidden_states = hidden_states + hidden_states_img

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states

    def _phaa_parallel(
        self,
        attn: Attention,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        encoder_hidden_states_img: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        rotary_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """PHAA (Parallel Head All2ALL) attention executes Flash Attention in parallel with all2all communication."""

        def get_chunk(tensor, chunk=0, dim=0):
            world_size = attn.parallel_manager.sp_size
            return torch.narrow(tensor, dim, world_size * chunk, world_size)

        def apply_pad(tensor, pad, dim=2):
            if pad > 0:
                pad_size = list(tensor.shape)
                pad_size[dim] = pad
                tensor = torch.cat([tensor, torch.zeros(pad_size, dtype=tensor.dtype, device=tensor.device)], dim=dim)
            return tensor

        def reshape_for_attention(x):
            # WS S B N D -> B N WS*S D
            x = x.flatten(0, 1).permute(1, 2, 0, 3)
            if get_pad("pad") > 0:
                x = x.narrow(2, 0, x.shape[2] - get_pad("pad"))
            return x

        def reshape_after_attention(x):
            x = apply_pad(x, pad=get_pad("pad"), dim=2)
            return x.unflatten(2, (W_S, -1)).permute(2, 3, 0, 1, 4).flatten(-2, -1)

        # Prepare shapes
        W_S = attn.parallel_manager.sp_size
        fa_split_num = get_phaa_split_num() or attn.heads // W_S
        attn_heads_per_split = attn.heads // W_S // fa_split_num
        B, Sq, H = query.shape
        Skv = key.shape[1]

        # Prepare events for synchronization
        all2all_before_attn_is_finished = [torch.npu.Event() for _ in range(fa_split_num)]
        fa_is_finished = [torch.npu.Event() for _ in range(fa_split_num)]
        all2all_img_before_attn_is_finished = [torch.npu.Event() for _ in range(fa_split_num)]  # noqa: F841
        fa_img_is_finished = [torch.npu.Event() for _ in range(fa_split_num)]

        # Prepare output tensors
        query_a2a_buf = torch.empty(
            (W_S * fa_split_num, Sq, B, attn_heads_per_split, H // attn.heads), device=query.device, dtype=query.dtype
        )
        key_a2a_buf = torch.empty(
            (W_S * fa_split_num, Skv, B, attn_heads_per_split, H // attn.heads), device=key.device, dtype=key.dtype
        )
        value_a2a_buf = torch.empty(
            (W_S * fa_split_num, Skv, B, attn_heads_per_split, H // attn.heads), device=value.device, dtype=value.dtype
        )
        hidden_states = torch.empty(
            (W_S * fa_split_num, Sq, B, H // W_S // fa_split_num), device=query.device, dtype=query.dtype
        )
        hidden_states_buf = [None] * fa_split_num
        hidden_states_img_buf = [None] * fa_split_num

        hidden_states_img = None

        # 1. Communicate `query`, `key`, `value` tensors while the next tensor is being reshaped
        # B S WorldSize*SplitNum*HeadNum*D -> WorldSize*SplitNum S B HeadNum D
        query = (
            query.unflatten(-1, (W_S * fa_split_num, attn_heads_per_split, -1)).permute([2, 1, 0, 3, 4]).contiguous()
        )
        self.comm_stream.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(self.comm_stream):
            torch.distributed.all_to_all_single(
                get_chunk(query_a2a_buf, 0), get_chunk(query, 0), group=attn.parallel_manager.sp_group, async_op=False
            )

        key = key.unflatten(-1, (W_S * fa_split_num, attn_heads_per_split, -1)).permute([2, 1, 0, 3, 4]).contiguous()
        self.comm_stream.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(self.comm_stream):
            torch.distributed.all_to_all_single(
                get_chunk(key_a2a_buf, 0), get_chunk(key, 0), group=attn.parallel_manager.sp_group, async_op=False
            )

        value = (
            value.unflatten(-1, (W_S * fa_split_num, attn_heads_per_split, -1)).permute([2, 1, 0, 3, 4]).contiguous()
        )
        self.comm_stream.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(self.comm_stream):
            torch.distributed.all_to_all_single(
                get_chunk(value_a2a_buf, 0), get_chunk(value, 0), group=attn.parallel_manager.sp_group, async_op=False
            )
            all2all_before_attn_is_finished[0].record(self.comm_stream)

        if encoder_hidden_states_img is not None:
            key_img = attn.add_k_proj(encoder_hidden_states_img)
            value_img = attn.add_v_proj(encoder_hidden_states_img)
            key_img = attn.norm_added_k(key_img)

            key_img = split_sequence(key_img, attn.parallel_manager.sp_group, dim=2)
            value_img = split_sequence(value_img, attn.parallel_manager.sp_group, dim=2)

            key_img = key_img.unflatten(2, (attn.heads // W_S, -1)).transpose(1, 2)
            value_img = value_img.unflatten(2, (attn.heads // W_S, -1)).transpose(1, 2)

            hidden_states_img = torch.empty(
                (W_S * fa_split_num, Sq, B, H // W_S // fa_split_num), dtype=query.dtype, device=query.device
            )

        for step in range(fa_split_num):
            # 2. Compute attention
            torch.npu.current_stream().wait_event(all2all_before_attn_is_finished[step])
            # WS S B N D -> B N WS*S D
            query_, key_, value_ = map(
                lambda x: reshape_for_attention(get_chunk(x, step)), [query_a2a_buf, key_a2a_buf, value_a2a_buf]
            )
            if rotary_emb is not None:
                query_, key_ = rope_manager.rope(query_, key_, *rotary_emb)

            hidden_states_buf[step] = attention_manager.attention(
                query_, key_, value_, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            ).type_as(query)
            # B N S D -> WS S//WS B N*D
            hidden_states_buf[step] = reshape_after_attention(hidden_states_buf[step])
            fa_is_finished[step].record()

            if encoder_hidden_states_img is not None:
                # 2.1 Compute img attention
                key_img_ = torch.narrow(key_img, 1, attn_heads_per_split * step, attn_heads_per_split)
                value_img_ = torch.narrow(value_img, 1, attn_heads_per_split * step, attn_heads_per_split)

                hidden_states_img_buf[step] = attention_manager.attention(
                    query_, key_img_, value_img_, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
                ).type_as(query)

                # B N S D -> WS S//WS B N*D
                hidden_states_img_buf[step] = reshape_after_attention(hidden_states_img_buf[step])
                fa_img_is_finished[step].record()

            with torch.npu.stream(self.comm_stream):
                # 1. Communicate tensors before attention
                if step + 1 < fa_split_num:
                    for source, target in zip([query, key, value], [query_a2a_buf, key_a2a_buf, value_a2a_buf]):
                        torch.distributed.all_to_all_single(
                            get_chunk(target, step + 1),
                            get_chunk(source, step + 1),
                            group=attn.parallel_manager.sp_group,
                            async_op=False,
                        )
                    all2all_before_attn_is_finished[step + 1].record(self.comm_stream)
                # 3. Communicate tensors after attention
                self.comm_stream.wait_event(fa_is_finished[step])
                torch.distributed.all_to_all_single(
                    get_chunk(hidden_states, step),
                    hidden_states_buf[step],
                    group=attn.parallel_manager.sp_group,
                    async_op=False,
                )
                # 3.1 Communicate img tensors after attention
                if hidden_states_img_buf[step] is not None:
                    self.comm_stream.wait_event(fa_img_is_finished[step])
                    torch.distributed.all_to_all_single(
                        get_chunk(hidden_states_img, step),
                        hidden_states_img_buf[step],
                        group=attn.parallel_manager.sp_group,
                        async_op=False,
                    )

        torch.npu.current_stream().wait_stream(self.comm_stream)
        if hidden_states_img is not None:
            hidden_states = hidden_states + hidden_states_img
        hidden_states = hidden_states.permute([2, 1, 0, 3]).flatten(-2, -1)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states


class AscendWanRotaryPosEmbed(nn.Module):
    def __init__(
        self,
        attention_head_dim: int,
        patch_size: tuple[int, int, int],
        max_seq_len: int,
        theta: float = 10000.0,
    ):
        super().__init__()

        self.attention_head_dim = attention_head_dim
        self.patch_size = patch_size
        self.max_seq_len = max_seq_len

        h_dim = w_dim = 2 * (attention_head_dim // 6)
        t_dim = attention_head_dim - h_dim - w_dim
        freqs_dtype = torch.float32 if torch.backends.mps.is_available() else torch.float64

        freqs_cos = []
        freqs_sin = []

        for dim in [t_dim, h_dim, w_dim]:
            freq_cos, freq_sin = get_1d_rotary_pos_embed(
                dim,
                max_seq_len,
                theta,
                use_real=True,
                repeat_interleave_real=True,
                freqs_dtype=freqs_dtype,
            )
            freqs_cos.append(freq_cos)
            freqs_sin.append(freq_sin)

        self.register_buffer("freqs_cos", torch.cat(freqs_cos, dim=1), persistent=False)
        self.register_buffer("freqs_sin", torch.cat(freqs_sin, dim=1), persistent=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.patch_size
        ppf, pph, ppw = num_frames // p_t, height // p_h, width // p_w

        split_sizes = [
            self.attention_head_dim - 2 * (self.attention_head_dim // 3),
            self.attention_head_dim // 3,
            self.attention_head_dim // 3,
        ]

        freqs_cos = self.freqs_cos.split(split_sizes, dim=1)
        freqs_sin = self.freqs_sin.split(split_sizes, dim=1)

        freqs_cos_f = freqs_cos[0][:ppf].view(ppf, 1, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_cos_h = freqs_cos[1][:pph].view(1, pph, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_cos_w = freqs_cos[2][:ppw].view(1, 1, ppw, -1).expand(ppf, pph, ppw, -1)

        freqs_sin_f = freqs_sin[0][:ppf].view(ppf, 1, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_sin_h = freqs_sin[1][:pph].view(1, pph, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_sin_w = freqs_sin[2][:ppw].view(1, 1, ppw, -1).expand(ppf, pph, ppw, -1)

        freqs_cos = torch.cat([freqs_cos_f, freqs_cos_h, freqs_cos_w], dim=-1).reshape(1, 1, ppf * pph * ppw, -1)
        freqs_sin = torch.cat([freqs_sin_f, freqs_sin_h, freqs_sin_w], dim=-1).reshape(1, 1, ppf * pph * ppw, -1)

        return freqs_cos, freqs_sin


@maybe_allow_in_graph
class AscendWanTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        qk_norm: str = "rms_norm_across_heads",
        cross_attn_norm: bool = False,
        eps: float = 1e-6,
        added_kv_proj_dim: int | None = None,
    ):
        super().__init__()

        # 1. Self-attention
        self.norm1 = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.attn1 = Attention(
            query_dim=dim,
            heads=num_heads,
            kv_heads=num_heads,
            dim_head=dim // num_heads,
            qk_norm=qk_norm,
            eps=eps,
            bias=True,
            cross_attention_dim=None,
            out_bias=True,
            processor=AscendWanAttnProcessor2_0(),
        )
        self.attn1.parallel_manager = None
        # 2. Cross-attention
        self.attn2 = Attention(
            query_dim=dim,
            heads=num_heads,
            kv_heads=num_heads,
            dim_head=dim // num_heads,
            qk_norm=qk_norm,
            eps=eps,
            bias=True,
            cross_attention_dim=None,
            out_bias=True,
            added_kv_proj_dim=added_kv_proj_dim,
            added_proj_bias=True,
            processor=AscendWanAttnProcessor2_0(),
        )

        self.attn2.parallel_manager = None
        self.norm2 = FP32LayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()

        # 3. Feed-forward
        self.ffn = FeedForward(dim, inner_dim=ffn_dim, activation_fn="gelu-approximate")
        self.norm3 = FP32LayerNorm(dim, eps, elementwise_affine=False)

        self.scale_shift_table = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        rotary_emb: torch.Tensor,
    ) -> torch.Tensor:
        if temb.ndim == 4:
            # temb: batch_size, seq_len, 6, inner_dim (wan2.2 ti2v)
            shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
                self.scale_shift_table.unsqueeze(0) + temb.float()
            ).chunk(6, dim=2)
            # batch_size, seq_len, 1, inner_dim
            shift_msa = shift_msa.squeeze(2)
            scale_msa = scale_msa.squeeze(2)
            gate_msa = gate_msa.squeeze(2)
            c_shift_msa = c_shift_msa.squeeze(2)
            c_scale_msa = c_scale_msa.squeeze(2)
            c_gate_msa = c_gate_msa.squeeze(2)
        else:
            # temb: batch_size, 6, inner_dim (wan2.1/wan2.2 14B)
            shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
                self.scale_shift_table + temb.float()
            ).chunk(6, dim=1)

        # 1. Self-attention
        norm_hidden_states = (self.norm1(hidden_states.float()) * (1 + scale_msa) + shift_msa).type_as(hidden_states)
        attn_output = self.attn1(hidden_states=norm_hidden_states, rotary_emb=rotary_emb)
        hidden_states = (hidden_states.float() + attn_output * gate_msa).type_as(hidden_states)

        # 2. Cross-attention
        norm_hidden_states = self.norm2(hidden_states.float()).type_as(hidden_states)
        attn_output = self.attn2(hidden_states=norm_hidden_states, encoder_hidden_states=encoder_hidden_states)
        hidden_states = hidden_states + attn_output

        # 3. Feed-forward
        norm_hidden_states = (self.norm3(hidden_states.float()) * (1 + c_scale_msa) + c_shift_msa).type_as(
            hidden_states
        )
        ff_output = self.ffn(norm_hidden_states)
        hidden_states = (hidden_states.float() + ff_output.float() * c_gate_msa).type_as(hidden_states)

        return hidden_states


class AscendWanTimeTextImageEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        time_freq_dim: int,
        time_proj_dim: int,
        text_embed_dim: int,
        image_embed_dim: int | None = None,
        pos_embed_seq_len: int | None = None,
    ):
        super().__init__()

        self.timesteps_proj = Timesteps(num_channels=time_freq_dim, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.time_embedder = TimestepEmbedding(in_channels=time_freq_dim, time_embed_dim=dim)
        self.act_fn = nn.SiLU()
        self.time_proj = nn.Linear(dim, time_proj_dim)
        self.text_embedder = PixArtAlphaTextProjection(text_embed_dim, dim, act_fn="gelu_tanh")

        self.image_embedder = None
        if image_embed_dim is not None:
            self.image_embedder = WanImageEmbedding(image_embed_dim, dim, pos_embed_seq_len=pos_embed_seq_len)

    def forward(
        self,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: torch.Tensor | None = None,
        timestep_seq_len: int | None = None,
    ):
        timestep = self.timesteps_proj(timestep)
        if timestep_seq_len is not None:
            timestep = timestep.unflatten(0, (1, timestep_seq_len))

        for p in self.time_embedder.parameters():
            time_embedder_dtype = p.dtype
            break
        else:
            buffers = list(self.time_embedder.buffers())
            time_embedder_dtype = buffers[0].dtype if buffers else encoder_hidden_states.dtype

        if timestep.dtype != time_embedder_dtype and time_embedder_dtype != torch.int8:
            timestep = timestep.to(time_embedder_dtype)
        temb = self.time_embedder(timestep).type_as(encoder_hidden_states)
        timestep_proj = self.time_proj(self.act_fn(temb))

        encoder_hidden_states = self.text_embedder(encoder_hidden_states)
        if encoder_hidden_states_image is not None:
            encoder_hidden_states_image = self.image_embedder(encoder_hidden_states_image)

        return temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image


class AscendWanTransformer3DModel(ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin, CacheMixin):
    r"""
    A Transformer model for video-like data used in the Wan model.

    Args:
        patch_size (`Tuple[int]`, defaults to `(1, 2, 2)`):
            3D patch dimensions for video embedding (t_patch, h_patch, w_patch).
        num_attention_heads (`int`, defaults to `40`):
            Fixed length for text embeddings.
        attention_head_dim (`int`, defaults to `128`):
            The number of channels in each head.
        in_channels (`int`, defaults to `16`):
            The number of channels in the input.
        out_channels (`int`, defaults to `16`):
            The number of channels in the output.
        text_dim (`int`, defaults to `512`):
            Input dimension for text embeddings.
        freq_dim (`int`, defaults to `256`):
            Dimension for sinusoidal time embeddings.
        ffn_dim (`int`, defaults to `13824`):
            Intermediate dimension in feed-forward network.
        num_layers (`int`, defaults to `40`):
            The number of layers of transformer blocks to use.
        window_size (`Tuple[int]`, defaults to `(-1, -1)`):
            Window size for local attention (-1 indicates global attention).
        cross_attn_norm (`bool`, defaults to `True`):
            Enable cross-attention normalization.
        qk_norm (`bool`, defaults to `True`):
            Enable query/key normalization.
        eps (`float`, defaults to `1e-6`):
            Epsilon value for normalization layers.
        add_img_emb (`bool`, defaults to `False`):
            Whether to use img_emb.
        added_kv_proj_dim (`int`, *optional*, defaults to `None`):
            The number of channels to use for the added key and value projections. If `None`, no projection is used.
    """

    _supports_gradient_checkpointing = True
    _skip_layerwise_casting_patterns = ["patch_embedding", "condition_embedder", "norm"]
    _no_split_modules = ["WanTransformerBlock"]
    _keep_in_fp32_modules = ["time_embedder", "scale_shift_table", "norm1", "norm2", "norm3"]
    _keys_to_ignore_on_load_unexpected = ["norm_added_q"]
    _repeated_blocks = ["WanTransformerBlock"]

    @register_to_config
    def __init__(
        self,
        patch_size: tuple[int] = (1, 2, 2),
        num_attention_heads: int = 40,
        attention_head_dim: int = 128,
        in_channels: int = 16,
        out_channels: int = 16,
        text_dim: int = 4096,
        freq_dim: int = 256,
        ffn_dim: int = 13824,
        num_layers: int = 40,
        cross_attn_norm: bool = True,
        qk_norm: str | None = "rms_norm_across_heads",
        eps: float = 1e-6,
        image_dim: int | None = None,
        added_kv_proj_dim: int | None = None,
        rope_max_seq_len: int = 1024,
        pos_embed_seq_len: int | None = None,
    ) -> None:
        super().__init__()

        inner_dim = num_attention_heads * attention_head_dim
        out_channels = out_channels or in_channels

        # 1. Patch & position embedding
        self.rope = AscendWanRotaryPosEmbed(attention_head_dim, patch_size, rope_max_seq_len)
        self.patch_embedding = nn.Conv3d(in_channels, inner_dim, kernel_size=patch_size, stride=patch_size)

        # teacache
        self.should_calc_even = None
        self.should_calc_odd = None

        # 2. Condition embeddings
        # image_embedding_dim=1280 for I2V model
        self.condition_embedder = AscendWanTimeTextImageEmbedding(
            dim=inner_dim,
            time_freq_dim=freq_dim,
            time_proj_dim=inner_dim * 6,
            text_embed_dim=text_dim,
            image_embed_dim=image_dim,
            pos_embed_seq_len=pos_embed_seq_len,
        )

        # 3. Transformer blocks
        self.blocks = nn.ModuleList(
            [
                AscendWanTransformerBlock(
                    inner_dim, ffn_dim, num_attention_heads, qk_norm, cross_attn_norm, eps, added_kv_proj_dim
                )
                for _ in range(num_layers)
            ]
        )

        # 4. Output norm & projection
        self.norm_out = FP32LayerNorm(inner_dim, eps, elementwise_affine=False)
        self.proj_out = nn.Linear(inner_dim, out_channels * math.prod(patch_size))
        self.scale_shift_table = nn.Parameter(torch.randn(1, 2, inner_dim) / inner_dim**0.5)

        self.gradient_checkpointing = False
        # parallel
        self.parallel_manager = None

    def enable_parallel(self, dp_size, sp_size, enable_cp, ulysses_size, ring_size):
        # update cfg parallel
        if enable_cp and sp_size % 2 == 0:
            sp_size = sp_size // 2
            cp_size = 2
        else:
            cp_size = 1

        self.parallel_manager: ParallelManager = ParallelManager(dp_size, cp_size, sp_size, ulysses_size, ring_size)

        if self.parallel_manager.enable_usp:
            from yunchang import LongContextAttention
            from yunchang.kernels import AttnType

            ring_impl_type = "basic"
            self.usp_attn = LongContextAttention(
                ring_impl_type=ring_impl_type,
                attn_type=AttnType.NPU,
            )
            for _, module in self.named_modules():
                if isinstance(module, Attention):
                    module.processor.usp_attn = self.usp_attn

        for _, module in self.named_modules():
            if hasattr(module, "parallel_manager"):
                module.parallel_manager = self.parallel_manager

    def forward_block(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep_proj: torch.Tensor,
        rotary_emb: torch.Tensor,
    ) -> torch.Tensor:
        if self.parallel_manager is not None and self.parallel_manager.enable_usp:
            set_pad("rope", rotary_emb[0].shape[2], self.parallel_manager.sp_group)
            rotary_emb_0 = split_sequence(rotary_emb[0], self.parallel_manager.sp_group, dim=2, pad=get_pad("rope"))
            rotary_emb_1 = split_sequence(rotary_emb[1], self.parallel_manager.sp_group, dim=2, pad=get_pad("rope"))
            rotary_emb = (rotary_emb_0, rotary_emb_1)

        if self.parallel_manager is not None and self.parallel_manager.sp_size > 1:
            set_pad("pad", hidden_states.shape[1], self.parallel_manager.sp_group)
            if hidden_states.shape[1] == timestep_proj.shape[1]:
                timestep_proj = split_sequence(timestep_proj, self.parallel_manager.sp_group, dim=1, pad=get_pad("pad"))
            hidden_states = split_sequence(hidden_states, self.parallel_manager.sp_group, dim=1, pad=get_pad("pad"))

            if encoder_hidden_states.size(1) <= LENGTH_CONTEXT:
                set_pad("encode_pad", encoder_hidden_states.shape[1], self.parallel_manager.sp_group)
                encoder_hidden_states = split_sequence(
                    encoder_hidden_states, self.parallel_manager.sp_group, dim=1, pad=get_pad("encode_pad")
                )

        # 4. Transformer blocks
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for block in self.blocks:
                hidden_states = self._gradient_checkpointing_func(
                    block, hidden_states, encoder_hidden_states, timestep_proj, rotary_emb
                )
        else:
            for block in self.blocks:
                hidden_states = block(hidden_states, encoder_hidden_states, timestep_proj, rotary_emb)

        if self.parallel_manager is not None and self.parallel_manager.sp_size > 1:
            hidden_states = gather_sequence(hidden_states, self.parallel_manager.sp_group, dim=1, pad=get_pad("pad"))
        return hidden_states

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: torch.Tensor | None = None,
        return_dict: bool = True,
        attention_kwargs: dict[str, Any] | None = None,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        attention_manager.update_t_idx()
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
                logger.warning("Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective.")

        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.config.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w

        rotary_emb = self.rope(hidden_states)

        hidden_states = self.patch_embedding(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)

        # timestep shape: batch_size, or batch_size, seq_len (wan 2.2 ti2v)
        if timestep.ndim == 2:
            ts_seq_len = timestep.shape[1]
            timestep = timestep.flatten()  # batch_size * seq_len
        else:
            ts_seq_len = None

        temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = self.condition_embedder(
            timestep, encoder_hidden_states, encoder_hidden_states_image, timestep_seq_len=ts_seq_len
        )
        if ts_seq_len is not None:
            # batch_size, seq_len, 6, inner_dim
            timestep_proj = timestep_proj.unflatten(2, (6, -1))
        else:
            # batch_size, 6, inner_dim
            timestep_proj = timestep_proj.unflatten(1, (6, -1))

        if encoder_hidden_states_image is not None:
            encoder_hidden_states = torch.concat([encoder_hidden_states_image, encoder_hidden_states], dim=1)

        hidden_states = self.forward_block(hidden_states, encoder_hidden_states, timestep_proj, rotary_emb)

        # 5. Output norm, projection & unpatchify
        if temb.ndim == 3:
            # batch_size, seq_len, inner_dim (wan 2.2 ti2v)
            shift, scale = (self.scale_shift_table.unsqueeze(0) + temb.unsqueeze(2)).chunk(2, dim=2)
            shift = shift.squeeze(2)
            scale = scale.squeeze(2)
        else:
            # batch_size, inner_dim
            shift, scale = (self.scale_shift_table + temb.unsqueeze(1)).chunk(2, dim=1)

        # Move the shift and scale tensors to the same device as hidden_states.
        # When using multi-GPU inference via accelerate these will be on the
        # first device rather than the last device, which hidden_states ends up
        # on.
        shift = shift.to(hidden_states.device)
        scale = scale.to(hidden_states.device)

        hidden_states = (self.norm_out(hidden_states.float()) * (1 + scale) + shift).type_as(hidden_states)
        hidden_states = self.proj_out(hidden_states)

        hidden_states = hidden_states.reshape(
            batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w, -1
        )
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        if self.parallel_manager is not None and self.parallel_manager.cp_size > 1:
            output = gather_sequence(output, self.parallel_manager.cp_group, dim=0)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)


transformer_wan.WanAttnProcessor2_0 = AscendWanAttnProcessor2_0
transformer_wan.WanRotaryPosEmbed = AscendWanRotaryPosEmbed
transformer_wan.WanTransformerBlock = AscendWanTransformerBlock
transformer_wan.WanTimeTextImageEmbedding = AscendWanTimeTextImageEmbedding
transformer_wan.WanTransformer3DModel = AscendWanTransformer3DModel
