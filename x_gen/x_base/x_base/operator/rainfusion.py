#!/usr/bin/env python
# Copyright 2025 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import math

import torch
from diffusers.utils import logging
from einops import rearrange
from torch import Tensor

logger = logging.get_logger("attention")

try:
    from mindiesd import attention_forward
    from mindiesd.utils import ParametersInvalid

    MINDIE_AVAILABLE = True
except ImportError:
    logger.error("Not install MINDIE")
    MINDIE_AVAILABLE = False

WIN_RATIO = 0.95
USE_LA = True


class Rainfusion(torch.nn.Module):
    def __init__(
        self,
        grid_size: list,
        skip_timesteps: int = 0,
        sparsity: float = 0.64,
    ) -> None:
        """
        参数:
            grid_size (list): latents的THW网格大小。
            skip_timesteps (int, optional): 从skip_timesteps步开始执行rainfusion。
            sparsity (float, optional): 稀疏度, 取值范围[0, 1]，默认为 0.50。
        """
        super().__init__()
        if not isinstance(grid_size, list):
            raise ParametersInvalid(f"The data type of input grid_size must be list, but got {type(grid_size)}.")
        if not isinstance(skip_timesteps, int):
            raise ParametersInvalid(
                f"The data type of input skip_timesteps must be int, but got {type(skip_timesteps)}."
            )
        if not isinstance(sparsity, float):
            raise ParametersInvalid(f"The data type of input sparsity must be float, but got {type(sparsity)}.")
        if skip_timesteps < 0:
            raise ParametersInvalid(f"The data type of input skip_timesteps must >= 0 , but got {skip_timesteps}.")
        if sparsity < 0.0 or sparsity > 1.0:
            raise ParametersInvalid(f"The data type of input sparsity value must in [0.0, 1.0] , but got {sparsity}.")

        # Rainfusion_param
        self.grid_size = grid_size
        self.skip_timesteps = skip_timesteps
        self.bandwidth = Rainfusion.cal_bandwidth(sparsity)
        self.use_rainfusion = False if math.isclose(sparsity, 1.0, rel_tol=1e-5) else True  # noqa: SIM211
        self.use_la = USE_LA

        self.layout_transform = True
        self.num_frames = self.grid_size[0]
        self.num_tokens_per_frame = self.grid_size[1] * self.grid_size[2]
        self.num_tokens_except_first_frame = (self.num_frames - 1) * self.num_tokens_per_frame

        if not self.use_la:
            import torch_atb

            self_attention_param = torch_atb.SelfAttentionParam()
            self_attention_param.head_num = 1
            self_attention_param.kv_head_num = 1
            self_attention_param.qk_scale = 1.0 / math.sqrt(1.0 * 128)
            self_attention_param.input_layout = torch_atb.TYPE_BNSD
            self_attention_param.calc_type = torch_atb.SelfAttentionParam.CalcType.PA_ENCODER
            self.self_attention = torch_atb.Operation(self_attention_param)

    @staticmethod
    def cal_bandwidth(sparsity):
        bandwidth = 1 - math.sqrt(sparsity)
        return round(bandwidth, 2)

    @staticmethod
    def get_grid_size(latent_size, patch_size):
        t, h, w = latent_size[-3:]
        return [t // patch_size[0], h // patch_size[1], w // patch_size[2]]

    @staticmethod
    def get_atten_mask(grid_size, sparsity):
        bandwidth = Rainfusion.cal_bandwidth(sparsity)
        bs = 1
        t, h, w = grid_size
        sq_len = h * w
        next_tokens = int(sq_len * bandwidth)
        pre_tokens = int(sq_len * bandwidth)
        atten_mask = Rainfusion.get_window_atten_mask(bs, sq_len, next_tokens, pre_tokens).to(torch.bool)
        return [atten_mask.npu()]

    @staticmethod
    def get_window_atten_mask(bs, size, next_tokens, pre_tokens, tile_h=1, tile_w=1, if_tile=False):
        shape = [bs, size, size]
        atten_mask_u = torch.triu(torch.ones(shape, dtype=torch.bool), diagonal=next_tokens + 1)
        atten_mask_l = torch.tril(torch.ones(shape, dtype=torch.bool), diagonal=-pre_tokens - 1)
        atten_masks = atten_mask_u + atten_mask_l
        if if_tile:
            atten_masks = torch.tile(atten_masks, (tile_h, tile_w))
        return atten_masks

    def get_rainfusion_fa(self, bandwidth, head_dim, text_len):
        import torch_atb

        rfa_param_local = torch_atb.RazorFusionAttentionParam()
        rfa_param_local.head_num = 1
        rfa_param_local.kv_head_num = 1
        rfa_param_local.qk_scale = 1 / math.sqrt(head_dim)
        rfa_param_local.razor_len = self.num_tokens_except_first_frame
        rfa_param_local.pre_tokens = int((bandwidth * self.num_tokens_per_frame) // 128 * (self.num_frames - 1) * 128)
        rfa_param_local.next_tokens = rfa_param_local.pre_tokens
        rfa_param_local.tile_q = 1
        rfa_param_local.tile_kv = 1
        rfa_param_local.text_q_len = text_len
        rfa_param_local.text_kv_len = text_len
        local_attention = torch_atb.Operation(rfa_param_local)

        rfa_param_global = torch_atb.RazorFusionAttentionParam()
        rfa_param_global.head_num = 1
        rfa_param_global.kv_head_num = 1
        rfa_param_global.qk_scale = 1 / math.sqrt(head_dim)
        rfa_param_global.razor_len = self.num_tokens_except_first_frame
        rfa_param_global.pre_tokens = int((bandwidth * rfa_param_global.razor_len) // 128 * 128)
        rfa_param_global.next_tokens = rfa_param_global.pre_tokens
        rfa_param_global.tile_q = 1
        rfa_param_global.tile_kv = 1
        rfa_param_global.text_q_len = text_len
        rfa_param_global.text_kv_len = text_len
        global_attention = torch_atb.Operation(rfa_param_global)

        return local_attention, global_attention

    def cal_mask_recall(self, query, key, head_num, head_dim, atten_mask):
        scale = 1 / math.sqrt(head_dim)
        attn_weight = torch.matmul(query, key.transpose(-2, -1)).mul(scale)
        attn_weight = torch.softmax(attn_weight, dim=-1)
        atten_mask_window_small = atten_mask
        small_attn_mask = atten_mask_window_small.repeat((1, head_num, 1, 1))
        attn_weight_masked = attn_weight * small_attn_mask
        ratio = 1 - torch.sum(attn_weight_masked, dim=(0, 2, 3)) / torch.sum(attn_weight, dim=(0, 2, 3))
        return ratio

    def get_recall_ratio(self, query_layer, key_layer, atten_mask, seq_len, head_num, head_dim, global_step):
        ratio_local = self.cal_mask_recall(
            query_layer[:, :, :seq_len, :], key_layer[:, :, :seq_len, :], head_num, head_dim, atten_mask
        )
        ratio_global = self.cal_mask_recall(
            query_layer[:, :, : self.num_tokens_except_first_frame, :][:, :, ::global_step, :],
            key_layer[:, :, : self.num_tokens_except_first_frame, :][:, :, ::global_step, :],
            head_num,
            head_dim,
            atten_mask,
        )
        return ratio_local, ratio_global

    def rearrange_to_global(self, query_in, key_in, value_in, text_len):
        query_in_img = query_in[:, :-text_len, :, :]
        query_in_txt = query_in[:, -text_len:, :, :]
        query_in_img = rearrange(
            query_in_img,
            "b (t h1 w) n h -> b (h1 w t) n h",
            t=self.grid_size[0] - 1,
            h1=self.grid_size[1],
            w=self.grid_size[2],
        )
        query_in = torch.cat([query_in_img, query_in_txt], 1)

        key_in_img = key_in[:, :-text_len, :, :]
        key_in_txt = key_in[:, -text_len:, :, :]
        key_in_img = rearrange(
            key_in_img,
            "b (t h1 w) n h -> b (h1 w t) n h",
            t=self.grid_size[0] - 1,
            h1=self.grid_size[1],
            w=self.grid_size[2],
        )
        key_in = torch.cat([key_in_img, key_in_txt], 1)

        value_in_img = value_in[:, :-text_len, :, :]
        value_in_txt = value_in[:, -text_len:, :, :]
        value_in_img = rearrange(
            value_in_img,
            "b (t h1 w) n h -> b (h1 w t) n h",
            t=self.grid_size[0] - 1,
            h1=self.grid_size[1],
            w=self.grid_size[2],
        )
        value_in = torch.cat([value_in_img, value_in_txt], 1)

        return query_in, key_in, value_in

    def check_input(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        atten_mask_all: Tensor,
        text_len: int = 0,
        t_idx: int = 0,
    ):
        if not isinstance(query, torch.Tensor):
            raise ParametersInvalid(f"The data type of input query must be torch.Tensor, but got {type(query)}.")
        if not isinstance(key, torch.Tensor):
            raise ParametersInvalid(f"The data type of input key must be torch.Tensor, but got {type(key)}.")
        if not isinstance(value, torch.Tensor):
            raise ParametersInvalid(f"The data type of input value must be torch.Tensor, but got {type(value)}.")
        if not isinstance(atten_mask_all, list):
            raise ParametersInvalid(
                f"The data type of input atten_mask_all must be list, but got {type(atten_mask_all)}."
            )
        if text_len < 0:
            raise ParametersInvalid(f"The data type of input text_len must >= 0 , but got {text_len}.")
        if t_idx < 0:
            raise ParametersInvalid(f"The data type of input t_idx must >= 0 , but got {t_idx}.")

    def forward_rainfusion_attention(self, ratio_local, ratio_global, query_in, key_in, value_in, text_len):
        use_local = False
        if ratio_local > WIN_RATIO or ratio_local > ratio_global:
            atten_func = self.local_attention
            use_local = True
        else:
            atten_func = self.global_attention

        if self.layout_transform and use_local:
            query_in, key_in, value_in = self.rearrange_to_global(query_in, key_in, value_in, text_len)

        query_in = rearrange(query_in, "b s n h -> (b s n) h")
        key_in = rearrange(key_in, "b s n h -> (b s n) h")
        value_in = rearrange(value_in, "b s n h -> (b s n) h")
        torch.npu.synchronize()
        out = atten_func.forward([query_in, key_in, value_in])[0]
        torch.npu.synchronize()
        out = rearrange(out, "(b s n) h -> b s n h", b=1, n=1)
        if self.layout_transform and use_local:
            out_img = out[:, :-text_len, :, :]
            out_txt = out[:, -text_len:, :, :]
            out_img = rearrange(
                out_img,
                "b (h1 w t) n h -> b (t h1 w) n h",
                t=self.grid_size[0] - 1,
                h1=self.grid_size[1],
                w=self.grid_size[2],
            )
            out = torch.cat([out_img, out_txt], 1)
        return out

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        atten_mask_all: Tensor,
        text_len: int = 0,
        t_idx: int = 0,
    ) -> Tensor:
        """
        参数:
            query (Tensor): 输入query, layout应为BSND。
            key (Tensor): 输入key, layout应为BSND。
            value (Tensor): 输入value, layout应为BSND。
            atten_mask_all (Tensor): 稀疏mask。
            text_len (int, optional): qkv的seqlen中的文本长度, 默认为 0。
            t_idx (int, optional): 当前step值, 默认为 0。

        返回:
            Tensor: 计算结果。
        """
        self.check_input(query, key, value, atten_mask_all, text_len, t_idx)

        b, s, n, d = query.shape
        for_loop = n
        grid_seqlen = self.num_frames * self.num_tokens_per_frame + text_len
        if s > grid_seqlen:
            query = query[:, :grid_seqlen, :, :]
            key = key[:, :grid_seqlen, :, :]
            value = value[:, :grid_seqlen, :, :]

        query_layer = torch.cat(
            [query[:, self.num_tokens_per_frame :, :, :], query[:, : self.num_tokens_per_frame, :, :]], dim=1
        )
        key_layer = torch.cat(
            [key[:, self.num_tokens_per_frame :, :, :], key[:, : self.num_tokens_per_frame, :, :]], dim=1
        )
        value_layer = torch.cat(
            [value[:, self.num_tokens_per_frame :, :, :], value[:, : self.num_tokens_per_frame, :, :]], dim=1
        )
        if self.use_rainfusion and t_idx > self.skip_timesteps:
            atten_mask = atten_mask_all[0]
            global_step = self.num_frames - 1
            seq_len = self.num_tokens_per_frame
            text_len = text_len + self.num_tokens_per_frame

            self.local_attention, self.global_attention = self.get_rainfusion_fa(
                bandwidth=self.bandwidth, head_dim=d, text_len=text_len
            )
            ratio_list, ratio_list_2 = self.get_recall_ratio(
                query_layer.transpose(1, 2),
                key_layer.transpose(1, 2),
                atten_mask,
                seq_len=seq_len,
                head_num=1,
                head_dim=d,
                global_step=global_step,
            )

        if self.use_la or (self.use_rainfusion and t_idx > self.skip_timesteps):
            query_layer_list = query_layer.split(1, dim=2)
            key_layer_list = key_layer.split(1, dim=2)
            value_layer_list = value_layer.split(1, dim=2)
        else:
            query_layer_list = query_layer.transpose(1, 2).split(1, dim=1)
            key_layer_list = key_layer.transpose(1, 2).split(1, dim=1)
            value_layer_list = value_layer.transpose(1, 2).split(1, dim=1)

        output = []
        for i in range(for_loop):
            if self.use_rainfusion and t_idx > self.skip_timesteps:
                ratio_local, ratio_global = ratio_list[i], ratio_list_2[i]
                out = self.forward_rainfusion_attention(
                    ratio_local, ratio_global, query_layer_list[i], key_layer_list[i], value_layer_list[i], text_len
                )
            else:
                if self.use_la:
                    out = attention_forward(
                        query_layer_list[i],
                        key_layer_list[i],
                        value_layer_list[i],
                        opt_mode="manual",
                        op_type="ascend_laser_attention",
                        layout="BNSD",
                    )
                else:
                    seqlen = torch.tensor([[query_layer.shape[1]], [key_layer.shape[1]]], dtype=torch.int32)
                    intensors = [query_layer_list[i], key_layer_list[i], value_layer_list[i], seqlen]
                    torch.npu.synchronize()
                    out = self.self_attention.forward(intensors)[0].transpose(1, 2)
                    torch.npu.synchronize()
            output.append(out)
        out = torch.cat(output, dim=2)
        out = torch.cat([out[:, -self.num_tokens_per_frame :, :, :], out[:, : -self.num_tokens_per_frame, :, :]], dim=1)
        if s > grid_seqlen:
            out = torch.cat([out, out.new_zeros(b, s - grid_seqlen, n, d)], dim=1)
        return out
