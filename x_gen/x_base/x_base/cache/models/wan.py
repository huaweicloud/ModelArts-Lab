"""
Wan 模型 Cache 加速实现

包含 TeaCache 和 MagCache 两种加速策略的前向传播实现。
"""

import os
from typing import Any

import numpy as np
import torch
from diffusers.models.modeling_outputs import Transformer2DModelOutput

from ..base import load_cache_config, nearest_interp
from ..utils import wan_post_forward, wan_pre_forward

# 环境变量配置
RESIDUAL_DIFF_THRESHOLD = float(os.getenv("RESIDUAL_DIFF_THRESHOLD", 0))
USE_RET_STEPS = int(os.getenv("USE_RET_STEPS", 1))
TURBO_THRESH_T2V = float(os.getenv("TURBO_THRESH_T2V", 0.1))
TURBO_THRESH_I2V = float(os.getenv("TURBO_THRESH_I2V", 0.18))
MAGCACHE_THRESH = float(os.getenv("MAGCACHE_THRESH", 0.24))
MAGCACHE_K = int(os.getenv("MAGCACHE_K", 6))
RETENTION_RATIO = float(os.getenv("RETENTION_RATIO", 0.2))


# ============ TeaCache 前向传播 ============
def teacache_wan_forward(
    self,
    hidden_states: torch.Tensor,
    timestep: torch.LongTensor,
    encoder_hidden_states: torch.Tensor,
    encoder_hidden_states_image: torch.Tensor | None = None,
    return_dict: bool = True,
    attention_kwargs: dict[str, Any] | None = None,
) -> torch.Tensor | dict[str, torch.Tensor]:
    """Wan TeaCache 前向传播"""
    # 前置处理
    (
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
    ) = wan_pre_forward(
        self, hidden_states, timestep, encoder_hidden_states, encoder_hidden_states_image, attention_kwargs
    )

    ori_x = hidden_states

    if self.enable_teacache:
        modulated_inp = timestep_proj if self.use_ref_steps else temb
        idx = self.cnt % 2

        # 判断是否跳过
        should_skip = _teacache_should_skip(self, idx, modulated_inp)

        if should_skip:
            hidden_states = hidden_states + _get_residual(self, idx)
        else:
            hidden_states = self.forward_block(hidden_states, encoder_hidden_states, timestep_proj, rotary_emb)
            _set_residual(self, idx, hidden_states - ori_x)
    else:
        hidden_states = self.forward_block(hidden_states, encoder_hidden_states, timestep_proj, rotary_emb)

    # 后置处理
    output = wan_post_forward(
        self,
        hidden_states,
        temb,
        batch_size,
        post_patch_num_frames,
        post_patch_height,
        post_patch_width,
        p_t,
        p_h,
        p_w,
        lora_scale,
    )

    _update_step_counter(self)
    return Transformer2DModelOutput(sample=output) if return_dict else (output,)


def teacache_wan_vace_forward(
    self,
    hidden_states: torch.Tensor,
    timestep: torch.LongTensor,
    encoder_hidden_states: torch.Tensor,
    encoder_hidden_states_image: torch.Tensor | None = None,
    control_hidden_states: torch.Tensor = None,
    control_hidden_states_scale: torch.Tensor = None,
    return_dict: bool = True,
    attention_kwargs: dict[str, Any] | None = None,
) -> torch.Tensor | dict[str, torch.Tensor]:
    """Wan VACE TeaCache 前向传播（视频编辑任务）"""
    (
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
    ) = wan_pre_forward(
        self, hidden_states, timestep, encoder_hidden_states, encoder_hidden_states_image, attention_kwargs
    )

    # VACE 控制状态处理
    control_hidden_states, control_hidden_states_scale = _vace_preprocess_control(
        self, control_hidden_states, control_hidden_states_scale, batch_size, hidden_states
    )

    ori_x = hidden_states

    if self.enable_teacache:
        modulated_inp = timestep_proj if self.use_ref_steps else temb
        idx = self.cnt % 2
        should_skip = _teacache_should_skip(self, idx, modulated_inp)

        if should_skip:
            hidden_states = hidden_states + _get_residual(self, idx)
        else:
            hidden_states = _vace_forward_step(
                self,
                hidden_states,
                encoder_hidden_states,
                timestep_proj,
                control_hidden_states,
                rotary_emb,
                control_hidden_states_scale,
            )
            _set_residual(self, idx, hidden_states - ori_x)
    else:
        hidden_states = _vace_forward_step(
            self,
            hidden_states,
            encoder_hidden_states,
            timestep_proj,
            control_hidden_states,
            rotary_emb,
            control_hidden_states_scale,
        )

    output = wan_post_forward(
        self,
        hidden_states,
        temb,
        batch_size,
        post_patch_num_frames,
        post_patch_height,
        post_patch_width,
        p_t,
        p_h,
        p_w,
        lora_scale,
    )

    _update_step_counter(self)
    return Transformer2DModelOutput(sample=output) if return_dict else (output,)


# ============ MagCache 前向传播 ============
def magcache_wan_forward(
    self,
    hidden_states: torch.Tensor,
    timestep: torch.LongTensor,
    encoder_hidden_states: torch.Tensor,
    encoder_hidden_states_image: torch.Tensor | None = None,
    return_dict: bool = True,
    attention_kwargs: dict[str, Any] | None = None,
) -> torch.Tensor | dict[str, torch.Tensor]:
    """Wan MagCache 前向传播"""
    (
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
    ) = wan_pre_forward(
        self, hidden_states, timestep, encoder_hidden_states, encoder_hidden_states_image, attention_kwargs
    )

    ori_x = hidden_states
    residual_x = None

    if getattr(self, "wan2", False):
        skip_forward, residual_x = _magcache_wan22_logic(self, ori_x)
    else:
        skip_forward, residual_x = _magcache_wan21_logic(self, ori_x)

    if skip_forward:
        hidden_states = ori_x + residual_x
    else:
        hidden_states = self.forward_block(hidden_states, encoder_hidden_states, timestep_proj, rotary_emb)
        residual_x = hidden_states - ori_x

    _update_residual_cache(self, residual_x)

    output = wan_post_forward(
        self,
        hidden_states,
        temb,
        batch_size,
        post_patch_num_frames,
        post_patch_height,
        post_patch_width,
        p_t,
        p_h,
        p_w,
        lora_scale,
    )

    _update_step_counter_magcache(self)
    return Transformer2DModelOutput(sample=output) if return_dict else (output,)


def magcache_wan_calibration(
    self,
    hidden_states: torch.Tensor,
    timestep: torch.LongTensor,
    encoder_hidden_states: torch.Tensor,
    encoder_hidden_states_image: torch.Tensor | None = None,
    return_dict: bool = True,
    attention_kwargs: dict[str, Any] | None = None,
) -> torch.Tensor | dict[str, torch.Tensor]:
    """MagCache 校准模式（生成 mag_ratios）"""
    import torch.distributed as dist

    (
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
    ) = wan_pre_forward(
        self, hidden_states, timestep, encoder_hidden_states, encoder_hidden_states_image, attention_kwargs
    )

    ori_x = hidden_states
    hidden_states = self.forward_block(hidden_states, encoder_hidden_states, timestep_proj, rotary_emb)
    residual_x = hidden_states - ori_x

    _calibration_update(self, residual_x)

    output = wan_post_forward(
        self,
        hidden_states,
        temb,
        batch_size,
        post_patch_num_frames,
        post_patch_height,
        post_patch_width,
        p_t,
        p_h,
        p_w,
        lora_scale,
    )

    self.cnt += 1
    if self.cnt >= self.num_steps and dist.get_rank() == 0:
        print(f"wan2_1_mag_ratio::{self.norm_ratio}")

    return Transformer2DModelOutput(sample=output) if return_dict else (output,)


# ============ 内部辅助函数 ============
def _teacache_should_skip(self, idx, modulated_inp):
    """判断 TeaCache 是否跳过"""
    if self.cnt < self.ret_steps or self.cnt >= self.cutoff_steps:
        _set_accumulated_distance(self, idx, 0)
        return False

    prev = _get_previous_input(self, idx)
    if prev is None:
        return False

    rel_diff = ((modulated_inp - prev).abs().mean() / prev.abs().mean()).cpu().item()
    rescale_func = np.poly1d(self.coefficients)
    accumulated = _get_accumulated_distance(self, idx) + rescale_func(rel_diff)
    _set_accumulated_distance(self, idx, accumulated)

    should_calc = accumulated >= self.teacache_thresh
    if should_calc:
        _set_accumulated_distance(self, idx, 0)

    _set_previous_input(self, idx, modulated_inp.clone())
    return not should_calc


def _vace_preprocess_control(self, control_states, control_scale, batch_size, hidden_states):
    """VACE 控制状态预处理"""
    if control_scale is None:
        control_scale = control_states.new_ones(len(self.config.vace_layers))
    control_scale = torch.unbind(control_scale)
    if len(control_scale) != len(self.config.vace_layers):
        raise ValueError(
            f"Length of `control_hidden_states_scale` {len(control_scale)} should be "
            f"equal to {len(self.config.vace_layers)}."
        )

    control_states = self.vace_patch_embedding(control_states)
    control_states = control_states.flatten(2).transpose(1, 2)
    padding = control_states.new_zeros(
        batch_size, hidden_states.size(1) - control_states.size(1), control_states.size(2)
    )
    control_states = torch.cat([control_states, padding], dim=1)

    return control_states, control_scale


def _vace_forward_step(
    self, hidden_states, encoder_hidden_states, timestep_proj, control_hidden_states, rotary_emb, control_scale
):
    """VACE 单步前向"""
    control_list = []
    for i, block in enumerate(self.vace_blocks):
        cond, control_hidden_states = block(
            hidden_states, encoder_hidden_states, control_hidden_states, timestep_proj, rotary_emb
        )
        control_list.append((cond, control_scale[i]))
    control_list = control_list[::-1]

    for i, block in enumerate(self.blocks):
        hidden_states, encoder_hidden_states = block(hidden_states, encoder_hidden_states, timestep_proj, rotary_emb)
        if i in self.config.vace_layers:
            hint, scale = control_list.pop()
            hidden_states = hidden_states + hint * scale

    return hidden_states


def _magcache_wan21_logic(self, ori_x):
    """Wan2.1 MagCache 逻辑"""
    skip_forward = False
    residual_x = None

    if self.cnt >= int(self.num_steps * self.retention_ratio):
        cur_mag_ratio = self.mag_ratios[self.cnt]
        _update_cache(self, cur_mag_ratio)

        if _check_skip_conditions(self):
            skip_forward = True
            residual_x = self.residual_cache[self.cnt % 2]
        else:
            _reset_cache(self)

    return skip_forward, residual_x


def _magcache_wan22_logic(self, ori_x):
    """Wan2.2 MagCache 逻辑"""
    skip_forward = False
    residual_x = None

    use_magcache = _compute_use_magcache_wan22(self)

    if use_magcache:
        cur_mag_ratio = self.mag_ratios[self.cnt]
        if self.is_distill:
            _update_cache_distill(self, cur_mag_ratio)
        else:
            _update_cache(self, cur_mag_ratio)

        if _check_skip_conditions(self):
            skip_forward = True
            residual_x = _get_cached_residual(self)
        else:
            _reset_cache(self)

    return skip_forward, residual_x


def _compute_use_magcache_wan22(self):
    if self.split_step is not None:
        if self.mode == "i2v":
            return self.cnt >= int(self.split_step + (self.num_steps - self.split_step) * self.retention_ratio)
        else:
            r = self.retention_ratio
            if self.cnt < int(self.split_step * r):
                return False
            if self.split_step <= self.cnt <= int((self.num_steps - self.split_step) * r + self.split_step):  # noqa: SIM103
                return False
            return True
    return self.cnt < int(self.num_steps * self.retention_ratio)


def _calibration_update(self, residual_x):
    """校准模式更新"""
    if self.is_distill:
        if self.cnt >= 1:
            ratio = (residual_x.norm(dim=-1) / self.residual_cache[0].norm(dim=-1)).mean().item()
            std = (residual_x.norm(dim=-1) / self.residual_cache[0].norm(dim=-1)).std().item()
            cos = (
                (1 - torch.nn.functional.cosine_similarity(residual_x, self.residual_cache[0], dim=-1, eps=1e-8))
                .mean()
                .item()
            )
            self.norm_ratio.append(round(ratio, 5))
            self.norm_std.append(round(std, 5))
            self.cos_dis.append(round(cos, 5))
        self.residual_cache[0] = residual_x
    else:
        if self.cnt >= 2:
            ratio = (residual_x.norm(dim=-1) / self.residual_cache[self.cnt % 2].norm(dim=-1)).mean().item()
            std = (residual_x.norm(dim=-1) / self.residual_cache[self.cnt % 2].norm(dim=-1)).std().item()
            cos = (
                (
                    1
                    - torch.nn.functional.cosine_similarity(
                        residual_x, self.residual_cache[self.cnt % 2], dim=-1, eps=1e-8
                    )
                )
                .mean()
                .item()
            )
            self.norm_ratio.append(round(ratio, 5))
            self.norm_std.append(round(std, 5))
            self.cos_dis.append(round(cos, 5))
        self.residual_cache[self.cnt % 2] = residual_x


# ============ 状态访问器 ============
def _get_accumulated_distance(self, idx):
    return self.accumulated_rel_l1_distance_even if idx == 0 else self.accumulated_rel_l1_distance_odd


def _set_accumulated_distance(self, idx, val):
    if idx == 0:
        self.accumulated_rel_l1_distance_even = val
    else:
        self.accumulated_rel_l1_distance_odd = val


def _get_previous_input(self, idx):
    return self.previous_e0_even if idx == 0 else self.previous_e0_odd


def _set_previous_input(self, idx, val):
    if idx == 0:
        self.previous_e0_even = val
    else:
        self.previous_e0_odd = val


def _get_residual(self, idx):
    return self.previous_residual_even if idx == 0 else self.previous_residual_odd


def _set_residual(self, idx, val):
    if idx == 0:
        self.previous_residual_even = val
    else:
        self.previous_residual_odd = val


def _update_step_counter(self):
    self.cnt += 1
    if self.cnt >= self.num_steps:
        self.cnt = 0


def _update_step_counter_magcache(self):
    self.cnt.add_(1)
    if self.cnt >= self.num_steps:
        self.cnt.zero_()
        n = 1 if self.is_distill else 2
        self.accumulated_ratio = [1.0] * n
        self.accumulated_err = [0.0] * n
        self.accumulated_steps = [0] * n


def _update_cache(self, ratio):
    idx = self.cnt % 2
    self.accumulated_ratio[idx] *= ratio
    self.accumulated_steps[idx] += 1
    self.accumulated_err[idx] += abs(1 - self.accumulated_ratio[idx])


def _update_cache_distill(self, ratio):
    self.accumulated_ratio[0] *= ratio
    self.accumulated_steps[0] += 1
    self.accumulated_err[0] += abs(1 - self.accumulated_ratio[0])


def _check_skip_conditions(self):
    if self.is_distill:
        return self.accumulated_err[0] < self.magcache_thresh and self.accumulated_steps[0] <= self.K
    idx = self.cnt % 2
    return self.accumulated_err[idx] < self.magcache_thresh and self.accumulated_steps[idx] <= self.K


def _get_cached_residual(self):
    return self.residual_cache[0] if self.is_distill else self.residual_cache[self.cnt % 2]


def _reset_cache(self):
    if self.is_distill:
        self.accumulated_err[0] = 0
        self.accumulated_steps[0] = 0
        self.accumulated_ratio[0] = 1.0
    else:
        idx = self.cnt % 2
        self.accumulated_err[idx] = 0
        self.accumulated_steps[idx] = 0
        self.accumulated_ratio[idx] = 1.0


def _update_residual_cache(self, residual):
    if self.is_distill:
        self.residual_cache[0] = residual
    else:
        self.residual_cache[self.cnt % 2] = residual


# ============ 初始化函数 ============
def teacache_init(pipe, args):
    """初始化 Wan TeaCache"""
    is_vace = "VACE" in args.model
    is_i2v = args.task_type == "i2v"

    pipe.transformer.__class__.enable_teacache = True
    pipe.transformer.__class__.forward = teacache_wan_vace_forward if is_vace else teacache_wan_forward
    pipe.transformer.__class__.cnt = 0
    pipe.transformer.__class__.num_steps = args.num_inference_steps * 2
    pipe.transformer.__class__.teacache_thresh = TURBO_THRESH_I2V if is_i2v else TURBO_THRESH_T2V

    _init_teacache_state(pipe.transformer.__class__, args, is_i2v)


def _init_teacache_state(cls, args, is_i2v):
    """初始化 TeaCache 状态"""
    from ..base import get_teacache_config

    cls.accumulated_rel_l1_distance_even = 0
    cls.accumulated_rel_l1_distance_odd = 0
    cls.previous_e0_even = None
    cls.previous_e0_odd = None
    cls.previous_residual_even = None
    cls.previous_residual_odd = None
    cls.use_ref_steps = USE_RET_STEPS

    # 推断模型配置键
    model_key = _infer_wan_model_key(args, is_i2v)

    # 从配置获取 coefficients
    teacache_cfg = get_teacache_config("wan", model_key, USE_RET_STEPS)
    cls.coefficients = teacache_cfg.get("coefficients", [])

    # 设置 ret_steps 和 cutoff_steps
    if USE_RET_STEPS:
        cls.ret_steps = teacache_cfg.get("ret_steps", 10)
        cls.cutoff_steps = args.num_inference_steps * 2
    else:
        cls.ret_steps = teacache_cfg.get("ret_steps", 2)
        cls.cutoff_steps = args.num_inference_steps * 2 - 2


def _infer_wan_model_key(args, is_i2v: bool) -> str:
    """从 args 推断 Wan 模型配置键

    Args:
        args: 包含 model 和 pretrained_model_name_or_path 属性的参数对象
        is_i2v: 是否为 I2V 模式

    Returns:
        配置文件中的模型键，如 "2.1-T2V-1.3B", "2.1-I2V-480P"
    """
    if "1.3B" in args.model:
        return "2.1-T2V-1.3B"
    elif "14B" in args.model:
        if is_i2v:
            if "480" in args.pretrained_model_name_or_path:
                return "2.1-I2V-480P"
            else:
                return "2.1-I2V-720P"
        else:
            return "2.1-T2V-14B"
    # 默认返回 1.3B 配置
    return "2.1-T2V-1.3B"


def magcache_calibration(pipe, args):
    """初始化 MagCache 校准模式"""
    from copy import deepcopy

    pipe.transformer.__class__.forward = magcache_wan_calibration
    pipe.transformer.__class__.cnt = torch.tensor(0)
    pipe.transformer.__class__.num_steps = args.num_inference_steps * 2
    pipe.transformer.__class__.is_distill = False
    pipe.transformer.__class__.residual_cache = deepcopy([None, None])
    if args.x:
        pipe.transformer.__class__.num_steps = args.num_inference_steps
        pipe.transformer.__class__.is_distill = True
        pipe.transformer.__class__.residual_cache = deepcopy([None])
    pipe.transformer.__class__.norm_ratio = deepcopy([])
    pipe.transformer.__class__.norm_std = deepcopy([])
    pipe.transformer.__class__.cos_dis = deepcopy([])


def magcache_init(pipe, args):
    """初始化 Wan MagCache"""
    config = load_cache_config()
    mag_ratios = _get_mag_ratios(args, config)

    pipe.transformer.__class__.forward = magcache_wan_forward
    pipe.transformer.__class__.cnt = torch.tensor(0)
    pipe.transformer.__class__.num_steps = args.num_inference_steps * 2
    pipe.transformer.__class__.magcache_thresh = MAGCACHE_THRESH
    pipe.transformer.__class__.K = MAGCACHE_K
    pipe.transformer.__class__.retention_ratio = RETENTION_RATIO
    pipe.transformer.__class__.accumulated_err = [0.0, 0.0]
    pipe.transformer.__class__.accumulated_steps = [0, 0]
    pipe.transformer.__class__.accumulated_ratio = [1.0, 1.0]
    pipe.transformer.__class__.residual_cache = [None, None]

    data_temp = np.concatenate(([1.0, 1.0], mag_ratios))
    pipe.transformer.__class__.mag_ratios = np.array(data_temp)

    if "Wan2.1" in args.model:
        _magcache_init_wan21(pipe, args)
    elif "Wan2.2" in args.model:
        _magcache_init_wan22(pipe, args, mag_ratios)
    else:
        raise ValueError(f"Unsupported magcache of wan model type: '{args.model}'.")


def _get_mag_ratios(args, config):
    """获取 mag_ratios"""
    mag_ratios = config.get("mag_ratios", {})

    if "Wan2.1-T2V-1.3B" in args.model:
        return mag_ratios.get("wan2.1-t2v-1.3b", [])
    if "Wan2.1-T2V-14B" in args.model:
        return mag_ratios.get("wan2.1-t2v-14b", [])
    if "Wan2.1-I2V-14B" in args.model:
        if "480" in args.pretrained_model_name_or_path:
            return mag_ratios.get("wan2.1-i2v-480p", [])
        return mag_ratios.get("wan2.1-i2v-720p", [])
    if "Wan2.2-T2V-A14B" in args.model:
        return mag_ratios.get("wan2.2-t2v-x" if args.x else "wan2.2-t2v-A14B", [])
    if "Wan2.2-I2V-A14B" in args.model:
        return mag_ratios.get("wan2.2-i2v-x" if args.x else "wan2.2-i2v-A14B", [])
    return []


def _magcache_init_wan21(pipe, args):
    pipe.transformer.__class__.wan2 = False
    pipe.transformer.__class__.is_distill = False

    if args.task_type == "t2v" and "T2V-1.3B" in args.pretrained_model_name_or_path:
        pipe.transformer.__class__.magcache_thresh = 0.12
        pipe.transformer.__class__.K = 4
    elif args.task_type == "i2v" and "I2V-14B" in args.pretrained_model_name_or_path:
        pipe.transformer.__class__.magcache_thresh = 0.05
        pipe.transformer.__class__.K = MAGCACHE_K
        pipe.transformer.__class__.retention_ratio = 0.1

    # 插值 mag_ratios
    if len(pipe.transformer.__class__.mag_ratios) != args.num_inference_steps * 2:
        mag_ratio_con = nearest_interp(pipe.transformer.__class__.mag_ratios[0::2], args.num_inference_steps)
        mag_ratio_ucon = nearest_interp(pipe.transformer.__class__.mag_ratios[1::2], args.num_inference_steps)
        interpolated = np.concatenate([mag_ratio_con.reshape(-1, 1), mag_ratio_ucon.reshape(-1, 1)], axis=1).reshape(-1)
        pipe.transformer.__class__.mag_ratios = interpolated


def _magcache_init_wan22(pipe, args, mag_ratios):
    pipe.transformer.__class__.is_distill = args.x
    if args.x:
        data_temp = np.concatenate(([1.0], mag_ratios))
        pipe.transformer.__class__.mag_ratios = np.array(data_temp)
    pipe.transformer.__class__.wan2 = True
    pipe.transformer.__class__.mode = args.task_type

    boundary = 0.900 if args.task_type == "i2v" else 0.875
    pipe.scheduler.set_timesteps(args.num_inference_steps, device=torch.device(f"npu:{torch.cuda.current_device()}"))
    high_noise_steps = (pipe.scheduler.timesteps >= (pipe.scheduler.config.num_train_timesteps * boundary)).sum().item()
    pipe.transformer.__class__.split_step = high_noise_steps * 2

    if args.x:
        pipe.transformer.__class__.split_step = high_noise_steps
        pipe.transformer.__class__.num_steps = args.num_inference_steps
        pipe.transformer.__class__.accumulated_err = [0.0]
        pipe.transformer.__class__.accumulated_steps = [0]
        pipe.transformer.__class__.accumulated_ratio = [1.0]
        pipe.transformer.__class__.residual_cache = [None]
        pipe.transformer.__class__.magcache_thresh = 0.1
        pipe.transformer.__class__.K = MAGCACHE_K
        pipe.transformer.__class__.retention_ratio = 0.1
    else:
        pipe.transformer.__class__.magcache_thresh = 0.06
        pipe.transformer.__class__.K = 2
        if "T2V-A14B" in args.pretrained_model_name_or_path:
            pipe.transformer.__class__.retention_ratio = 0.4
        if "I2V-A14B" in args.pretrained_model_name_or_path:
            pipe.transformer.__class__.retention_ratio = 0.1
