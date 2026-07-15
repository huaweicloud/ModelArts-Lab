#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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
#

"""Patch vllm_ascend.ops.layernorm.LayerNormFn.forward to use cloud_rmsnorm_silu.

When VLLM_ASCEND_DISABLE_CLOUD_OPS_TURBO is 0 (default), the fused
cloud_ops_turbo.cloud_rmsnorm_silu AscendC operator replaces the Triton
layer_norm_fwd_npu path. Setting the env var to 1 keeps the legacy Triton
implementation for debugging.
"""

import torch
import vllm_ascend.envs as envs
import vllm_ascend.ops.layernorm as _layernorm
from vllm_ascend.ops.triton.layernorm_gated import layer_norm_fwd_npu

# cloud_ops_turbo is imported lazily inside the runtime branch. A module-level
# import triggers device property queries before init_device_properties_triton()
# runs during worker setup, causing "Device properties not initialized" errors.
_USE_CLOUD_OPS = not envs.VLLM_ASCEND_DISABLE_CLOUD_OPS_TURBO

_LayerNormFn = _layernorm.LayerNormFn


def _patched_forward(
    ctx,
    x,
    weight,
    bias,
    z=None,
    eps=1e-6,
    group_size=None,
    norm_before_gate=True,
    is_rms_norm=False,
    activation: str = "swish",
):
    """If z is not None, we do norm(x) * silu(z) if norm_before_gate, else norm(x * silu(z))"""

    x_shape_og = x.shape
    # reshape input data into 2D tensor
    x = x.reshape(-1, x.shape[-1])
    if x.stride(-1) != 1:
        x = x.contiguous()
    if z is not None:
        assert z.shape == x_shape_og
        z = z.reshape(-1, z.shape[-1])
        if z.stride(-1) != 1:
            z = z.contiguous()
    weight = weight.contiguous()
    if bias is not None:
        bias = bias.contiguous()
    if _USE_CLOUD_OPS:
        import cloud_ops_turbo  # noqa: F401

        y = torch.ops.cloud_ops_turbo.cloud_rmsnorm_silu(x, weight, z, eps)
        mean = None
        rstd = None
    else:
        y, mean, rstd = layer_norm_fwd_npu(
            x,
            weight,
            bias,
            eps,
            z=z,
            group_size=group_size,
            norm_before_gate=norm_before_gate,
            is_rms_norm=is_rms_norm,
        )
    ctx.save_for_backward(x, weight, bias, mean, rstd, z)
    ctx.x_shape_og = x_shape_og
    ctx.eps = eps
    ctx.group_size = group_size
    ctx.norm_before_gate = norm_before_gate
    ctx.is_rms_norm = is_rms_norm
    return y.reshape(x_shape_og)


_LayerNormFn.forward = staticmethod(_patched_forward)
