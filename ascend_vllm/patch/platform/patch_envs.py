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

"""Patch vllm_ascend.envs to register VLLM_ASCEND_DISABLE_CLOUD_OPS_TURBO.

This env var toggles cloud_ops_turbo custom AscendC operators
(cloud_chunk_scaled_dot_kkt / cloud_solve_tril / cloud_recompute_wu /
cloud_rmsnorm_silu). When set to 1 the operators fall back to the legacy
Triton implementations for debugging.
"""

import os

import vllm_ascend.envs as _envs

_ENV_VAR = "VLLM_ASCEND_DISABLE_CLOUD_OPS_TURBO"

if _ENV_VAR not in _envs.env_variables:
    _envs.env_variables[_ENV_VAR] = lambda: bool(int(os.getenv(_ENV_VAR, "0")))
