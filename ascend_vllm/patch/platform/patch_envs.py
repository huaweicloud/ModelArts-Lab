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

"""Patch vllm_ascend.envs to register cloud runtime env vars.

- VLLM_ASCEND_DISABLE_CLOUD_OPS_TURBO toggles cloud_ops_turbo custom AscendC
  operators (cloud_chunk_scaled_dot_kkt / cloud_solve_tril / cloud_recompute_wu
  / cloud_rmsnorm_silu). When set to 1 the operators fall back to the legacy
  Triton implementations for debugging.
- LoPT env vars control the lossless overlapping parallel tokenization.
"""

import os
from collections.abc import Callable
from typing import Any

from vllm_ascend import envs

env_variables: dict[str, Callable[[], Any]] = {
    "VLLM_ASCEND_DISABLE_CLOUD_OPS_TURBO": lambda: bool(int(os.getenv("VLLM_ASCEND_DISABLE_CLOUD_OPS_TURBO", "0"))),
    "VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT_COEFFICIENT": lambda: float(
        os.getenv("VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT_COEFFICIENT", 0.95)
    ),
    # Peer-level circuit breaker for decode-side KV pulls: after this many
    # consecutive transfer failures to a P host, its circuit opens for
    # MODELARTS_KV_CIRCUIT_BREAKER_WINDOW_SECONDS and pulls are skipped (the
    # destination blocks are marked invalid and recomputed locally).
    "MODELARTS_KV_CIRCUIT_BREAKER_THRESHOLD": lambda: int(os.getenv("MODELARTS_KV_CIRCUIT_BREAKER_THRESHOLD", 3)),
    "MODELARTS_KV_CIRCUIT_BREAKER_WINDOW_SECONDS": lambda: float(
        os.getenv("MODELARTS_KV_CIRCUIT_BREAKER_WINDOW_SECONDS", 60)
    ),
    # Number of long-lived worker threads used to tokenize overlapping chunks.
    "VLLM_ASCEND_LOPT_THREAD_WORKERS": lambda: int(os.getenv("VLLM_ASCEND_LOPT_THREAD_WORKERS", "4")),
    # Minimum prompt length, in Python Unicode characters, required for LoPT.
    "VLLM_ASCEND_LOPT_MIN_CHARS": lambda: int(os.getenv("VLLM_ASCEND_LOPT_MIN_CHARS", "32768")),
    # Non-overlapping body length of each LoPT text chunk, in characters.
    "VLLM_ASCEND_LOPT_CHUNK_CHARS": lambda: int(os.getenv("VLLM_ASCEND_LOPT_CHUNK_CHARS", "32768")),
    # Character overlap appended to adjacent LoPT chunks.
    "VLLM_ASCEND_LOPT_OVERLAP_CHARS": lambda: int(os.getenv("VLLM_ASCEND_LOPT_OVERLAP_CHARS", "512")),
    # Minimum number of position-identical tokens required to splice chunks.
    "VLLM_ASCEND_LOPT_MIN_MATCH_TOKENS": lambda: int(os.getenv("VLLM_ASCEND_LOPT_MIN_MATCH_TOKENS", "2")),
    # Maximum number of retries that double the LoPT chunk body length.
    "VLLM_ASCEND_LOPT_MAX_RETRIES": lambda: int(os.getenv("VLLM_ASCEND_LOPT_MAX_RETRIES", "3")),
    # Compare LoPT output with standard tokenization before returning it.
    # This is intended for validation and has the cost of tokenizing twice.
    "VLLM_ASCEND_LOPT_VERIFY": lambda: bool(int(os.getenv("VLLM_ASCEND_LOPT_VERIFY", "0"))),
}


def add_dynamic_module_envs():
    for env_name, value in env_variables.items():
        if env_name not in envs.env_variables:
            envs.env_variables[env_name] = value


add_dynamic_module_envs()


def __getattr__(name: str):
    # lazy evaluation of environment variables
    if name in env_variables:
        return env_variables[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(env_variables.keys())
