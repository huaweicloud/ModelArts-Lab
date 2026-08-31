import os
from typing import Any, Callable, Dict

from vllm.logger import logger
from vllm_ascend import envs

env_variables: Dict[str, Callable[[], Any]] = {
    "VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT_COEFFICIENT":
        lambda: float(os.getenv("VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT_COEFFICIENT", 0.95)),
    # Peer-level circuit breaker for decode-side KV pulls: after this many
    # consecutive transfer failures to a P host, its circuit opens for
    # MODELARTS_KV_CIRCUIT_BREAKER_WINDOW_SECONDS and pulls are skipped (the
    # destination blocks are marked invalid and recomputed locally).
    "MODELARTS_KV_CIRCUIT_BREAKER_THRESHOLD":
        lambda: int(os.getenv("MODELARTS_KV_CIRCUIT_BREAKER_THRESHOLD", 3)),
    "MODELARTS_KV_CIRCUIT_BREAKER_WINDOW_SECONDS":
        lambda: float(os.getenv("MODELARTS_KV_CIRCUIT_BREAKER_WINDOW_SECONDS", 60)),
}

def add_dynamic_module_envs():
    for env_name, value in env_variables.items():
        if env_name not in envs.env_variables:
            envs.env_variables[env_name] = value

add_dynamic_module_envs()