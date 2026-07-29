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
from vllm_ascend.utils import vllm_version_is

_GLOBAL_PATCH_APPLIED = False


def _ensure_global_patch():
    """Apply vllm-ascend's process-wide patches once per process."""

    global _GLOBAL_PATCH_APPLIED
    if _GLOBAL_PATCH_APPLIED:
        return

    from ascend_vllm.utils import adapt_patch

    adapt_patch(is_global_patch=True)
    _GLOBAL_PATCH_APPLIED = True


def register():
    """Register the PATCH NPU platform."""
    return "ascend_vllm.platform.PatchNPUPlatform"


def register_connector():
    _ensure_global_patch()

    from vllm_ascend.distributed.kv_transfer import register_connector

    register_connector()


def register_model_loader():
    _ensure_global_patch()

    from vllm_ascend.model_loader.netloader import register_netloader
    from vllm_ascend.model_loader.rfork import register_rforkloader

    register_netloader()
    register_rforkloader()


def register_service_profiling():
    _ensure_global_patch()

    from vllm_ascend.profiling_config import generate_service_profiling_config

    generate_service_profiling_config()


def register_model():
    from vllm_ascend.models import register_model

    register_model()


def register_kv_failure_patch():
    """Load patches for Mooncake KV load failure reporting and recovery."""
    # The KV failure fix spans both scheduler and worker processes:
    # scheduler handles invalid KV blocks, while worker reports Mooncake KV
    # load failures from the decode recv thread. This function is called from a
    # vllm.general_plugins entry point so both sides can be patched in the
    # processes where vLLM loads general plugins.
    from ascend_vllm.patch.platform import patch_recompute_scheduler  # noqa: F401
    from ascend_vllm.patch.worker import patch_mooncake_hybrid_connector  # noqa: F401


def register_general_plugin_patch():
    """Load ModelArts patches that require vLLM general-plugin timing."""
    if vllm_version_is("0.21.0"):
        register_kv_failure_patch()
    from ascend_vllm.patch.platform import patch_scheduler  # noqa: F401
