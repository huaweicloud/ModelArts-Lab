# ruff: noqa: I001

from vllm_ascend.utils import (
    vllm_version_is,
    AscendDeviceType,
    get_ascend_device_type,
)
from ascend_vllm.patch.platform import patch_health as patch_health
from ascend_vllm.patch.platform import patch_request as patch_request
from ascend_vllm.patch.platform import patch_detokenizer as patch_detokenizer
from ascend_vllm.patch.platform import patch_scheduler as patch_scheduler
from ascend_vllm.patch.platform import (
    patch_deepseek_v4_validation as patch_deepseek_v4_validation,
)
from ascend_vllm.patch.platform import (
    patch_recompute_scheduler as patch_recompute_scheduler,
)
from ascend_vllm.patch.platform import (
    patch_generate_base_serving as patch_generate_base_serving,
)
from ascend_vllm.patch.platform import (
    patch_chat_completion_serving as patch_chat_completion_serving,
)
from ascend_vllm.patch.platform import patch_vllm_registry as patch_vllm_registry

if get_ascend_device_type() == AscendDeviceType.A5 and vllm_version_is("0.25.1"):
    from ascend_vllm.patch.platform import patch_vllm_ascend_utils as patch_vllm_ascend_utils
