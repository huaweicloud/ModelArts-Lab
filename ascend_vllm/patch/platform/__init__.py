# ruff: noqa: I001
from vllm_ascend.utils import vllm_version_is

from ascend_vllm.patch.platform import patch_health as patch_health

if vllm_version_is("0.21.0"):
    from ascend_vllm.patch.platform import (
        patch_disable_completion_tokens_details as patch_disable_completion_tokens_details,
    )
    from ascend_vllm.patch.platform import (
        patch_recompute_scheduler as patch_recompute_scheduler,
    )
    from ascend_vllm.patch.platform import (
        patch_deepseek_v4_validation as patch_deepseek_v4_validation,
    )
