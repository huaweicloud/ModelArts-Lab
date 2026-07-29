# ruff: noqa: I001
from vllm_ascend.utils import vllm_version_is

from ascend_vllm.patch.platform import patch_health as patch_health
from ascend_vllm.patch.platform import patch_request as patch_request
from ascend_vllm.patch.platform import patch_detokenizer as patch_detokenizer
from ascend_vllm.patch.platform import patch_scheduler as patch_scheduler
from ascend_vllm.patch.platform import patch_recompute_scheduler_023 as patch_recompute_scheduler_023

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
