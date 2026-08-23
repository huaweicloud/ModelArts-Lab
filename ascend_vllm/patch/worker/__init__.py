# ruff: noqa: I001

from vllm_ascend.utils import (
    vllm_version_is,
    AscendDeviceType,
    get_ascend_device_type,
)
from ascend_vllm.patch.worker import (
    patch_mooncake_hybrid_connector as patch_mooncake_hybrid_connector,
)
from ascend_vllm.patch.worker import (
    patch_dspark_proposer as patch_dspark_proposer,
)
from ascend_vllm.patch.worker import (
    patch_spec_decode_drafter as patch_spec_decode_drafter,
)
from ascend_vllm.patch.worker import (
    patch_spec_decode_utils as patch_spec_decode_utils,
)
from ascend_vllm.patch.worker import (
    patch_model_runner_v1 as patch_model_runner_v1,
)

if get_ascend_device_type() == AscendDeviceType.A5 and vllm_version_is("0.25.1"):
    from ascend_vllm.patch.worker import patch_deepseek_v4_dspark as patch_deepseek_v4_dspark
    from ascend_vllm.patch.worker import patch_fused_moe as patch_fused_moe
