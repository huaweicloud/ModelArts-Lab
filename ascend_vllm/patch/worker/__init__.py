# ruff: noqa: I001
from vllm_ascend.utils import vllm_version_is

if vllm_version_is("0.21.0"):
    from ascend_vllm.patch.worker import (
        patch_mooncake_hybrid_connector as patch_mooncake_hybrid_connector,
    )
