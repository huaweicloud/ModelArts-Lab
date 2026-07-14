"""Cloud-specific platform patches.

These are imported when adapt_patch(is_global_patch=True) is called.
Each patch module applies its monkey-patches at import time.

Add new platform patches here by importing the patch module:

    from ascend_vllm.patch.platform import patch_<name>  # noqa: F401
"""

# ruff: noqa: I001
# Env var registration must run first so later patches can read
# VLLM_ASCEND_DISABLE_CLOUD_OPS_TURBO via vllm_ascend.envs.
from ascend_vllm.patch.platform import patch_envs as patch_envs  # noqa: F401

# patch_chunk_fla computes chunk_offsets_idx on-the-fly from cu_seqlens
# (fallback), so patch_gdn_attn_builder / patch_gdn_chunk_meta are no longer
# needed — the upstream gdn_attn_builder.py was refactored to drop the old
# internal helpers those patches relied on.
from ascend_vllm.patch.platform import patch_layernorm as patch_layernorm  # noqa: F401
from ascend_vllm.patch.platform import patch_chunk_fla as patch_chunk_fla  # noqa: F401
from ascend_vllm.patch.platform import patch_health as patch_health
from ascend_vllm.patch.platform import (
    patch_disable_completion_tokens_details as patch_disable_completion_tokens_details,
)
