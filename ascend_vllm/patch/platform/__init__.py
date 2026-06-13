"""Cloud-specific platform patches.

These are imported when adapt_patch(is_global_patch=True) is called.
Each patch module applies its monkey-patches at import time.

Add new platform patches here by importing the patch module:

    from ascend_vllm.patch.platform import patch_<name>  # noqa: F401
"""
