"""Cloud-specific worker patches.

These are imported when adapt_patch(is_global_patch=False) is called.
Each patch module applies its monkey-patches at import time.

Add new worker patches here by importing the patch module:

    from ascend_vllm.patch.worker import patch_<name>  # noqa: F401
"""
