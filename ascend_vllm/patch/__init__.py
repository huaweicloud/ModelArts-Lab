"""Additional monkey-patches applied on top of vllm-ascend patches.

Patches are applied by ascend_vllm.utils.adapt_patch() after vllm-ascend's own
patches, so ascend_vllm patches take precedence for overlapping targets.
"""
