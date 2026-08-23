from __future__ import annotations

import functools

import torch
import torch.distributed as dist
from vllm.config import CUDAGraphMode
from vllm.distributed.parallel_state import get_dp_group
from vllm_ascend.utils import (
    is_moe_model,
    should_skip_allreduce_across_dp_group,
)

_PATCH_APPLIED = False
_PATCH_MARKER = "_modelarts_moe_dp_metadata_sync_applied"


def apply_patch() -> None:
    """Keep MoE communication and ACLGraph modes consistent across DP ranks."""
    global _PATCH_APPLIED

    if _PATCH_APPLIED:
        return

    from vllm_ascend.worker.model_runner_v1 import NPUModelRunner

    original_sync_metadata_across_dp = NPUModelRunner._sync_metadata_across_dp

    if getattr(
        original_sync_metadata_across_dp,
        _PATCH_MARKER,
        False,
    ):
        _PATCH_APPLIED = True
        return

    @functools.wraps(original_sync_metadata_across_dp)
    def sync_metadata_across_dp_with_moe_consistency(
        self,
        num_tokens: int,
        is_draft_model: bool = False,
        cudagraph_mode: CUDAGraphMode = CUDAGraphMode.NONE,
        allow_dp_padding: bool = False,
    ) -> tuple[int, torch.Tensor | None, CUDAGraphMode]:
        needs_moe_metadata_sync = (
            self.dp_size > 1
            and not is_draft_model
            and is_moe_model(self.vllm_config)
            and should_skip_allreduce_across_dp_group(
                self.vllm_config,
                is_draft_model,
            )
        )

        if not needs_moe_metadata_sync:
            return original_sync_metadata_across_dp(
                self,
                num_tokens,
                is_draft_model,
                cudagraph_mode,
                allow_dp_padding,
            )

        # Each rank writes its metadata into its own column. After SUM
        # all-reduce, every rank obtains the complete DP metadata.
        dp_metadata = torch.zeros(
            (2, self.dp_size),
            device="cpu",
            dtype=torch.int32,
        )
        dp_metadata[0, self.dp_rank] = num_tokens
        dp_metadata[1, self.dp_rank] = cudagraph_mode.value

        dist.all_reduce(
            dp_metadata,
            group=get_dp_group().cpu_group,
        )

        tokens_across_dp = dp_metadata[0]
        max_tokens_across_dp = int(tokens_across_dp.max().item())

        # CUDAGraphMode: NONE=0, PIECEWISE=1, FULL=2.
        # Use the most conservative mode supported by every rank.
        synced_cudagraph_mode = CUDAGraphMode(int(dp_metadata[1].min().item()))

        # Preserve each rank's local token count. Downstream code uses the
        # global maximum only to choose one consistent MoE communication mode.
        return (
            max_tokens_across_dp,
            tokens_across_dp,
            synced_cudagraph_mode,
        )

    setattr(
        sync_metadata_across_dp_with_moe_consistency,
        _PATCH_MARKER,
        True,
    )

    NPUModelRunner._sync_metadata_across_dp = sync_metadata_across_dp_with_moe_consistency

    _PATCH_APPLIED = True


apply_patch()
