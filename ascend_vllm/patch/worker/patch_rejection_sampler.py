from __future__ import annotations

import torch

_PATCH_APPLIED = False
_PATCH_MARKER = "_modelarts_recovered_tokens_clamped"


def apply_patch() -> None:
    """Clamp spec-decode recovered tokens to >= 0 after the triton producer.

    ModelArts-layer override of
    ``vllm_ascend.sample.rejection_sampler.sample_recovered_tokens``.

    Background: on NPU the triton ``sample_recovered_tokens_kernel`` can store
    the ``PLACEHOLDER_TOKEN_ID`` (-1) sentinel when its running-max seed is
    never strictly beaten (degenerate rows: q==0 / NaN / fully masked pads).
    The random/greedy kernels forward that recovered value verbatim at
    reject / ``draft==-1`` slots, so a ``-1`` at a row head produces an
    all-``-1`` empty output row -> no tokens appended -> PD-decode hang
    ("Avg generation throughput: 0.0").

    Instead of re-routing recovered sampling through torch ops (which is
    ~O(num_tokens x vocab) over the full row), this patch keeps the pristine
    producer -- the triton kernel fast path under HAS_TRITON, the torch ops
    path otherwise -- and only clamps the returned token ids to >= 0 (an
    O(num_tokens) int elementwise, negligible next to the sampling itself).
    Clamping makes the ``-1`` sentinel structurally inexpressible downstream
    regardless of its origin (a degenerate row, or a buffer slot the kernel
    left untouched): every negative value becomes 0, which is exactly the
    token the torch path would also fall back to for a fully-degenerate row
    (whole-row argmax over the clamped floor -> column 0). Healthy rows pass
    through unchanged (no-op).

    The function is rebound on the ``vllm_ascend.sample.rejection_sampler``
    module because ``rejection_sample`` (same module) resolves the bare global
    ``sample_recovered_tokens`` at call time. The original function is captured
    before the rebind and called as the fast-path producer.
    """

    global _PATCH_APPLIED
    if _PATCH_APPLIED:
        return

    import vllm_ascend.sample.rejection_sampler as rejection_sampler_module

    original = rejection_sampler_module.sample_recovered_tokens
    if getattr(original, _PATCH_MARKER, False):
        _PATCH_APPLIED = True
        return

    def sample_recovered_tokens_clamped(
        max_spec_len: int,
        num_draft_tokens: list[int],
        cu_num_draft_tokens: torch.Tensor,
        draft_token_ids: torch.Tensor,
        draft_probs: torch.Tensor | None,
        target_probs: torch.Tensor,
        sampling_metadata: "SamplingMetadata",
        device: torch.device,
        use_block_verify: bool = False,
        target_indices: torch.Tensor | None = None,
        global_vocab_size: int | None = None,
        enable_reduce_sampling: bool = False,
    ) -> torch.Tensor:
        # Delegate to the pristine producer: triton kernel under HAS_TRITON
        # (torch ops otherwise), preserving the fast-path sampling performance.
        recovered_token_ids = original(
            max_spec_len,
            num_draft_tokens,
            cu_num_draft_tokens,
            draft_token_ids,
            draft_probs,
            target_probs,
            sampling_metadata,
            device,
            use_block_verify=use_block_verify,
            target_indices=target_indices,
            global_vocab_size=global_vocab_size,
            enable_reduce_sampling=enable_reduce_sampling,
        )
        # Belt: no -1 may reach the triton random/greedy consumer kernels.
        recovered_token_ids.clamp_min_(0)
        return recovered_token_ids

    sample_recovered_tokens_clamped.__name__ = original.__name__
    sample_recovered_tokens_clamped.__doc__ = original.__doc__
    setattr(sample_recovered_tokens_clamped, _PATCH_MARKER, True)

    rejection_sampler_module.sample_recovered_tokens = sample_recovered_tokens_clamped
    _PATCH_APPLIED = True


apply_patch()
