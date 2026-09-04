from __future__ import annotations

import torch

_PATCH_APPLIED = False
_PATCH_MARKER = "_modelarts_recovered_tokens_always_torch"


def apply_patch() -> None:
    """Route spec-decode recovered-token sampling through torch ops.

    ModelArts-layer override of
    ``vllm_ascend.sample.rejection_sampler.sample_recovered_tokens``.

    Background: on NPU the triton ``sample_recovered_tokens_kernel`` can store
    the ``PLACEHOLDER_TOKEN_ID`` (-1) sentinel when its running-max seed is
    never strictly beaten (degenerate rows: q==0 / NaN / fully masked pads).
    The random/greedy kernels forward that recovered value verbatim at
    reject / ``draft==-1`` slots, so a ``-1`` at a row head produces an
    all-``-1`` empty output row -> no tokens appended -> PD-decode hang
    ("Avg generation throughput: 0.0").

    The torch ops path is structurally immune: q==0/isinf columns are clamped
    to a finite floor and a single whole-row ``torch.argmax`` always returns a
    valid vocab index (>= 0), even when the whole row degenerates. This patch
    therefore forces recovered sampling through the torch ops unconditionally
    (regardless of HAS_TRITON). The triton random/greedy kernels still consume
    the result afterwards, so no ``-1`` can reach output_token_ids.

    The function is rebound on the ``vllm_ascend.sample.rejection_sampler``
    module because ``rejection_sample`` (same module) resolves the bare global
    ``sample_recovered_tokens`` at call time.
    """

    global _PATCH_APPLIED
    if _PATCH_APPLIED:
        return

    import vllm_ascend.sample.rejection_sampler as rejection_sampler_module

    original = rejection_sampler_module.sample_recovered_tokens
    if getattr(original, _PATCH_MARKER, False):
        _PATCH_APPLIED = True
        return

    def sample_recovered_tokens_always_torch(
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
        batch_size = len(num_draft_tokens)
        vocab_size = target_probs.shape[-1]

        # Per-request normalization noise. Drawn again (with the request RNG)
        # for rows that actually have draft tokens; other rows keep the bulk
        # fill because their recovered value is never consumed.
        q = torch.empty(
            (batch_size, vocab_size),
            dtype=torch.float32,
            device=device,
        )
        q.exponential_()

        num_draft_tensor = torch.tensor(num_draft_tokens, pin_memory=True).to(device, non_blocking=True)
        has_draft_mask = num_draft_tensor > 0

        for i, generator in sampling_metadata.generators.items():
            temp_q = torch.empty_like(q[i])
            temp_q.exponential_(generator=generator)
            q[i] = torch.where(has_draft_mask[i], temp_q, q[i])

        recovered_token_ids = torch.empty_like(draft_token_ids)
        if use_block_verify:
            rejection_sampler_module.sample_recovered_tokens_blockwise_pytorch(
                recovered_token_ids,
                cu_num_draft_tokens,
                draft_token_ids,
                draft_probs,
                target_probs,
                q,
                vocab_size,
                IS_NGRAM=draft_probs is None,
                target_indices=target_indices,
                enable_reduce_sampling=enable_reduce_sampling,
            )
        else:
            rejection_sampler_module.sample_recovered_tokens_pytorch(
                recovered_token_ids,
                cu_num_draft_tokens,
                draft_token_ids,
                draft_probs,
                target_probs,
                q,
                vocab_size,
                IS_NGRAM=draft_probs is None,
                target_indices=target_indices,
                enable_reduce_sampling=enable_reduce_sampling,
            )
        return recovered_token_ids

    sample_recovered_tokens_always_torch.__name__ = original.__name__
    sample_recovered_tokens_always_torch.__doc__ = original.__doc__
    setattr(sample_recovered_tokens_always_torch, _PATCH_MARKER, True)

    rejection_sampler_module.sample_recovered_tokens = sample_recovered_tokens_always_torch
    _PATCH_APPLIED = True


apply_patch()
