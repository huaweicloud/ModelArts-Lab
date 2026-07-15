import torch
from diffusers.utils import logging

logger = logging.get_logger("attention")

try:
    import ascend_cloud  # noqa: F401
    import cann_ops  # noqa: F401

    ASCEND_CLOUD_AVAILABLE = True
except ImportError:
    logger.error("Not install CANN-OPS or ascend_cloud")
    ASCEND_CLOUD_AVAILABLE = False

try:
    from mindiesd import rotary_position_embedding

    AVAILABLE_MINDIESD = True
except ImportError:
    logger.warning("Not install MINDIE")
    AVAILABLE_MINDIESD = False


def apply_rotary_emb(
    hidden_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
):
    x = hidden_states.view(*hidden_states.shape[:-1], -1, 2)
    x1, x2 = x[..., 0], x[..., 1]
    cos = cos[..., 0::2]
    sin = sin[..., 1::2]
    out = torch.empty_like(hidden_states)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos
    return out.type_as(hidden_states)


class RopeManager:
    def __init__(self):
        self.is_rope_fused = False

    def enable_rope_fused(self):
        if not ASCEND_CLOUD_AVAILABLE and not AVAILABLE_MINDIESD:
            logger.error("Not install CANN-OPS or MINDIE. Now using base rope")
        else:
            self.is_rope_fused = True

    @staticmethod
    def rope_flux(
        query: torch.Tensor,
        key: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ):
        query = query.contiguous()
        key = key.contiguous()
        if cos.dtype != query.dtype:
            cos = cos.to(dtype=query.dtype)
        if sin.dtype != query.dtype:
            sin = sin.to(dtype=query.dtype)
        cos = cos[..., 0::2]
        sin = sin[..., 1::2]
        row1 = torch.stack([cos, sin], dim=-1)
        row2 = torch.stack([-sin, cos], dim=-1)
        freqs_cis = torch.stack([row1, row2], dim=-1)
        return torch.ops.ascend_cloud.rope_flux(query, key, freqs_cis)

    @staticmethod
    def rope_base(
        query: torch.Tensor,
        key: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ):
        query = apply_rotary_emb(query, cos, sin)
        key = apply_rotary_emb(key, cos, sin)
        return query, key

    @staticmethod
    def rope_mindie(
        query: torch.Tensor,
        key: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ):
        query = rotary_position_embedding(query, cos, sin, rotated_mode="rotated_interleaved", fused=True)
        key = rotary_position_embedding(key, cos, sin, rotated_mode="rotated_interleaved", fused=True)
        return query, key

    def rope(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ):
        if self.is_rope_fused and AVAILABLE_MINDIESD:
            return self.rope_mindie(query, key, cos, sin)
        elif self.is_rope_fused and ASCEND_CLOUD_AVAILABLE:
            return self.rope_flux(query, key, cos, sin)
        else:
            return self.rope_base(query, key, cos, sin)


rope_manager = RopeManager()
