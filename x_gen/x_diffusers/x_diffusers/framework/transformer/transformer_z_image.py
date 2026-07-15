import torch
from diffusers.models.transformers.transformer_z_image import (
    RopeEmbedder,
)


def z_image_rope_embedder_call__(self, ids: torch.Tensor):
    device = ids.device

    if self.freqs_cis is None:
        self.freqs_cis = self.precompute_freqs_cis(self.axes_dims, self.axes_lens, theta=self.theta)
        self.freqs_cis = [freqs_cis.to(device) for freqs_cis in self.freqs_cis]
    else:
        # Ensure freqs_cis are on the same device as ids
        if self.freqs_cis[0].device != device:
            self.freqs_cis = [freqs_cis.to(device) for freqs_cis in self.freqs_cis]

    result = []
    for i in range(len(self.axes_dims)):
        index = ids[:, i]
        # ================== for npu ==================
        result.append(torch.index_select(self.freqs_cis[i], 0, index))
        # ================== for npu ==================
    return torch.cat(result, dim=-1)


RopeEmbedder.__call__ = z_image_rope_embedder_call__
