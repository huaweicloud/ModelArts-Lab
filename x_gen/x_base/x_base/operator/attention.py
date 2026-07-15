import logging
import os

import torch
import torch.nn.functional as F
from diffusers.utils import logging  # noqa: F811

from .rainfusion import Rainfusion

logger = logging.get_logger("attention")

try:
    import cann_ops

    CANN_OPS_AVAILABLE = True
except ImportError:
    logger.error("Not install CANN-OPS")
    CANN_OPS_AVAILABLE = False

try:
    from mindiesd import attention_forward

    LA_AVAILABLE = True
except ImportError:
    logger.error("Not install mindiesd, not laserattention")
    LA_AVAILABLE = False

SAGE_SEQ_LEN_THRESHOLD = int(os.getenv("SAGE_SEQ_LEN_THRESHOLD", 10000))


def base_attention_infer(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
    return F.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
    )


def sage_attention_infer_quant_scale(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
    # scale移到quant，性能下降
    B, N, Sq, D = query.shape
    B, N, Sk, D = key.shape
    if D == 128 and Sq > SAGE_SEQ_LEN_THRESHOLD and Sk > SAGE_SEQ_LEN_THRESHOLD:
        q_int8, deq_scale1 = cann_ops.block_quant(query, sm_scale=D**-0.5, block_size=128, is_query=True)
        k_int8, deq_scalek = cann_ops.block_quant(key, sm_scale=None, block_size=1024, is_query=False)
        return cann_ops.npu_sage_attention(
            q_int8.contiguous(),
            k_int8.contiguous(),
            value.contiguous(),
            deq_scale1=deq_scale1,
            deq_scalek=deq_scalek,
            num_heads=N,
            input_layout="BNSD",
            scale_value=0.0,
            num_key_value_heads=N,
            next_tokens=65535,
            atten_mask=attn_mask,
        )
    else:
        return base_attention_infer(query, key, value)


def sage_attention_infer(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
    B, N, Sq, D = query.shape
    B, N, Sk, D = key.shape
    if D == 128 and Sq > SAGE_SEQ_LEN_THRESHOLD and Sk > SAGE_SEQ_LEN_THRESHOLD:
        q_int8, deq_scale1 = cann_ops.block_quant(query, block_size=128, is_query=True)
        k_int8, deq_scalek = cann_ops.block_quant(key, block_size=1024, is_query=False)
        return cann_ops.npu_sage_attention(
            q_int8.contiguous(),
            k_int8.contiguous(),
            value.contiguous(),
            deq_scale1=deq_scale1,
            deq_scalek=deq_scalek,
            num_heads=N,
            input_layout="BNSD",
            scale_value=D**-0.5,
            num_key_value_heads=N,
            next_tokens=65535,
            atten_mask=attn_mask,
        )
    else:
        return base_attention_infer(query, key, value)


def laser_attention_infer(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
    B, N, Sq, D = query.shape
    B, N, Sk, D = key.shape
    scale = D**-0.5  # noqa: F841
    MIN_SEQLEN_SELF = 4000
    if Sk >= MIN_SEQLEN_SELF:
        return attention_forward(
            query, key, value, opt_mode="manual", op_type="ascend_laser_attention", layout="BNSD", head_first=True
        )
    else:
        return base_attention_infer(query, key, value)


class AttentionManager:
    def __init__(self):
        self.is_sage_attention = False
        self.is_laser_attention = False
        self.is_sparse_attention = False

        # rainfusion
        self.is_rainfusion_attention = False
        self.rainfusion_fa = None
        self.atten_mask_all = None
        self.t_idx = -1

    def enable_rainfusion_attention(self):
        self.is_rainfusion_attention = True

    def disable_rainfusion_attention(self):
        self.is_rainfusion_attention = False

    def enable_sage_attention(self):
        if not CANN_OPS_AVAILABLE:
            logger.error("Not install CANN-OPS")
        else:
            self.is_sage_attention = True

    def disable_sage_attention(self):
        self.is_sage_attention = False

    def enable_laser_attention(self):
        if not LA_AVAILABLE:
            logger.error("Not install laserattention")
        else:
            self.is_laser_attention = True

    def disable_laser_attention(self):
        self.is_laser_attention = False

    def enable_sparse_attention(self):
        self.is_sparse_attention = True

    def disable_sparse_attention(self):
        self.is_sparse_attention = False

    def set_t_idx(self, t_idx):
        self.t_idx = t_idx

    def update_t_idx(self):
        self.t_idx += 1

    def set_rainfusion_attention(self, latent_shape: list, skip_timesteps: int = 0, sparsity: float = 0.64):
        """
        每次推理前需要根据输入初始化 Rainfusion
        :param latent_shape: latents的THW网格大小
        """
        print(f"init latent_shape:{latent_shape} skip_timesteps:{skip_timesteps}")
        if not self.is_rainfusion_attention:
            logger.error("Before setting the parameters for rainfusion, rainfusion enable is required！")
        grid_size = Rainfusion.get_grid_size(latent_shape, [1, 2, 2])
        self.rainfusion_fa = Rainfusion(
            grid_size=grid_size,
            skip_timesteps=skip_timesteps,
            sparsity=sparsity,
        )
        self.atten_mask_all = Rainfusion.get_atten_mask(grid_size=grid_size, sparsity=sparsity)
        self.t_idx = -1

    def set_ada_bsa_sparse_flash_attention(self, sparsity: float = 0.7):
        self.ada_sparsity = sparsity

    def ada_bsa_sparse_flash_attention_infer(
        self, query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None
    ):
        B, N, Sq, D = query.shape
        B, N, Sk, D = key.shape
        scale = D**-0.5
        if Sq == Sk:
            # Going to sparse attention
            head_num = None  # noqa: F841
            keep_sink = True
            eep_recent = True  # noqa: F841
            keep_sink = True
            keep_recent = True
            sparsity = self.ada_sparsity
            cdf_threshold = 1.0
            sparse_size = 128
            stride = 8
            smask, sct = torch.ops.mindiesd.sparse_block_estimate(
                query,
                key,
                actual_seq_lengths=None,
                actual_seq_lengths_kv=None,
                input_layout="BNSD",
                stride=stride,
                sparse_size=sparse_size,
                num_heads=N,
                num_key_value_heads=N,
                scale_value=scale / stride,
                threshold=cdf_threshold,
                causal=is_causal,
                keep_sink=keep_sink,
                keep_recent=keep_recent,
                row_sparse=1.0 - sparsity,
            )
            value = value.contiguous()
            out = torch.ops.mindiesd.block_sparse_attention(
                query,
                key,
                value,
                num_heads=N,
                num_key_value_heads=N,
                input_layout="BNSD",
                scale_value=scale,
                causal=is_causal,
                sparse_size=sparse_size,
                sparse_mask=smask,
                sparse_count_table=sct,
            )
            return out
        else:
            return base_attention_infer(query, key, value)

    def attention(self, query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
        if self.is_sage_attention:
            return sage_attention_infer(
                query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale
            )
        elif self.is_laser_attention:
            return laser_attention_infer(
                query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale
            )
        elif self.is_rainfusion_attention:
            return self.rainfusion_attention_infer(
                query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale
            )
        elif self.is_sparse_attention:
            return self.ada_bsa_sparse_flash_attention_infer(
                query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale
            )
        else:
            return base_attention_infer(
                query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale
            )


attention_manager = AttentionManager()
