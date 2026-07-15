"""
BaseLongCatImagePipelineMixin - LongCatImage Pipeline 共享逻辑基类

由于 xDiffusersLongCatImagePipeline 和 xDiffusersLongCatImageEditPipeline
继承自不同的 diffusers 基类，使用 Mixin 模式实现代码复用。
"""

import numpy as np
import torch
import torch.distributed as dist
from diffusers.pipelines.longcat_image.pipeline_longcat_image import (
    calculate_shift,
    retrieve_timesteps,
)
from diffusers.pipelines.longcat_image.pipeline_output import LongCatImagePipelineOutput
from diffusers.utils import is_torch_xla_available, logging
from x_base import get_cfg_group

logger = logging.get_logger(__name__)


if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False


class BaseLongCatImagePipelineMixin:
    """LongCatImage Pipeline 共享逻辑 Mixin 基类

    只抽象真正值得复用的核心逻辑：
    - Timesteps 准备
    - 去噪循环（含 CFG 非并行/并行模式）
    - Latents 解码
    """

    def _prepare_timesteps(
        self,
        num_inference_steps: int,
        sigmas: list[float] | None,
        latents: torch.FloatTensor,
        device: torch.device,
    ) -> tuple[torch.Tensor, int, int]:
        """准备 timesteps"""
        sigmas = np.linspace(1.0, 1.0 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
        image_seq_len = latents.shape[1]
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.15),
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
            mu=mu,
        )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)
        return timesteps, num_inference_steps, num_warmup_steps

    def _run_denoising_loop(
        self,
        latents: torch.FloatTensor,
        timesteps: torch.Tensor,
        num_inference_steps: int,
        num_warmup_steps: int,
        prompt_embeds: torch.FloatTensor,
        negative_prompt_embeds: torch.FloatTensor | None,
        text_ids: torch.Tensor,
        negative_text_ids: torch.Tensor | None,
        latent_image_ids: torch.Tensor,
        cfg_parallel_size: int,
        enable_cfg_renorm: bool = False,
        cfg_renorm_min: float = 0.0,
        image_seq_len: int | None = None,
        image_latents: torch.FloatTensor | None = None,
    ) -> torch.FloatTensor:
        """执行去噪循环"""
        if self.joint_attention_kwargs is None:
            self._joint_attention_kwargs = {}

        local_rank = dist.get_rank() if cfg_parallel_size == 2 else 0

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                self._current_timestep = t

                # 准备 transformer 输入
                if image_latents is not None:
                    latent_model_input = torch.cat([latents, image_latents], dim=1)
                else:
                    latent_model_input = latents
                timestep = t.expand(latent_model_input.shape[0]).to(latents.dtype)

                # CFG 非并行模式
                if cfg_parallel_size == 1:
                    noise_pred = self._denoise_step_non_parallel(
                        latent_model_input,
                        timestep,
                        prompt_embeds,
                        negative_prompt_embeds,
                        text_ids,
                        negative_text_ids,
                        latent_image_ids,
                        enable_cfg_renorm,
                        cfg_renorm_min,
                        image_seq_len,
                    )
                # CFG 并行模式
                else:
                    noise_pred = self._denoise_step_parallel(
                        latent_model_input,
                        timestep,
                        prompt_embeds,
                        text_ids,
                        latent_image_ids,
                        local_rank,
                        enable_cfg_renorm,
                        cfg_renorm_min,
                        image_seq_len,
                    )

                # Scheduler step
                latents_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                if latents.dtype != latents_dtype and torch.backends.mps.is_available():
                    latents = latents.to(latents_dtype)

                # 更新进度
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        self._current_timestep = None
        return latents

    def _denoise_step_non_parallel(
        self,
        latent_model_input: torch.FloatTensor,
        timestep: torch.Tensor,
        prompt_embeds: torch.FloatTensor,
        negative_prompt_embeds: torch.FloatTensor | None,
        text_ids: torch.Tensor,
        negative_text_ids: torch.Tensor | None,
        latent_image_ids: torch.Tensor,
        enable_cfg_renorm: bool,
        cfg_renorm_min: float,
        image_seq_len: int | None,
    ) -> torch.FloatTensor:
        """非并行去噪步骤"""
        # 条件预测
        with self.transformer.cache_context("cond"):
            noise_pred_text = self.transformer(
                hidden_states=latent_model_input,
                timestep=timestep / 1000,
                guidance=None,
                encoder_hidden_states=prompt_embeds,
                txt_ids=text_ids,
                img_ids=latent_image_ids,
                return_dict=False,
            )[0]
            if image_seq_len is not None:
                noise_pred_text = noise_pred_text[:, :image_seq_len]

        # CFG
        if self.do_classifier_free_guidance and negative_prompt_embeds is not None:
            with self.transformer.cache_context("uncond"):
                noise_pred_uncond = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep / 1000,
                    guidance=None,
                    encoder_hidden_states=negative_prompt_embeds,
                    txt_ids=negative_text_ids,
                    img_ids=latent_image_ids,
                    return_dict=False,
                )[0]
                if image_seq_len is not None:
                    noise_pred_uncond = noise_pred_uncond[:, :image_seq_len]

            noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

            # CFG renorm
            if enable_cfg_renorm:
                cond_norm = torch.norm(noise_pred_text, dim=-1, keepdim=True)
                noise_norm = torch.norm(noise_pred, dim=-1, keepdim=True)
                scale = (cond_norm / (noise_norm + 1e-8)).clamp(min=cfg_renorm_min, max=1.0)
                noise_pred = noise_pred * scale
        else:
            noise_pred = noise_pred_text

        return noise_pred

    def _denoise_step_parallel(
        self,
        latent_model_input: torch.FloatTensor,
        timestep: torch.Tensor,
        prompt_embeds: torch.FloatTensor,
        text_ids: torch.Tensor,
        latent_image_ids: torch.Tensor,
        local_rank: int,
        enable_cfg_renorm: bool,
        cfg_renorm_min: float,
        image_seq_len: int | None,
    ) -> torch.FloatTensor:
        """并行去噪步骤 (CFG parallel)"""
        cache_index = "cond" if local_rank == 0 else "uncond"

        with self.transformer.cache_context(cache_index):
            noise_pred_temp = self.transformer(
                hidden_states=latent_model_input,
                timestep=timestep / 1000,
                guidance=None,
                encoder_hidden_states=prompt_embeds,
                txt_ids=text_ids,
                img_ids=latent_image_ids,
                return_dict=False,
            )[0]
            if image_seq_len is not None:
                noise_pred_temp = noise_pred_temp[:, :image_seq_len]
            torch.cuda.synchronize()

        # 合并 CFG 并行结果
        comb_pred = get_cfg_group().all_gather(noise_pred_temp, dim=0)
        noise_pred = comb_pred[1] + self.guidance_scale * (comb_pred[0] - comb_pred[1])

        # CFG renorm
        if enable_cfg_renorm:
            cond_norm = torch.norm(comb_pred[0], dim=-1, keepdim=True)
            noise_norm = torch.norm(noise_pred, dim=-1, keepdim=True)
            scale = (cond_norm / (noise_norm + 1e-8)).clamp(min=cfg_renorm_min, max=1.0)
            noise_pred = noise_pred * scale

        return noise_pred

    def _decode_latents(
        self,
        latents: torch.FloatTensor,
        height: int,
        width: int,
        output_type: str,
    ) -> torch.FloatTensor:
        """解码 latents"""
        if output_type == "latent":
            return latents

        latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
        latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor

        if latents.dtype != self.vae.dtype:
            latents = latents.to(dtype=self.vae.dtype)

        image = self.vae.decode(latents, return_dict=False)[0]
        return self.image_processor.postprocess(image, output_type=output_type)

    def _finalize_output(self, image: torch.FloatTensor, return_dict: bool):
        """最终输出处理"""
        self.maybe_free_model_hooks()
        if not return_dict:
            return (image,)
        return LongCatImagePipelineOutput(images=image)
