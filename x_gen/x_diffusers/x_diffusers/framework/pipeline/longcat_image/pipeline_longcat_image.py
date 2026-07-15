"""
xDiffusersLongCatImagePipeline - LongCatImage Pipeline 的扩展实现

支持 CFG 并行、CFG renorm、prompt rewrite 等特性。
"""

from typing import Any

import torch
import torch.distributed as dist
from diffusers import LongCatImagePipeline
from diffusers.utils import logging

from ..registry import register_hf_pipeline_class
from .base_longcat_image import BaseLongCatImagePipelineMixin

logger = logging.get_logger(__name__)


@register_hf_pipeline_class("LongCatImagePipeline")
class LongCatImagePipeline(BaseLongCatImagePipelineMixin, LongCatImagePipeline):
    """LongCatImage Pipeline 扩展类

    继承自 LongCatImagePipeline，混入 BaseLongCatImagePipelineMixin
    提供 CFG 并行、CFG renorm、prompt rewrite 等增强特性。
    """

    @torch.no_grad()
    def __call__(
        self,
        prompt: str | list[str] = None,
        negative_prompt: str | list[str] = None,
        height: int | None = None,
        width: int | None = None,
        num_inference_steps: int = 50,
        sigmas: list[float] | None = None,
        guidance_scale: float = 4.5,
        num_images_per_prompt: int | None = 1,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.FloatTensor | None = None,
        prompt_embeds: torch.FloatTensor | None = None,
        negative_prompt_embeds: torch.FloatTensor | None = None,
        output_type: str | None = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: dict[str, Any] | None = None,
        enable_cfg_renorm: bool | None = True,
        cfg_renorm_min: float | None = 0.0,
        enable_prompt_rewrite: bool | None = True,
        cfg_parallel_size: int = 2,
    ):
        # ========== 1. 计算图像尺寸 ==========
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        # ========== 2. 输入校验 ==========
        self.check_inputs(
            prompt,
            height,
            width,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )

        # ========== 3. 初始化状态 ==========
        self._guidance_scale = guidance_scale
        self._joint_attention_kwargs = joint_attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        # ========== 4. 准备 batch 参数 ==========
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]
        device = self._execution_device

        # ========== 5. Prompt rewrite ==========
        if enable_prompt_rewrite:
            prompt = self.rewire_prompt(prompt, device)
            logger.info(f"Rewrite prompt {prompt}!")  # noqa: G004

        negative_prompt = "" if negative_prompt is None else negative_prompt

        # ========== 6. 编码 prompt ==========
        # CFG 并行验证
        if cfg_parallel_size == 2 and not self.do_classifier_free_guidance:
            raise ValueError(
                f"CFG parallel size is {cfg_parallel_size} must need 'guidance_scale > 1 and has_neg_prompt', "
                f"but guidance_scale is {self._guidance_scale}"
            )

        # CFG 非并行模式
        if cfg_parallel_size == 1:
            prompt_embeds, text_ids = self.encode_prompt(
                prompt=prompt,
                prompt_embeds=prompt_embeds,
                num_images_per_prompt=num_images_per_prompt,
            )
            if self.do_classifier_free_guidance:
                negative_prompt_embeds, negative_text_ids = self.encode_prompt(
                    prompt=negative_prompt,
                    prompt_embeds=negative_prompt_embeds,
                    num_images_per_prompt=num_images_per_prompt,
                )
            else:
                negative_prompt_embeds, negative_text_ids = None, None
        # CFG 并行模式
        else:
            local_rank = dist.get_rank()
            if local_rank == 1:
                prompt, prompt_embeds = negative_prompt, negative_prompt_embeds
            prompt_embeds, text_ids = self.encode_prompt(
                prompt=prompt,
                prompt_embeds=prompt_embeds,
                num_images_per_prompt=num_images_per_prompt,
            )
            negative_prompt_embeds, negative_text_ids = None, None

        # ========== 7. 准备 latents ==========
        latents, latent_image_ids = self.prepare_latents(
            batch_size * num_images_per_prompt,
            16,  # num_channels_latents
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # ========== 8. 准备 timesteps ==========
        timesteps, num_inference_steps, num_warmup_steps = self._prepare_timesteps(
            num_inference_steps, sigmas, latents, device
        )

        # ========== 9. 去噪循环 ==========
        latents = self._run_denoising_loop(
            latents=latents,
            timesteps=timesteps,
            num_inference_steps=num_inference_steps,
            num_warmup_steps=num_warmup_steps,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            text_ids=text_ids,
            negative_text_ids=negative_text_ids,
            latent_image_ids=latent_image_ids,
            cfg_parallel_size=cfg_parallel_size,
            enable_cfg_renorm=enable_cfg_renorm,
            cfg_renorm_min=cfg_renorm_min,
        )

        # ========== 10. 解码输出 ==========
        image = self._decode_latents(latents, height, width, output_type)

        # ========== 11. 最终处理 ==========
        return self._finalize_output(image, return_dict)
