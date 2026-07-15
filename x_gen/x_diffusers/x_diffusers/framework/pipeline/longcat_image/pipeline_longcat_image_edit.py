"""
xDiffusersLongCatImageEditPipeline - LongCatImage Edit Pipeline 的扩展实现

支持图像编辑、CFG 并行等特性。
"""

from typing import Any

import PIL
import torch
import torch.distributed as dist
from diffusers import LongCatImageEditPipeline
from diffusers.utils import logging

from ..registry import register_hf_pipeline_class
from .base_longcat_image import BaseLongCatImagePipelineMixin

logger = logging.get_logger(__name__)


@register_hf_pipeline_class("LongCatImageEditPipeline")
class LongCatImageEditPipeline(BaseLongCatImagePipelineMixin, LongCatImageEditPipeline):
    """LongCatImage Edit Pipeline 扩展类

    继承自 LongCatImageEditPipeline，混入 BaseLongCatImagePipelineMixin
    提供图像编辑、CFG 并行等特性。
    """

    @torch.no_grad()
    def __call__(
        self,
        image: PIL.Image.Image | None = None,
        prompt: str | list[str] = None,
        negative_prompt: str | list[str] = None,
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
        cfg_parallel_size: int = 2,
    ):
        # ========== 1. 计算图像尺寸 ==========
        image_size = image[0].size if isinstance(image, list) else image.size
        calculated_width, calculated_height = image_size[0], image_size[1]

        # ========== 2. 输入校验 ==========
        self.check_inputs(
            prompt,
            calculated_height,
            calculated_width,
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
        batch_size = self._calculate_batch_size(prompt, prompt_embeds)
        device = self._execution_device

        # ========== 5. 预处理图像 ==========
        image, prompt_image = self._preprocess_input_image(image, calculated_height, calculated_width)

        # ========== 6. 编码 prompt ==========
        prompt_embeds, text_ids, negative_prompt_embeds, negative_text_ids = self._encode_prompt_with_cfg(
            prompt=prompt,
            negative_prompt=negative_prompt,
            prompt_image=prompt_image,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            num_images_per_prompt=num_images_per_prompt,
            cfg_parallel_size=cfg_parallel_size,
        )

        # ========== 7. 准备 latents ==========
        latents, image_latents, latents_ids, image_latents_ids = self.prepare_latents(
            image,
            batch_size * num_images_per_prompt,
            16,  # num_channels_latents
            calculated_height,
            calculated_width,
            prompt_embeds.dtype,
            prompt_embeds.shape[1],
            device,
            generator,
            latents,
        )
        image_seq_len = latents.shape[1]

        # ========== 8. 准备 latent_image_ids ==========
        latent_image_ids = self._merge_latent_image_ids(latents_ids, image_latents_ids)

        # ========== 9. 准备 timesteps ==========
        timesteps, num_inference_steps, num_warmup_steps = self._prepare_timesteps(
            num_inference_steps, sigmas, latents, device
        )

        # ========== 10. 去噪循环 ==========
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
            enable_cfg_renorm=False,  # Edit pipeline 不支持
            cfg_renorm_min=0.0,
            image_seq_len=image_seq_len,
            image_latents=image_latents,
        )

        # ========== 11. 解码输出 ==========
        image = self._decode_latents(latents, calculated_height, calculated_width, output_type)

        # ========== 12. 最终处理 ==========
        return self._finalize_output(image, return_dict)

    def _calculate_batch_size(
        self,
        prompt: str | list[str] | None,
        prompt_embeds: torch.FloatTensor | None,
    ) -> int:
        """计算 batch_size"""
        if prompt is not None and isinstance(prompt, str):
            return 1
        elif prompt is not None and isinstance(prompt, list):
            return len(prompt)
        else:
            return prompt_embeds.shape[0]

    def _preprocess_input_image(
        self,
        image: PIL.Image.Image | torch.Tensor | None,
        calculated_height: int,
        calculated_width: int,
    ) -> tuple[torch.Tensor | PIL.Image.Image | None, PIL.Image.Image | None]:
        """预处理输入图像

        Returns:
            tuple: (processed_image, prompt_image)
        """
        if image is None:
            return None, None

        if isinstance(image, torch.Tensor) and image.size(1) == self.latent_channels:
            return image, None

        image = self.image_processor.resize(image, calculated_height, calculated_width)
        prompt_image = self.image_processor.resize(image, calculated_height // 2, calculated_width // 2)
        image = self.image_processor.preprocess(image, calculated_height, calculated_width)
        return image, prompt_image

    def _encode_prompt_with_cfg(
        self,
        prompt: str | list[str] | None,
        negative_prompt: str | list[str] | None,
        prompt_image: PIL.Image.Image | None,
        prompt_embeds: torch.FloatTensor | None,
        negative_prompt_embeds: torch.FloatTensor | None,
        num_images_per_prompt: int,
        cfg_parallel_size: int,
    ) -> tuple[torch.FloatTensor, Any, torch.FloatTensor | None, Any | None]:
        """CFG 模式下的 prompt 编码

        Returns:
            tuple: (prompt_embeds, text_ids, negative_prompt_embeds, negative_text_ids)
        """
        # CFG 并行验证
        if cfg_parallel_size == 2 and not self.do_classifier_free_guidance:
            raise ValueError(
                f"CFG parallel size is {cfg_parallel_size} must need 'guidance_scale > 1 and has_neg_prompt', "
                f"but guidance_scale is {self._guidance_scale}"
            )

        negative_prompt = "" if negative_prompt is None else negative_prompt

        # CFG 非并行模式
        if cfg_parallel_size == 1:
            prompt_embeds, text_ids = self.encode_prompt(
                prompt=prompt,
                image=prompt_image,
                prompt_embeds=prompt_embeds,
                num_images_per_prompt=num_images_per_prompt,
            )
            if self.do_classifier_free_guidance:
                negative_prompt_embeds, negative_text_ids = self.encode_prompt(
                    prompt=negative_prompt,
                    image=prompt_image,
                    prompt_embeds=negative_prompt_embeds,
                    num_images_per_prompt=num_images_per_prompt,
                )
            else:
                negative_prompt_embeds, negative_text_ids = None, None
            return prompt_embeds, text_ids, negative_prompt_embeds, negative_text_ids

        # CFG 并行模式
        local_rank = dist.get_rank()
        if local_rank == 1:
            prompt, prompt_embeds = negative_prompt, negative_prompt_embeds
        prompt_embeds, text_ids = self.encode_prompt(
            prompt=prompt,
            image=prompt_image,
            prompt_embeds=prompt_embeds,
            num_images_per_prompt=num_images_per_prompt,
        )
        return prompt_embeds, text_ids, None, None

    def _merge_latent_image_ids(
        self,
        latents_ids: torch.Tensor,
        image_latents_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        """合并 latent_image_ids"""
        if image_latents_ids is not None:
            return torch.cat([latents_ids, image_latents_ids], dim=0)
        return latents_ids
