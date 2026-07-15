from collections.abc import Callable
from typing import Any

import torch
from diffusers.image_processor import PipelineImageInput
from diffusers.pipelines import QwenImageEditPlusPipeline
from diffusers.pipelines.qwenimage.pipeline_output import QwenImagePipelineOutput
from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus import (
    CONDITION_IMAGE_SIZE,
    VAE_IMAGE_SIZE,
    calculate_dimensions,
    calculate_shift,
    retrieve_timesteps,
)
from diffusers.utils import logging

from ..registry import register_hf_pipeline_class
from .utils import (
    PIPELINE_CONFIGS,
    prepare_dimensions,
    run_qwenimage_pipeline_core,
)

logger = logging.get_logger(__name__)


def _get_encode_image(pipeline, image):
    """Get condition images for encode_prompt."""
    if image is None or (isinstance(image, torch.Tensor) and image.size(1) == pipeline.latent_channels):
        return None

    images = image if isinstance(image, list) else [image]
    condition_images = []
    for img in images:
        image_width, image_height = img.size
        condition_width, condition_height = calculate_dimensions(CONDITION_IMAGE_SIZE, image_width / image_height)
        condition_images.append(pipeline.image_processor.resize(img, condition_height, condition_width))
    return condition_images


def _prepare_latents_data(
    pipeline, image, batch_size, num_images_per_prompt, height, width, dtype, device, generator, latents
):
    """Prepare latents and img_shapes for QwenImageEditPlusPipeline."""
    if image is None or (isinstance(image, torch.Tensor) and image.size(1) == pipeline.latent_channels):
        vae_images = image
        vae_image_sizes = []
    else:
        images = image if isinstance(image, list) else [image]
        vae_images = []
        vae_image_sizes = []
        for img in images:
            image_width, image_height = img.size
            vae_width, vae_height = calculate_dimensions(VAE_IMAGE_SIZE, image_width / image_height)
            vae_image_sizes.append((vae_width, vae_height))
            vae_images.append(pipeline.image_processor.preprocess(img, vae_height, vae_width).unsqueeze(2))

    # Prepare latents
    num_channels_latents = pipeline.transformer.config.in_channels // 4
    latents, image_latents = pipeline.prepare_latents(
        vae_images,
        batch_size * num_images_per_prompt,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents,
    )

    # Prepare img_shapes
    img_shapes = [
        [
            (1, height // pipeline.vae_scale_factor // 2, width // pipeline.vae_scale_factor // 2),
            *[
                (1, h // pipeline.vae_scale_factor // 2, w // pipeline.vae_scale_factor // 2)
                for w, h in vae_image_sizes
            ],
        ]
    ] * batch_size

    return latents, img_shapes, {"image_latents": image_latents}


@register_hf_pipeline_class("QwenImageEditPlusPipeline")
class QwenImageEditPlusPipeline(QwenImageEditPlusPipeline):
    @torch.no_grad()
    def __call__(
        self,
        image: PipelineImageInput | None = None,
        prompt: str | list[str] = None,
        negative_prompt: str | list[str] = None,
        true_cfg_scale: float = 4.0,
        height: int | None = None,
        width: int | None = None,
        num_inference_steps: int = 50,
        cfg_parallel_size: int = 2,
        sigmas: list[float] | None = None,
        guidance_scale: float | None = None,
        num_images_per_prompt: int = 1,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.Tensor | None = None,
        prompt_embeds: torch.Tensor | None = None,
        prompt_embeds_mask: torch.Tensor | None = None,
        negative_prompt_embeds: torch.Tensor | None = None,
        negative_prompt_embeds_mask: torch.Tensor | None = None,
        output_type: str | None = "pil",
        return_dict: bool = True,
        attention_kwargs: dict[str, Any] | None = None,
        callback_on_step_end: Callable | None = None,
        callback_on_step_end_tensor_inputs: list[str] | None = None,
        max_sequence_length: int = 512,
    ):
        return run_qwenimage_pipeline_core(
            self,
            PIPELINE_CONFIGS["qwenimage_edit_plus"],
            prompt=prompt,
            negative_prompt=negative_prompt,
            true_cfg_scale=true_cfg_scale,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            cfg_parallel_size=cfg_parallel_size,
            sigmas=sigmas,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_images_per_prompt,
            generator=generator,
            latents=latents,
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_prompt_embeds_mask=negative_prompt_embeds_mask,
            output_type=output_type,
            return_dict=return_dict,
            attention_kwargs=attention_kwargs,
            callback_on_step_end=callback_on_step_end,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
            image=image,
            prepare_dimensions_fn=prepare_dimensions,
            prepare_latents_data_fn=_prepare_latents_data,
            get_encode_image_fn=_get_encode_image,
            calculate_shift_fn=calculate_shift,
            retrieve_timesteps_fn=retrieve_timesteps,
            output_class=QwenImagePipelineOutput,
        )
