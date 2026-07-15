from collections.abc import Callable
from typing import Any

import torch
from diffusers.image_processor import PipelineImageInput
from diffusers.pipelines import QwenImageEditPipeline
from diffusers.pipelines.qwenimage.pipeline_output import QwenImagePipelineOutput
from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit import calculate_shift, retrieve_timesteps
from diffusers.utils import logging

from ..registry import register_hf_pipeline_class
from .utils import (
    PIPELINE_CONFIGS,
    get_calculated_dimensions,
    prepare_dimensions,
    run_qwenimage_pipeline_core,
)

logger = logging.get_logger(__name__)


def _get_encode_image(pipeline, image):
    """Get image for encode_prompt."""
    if image is None or (isinstance(image, torch.Tensor) and image.size(1) == pipeline.latent_channels):
        return None
    calculated_width, calculated_height = get_calculated_dimensions(pipeline, image)
    return pipeline.image_processor.resize(image, calculated_height, calculated_width)


def _prepare_latents_data(
    pipeline, image, batch_size, num_images_per_prompt, height, width, dtype, device, generator, latents
):
    """Prepare latents and img_shapes for QwenImageEditPipeline."""
    calculated_width, calculated_height = get_calculated_dimensions(pipeline, image)

    # Preprocess image for VAE
    if image is not None and not (isinstance(image, torch.Tensor) and image.size(1) == pipeline.latent_channels):
        vae_image = pipeline.image_processor.resize(image, calculated_height, calculated_width)
        vae_image = pipeline.image_processor.preprocess(vae_image, calculated_height, calculated_width).unsqueeze(2)
    else:
        vae_image = image

    # Prepare latents
    num_channels_latents = pipeline.transformer.config.in_channels // 4
    latents, image_latents = pipeline.prepare_latents(
        vae_image,
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
            (
                1,
                calculated_height // pipeline.vae_scale_factor // 2,
                calculated_width // pipeline.vae_scale_factor // 2,
            ),
        ]
    ] * batch_size

    return latents, img_shapes, {"image_latents": image_latents}


@register_hf_pipeline_class("QwenImageEditPipeline")
class QwenImageEditPipeline(QwenImageEditPipeline):
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
            PIPELINE_CONFIGS["qwenimage_edit"],
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
