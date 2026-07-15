from collections.abc import Callable
from typing import Any

import torch
from diffusers.pipelines import QwenImagePipeline
from diffusers.pipelines.qwenimage.pipeline_output import QwenImagePipelineOutput
from diffusers.pipelines.qwenimage.pipeline_qwenimage import calculate_shift, retrieve_timesteps
from diffusers.utils import logging

from ..registry import register_hf_pipeline_class
from .utils import (
    PIPELINE_CONFIGS,
    run_qwenimage_pipeline_core,
)

logger = logging.get_logger(__name__)


def _prepare_dimensions(pipeline, image, height, width):
    """Prepare dimensions for QwenImagePipeline."""
    height = height or pipeline.default_sample_size * pipeline.vae_scale_factor
    width = width or pipeline.default_sample_size * pipeline.vae_scale_factor
    return height, width


@register_hf_pipeline_class("QwenImagePipeline")
class QwenImagePipeline(QwenImagePipeline):
    @torch.no_grad()
    def __call__(
        self,
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
            config=PIPELINE_CONFIGS["qwenimage"],
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
            prepare_dimensions_fn=_prepare_dimensions,
            calculate_shift_fn=calculate_shift,
            retrieve_timesteps_fn=retrieve_timesteps,
            output_class=QwenImagePipelineOutput,
        )
