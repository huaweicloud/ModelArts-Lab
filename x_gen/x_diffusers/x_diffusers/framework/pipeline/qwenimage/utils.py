"""Common utilities for QwenImage pipelines."""

from collections.abc import Callable
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus import calculate_dimensions
from diffusers.utils import is_torch_xla_available, logging
from x_base import get_cfg_group

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

logger = logging.get_logger(__name__)


# =============================================================================
# Basic Utilities
# =============================================================================
def get_calculated_dimensions(pipeline, image):
    """Calculate dimensions from image."""
    image_size = image[-1].size if isinstance(image, list) else image.size
    calculated_width, calculated_height = calculate_dimensions(1024 * 1024, image_size[0] / image_size[1])
    multiple_of = pipeline.vae_scale_factor * 2
    return calculated_width // multiple_of * multiple_of, calculated_height // multiple_of * multiple_of


def prepare_dimensions(pipeline, image, height, width):
    """Prepare dimensions for QwenImageEditPipeline,QwenImageEditPlusPipeline."""
    calculated_width, calculated_height = get_calculated_dimensions(pipeline, image)
    return height or calculated_height, width or calculated_width


def get_batch_size(prompt: str | list[str] | None, prompt_embeds: torch.Tensor | None) -> int:
    """Calculate batch size from prompt or prompt_embeds."""
    if prompt is not None and isinstance(prompt, str):
        return 1
    elif prompt is not None and isinstance(prompt, list):
        return len(prompt)
    else:
        return prompt_embeds.shape[0]


def get_has_neg_prompt(
    negative_prompt: str | list[str] | None,
    negative_prompt_embeds: torch.Tensor | None,
    negative_prompt_embeds_mask: torch.Tensor | None,
) -> bool:
    """Check if negative prompt is provided."""
    return negative_prompt is not None or (
        negative_prompt_embeds is not None and negative_prompt_embeds_mask is not None
    )


def warn_cfg_settings(true_cfg_scale: float, has_neg_prompt: bool) -> None:
    """Log warnings for CFG settings."""
    if true_cfg_scale > 1 and not has_neg_prompt:
        logger.warning(
            f"true_cfg_scale is passed as {true_cfg_scale}, but classifier-free guidance is not enabled since no negative_prompt is provided."  # noqa: G004, E501
        )
    elif true_cfg_scale <= 1 and has_neg_prompt:
        logger.warning(
            " negative_prompt is passed but classifier-free guidance is not enabled since true_cfg_scale <= 1"
        )


# =============================================================================
# CFG Utilities
# =============================================================================


def apply_cfg(
    noise_pred: torch.Tensor,
    neg_noise_pred: torch.Tensor,
    true_cfg_scale: float,
) -> torch.Tensor:
    """Apply classifier-free guidance with normalization."""
    comb_pred = neg_noise_pred + true_cfg_scale * (noise_pred - neg_noise_pred)

    cond_norm = torch.norm(noise_pred, dim=-1, keepdim=True)
    noise_norm = torch.norm(comb_pred, dim=-1, keepdim=True)
    return comb_pred * (cond_norm / noise_norm)


def apply_cfg_parallel(
    noise_pred: torch.Tensor,
    true_cfg_scale: float,
    cfg_group,
) -> torch.Tensor:
    """Apply CFG in parallel mode with all_gather."""
    noise_pred_temp = cfg_group.all_gather(noise_pred, dim=0)
    comb_pred = noise_pred_temp[1] + true_cfg_scale * (noise_pred_temp[0] - noise_pred_temp[1])

    cond_norm = torch.norm(noise_pred_temp[0], dim=-1, keepdim=True)
    noise_norm = torch.norm(comb_pred, dim=-1, keepdim=True)
    return comb_pred * (cond_norm / noise_norm)


# =============================================================================
# Pipeline Configuration
# =============================================================================


class PipelineConfig:
    """Configuration for different pipeline types."""

    def __init__(
        self,
        needs_image: bool = False,
        needs_multiple_images: bool = False,
        batch_size_limit: int | None = None,
        needs_edit_preprocess: bool = False,
    ):
        self.needs_image = needs_image
        self.needs_multiple_images = needs_multiple_images
        self.batch_size_limit = batch_size_limit
        self.needs_edit_preprocess = needs_edit_preprocess


# Predefined configs for each pipeline type
PIPELINE_CONFIGS = {
    "qwenimage": PipelineConfig(
        needs_image=False,
        needs_edit_preprocess=False,
    ),
    "qwenimage_edit": PipelineConfig(
        needs_image=True,
        needs_multiple_images=False,
        needs_edit_preprocess=True,
    ),
    "qwenimage_edit_plus": PipelineConfig(
        needs_image=True,
        needs_multiple_images=True,
        batch_size_limit=1,
        needs_edit_preprocess=True,
    ),
}


# =============================================================================
# Pipeline Common Operations
# =============================================================================


def init_pipeline_state(
    pipeline,
    guidance_scale: float | None,
    attention_kwargs: dict[str, Any] | None,
) -> None:
    """Initialize pipeline internal state.

    Args:
        pipeline: Pipeline instance
        guidance_scale: Guidance scale value
        attention_kwargs: Attention kwargs
    """
    pipeline._guidance_scale = guidance_scale
    pipeline._attention_kwargs = attention_kwargs
    pipeline._current_timestep = None
    pipeline._interrupt = False


def get_cfg_setup(
    true_cfg_scale: float,
    negative_prompt: str | list[str] | None,
    negative_prompt_embeds: torch.Tensor | None,
    negative_prompt_embeds_mask: torch.Tensor | None,
) -> tuple[bool, bool]:
    """Get CFG setup parameters.

    Args:
        true_cfg_scale: CFG scale value
        negative_prompt: Negative prompt text
        negative_prompt_embeds: Pre-computed negative prompt embeddings
        negative_prompt_embeds_mask: Negative prompt embeddings mask

    Returns:
        Tuple of (has_neg_prompt, do_true_cfg)
    """
    has_neg_prompt = get_has_neg_prompt(negative_prompt, negative_prompt_embeds, negative_prompt_embeds_mask)
    warn_cfg_settings(true_cfg_scale, has_neg_prompt)
    do_true_cfg = true_cfg_scale > 1 and has_neg_prompt
    return has_neg_prompt, do_true_cfg


def encode_prompts_with_cfg(
    pipeline,
    cfg_parallel_size: int,
    do_true_cfg: bool,
    device,
    num_images_per_prompt: int,
    max_sequence_length: int,
    prompt: str | list[str] | None = None,
    negative_prompt: str | list[str] | None = None,
    prompt_embeds: torch.Tensor | None = None,
    prompt_embeds_mask: torch.Tensor | None = None,
    negative_prompt_embeds: torch.Tensor | None = None,
    negative_prompt_embeds_mask: torch.Tensor | None = None,
    image=None,
    condition_images=None,
) -> tuple:
    """Encode prompts with CFG parallel support.

    Args:
        pipeline: Pipeline instance with encode_prompt method
        cfg_parallel_size: CFG parallel size (1 or 2)
        do_true_cfg: Whether to do true CFG
        device: Device to use
        num_images_per_prompt: Number of images per prompt
        max_sequence_length: Max sequence length
        prompt: Prompt text
        negative_prompt: Negative prompt text
        prompt_embeds: Pre-computed prompt embeddings
        prompt_embeds_mask: Prompt embeddings mask
        negative_prompt_embeds: Pre-computed negative prompt embeddings
        negative_prompt_embeds_mask: Negative prompt embeddings mask
        image: Image for encode_prompt (for edit pipelines)
        condition_images: Condition images for encode_prompt (for edit_plus pipeline)

    Returns:
        Tuple of (prompt_embeds, prompt_embeds_mask, negative_prompt_embeds,
                  negative_prompt_embeds_mask, local_rank)
    """
    # Prepare encode_prompt kwargs
    encode_kwargs = {
        "prompt": prompt,
        "prompt_embeds": prompt_embeds,
        "prompt_embeds_mask": prompt_embeds_mask,
        "device": device,
        "num_images_per_prompt": num_images_per_prompt,
        "max_sequence_length": max_sequence_length,
    }
    if image is not None:
        encode_kwargs["image"] = image
    if condition_images is not None:
        encode_kwargs["image"] = condition_images

    if cfg_parallel_size == 1:
        local_rank = 0
        prompt_embeds, prompt_embeds_mask = pipeline.encode_prompt(**encode_kwargs)

        if do_true_cfg:
            neg_encode_kwargs = encode_kwargs.copy()
            neg_encode_kwargs["prompt"] = negative_prompt
            neg_encode_kwargs["prompt_embeds"] = negative_prompt_embeds
            neg_encode_kwargs["prompt_embeds_mask"] = negative_prompt_embeds_mask
            negative_prompt_embeds, negative_prompt_embeds_mask = pipeline.encode_prompt(**neg_encode_kwargs)
        else:
            negative_prompt_embeds = None
            negative_prompt_embeds_mask = None

    elif cfg_parallel_size == 2:
        if not do_true_cfg:
            has_neg_prompt = get_has_neg_prompt(negative_prompt, negative_prompt_embeds, negative_prompt_embeds_mask)
            raise ValueError(
                f"CFG parallel size is {cfg_parallel_size} must need 'true_cfg_scale > 1 and has_neg_prompt', "
                f"but true_cfg_scale <= 1 or has_neg_prompt is {has_neg_prompt}"
            )

        local_rank = dist.get_rank()
        if local_rank == 1:
            prompt = negative_prompt
            prompt_embeds = negative_prompt_embeds
            prompt_embeds_mask = negative_prompt_embeds_mask

        encode_kwargs["prompt"] = prompt
        encode_kwargs["prompt_embeds"] = prompt_embeds
        encode_kwargs["prompt_embeds_mask"] = prompt_embeds_mask
        prompt_embeds, prompt_embeds_mask = pipeline.encode_prompt(**encode_kwargs)

        # For parallel CFG, negative embeds will be gathered from other ranks
        negative_prompt_embeds = None
        negative_prompt_embeds_mask = None
    else:
        raise ValueError(f"Invalid cfg_parallel_size: {cfg_parallel_size}, must be 1 or 2")

    return (
        prompt_embeds,
        prompt_embeds_mask,
        negative_prompt_embeds,
        negative_prompt_embeds_mask,
        local_rank,
    )


def prepare_timesteps(
    scheduler,
    num_inference_steps: int,
    device,
    latents: torch.Tensor,
    sigmas: list[float] | None = None,
    calculate_shift_fn=None,
    retrieve_timesteps_fn=None,
):
    """Prepare timesteps for denoising loop."""
    sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
    image_seq_len = latents.shape[1]
    mu = calculate_shift_fn(
        image_seq_len,
        scheduler.config.get("base_image_seq_len", 256),
        scheduler.config.get("max_image_seq_len", 4096),
        scheduler.config.get("base_shift", 0.5),
        scheduler.config.get("max_shift", 1.15),
    )
    timesteps, num_inference_steps = retrieve_timesteps_fn(
        scheduler,
        num_inference_steps,
        device,
        sigmas=sigmas,
        mu=mu,
    )
    num_warmup_steps = max(len(timesteps) - num_inference_steps * scheduler.order, 0)
    return timesteps, num_inference_steps, num_warmup_steps


def prepare_timesteps_and_guidance(
    pipeline,
    num_inference_steps: int,
    device,
    latents: torch.Tensor,
    guidance_scale: float | None,
    sigmas: list[float] | None = None,
    calculate_shift_fn=None,
    retrieve_timesteps_fn=None,
) -> tuple:
    """Prepare timesteps and guidance for pipeline.

    Args:
        pipeline: Pipeline instance
        num_inference_steps: Number of inference steps
        device: Device to use
        latents: Latents tensor
        guidance_scale: Guidance scale value
        sigmas: Optional sigmas
        calculate_shift_fn: Calculate shift function
        retrieve_timesteps_fn: Retrieve timesteps function

    Returns:
        Tuple of (timesteps, num_inference_steps, num_warmup_steps, guidance)
    """
    timesteps, num_inference_steps, num_warmup_steps = prepare_timesteps(
        pipeline.scheduler,
        num_inference_steps,
        device,
        latents,
        sigmas,
        calculate_shift_fn=calculate_shift_fn,
        retrieve_timesteps_fn=retrieve_timesteps_fn,
    )
    pipeline._num_timesteps = len(timesteps)

    guidance = handle_guidance(pipeline.transformer, guidance_scale, latents, device)

    return timesteps, num_inference_steps, num_warmup_steps, guidance


def handle_guidance(
    transformer,
    guidance_scale: float | None,
    latents: torch.Tensor,
    device,
) -> torch.Tensor | None:
    """Handle guidance scale for transformer."""
    if transformer.config.guidance_embeds and guidance_scale is None:
        raise ValueError("guidance_scale is required for guidance-distilled model.")
    elif transformer.config.guidance_embeds:
        guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
        guidance = guidance.expand(latents.shape[0])
        return guidance
    elif not transformer.config.guidance_embeds and guidance_scale is not None:
        logger.warning(
            f"guidance_scale is passed as {guidance_scale}, but ignored since the model is not guidance-distilled."  # noqa: G004
        )
        return None
    elif not transformer.config.guidance_embeds and guidance_scale is None:
        return None
    return None


def finalize_output(
    pipeline,
    latents: torch.Tensor,
    height: int,
    width: int,
    output_type: str = "pil",
    return_dict: bool = True,
    output_class=None,
):
    """Finalize pipeline output."""
    image = decode_latents_to_image(pipeline, latents, height, width, output_type)
    pipeline.maybe_free_model_hooks()

    if not return_dict:
        return (image,)
    return output_class(images=image)


# =============================================================================
# Core Pipeline Runner
# =============================================================================


def run_qwenimage_pipeline_core(
    pipeline,
    config: PipelineConfig,
    # All pipeline parameters
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
    image=None,
    # Required callbacks for pipeline-specific logic
    prepare_dimensions_fn: Callable | None = None,
    prepare_latents_data_fn: Callable | None = None,
    get_encode_image_fn: Callable | None = None,
    calculate_shift_fn=None,
    retrieve_timesteps_fn=None,
    output_class=None,
):
    """Core pipeline runner - handles common logic for all QwenImage pipelines.

    Args:
        pipeline: Pipeline instance
        config: PipelineConfig describing pipeline characteristics
        prepare_dimensions_fn: Callable to prepare height/width, returns (height, width)
        prepare_latents_data_fn: Callable to prepare latents, returns (latents, img_shapes, extra_data)
        get_encode_image_fn: Callable to get image for encode_prompt
        ... (other args same as pipeline __call__)

    Returns:
        Final output (image tuple or output class instance)
    """
    # Handle mutable default argument
    if callback_on_step_end_tensor_inputs is None:
        callback_on_step_end_tensor_inputs = ["latents"]

    # 1. Prepare dimensions (pipeline-specific)
    if prepare_dimensions_fn is not None:
        height, width = prepare_dimensions_fn(pipeline, image, height, width)

    # 2. Check inputs
    pipeline.check_inputs(
        prompt,
        height,
        width,
        negative_prompt=negative_prompt,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        prompt_embeds_mask=prompt_embeds_mask,
        negative_prompt_embeds_mask=negative_prompt_embeds_mask,
        callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
        max_sequence_length=max_sequence_length,
    )

    # 3. Initialize state
    init_pipeline_state(pipeline, guidance_scale, attention_kwargs)

    # 4. Calculate batch size and device
    batch_size = get_batch_size(prompt, prompt_embeds)

    # Check batch size limit
    if config.batch_size_limit is not None and batch_size > config.batch_size_limit:
        raise ValueError(
            f"Pipeline currently only supports batch_size={config.batch_size_limit}, "
            f"but received batch_size={batch_size}."
        )

    device = pipeline._execution_device

    # 5. Handle negative prompt
    _, do_true_cfg = get_cfg_setup(true_cfg_scale, negative_prompt, negative_prompt_embeds, negative_prompt_embeds_mask)

    # 6. Get image for encode_prompt (pipeline-specific)
    encode_image = None
    if get_encode_image_fn is not None:
        encode_image = get_encode_image_fn(pipeline, image)

    # 7. Encode prompts
    (prompt_embeds, prompt_embeds_mask, negative_prompt_embeds, negative_prompt_embeds_mask, local_rank) = (
        encode_prompts_with_cfg(
            pipeline,
            cfg_parallel_size,
            do_true_cfg,
            device,
            num_images_per_prompt,
            max_sequence_length,
            prompt=prompt,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_prompt_embeds_mask=negative_prompt_embeds_mask,
            image=encode_image if not config.needs_multiple_images else None,
            condition_images=encode_image if config.needs_multiple_images else None,
        )
    )

    # 8. Prepare latents and img_shapes (pipeline-specific)
    if prepare_latents_data_fn is not None:
        latents, img_shapes, extra_data = prepare_latents_data_fn(
            pipeline,
            image,
            batch_size,
            num_images_per_prompt,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )
    else:
        # Default for QwenImagePipeline
        num_channels_latents = pipeline.transformer.config.in_channels // 4
        latents = pipeline.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )
        img_shapes = [
            [(1, height // pipeline.vae_scale_factor // 2, width // pipeline.vae_scale_factor // 2)]
        ] * batch_size
        extra_data = {}

    # 9. Prepare timesteps and guidance
    timesteps, num_inference_steps, num_warmup_steps, guidance = prepare_timesteps_and_guidance(
        pipeline,
        num_inference_steps,
        device,
        latents,
        guidance_scale,
        sigmas,
        calculate_shift_fn=calculate_shift_fn,
        retrieve_timesteps_fn=retrieve_timesteps_fn,
    )

    # 10. Get preprocess/postprocess functions for edit pipelines
    latent_preprocess = None
    noise_postprocess = None
    if config.needs_edit_preprocess:
        image_latents = extra_data.get("image_latents")
        latent_preprocess = lambda lats: torch.cat([lats, image_latents], dim=1) if image_latents is not None else lats
        noise_postprocess = lambda pred, lats: pred[:, : lats.size(1)]

    # 11. Run denoising loop
    latents = run_denoising_loop(
        pipeline,
        timesteps=timesteps,
        num_inference_steps=num_inference_steps,
        num_warmup_steps=num_warmup_steps,
        latents=latents,
        guidance=guidance,
        prompt_embeds=prompt_embeds,
        prompt_embeds_mask=prompt_embeds_mask,
        negative_prompt_embeds=negative_prompt_embeds,
        negative_prompt_embeds_mask=negative_prompt_embeds_mask,
        img_shapes=img_shapes,
        do_true_cfg=do_true_cfg,
        true_cfg_scale=true_cfg_scale,
        cfg_parallel_size=cfg_parallel_size,
        local_rank=local_rank,
        latent_model_input_preprocess=latent_preprocess,
        noise_pred_postprocess=noise_postprocess,
        callback_on_step_end=callback_on_step_end,
        callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
        attention_kwargs=attention_kwargs,
    )

    # 12. Finalize output
    return finalize_output(pipeline, latents, height, width, output_type, return_dict, output_class=output_class)


def decode_latents_to_image(
    pipeline,
    latents: torch.Tensor,
    height: int,
    width: int,
    output_type: str = "pil",
) -> torch.Tensor | Any:
    """Decode latents to image using VAE."""
    if output_type == "latent":
        return latents

    latents = pipeline._unpack_latents(latents, height, width, pipeline.vae_scale_factor)
    latents = latents.to(pipeline.vae.dtype)

    latents_mean = (
        torch.tensor(pipeline.vae.config.latents_mean)
        .view(1, pipeline.vae.config.z_dim, 1, 1, 1)
        .to(latents.device, latents.dtype)
    )
    latents_std = 1.0 / torch.tensor(pipeline.vae.config.latents_std).view(1, pipeline.vae.config.z_dim, 1, 1, 1).to(
        latents.device, latents.dtype
    )
    latents = latents / latents_std + latents_mean
    image = pipeline.vae.decode(latents, return_dict=False)[0][:, :, 0]
    image = pipeline.image_processor.postprocess(image, output_type=output_type)

    return image


# =============================================================================
# Denoising Loop Helpers
# =============================================================================


def _run_single_gpu_step(
    pipeline,
    latent_model_input,
    timestep,
    guidance,
    prompt_embeds,
    prompt_embeds_mask,
    negative_prompt_embeds,
    negative_prompt_embeds_mask,
    img_shapes,
    attn_kwargs,
    do_true_cfg,
    true_cfg_scale,
    noise_pred_postprocess,
    latents,
):
    """Run single GPU denoising step."""
    with pipeline.transformer.cache_context("cond"):
        noise_pred = pipeline.transformer(
            hidden_states=latent_model_input,
            timestep=timestep / 1000,
            guidance=guidance,
            encoder_hidden_states_mask=prompt_embeds_mask,
            encoder_hidden_states=prompt_embeds,
            img_shapes=img_shapes,
            attention_kwargs=attn_kwargs,
            return_dict=False,
        )[0]

    if noise_pred_postprocess is not None:
        noise_pred = noise_pred_postprocess(noise_pred, latents)

    if do_true_cfg:
        with pipeline.transformer.cache_context("uncond"):
            neg_noise_pred = pipeline.transformer(
                hidden_states=latent_model_input,
                timestep=timestep / 1000,
                guidance=guidance,
                encoder_hidden_states_mask=negative_prompt_embeds_mask,
                encoder_hidden_states=negative_prompt_embeds,
                img_shapes=img_shapes,
                attention_kwargs=attn_kwargs,
                return_dict=False,
            )[0]

        if noise_pred_postprocess is not None:
            neg_noise_pred = noise_pred_postprocess(neg_noise_pred, latents)

        noise_pred = apply_cfg(noise_pred, neg_noise_pred, true_cfg_scale)

    return noise_pred


def _run_parallel_cfg_step(
    pipeline,
    latent_model_input,
    timestep,
    guidance,
    prompt_embeds,
    prompt_embeds_mask,
    img_shapes,
    attn_kwargs,
    true_cfg_scale,
    noise_pred_postprocess,
    latents,
    local_rank,
):
    """Run parallel CFG denoising step."""
    cache_index = "cond" if local_rank == 0 else "uncond"
    with pipeline.transformer.cache_context(cache_index):
        noise_pred = pipeline.transformer(
            hidden_states=latent_model_input,
            timestep=timestep / 1000,
            guidance=guidance,
            encoder_hidden_states_mask=prompt_embeds_mask,
            encoder_hidden_states=prompt_embeds,
            img_shapes=img_shapes,
            attention_kwargs=attn_kwargs,
            return_dict=False,
        )[0]

        if noise_pred_postprocess is not None:
            noise_pred = noise_pred_postprocess(noise_pred, latents)
        torch.cuda.synchronize()

    return apply_cfg_parallel(noise_pred, true_cfg_scale, get_cfg_group())


def _handle_callback(
    callback_on_step_end,
    callback_tensor_inputs,
    pipeline,
    i,
    t,
    latents,
    prompt_embeds,
):
    """Handle step end callback."""
    if callback_on_step_end is None:
        return latents, prompt_embeds

    callback_kwargs = {k: locals()[k] for k in callback_tensor_inputs}
    callback_outputs = callback_on_step_end(pipeline, i, t, callback_kwargs)
    latents = callback_outputs.pop("latents", latents)
    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
    return latents, prompt_embeds


def _fix_dtype(latents, latents_dtype):
    """Fix dtype mismatch on MPS backend."""
    if latents.dtype != latents_dtype and torch.backends.mps.is_available():
        return latents.to(latents_dtype)
    return latents


def run_denoising_loop(
    pipeline,
    timesteps,
    num_inference_steps: int,
    num_warmup_steps: int,
    latents: torch.Tensor,
    guidance: torch.Tensor | None,
    prompt_embeds: torch.Tensor,
    prompt_embeds_mask: torch.Tensor,
    negative_prompt_embeds: torch.Tensor | None,
    negative_prompt_embeds_mask: torch.Tensor | None,
    img_shapes: list,
    do_true_cfg: bool,
    true_cfg_scale: float,
    cfg_parallel_size: int,
    local_rank: int = 0,
    latent_model_input_preprocess: Callable | None = None,
    noise_pred_postprocess: Callable | None = None,
    callback_on_step_end: Callable | None = None,
    callback_on_step_end_tensor_inputs: list[str] | None = None,
    attention_kwargs: dict[str, Any] | None = None,
):
    """Run the denoising loop."""

    callback_inputs = (
        callback_on_step_end_tensor_inputs if callback_on_step_end_tensor_inputs is not None else ["latents"]
    )
    pipeline.scheduler.set_begin_index(0)

    with pipeline.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            if pipeline.interrupt:
                continue

            pipeline._current_timestep = t

            # Prepare latent model input
            latent_model_input = latents
            if latent_model_input_preprocess is not None:
                latent_model_input = latent_model_input_preprocess(latents)

            # Broadcast timestep
            timestep = t.expand(latents.shape[0]).to(latents.dtype)

            # Get attention kwargs
            attn_kwargs = attention_kwargs if attention_kwargs is not None else {}

            # Run denoising step
            if cfg_parallel_size == 1:
                noise_pred = _run_single_gpu_step(
                    pipeline,
                    latent_model_input,
                    timestep,
                    guidance,
                    prompt_embeds,
                    prompt_embeds_mask,
                    negative_prompt_embeds,
                    negative_prompt_embeds_mask,
                    img_shapes,
                    attn_kwargs,
                    do_true_cfg,
                    true_cfg_scale,
                    noise_pred_postprocess,
                    latents,
                )
            else:
                noise_pred = _run_parallel_cfg_step(
                    pipeline,
                    latent_model_input,
                    timestep,
                    guidance,
                    prompt_embeds,
                    prompt_embeds_mask,
                    img_shapes,
                    attn_kwargs,
                    true_cfg_scale,
                    noise_pred_postprocess,
                    latents,
                    local_rank,
                )

            # Scheduler step
            latents_dtype = latents.dtype
            latents = pipeline.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
            latents = _fix_dtype(latents, latents_dtype)

            # Callback
            latents, prompt_embeds = _handle_callback(
                callback_on_step_end, callback_inputs, pipeline, i, t, latents, prompt_embeds
            )

            # Update progress bar
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % pipeline.scheduler.order == 0):
                progress_bar.update()

            if XLA_AVAILABLE:
                xm.mark_step()

    pipeline._current_timestep = None
    return latents
