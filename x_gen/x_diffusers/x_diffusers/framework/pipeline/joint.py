from collections.abc import Callable
from typing import Any

import torch
from diffusers import WanImageToVideoPipeline
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.image_processor import PipelineImageInput
from diffusers.models import AutoencoderKLWan, WanTransformer3DModel
from diffusers.pipelines.wan.pipeline_output import WanPipelineOutput
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import is_torch_xla_available, logging
from transformers import AutoTokenizer, CLIPImageProcessor, CLIPVisionModel, UMT5EncoderModel

if is_torch_xla_available():
    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class WanImageToVideoPipelineJoint(WanImageToVideoPipeline):
    model_cpu_offload_seq = WanImageToVideoPipeline.model_cpu_offload_seq.replace("->vae", "->transformer_3->vae")
    _optional_components = WanImageToVideoPipeline._optional_components + ["transformer_3"]

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        text_encoder: UMT5EncoderModel,
        transformer: WanTransformer3DModel,
        vae: AutoencoderKLWan,
        scheduler: FlowMatchEulerDiscreteScheduler,
        image_processor: CLIPImageProcessor = None,
        image_encoder: CLIPVisionModel = None,
        transformer_2: WanTransformer3DModel = None,
        transformer_3: WanTransformer3DModel = None,
        boundary_ratio: float | None = None,
        expand_timesteps: bool = False,
    ):
        super().__init__(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            transformer=transformer,
            vae=vae,
            scheduler=scheduler,
            image_processor=image_processor,
            image_encoder=image_encoder,
            transformer_2=transformer_2,
            boundary_ratio=boundary_ratio,
            expand_timesteps=expand_timesteps,
        )

        self.register_modules(transformer_3=transformer_3)

    def calc_frames(self, num_frames):
        if num_frames % self.vae_scale_factor_temporal != 1:
            logger.warning(
                f"`num_frames - 1` has to be divisible by {self.vae_scale_factor_temporal}. Rounding to the nearest number."  # noqa: G004, E501
            )
            num_frames = num_frames // self.vae_scale_factor_temporal * self.vae_scale_factor_temporal + 1
        num_frames = max(num_frames, 1)
        return num_frames

    def calc_bs(self, prompt, prompt_embeds):
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]
        return batch_size

    def calc_boundaryts(self):
        if self.config.boundary_ratio is not None:
            boundary_timestep = self.config.boundary_ratio * self.scheduler.config.num_train_timesteps
        else:
            boundary_timestep = None
        return boundary_timestep

    def switch_model(self, t_s, high_noise, low_noise, small_boundary):
        if t_s in high_noise:
            print(f"Large: Wan2.2 14B high-noise at timestep {t_s}", flush=True)
            current_model = self.transformer
            current_guidance_scale = self._guidance_scale
        elif t_s in low_noise:
            print(f"Large: Wan2.2 14B low-noise at timestep {t_s}", flush=True)
            current_model = self.transformer_2
            current_guidance_scale = self._guidance_scale_2
        elif t_s < small_boundary:
            print(f"Small: Wan2.1 1.3B at timestep {t_s}", flush=True)
            current_model = self.transformer_3
            current_guidance_scale = self._guidance_scale_3
        else:
            print(f"Skip: at timestep {t_s}", flush=True)
            current_model = None
            current_guidance_scale = None
        return current_model, current_guidance_scale

    def calc_latent_input(self, latents, condition, transformer_dtype, t_s, small_boundary):
        if t_s < small_boundary:
            latent_model_input = latents.to(transformer_dtype)
            print(f"Latent channel to 16 at small model stage, timestep {t_s}", flush=True)
        else:
            latent_model_input = torch.cat([latents, condition], dim=1).to(transformer_dtype)
        return latent_model_input

    def calc_video(self, latents, output_type):
        if output_type != "latent":
            latents = latents.to(self.vae.dtype)
            latents_mean = (
                torch.tensor(self.vae.config.latents_mean)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
                latents.device, latents.dtype
            )
            latents = latents / latents_std + latents_mean
            video = self.vae.decode(latents, return_dict=False)[0]
            video = self.video_processor.postprocess_video(video, output_type=output_type)
        else:
            video = latents
        return video

    @torch.no_grad()
    def __call__(
        self,
        image: PipelineImageInput,
        prompt: str | list[str] = None,
        negative_prompt: str | list[str] = None,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        guidance_scale_2: float | None = None,
        guidance_scale_3: float | None = None,
        num_videos_per_prompt: int | None = 1,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.Tensor | None = None,
        prompt_embeds: torch.Tensor | None = None,
        negative_prompt_embeds: torch.Tensor | None = None,
        image_embeds: torch.Tensor | None = None,
        last_image: torch.Tensor | None = None,
        output_type: str | None = "np",
        return_dict: bool = True,
        attention_kwargs: dict[str, Any] | None = None,
        callback_on_step_end: Callable[[int, int, dict], None]
        | PipelineCallback
        | MultiPipelineCallbacks
        | None = None,
        callback_on_step_end_tensor_inputs: list[str] = None,
        max_sequence_length: int = 512,
    ):
        if callback_on_step_end_tensor_inputs is None:
            callback_on_step_end_tensor_inputs = ["latents"]

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            negative_prompt,
            image,
            height,
            width,
            prompt_embeds,
            negative_prompt_embeds,
            image_embeds,
            callback_on_step_end_tensor_inputs,
            guidance_scale_2,
        )

        num_frames = self.calc_frames(num_frames)

        if self.config.boundary_ratio is not None and guidance_scale_2 is None and guidance_scale_3 is None:
            guidance_scale_2 = guidance_scale
            guidance_scale_3 = guidance_scale

        self._guidance_scale = guidance_scale
        self._guidance_scale_2 = guidance_scale_2
        self._guidance_scale_3 = guidance_scale_3
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        device = self._execution_device

        # 2. Define call parameters

        batch_size = self.calc_bs(prompt, prompt_embeds)

        # 3. Encode input prompt
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            max_sequence_length=max_sequence_length,
            device=device,
        )

        # Encode image embedding
        transformer_dtype = self.transformer.dtype
        prompt_embeds = prompt_embeds.to(transformer_dtype)
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(transformer_dtype)

        # 4. Prepare timesteps
        TOT_DISTILL_STEPS = num_inference_steps  #
        small_steps = 4
        num_inference_steps = TOT_DISTILL_STEPS * (small_steps + 1)  # tot_distill_steps * (small_steps+1)
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.vae.config.z_dim
        image = self.video_processor.preprocess(image, height=height, width=width).to(device, dtype=torch.float32)
        if last_image is not None:
            last_image = self.video_processor.preprocess(last_image, height=height, width=width).to(
                device, dtype=torch.float32
            )

        latents_outputs = self.prepare_latents(
            image,
            batch_size * num_videos_per_prompt,
            num_channels_latents,
            height,
            width,
            num_frames,
            torch.float32,
            device,
            generator,
            latents,
            last_image,
        )

        latents, condition = latents_outputs

        # 6. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)

        boundary_timestep = self.calc_boundaryts()

        distill_timesteps = timesteps[:: (small_steps + 1)]
        distill_steps = max(TOT_DISTILL_STEPS - 3, 5)
        distill_timesteps_trunc = distill_timesteps[:distill_steps]
        high_noise = distill_timesteps_trunc[distill_timesteps_trunc >= boundary_timestep]
        low_mode = 0
        if low_mode == 0:
            low_noise = distill_timesteps_trunc[distill_timesteps_trunc < boundary_timestep]
        else:
            low_len = len(distill_timesteps) - len(high_noise)
            low_noise = distill_timesteps[-low_len:]
        small_boundary = distill_timesteps[-1]
        small_noise = timesteps[timesteps < small_boundary]

        with self.progress_bar(total=distill_steps + len(small_noise)) as progress_bar:
            for i, t in enumerate(timesteps):
                self._current_timestep = t

                current_model, current_guidance_scale = self.switch_model(
                    t.item(), high_noise, low_noise, small_boundary
                )
                if current_model is None:
                    latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]  # noqa: F821
                    continue

                latent_model_input = self.calc_latent_input(
                    latents, condition, transformer_dtype, t.item(), small_boundary
                )

                timestep = t.expand(latents.shape[0])

                with current_model.cache_context("cond"):
                    noise_pred = current_model(
                        hidden_states=latent_model_input,
                        timestep=timestep,
                        encoder_hidden_states=prompt_embeds,
                        encoder_hidden_states_image=image_embeds,
                        attention_kwargs=attention_kwargs,
                        return_dict=False,
                    )[0]

                if self.do_classifier_free_guidance:
                    with current_model.cache_context("uncond"):
                        noise_uncond = current_model(
                            hidden_states=latent_model_input,
                            timestep=timestep,
                            encoder_hidden_states=negative_prompt_embeds,
                            encoder_hidden_states_image=image_embeds,
                            attention_kwargs=attention_kwargs,
                            return_dict=False,
                        )[0]
                        noise_pred = noise_uncond + current_guidance_scale * (noise_pred - noise_uncond)

                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        self._current_timestep = None

        video = self.calc_video(latents, output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (video,)

        return WanPipelineOutput(frames=video)
