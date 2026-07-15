import torch
import torch.distributed as dist
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.autoencoders import autoencoder_kl_hunyuan_video
from diffusers.models.autoencoders.autoencoder_kl_hunyuan_video import (
    HunyuanVideoDecoder3D,
    HunyuanVideoEncoder3D,
)
from diffusers.models.autoencoders.vae import DecoderOutput, DiagonalGaussianDistribution
from diffusers.models.modeling_outputs import AutoencoderKLOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils.accelerate_utils import apply_forward_hook
from x_base.utils.infer_info import infer_info
from x_base.vae_parallelism.utils import enable_lightning
from x_base.vae_parallelism.vae_mgr import (
    blend_t,
    parallel_spatial_tiled_decode,
    temporal_tiled_decode,
    tiled_decode_parallel,
    tiled_encode_parallel,
)


class AutoencoderKLHunyuanVideo(ModelMixin, ConfigMixin):
    r"""
    A VAE model with KL loss for encoding videos into latents and decoding latent representations into videos.
    Introduced in [HunyuanVideo](https://huggingface.co/papers/2412.03603).

    This model inherits from [`ModelMixin`]. Check the superclass documentation for it's generic methods implemented
    for all models (such as downloading or saving).
    """

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        latent_channels: int = 16,
        down_block_types: tuple[str, ...] = (
            "HunyuanVideoDownBlock3D",
            "HunyuanVideoDownBlock3D",
            "HunyuanVideoDownBlock3D",
            "HunyuanVideoDownBlock3D",
        ),
        up_block_types: tuple[str, ...] = (
            "HunyuanVideoUpBlock3D",
            "HunyuanVideoUpBlock3D",
            "HunyuanVideoUpBlock3D",
            "HunyuanVideoUpBlock3D",
        ),
        block_out_channels: tuple[int] = (128, 256, 512, 512),
        layers_per_block: int = 2,
        act_fn: str = "silu",
        norm_num_groups: int = 32,
        scaling_factor: float = 0.476986,
        spatial_compression_ratio: int = 8,
        temporal_compression_ratio: int = 4,
        mid_block_add_attention: bool = True,
    ) -> None:
        super().__init__()

        self.time_compression_ratio = temporal_compression_ratio

        self.lightning = False
        self.encoder = HunyuanVideoEncoder3D(
            in_channels=in_channels,
            out_channels=latent_channels,
            down_block_types=down_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            norm_num_groups=norm_num_groups,
            act_fn=act_fn,
            double_z=True,
            mid_block_add_attention=mid_block_add_attention,
            temporal_compression_ratio=temporal_compression_ratio,
            spatial_compression_ratio=spatial_compression_ratio,
        )

        self.decoder = HunyuanVideoDecoder3D(
            in_channels=latent_channels,
            out_channels=out_channels,
            up_block_types=up_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            norm_num_groups=norm_num_groups,
            act_fn=act_fn,
            time_compression_ratio=temporal_compression_ratio,
            spatial_compression_ratio=spatial_compression_ratio,
            mid_block_add_attention=mid_block_add_attention,
        )

        self.quant_conv = nn.Conv3d(2 * latent_channels, 2 * latent_channels, kernel_size=1)
        self.post_quant_conv = nn.Conv3d(latent_channels, latent_channels, kernel_size=1)

        self.spatial_compression_ratio = spatial_compression_ratio
        self.temporal_compression_ratio = temporal_compression_ratio

        # When decoding a batch of video latents at a time, one can save memory by slicing across the batch dimension
        # to perform decoding of a single video latent at a time.
        self.use_slicing = False

        # When decoding spatially large video latents, the memory requirement is very high. By breaking the video latent
        # frames spatially into smaller tiles and performing multiple forward passes for decoding, and then blending the
        # intermediate tiles together, the memory requirement can be lowered.
        self.use_tiling = False

        # When decoding temporally long video latents, the memory requirement is very high. By decoding latent frames
        # at a fixed frame batch size (based on `self.tile_sample_min_num_frames`), the memory requirement can be lowered.  # noqa: E501
        self.use_framewise_encoding = True
        self.use_framewise_decoding = True

        # The minimal tile height and width for spatial tiling to be used
        self.tile_sample_min_height = 256
        self.tile_sample_min_width = 256
        self.tile_sample_min_num_frames = 16

        # The minimal distance between two spatial tiles
        self.tile_sample_stride_height = 192
        self.tile_sample_stride_width = 192
        self.tile_sample_stride_num_frames = 12

        self.max_shape_encode = (1, 32, 32, 32, 32)
        self.tile_sample_min_size = min(self.tile_sample_min_height, self.tile_sample_min_width)
        self.tile_latent_min_size = int(self.tile_sample_min_size / (2 ** (len(self.config.block_out_channels) - 1)))
        self.tile_latent_min_height = int(
            self.tile_sample_min_height / (2 ** (len(self.config.block_out_channels) - 1))
        )
        self.tile_latent_min_width = int(self.tile_sample_min_width / (2 ** (len(self.config.block_out_channels) - 1)))

        self.tile_sample_min_tsize = 64
        self.tile_latent_min_tsize = self.tile_sample_min_tsize // temporal_compression_ratio
        self.tile_overlap_factor = 0.25

    def enable_tiling(
        self,
        tile_sample_min_height: int | None = None,
        tile_sample_min_width: int | None = None,
        tile_sample_min_num_frames: int | None = None,
        tile_sample_stride_height: float | None = None,
        tile_sample_stride_width: float | None = None,
        tile_sample_stride_num_frames: float | None = None,
    ) -> None:
        r"""
        Enable tiled VAE decoding. When this option is enabled, the VAE will split the input tensor into tiles to
        compute decoding and encoding in several steps. This is useful for saving a large amount of memory and to allow
        processing larger images.

        Args:
            tile_sample_min_height (`int`, *optional*):
                The minimum height required for a sample to be separated into tiles across the height dimension.
            tile_sample_min_width (`int`, *optional*):
                The minimum width required for a sample to be separated into tiles across the width dimension.
            tile_sample_min_num_frames (`int`, *optional*):
                The minimum number of frames required for a sample to be separated into tiles across the frame
                dimension.
            tile_sample_stride_height (`int`, *optional*):
                The minimum amount of overlap between two consecutive vertical tiles. This is to ensure that there are
                no tiling artifacts produced across the height dimension.
            tile_sample_stride_width (`int`, *optional*):
                The stride between two consecutive horizontal tiles. This is to ensure that there are no tiling
                artifacts produced across the width dimension.
            tile_sample_stride_num_frames (`int`, *optional*):
                The stride between two consecutive frame tiles. This is to ensure that there are no tiling artifacts
                produced across the frame dimension.
        """
        self.use_tiling = True
        self.tile_sample_min_height = tile_sample_min_height or self.tile_sample_min_height
        self.tile_sample_min_width = tile_sample_min_width or self.tile_sample_min_width
        self.tile_sample_min_num_frames = tile_sample_min_num_frames or self.tile_sample_min_num_frames
        self.tile_sample_stride_height = tile_sample_stride_height or self.tile_sample_stride_height
        self.tile_sample_stride_width = tile_sample_stride_width or self.tile_sample_stride_width
        self.tile_sample_stride_num_frames = tile_sample_stride_num_frames or self.tile_sample_stride_num_frames

    def disable_tiling(self) -> None:
        r"""
        Disable tiled VAE decoding. If `enable_tiling` was previously enabled, this method will go back to computing
        decoding in one step.
        """
        self.use_tiling = False

    def enable_slicing(self) -> None:
        r"""
        Enable sliced VAE decoding. When this option is enabled, the VAE will split the input tensor in slices to
        compute decoding in several steps. This is useful to save some memory and allow larger batch sizes.
        """
        self.use_slicing = True

    def disable_slicing(self) -> None:
        r"""
        Disable sliced VAE decoding. If `enable_slicing` was previously enabled, this method will go back to computing
        decoding in one step.
        """
        self.use_slicing = False

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, num_frames, height, width = x.shape

        # Parallel encoding turns out to be slower in practice for image processing,
        # so it's not an efficient option in this case.
        if num_frames != 1:
            if self.use_framewise_encoding and num_frames > self.tile_sample_min_num_frames:
                return self.temporal_tiled_encode(x)

            if self.use_tiling and (width > self.tile_sample_min_width or height > self.tile_sample_min_height):
                return self._tiled_encode_parallel(x)

        x = self.encoder(x)
        enc = self.quant_conv(x)
        return enc

    @apply_forward_hook
    def encode(
        self, x: torch.Tensor, return_dict: bool = True
    ) -> AutoencoderKLOutput | tuple[DiagonalGaussianDistribution]:
        r"""
        Encode a batch of images into latents.

        Args:
            x (`torch.Tensor`): Input batch of images.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return a [`~models.autoencoder_kl.AutoencoderKLOutput`] instead of a plain tuple.

        Returns:
                The latent representations of the encoded videos. If `return_dict` is True, a
                [`~models.autoencoder_kl.AutoencoderKLOutput`] is returned, otherwise a plain `tuple` is returned.
        """
        if self.use_slicing and x.shape[0] > 1:
            encoded_slices = [self._encode(x_slice) for x_slice in x.split(1)]
            h = torch.cat(encoded_slices)
        else:
            h = self._encode(x)

        posterior = DiagonalGaussianDistribution(h)

        if not return_dict:
            return (posterior,)
        return AutoencoderKLOutput(latent_dist=posterior)

    def _decode(self, z: torch.Tensor, return_dict: bool = True) -> DecoderOutput | torch.Tensor:
        batch_size, num_channels, num_frames, height, width = z.shape
        tile_latent_min_height = self.tile_sample_min_height // self.spatial_compression_ratio
        tile_latent_min_width = self.tile_sample_min_width // self.spatial_compression_ratio

        if self.use_tiling and (width > tile_latent_min_width or height > tile_latent_min_height):
            if self.lightning:
                sample_shape = (infer_info.frames, infer_info.height, infer_info.width)
                result = self.temporal_tiled_decode(
                    z,
                    return_dict=return_dict,
                    sample_shape=sample_shape,
                    save_path=infer_info.save_path,
                    use_conv_cache=False,
                )
            else:
                result = self.tiled_decode_parallel(z, return_dict=return_dict, use_conv_cache=False)
            return result

        z = self.post_quant_conv(z)
        dec = self.decoder(z)

        if not return_dict:
            return (dec,)

        return DecoderOutput(sample=dec)

    @apply_forward_hook
    def decode(self, z: torch.Tensor, return_dict: bool = True) -> DecoderOutput | torch.Tensor:
        r"""
        Decode a batch of images.

        Args:
            z (`torch.Tensor`): Input batch of latent vectors.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return a [`~models.vae.DecoderOutput`] instead of a plain tuple.

        Returns:
            [`~models.vae.DecoderOutput`] or `tuple`:
                If return_dict is True, a [`~models.vae.DecoderOutput`] is returned, otherwise a plain `tuple` is
                returned.
        """
        if self.use_slicing and z.shape[0] > 1:
            decoded_slices = [self._decode(z_slice).sample for z_slice in z.split(1)]
            decoded = torch.cat(decoded_slices)
        else:
            decoded = self._decode(z).sample

        if not return_dict:
            return (decoded,)

        return DecoderOutput(sample=decoded)

    def temporal_tiled_encode(self, x: torch.Tensor) -> AutoencoderKLOutput:
        B, C, T, H, W = x.shape

        overlap_size = int(self.tile_sample_min_tsize * (1.0 - self.tile_overlap_factor))
        blend_extent_default = int(self.tile_sample_min_tsize * self.tile_overlap_factor)

        latent_ratio = self.temporal_compression_ratio
        encoded_tiles = []
        last_encoded = None

        for i in range(0, T, overlap_size):
            sample_end = min(i + self.tile_sample_min_tsize + 1, T)
            tile = x[:, :, i:sample_end, :, :]

            if self.use_tiling and (self.tile_sample_min_height < H or self.tile_sample_min_width < W):
                z_tile = self._tiled_encode_parallel(tile)
            else:
                z_tile = self.quant_conv(self.encoder(tile))

            Tz = z_tile.shape[2]
            curr_blend = min(blend_extent_default // latent_ratio, Tz - 1)
            is_first = i == 0
            is_last = sample_end == T

            if not is_first:
                z_tile = z_tile[:, :, 1:, :, :]
                Tz -= 1
                z_tile = blend_t(last_encoded, z_tile, curr_blend)

            if is_last:
                z_chunk = z_tile
            else:
                keep = Tz - curr_blend
                z_chunk = z_tile[:, :, :keep, :, :]

            encoded_tiles.append(z_chunk)
            last_encoded = z_tile[:, :, -curr_blend:, :, :]

        enc = torch.cat(encoded_tiles, dim=2)[:, :, : (T // latent_ratio + 1)]
        return enc

    def _tiled_encode_parallel(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: Disabled for world_size=8, as a different tile partitioning is required.
        # Partition matching also does not give performance gain over synchronous mode
        # due to long communication time.
        if dist.get_world_size() == 8:
            x = self.encoder(x)
            enc = self.quant_conv(x)
            return enc

        batch_size, num_channels, num_frames, height, width = x.shape
        latent_height = height // self.spatial_compression_ratio
        latent_width = width // self.spatial_compression_ratio

        tile_latent_min_height = self.tile_sample_min_height // self.spatial_compression_ratio
        tile_latent_min_width = self.tile_sample_min_width // self.spatial_compression_ratio
        tile_latent_stride_height = self.tile_sample_stride_height // self.spatial_compression_ratio
        tile_latent_stride_width = self.tile_sample_stride_width // self.spatial_compression_ratio

        blend_height = tile_latent_min_height - tile_latent_stride_height
        blend_width = tile_latent_min_width - tile_latent_stride_width
        # Create a grid of tile indices

        num_tiles_height = (height + self.tile_sample_stride_height - 1) // self.tile_sample_stride_height
        num_tiles_width = (width + self.tile_sample_stride_width - 1) // self.tile_sample_stride_width

        enc = self.tiled_encode_parallel(
            x,
            overlap_height=self.tile_sample_stride_height,
            overlap_width=self.tile_sample_stride_width,
            blend_extent_height=blend_height,
            blend_extent_width=blend_width,
            row_limit_height=tile_latent_stride_height,
            row_limit_width=tile_latent_stride_width,
            num_tiles_height=num_tiles_height,
            num_tiles_width=num_tiles_width,
            use_conv_cache=False,
        )
        return enc[:, :, :, :latent_height, :latent_width]

    def forward(
        self,
        sample: torch.Tensor,
        sample_posterior: bool = False,
        return_dict: bool = True,
        generator: torch.Generator | None = None,
    ) -> DecoderOutput | torch.Tensor:
        r"""
        Args:
            sample (`torch.Tensor`): Input sample.
            sample_posterior (`bool`, *optional*, defaults to `False`):
                Whether to sample from the posterior.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`DecoderOutput`] instead of a plain tuple.
        """
        x = sample
        posterior = self.encode(x).latent_dist
        if sample_posterior:
            z = posterior.sample(generator=generator)
        else:
            z = posterior.mode()
        dec = self.decode(z, return_dict=return_dict)
        return dec


AutoencoderKLHunyuanVideo.parallel_spatial_tiled_decode = parallel_spatial_tiled_decode
AutoencoderKLHunyuanVideo.temporal_tiled_decode = temporal_tiled_decode
AutoencoderKLHunyuanVideo.tiled_decode_parallel = tiled_decode_parallel
AutoencoderKLHunyuanVideo.tiled_encode_parallel = tiled_encode_parallel
autoencoder_kl_hunyuan_video.AutoencoderKLHunyuanVideo = AutoencoderKLHunyuanVideo
autoencoder_kl_hunyuan_video.AutoencoderKLHunyuanVideo.enable_lightning = enable_lightning
