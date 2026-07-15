import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import ConfigMixin, ModelMixin
from diffusers.configuration_utils import register_to_config
from diffusers.loaders import FromOriginalModelMixin
from diffusers.models.autoencoders import autoencoder_kl_cogvideox
from diffusers.models.autoencoders.autoencoder_kl_cogvideox import (
    CogVideoXDecoder3D,
    CogVideoXEncoder3D,
    CogVideoXSafeConv3d,
    DecoderOutput,
)
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from diffusers.models.modeling_outputs import AutoencoderKLOutput
from diffusers.utils.accelerate_utils import apply_forward_hook
from x_base.utils.infer_info import infer_info
from x_base.vae_parallelism.vae_mgr import (
    parallel_spatial_tiled_decode,
    temporal_tiled_decode,
    tiled_decode_parallel,
    tiled_encode_parallel,
)


class AscendCogVideoXCausalConv3d(nn.Module):
    r"""A 3D causal convolution layer that pads the input tensor to ensure causality in CogVideoX Model.

    Args:
        in_channels (`int`): Number of channels in the input tensor.
        out_channels (`int`): Number of output channels produced by the convolution.
        kernel_size (`int` or `Tuple[int, int, int]`): Kernel size of the convolutional kernel.
        stride (`int`, defaults to `1`): Stride of the convolution.
        dilation (`int`, defaults to `1`): Dilation rate of the convolution.
        pad_mode (`str`, defaults to `"constant"`): Padding mode.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int, int],
        stride: int = 1,
        dilation: int = 1,
        pad_mode: str = "constant",
    ):
        super().__init__()

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 3

        time_kernel_size, height_kernel_size, width_kernel_size = kernel_size

        self.pad_mode = pad_mode
        time_pad = dilation * (time_kernel_size - 1) + (1 - stride)
        height_pad = height_kernel_size // 2
        width_pad = width_kernel_size // 2

        self.height_pad = height_pad
        self.width_pad = width_pad
        self.time_pad = time_pad
        self.time_causal_padding = (width_pad, width_pad, height_pad, height_pad, time_pad, 0)

        self.temporal_dim = 2
        self.time_kernel_size = time_kernel_size

        stride = (stride, 1, 1)
        dilation = (dilation, 1, 1)
        self.conv = CogVideoXSafeConv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
        )

    def fake_context_parallel_forward(
        self, inputs: torch.Tensor, conv_cache: torch.Tensor | None = None
    ) -> torch.Tensor:
        kernel_size = self.time_kernel_size
        if kernel_size > 1:
            cached_inputs = [conv_cache] if conv_cache is not None else [inputs[:, :, :1]] * (kernel_size - 1)
            inputs = torch.cat(cached_inputs + [inputs], dim=2)
        return inputs

    def forward(self, inputs: torch.Tensor, conv_cache: torch.Tensor | None = None) -> torch.Tensor:
        inputs = self.fake_context_parallel_forward(inputs, conv_cache)
        conv_cache = inputs[:, :, -self.time_kernel_size + 1 :].clone()

        padding_2d = (self.width_pad, self.width_pad, self.height_pad, self.height_pad)
        inputs = F.pad(inputs, padding_2d, mode="constant", value=0)

        output = self.conv(inputs)
        return output, conv_cache


class AscendAutoencoderKLCogVideoX(ModelMixin, ConfigMixin, FromOriginalModelMixin):
    _supports_gradient_checkpointing = True
    _no_split_modules = ["CogVideoXResnetBlock3D"]

    @register_to_config
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        down_block_types: tuple[str] = (
            "CogVideoXDownBlock3D",
            "CogVideoXDownBlock3D",
            "CogVideoXDownBlock3D",
            "CogVideoXDownBlock3D",
        ),
        up_block_types: tuple[str] = (
            "CogVideoXUpBlock3D",
            "CogVideoXUpBlock3D",
            "CogVideoXUpBlock3D",
            "CogVideoXUpBlock3D",
        ),
        block_out_channels: tuple[int] = (128, 256, 256, 512),
        latent_channels: int = 16,
        layers_per_block: int = 3,
        act_fn: str = "silu",
        norm_eps: float = 1e-6,
        norm_num_groups: int = 32,
        temporal_compression_ratio: float = 4,
        sample_height: int = 480,
        sample_width: int = 720,
        scaling_factor: float = 1.15258426,
        shift_factor: float | None = None,
        latents_mean: tuple[float] | None = None,
        latents_std: tuple[float] | None = None,
        force_upcast: float = True,
        use_quant_conv: bool = False,
        use_post_quant_conv: bool = False,
    ):
        super().__init__()

        self.lightning = False
        self.encoder = CogVideoXEncoder3D(
            in_channels=in_channels,
            out_channels=latent_channels,
            down_block_types=down_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            act_fn=act_fn,
            norm_eps=norm_eps,
            norm_num_groups=norm_num_groups,
            temporal_compression_ratio=temporal_compression_ratio,
        )
        self.decoder = CogVideoXDecoder3D(
            in_channels=latent_channels,
            out_channels=out_channels,
            up_block_types=up_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            act_fn=act_fn,
            norm_eps=norm_eps,
            norm_num_groups=norm_num_groups,
            temporal_compression_ratio=temporal_compression_ratio,
        )
        self.quant_conv = CogVideoXSafeConv3d(2 * out_channels, 2 * out_channels, 1) if use_quant_conv else None
        self.post_quant_conv = CogVideoXSafeConv3d(out_channels, out_channels, 1) if use_post_quant_conv else None

        self.use_slicing = False
        self.use_tiling = False

        # Can be increased to decode more latent frames at once, but comes at a reasonable memory cost and it is not
        # recommended because the temporal parts of the VAE, here, are tricky to understand.
        # If you decode X latent frames together, the number of output frames is:
        #     (X + (2 conv cache) + (2 time upscale_1) + (4 time upscale_2) - (2 causal conv downscale)) => X + 6 frames
        #
        # It has been implemented this way so as to not have "magic values" in the code base that would be hard to explain. Note that  # noqa: E501
        # setting it to anything other than 2 would give poor results because the VAE hasn't been trained to be adaptive with different  # noqa: E501
        # number of temporal frames.
        self.num_latent_frames_batch_size = 2
        self.num_sample_frames_batch_size = 8

        # 5B
        # We make the minimum height and width of sample for tiling half that of the generally supported 5B
        self.tile_sample_min_height = sample_height // 3
        self.tile_sample_min_width = sample_width // 5
        self.tile_sample_min_size = 128
        self.tile_latent_min_size = int(self.tile_sample_min_size / (2 ** (len(self.config.block_out_channels) - 1)))

        self.max_shape_encode = (1, 32, 20, 20, 18)
        self.max_shape_decode = (1, 3, 49, 160, 144)  # Maximum shape for padding
        self.tile_latent_min_height = int(
            self.tile_sample_min_height / (2 ** (len(self.config.block_out_channels) - 1))
        )
        self.tile_latent_min_width = int(self.tile_sample_min_width / (2 ** (len(self.config.block_out_channels) - 1)))

        # These are experimental overlap factors that were chosen based on experimentation and seem to work best for
        # 720x480 (WxH) resolution. The above resolution is the strongly recommended generation resolution in CogVideoX
        # and so the tiling implementation has only been tested on those specific resolutions.
        # 要能被max_shape_encode整除
        self.tile_overlap_factor_height = 1 / 5
        self.tile_overlap_factor_width = 1 / 6

        self.tile_sample_min_tsize = 64
        self.tile_latent_min_tsize = self.tile_sample_min_tsize // 4
        self.tile_overlap_factor = 0.25

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, (CogVideoXEncoder3D, CogVideoXDecoder3D)):  # noqa: UP038
            module.gradient_checkpointing = value

    def enable_tiling(
        self,
        tile_sample_min_height: int | None = None,
        tile_sample_min_width: int | None = None,
        tile_overlap_factor_height: float | None = None,
        tile_overlap_factor_width: float | None = None,
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
            tile_overlap_factor_height (`int`, *optional*):
                The minimum amount of overlap between two consecutive vertical tiles. This is to ensure that there are
                no tiling artifacts produced across the height dimension. Must be between 0 and 1. Setting a higher
                value might cause more tiles to be processed leading to slow down of the decoding process.
            tile_overlap_factor_width (`int`, *optional*):
                The minimum amount of overlap between two consecutive horizontal tiles. This is to ensure that there
                are no tiling artifacts produced across the width dimension. Must be between 0 and 1. Setting a higher
                value might cause more tiles to be processed leading to slow down of the decoding process.
        """
        self.use_tiling = True
        self.tile_sample_min_height = tile_sample_min_height or self.tile_sample_min_height
        self.tile_sample_min_width = tile_sample_min_width or self.tile_sample_min_width
        self.tile_latent_min_height = int(
            self.tile_sample_min_height / (2 ** (len(self.config.block_out_channels) - 1))
        )
        self.tile_latent_min_width = int(self.tile_sample_min_width / (2 ** (len(self.config.block_out_channels) - 1)))
        self.tile_overlap_factor_height = tile_overlap_factor_height or self.tile_overlap_factor_height
        self.tile_overlap_factor_width = tile_overlap_factor_width or self.tile_overlap_factor_width

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
        if self.use_tiling and (width > self.tile_sample_min_width or height > self.tile_sample_min_height):
            result = self._tiled_encode_parallel(self, x)
            return result
        frame_batch_size = self.num_sample_frames_batch_size
        # Note: We expect the number of frames to be either `1` or `frame_batch_size * k` or `frame_batch_size * k + 1` for some k.  # noqa: E501
        # As the extra single frame is handled inside the loop, it is not required to round up here.
        num_batches = max(num_frames // frame_batch_size, 1)
        conv_cache = None
        enc = []

        for i in range(num_batches):
            remaining_frames = num_frames % frame_batch_size
            start_frame = frame_batch_size * i + (0 if i == 0 else remaining_frames)
            end_frame = frame_batch_size * (i + 1) + remaining_frames
            x_intermediate = x[:, :, start_frame:end_frame]
            x_intermediate, conv_cache = self.encoder(x_intermediate, conv_cache=conv_cache)
            if self.quant_conv is not None:
                x_intermediate = self.quant_conv(x_intermediate)
            enc.append(x_intermediate)

        enc = torch.cat(enc, dim=2)
        return enc

    @apply_forward_hook
    def encode(
        self, x: torch.Tensor, return_dict: bool = True
    ) -> AutoencoderKLOutput | tuple[DiagonalGaussianDistribution]:
        """
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
        sample_shape = (infer_info.frames, infer_info.height, infer_info.width)
        if self.use_tiling and (width > self.tile_latent_min_width or height > self.tile_latent_min_height):
            if self.lightning:
                result = self.temporal_tiled_decode(
                    z, return_dict=return_dict, sample_shape=sample_shape, save_path=infer_info.save_path
                )
            else:
                result = self.tiled_decode_parallel(z, return_dict=return_dict)
            return result
        frame_batch_size = self.num_latent_frames_batch_size
        num_batches = max(num_frames // frame_batch_size, 1)
        conv_cache = None
        dec = []

        for i in range(num_batches):
            remaining_frames = num_frames % frame_batch_size
            start_frame = frame_batch_size * i + (0 if i == 0 else remaining_frames)
            end_frame = frame_batch_size * (i + 1) + remaining_frames
            z_intermediate = z[:, :, start_frame:end_frame]
            if self.post_quant_conv is not None:
                z_intermediate = self.post_quant_conv(z_intermediate)
            z_intermediate, conv_cache = self.decoder(z_intermediate, conv_cache=conv_cache)
            dec.append(z_intermediate)

        dec = torch.cat(dec, dim=2)

        if not return_dict:
            return (dec,)

        return DecoderOutput(sample=dec)

    @apply_forward_hook
    def decode(self, z: torch.Tensor, return_dict: bool = True) -> DecoderOutput | torch.Tensor:
        """
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

    def _tiled_encode_parallel(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, num_frames, height, width = x.shape
        overlap_height = int(self.tile_sample_min_height * (1 - self.tile_overlap_factor_height))
        overlap_width = int(self.tile_sample_min_width * (1 - self.tile_overlap_factor_width))
        blend_extent_height = int(self.tile_latent_min_height * self.tile_overlap_factor_height)
        blend_extent_width = int(self.tile_latent_min_width * self.tile_overlap_factor_width)
        row_limit_height = self.tile_latent_min_height - blend_extent_height
        row_limit_width = self.tile_latent_min_width - blend_extent_width
        # Create a grid of tile indices

        num_tiles_height = (height + overlap_height - 1) // overlap_height
        num_tiles_width = (width + overlap_width - 1) // overlap_width
        enc = self.tiled_encode_parallel(
            x,
            overlap_height=overlap_height,
            overlap_width=overlap_width,
            blend_extent_height=blend_extent_height,
            blend_extent_width=blend_extent_width,
            row_limit_height=row_limit_height,
            row_limit_width=row_limit_width,
            num_tiles_height=num_tiles_height,
            num_tiles_width=num_tiles_width,
            use_conv_cache=True,
        )
        return enc

    def forward(
        self,
        sample: torch.Tensor,
        sample_posterior: bool = False,
        return_dict: bool = True,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor | torch.Tensor:
        x = sample
        posterior = self.encode(x).latent_dist
        if sample_posterior:
            z = posterior.sample(generator=generator)
        else:
            z = posterior.mode()

        dec = self.decode(z)
        if not return_dict:
            return (dec,)
        return dec


AscendAutoencoderKLCogVideoX.parallel_spatial_tiled_decode = parallel_spatial_tiled_decode
AscendAutoencoderKLCogVideoX.temporal_tiled_decode = temporal_tiled_decode
AscendAutoencoderKLCogVideoX.tiled_decode_parallel = tiled_decode_parallel
AscendAutoencoderKLCogVideoX.tiled_encode_parallel = tiled_encode_parallel
autoencoder_kl_cogvideox.CogVideoXCausalConv3d = AscendCogVideoXCausalConv3d
autoencoder_kl_cogvideox.AutoencoderKLCogVideoX = AscendAutoencoderKLCogVideoX
