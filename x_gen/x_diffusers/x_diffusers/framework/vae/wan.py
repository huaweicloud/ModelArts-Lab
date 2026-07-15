import pathlib
import time
import typing
from functools import wraps
from pathlib import Path

import numpy as np
import PIL
import torch
import torch.distributed as dist
import torch.nn as nn
from diffusers.models.autoencoders import autoencoder_kl_wan
from diffusers.models.autoencoders.autoencoder_kl_wan import (
    CACHE_T,
    AutoencoderKLOutput,
    AutoencoderKLWan,
    DecoderOutput,
    DiagonalGaussianDistribution,
    WanCausalConv3d,
    WanUpsample,
    unpatchify,
)
from diffusers.video_processor import VideoProcessor
from x_base.utils.infer_info import infer_info
from x_base.vae_parallelism.utils import add_lightning_init, enable_lightning
from x_base.vae_parallelism.vae_mgr import VAEManager

from .IFRNet_S_arch import IRFNet_S
from .vfi_utils import InterpolationStateList, generic_frame_loop

MODEL_TYPE = pathlib.Path(__file__).parent.name


def vfi(
    frames: torch.Tensor,
    device,
    interpolation_model,
    batch_size=2,
    clear_cache_after_n_frames: typing.SupportsInt = 80,
    multiplier: typing.SupportsInt = 2,
    is_skip=False,
    scale_factor: typing.SupportsFloat = 1.0,
    optional_interpolation_states: InterpolationStateList = None,
):
    interpolation_model.eval().to(device)
    interpolation_model.to(torch.float16)
    frames.to(torch.float16)

    def return_middle_frame(frame_0, frame_1, timestep, model, scale_factor):
        return model(frame_0, frame_1, timestep, scale_factor)

    args = [interpolation_model, scale_factor]
    out = generic_frame_loop(
        frames,
        batch_size,
        device,
        clear_cache_after_n_frames,
        multiplier,
        return_middle_frame,
        *args,
        interpolation_states=optional_interpolation_states,
        dtype=torch.float16,
        is_skip=is_skip,
    )
    return out


def time_count(warmup=2, repeat=5):
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            if not bool(getattr(infer_info, "enable_vae_time_count", False)):
                return func(self, *args, **kwargs)

            for _ in range(warmup):
                func(self, *args, **kwargs)

            times = []
            results = None

            for _ in range(repeat):
                if dist.is_initialized():
                    dist.barrier()
                torch.cuda.synchronize()

                start_time = time.time()
                results = func(self, *args, **kwargs)

                torch.cuda.synchronize()
                times.append((time.time() - start_time) * 1000)

            avg_t, max_t, min_t = sum(times) / len(times), max(times), min(times)

            # 将平均时间记录到 infer_info 中，方便外部获取
            func_name = func.__name__.lower()
            if "encode" in func_name:
                infer_info.vae_last_encode_time = f"{avg_t:.2f}"
            elif "decode" in func_name:
                infer_info.vae_last_decode_time = f"{avg_t:.2f}"

            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
            if rank == 0:
                print(f"[{func.__name__}] Avg: {avg_t:.2f}ms, Max: {max_t:.2f}ms, Min: {min_t:.2f}ms")
            return results

        return wrapper

    return decorator


def _init_frame_interpolation():
    """抽离：插帧模型的初始化逻辑"""
    if not (infer_info.frame_interpolation and not infer_info.ten_second):
        return None, 1, False  # model, multiplier, is_skip

    interpolation_model = IRFNet_S()
    interpolation_model.load_state_dict(torch.load(infer_info.frame_model_path))

    if infer_info.fps % 16 == 0:
        multiplier = infer_info.fps // 16
        is_skip = False
    else:
        multiplier = infer_info.fps // 16 + 1
        is_skip = infer_info.fps % 8 < 4

    return interpolation_model, multiplier, is_skip


def _apply_frame_interpolation(out_, first_frame, i, skip, is_skip, multiplier, interp_model):
    """抽离：循环内的插帧计算逻辑"""
    if i == 0:
        first_frame = out_
    elif not is_skip or (is_skip and skip % 2 == 0):
        input_ = torch.cat([first_frame, out_], dim=2).transpose(1, 2).squeeze(0)
        out_ = (
            vfi(input_, input_.device, interp_model, input_.shape[0] - 1, multiplier=multiplier)[1:]
            .transpose(0, 1)
            .unsqueeze(0)
            .contiguous()
        )

    first_frame = out_[:, :, -1:, :, :]
    return out_, first_frame


def _decode_ascend(self, z: torch.Tensor, return_dict: bool = True) -> DecoderOutput | torch.Tensor:
    # 直接调用普通函数，不要加 self.
    interp_model, multiplier, is_skip = _init_frame_interpolation()
    use_interp = interp_model is not None

    self.clear_cache()
    num_frame = z.shape[2]
    x = self.post_quant_conv(z)

    is_lightning_mode = infer_info.vae_lightning in ["decoder", "encoder_and_decoder"]
    vae_manager = VAEManager(
        x.shape,
        is_encode=False,
        pad_mode=infer_info.vae_pad_mode,
        pad_content=infer_info.vae_pad_content,
        use_blend=infer_info.vae_use_blend,
    )

    if not self.return_output:
        vae_manager.create_video_stream()

    tile = vae_manager.get_tile_from_x(x) if is_lightning_mode else x

    output = []
    skip = 0
    first_frame = None

    for i in range(num_frame):
        self._conv_idx = [0]
        out_ = self.decoder(
            tile[:, :, i : i + 1, :, :], feat_cache=self._feat_map, feat_idx=self._conv_idx, first_chunk=(i == 0)
        )

        # 直接调用普通函数
        if use_interp:
            out_, first_frame = _apply_frame_interpolation(
                out_, first_frame, i, skip, is_skip, multiplier, interp_model
            )
        skip += 1  # noqa: SIM113

        if is_lightning_mode:
            out_ = vae_manager.align_out(out_)
            out_ = vae_manager.collect_out(out_)

        if self.config.patch_size is not None:
            out_ = unpatchify(out_, patch_size=self.config.patch_size)

        if self.return_output:
            output.append(out_)
        else:
            vae_manager.write_video_stream(out_)

    if self.return_output:
        output = torch.cat(output, dim=2)
    else:
        vae_manager.close_video_stream()
        output = None

    self.clear_cache()

    if not return_dict:
        return (output,)

    return DecoderOutput(sample=output)


@time_count(warmup=2, repeat=5)
def decode_ascend(self, z: torch.Tensor, return_dict: bool = True) -> DecoderOutput | torch.Tensor:
    decoded = self._decode_ascend(z).sample
    if not return_dict:
        return (decoded,)
    return DecoderOutput(sample=decoded)


def _encode_ascend(self, x: torch.Tensor):
    _, _, num_frame, height, width = x.shape
    self.clear_cache()
    iter_ = (num_frame + 3) // 4

    is_lightning_mode = infer_info.vae_lightning in ["encoder", "encoder_and_decoder"]
    vae_manager = VAEManager(
        x.shape,
        is_encode=True,
        pad_mode=infer_info.vae_pad_mode,
        pad_content=infer_info.vae_pad_content,
        use_blend=infer_info.vae_use_blend,
    )
    if is_lightning_mode:
        tile = vae_manager.get_tile_from_x(x)
    else:
        tile = x

    for i in range(iter_):
        self._enc_conv_idx = [0]
        if i == 0:
            out = self.encoder(tile[:, :, :1, :, :], feat_cache=self._enc_feat_map, feat_idx=self._enc_conv_idx)
        else:
            out_ = self.encoder(
                tile[:, :, 1 + 4 * (i - 1) : 1 + 4 * i, :, :],
                feat_cache=self._enc_feat_map,
                feat_idx=self._enc_conv_idx,
            )
            out = torch.cat([out, out_], 2)

    if is_lightning_mode:
        out = vae_manager.align_out(out)
        out = vae_manager.collect_out(out)
    enc = self.quant_conv(out)
    self.clear_cache()
    return enc


@time_count(warmup=2, repeat=5)
def encode_ascend(
    self, x: torch.Tensor, return_dict: bool = True
) -> AutoencoderKLOutput | tuple[DiagonalGaussianDistribution]:
    h = self._encode_ascend(x)

    if infer_info.vae_output_dir and dist.get_rank() == 0:
        out_dir = Path(infer_info.vae_output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        save_path = out_dir / "encode_output.npy"
        np.save(save_path, h.detach().float().cpu().numpy())
        print(f"vae encode输出结果已保存至: {save_path.resolve()}")

    posterior = DiagonalGaussianDistribution(h)
    if not return_dict:
        return (posterior,)
    return AutoencoderKLOutput(latent_dist=posterior)


def postprocess_video_ascend(
    self, video: torch.Tensor, output_type: str = "np"
) -> np.ndarray | torch.Tensor | list[PIL.Image.Image]:
    r"""
    Converts a video tensor to a list of frames for export.

    Args:
        video (`torch.Tensor`): The video as a tensor.
        output_type (`str`, defaults to `"np"`): Output type of the postprocessed `video` tensor.
    """
    if self.lightning and not self.return_output:
        return [None]
    batch_size = video.shape[0]
    outputs = []
    for batch_idx in range(batch_size):
        batch_vid = video[batch_idx].permute(1, 0, 2, 3)
        batch_output = self.postprocess(batch_vid, output_type)
        outputs.append(batch_output)

    if output_type == "np":
        outputs = np.stack(outputs)
    elif output_type == "pt":
        outputs = torch.stack(outputs)
    elif output_type != "pil":
        raise ValueError(f"{output_type} does not exist. Please choose one of ['np', 'pt', 'pil']")

    return outputs


class AscendWanResample(nn.Module):
    r"""
    A custom resampling module for 2D and 3D data.

    Args:
        dim (int): The number of input/output channels.
        mode (str): The resampling mode. Must be one of:
            - 'none': No resampling (identity operation).
            - 'upsample2d': 2D upsampling with nearest-exact interpolation and convolution.
            - 'upsample3d': 3D upsampling with nearest-exact interpolation, convolution, and causal 3D convolution.
            - 'downsample2d': 2D downsampling with zero-padding and convolution.
            - 'downsample3d': 3D downsampling with zero-padding, convolution, and causal 3D convolution.
    """

    def __init__(self, dim: int, mode: str, upsample_out_dim: int = None) -> None:
        super().__init__()
        self.dim = dim
        self.mode = mode

        # default to dim //2
        if upsample_out_dim is None:
            upsample_out_dim = dim // 2

        # layers
        if mode == "upsample2d":
            self.resample = nn.Sequential(
                WanUpsample(scale_factor=(2.0, 2.0), mode="nearest-exact"),
                nn.Conv2d(dim, upsample_out_dim, 3, padding=1),
            )
        elif mode == "upsample3d":
            self.resample = nn.Sequential(
                WanUpsample(scale_factor=(2.0, 2.0), mode="nearest-exact"),
                nn.Conv2d(dim, upsample_out_dim, 3, padding=1),
            )
            self.time_conv = WanCausalConv3d(dim, dim * 2, (3, 1, 1), padding=(1, 0, 0))

        elif mode == "downsample2d":
            self.resample = nn.Sequential(nn.ZeroPad2d((0, 1, 0, 1)), nn.Conv2d(dim, dim, 3, stride=(2, 2)))
        elif mode == "downsample3d":
            self.resample = nn.Sequential(nn.ZeroPad2d((0, 1, 0, 1)), nn.Conv2d(dim, dim, 3, stride=(2, 2)))
            self.time_conv = WanCausalConv3d(dim, dim, (3, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0))

        else:
            self.resample = nn.Identity()

        self._init_module = True

    def _init_module_test(self):
        for name, layer in self.resample.named_modules():
            if isinstance(layer, nn.Conv2d):
                layer = layer.to(torch.bfloat16)
        self._init_module = False

    def forward(self, x, feat_cache=None, feat_idx=None):
        if self._init_module:
            self._init_module_test()

        b, c, t, h, w = x.size()
        if self.mode == "upsample3d":
            if feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    feat_cache[idx] = "Rep"
                    feat_idx[0] += 1
                else:
                    cache_x = x[:, :, -CACHE_T:, :, :].clone()
                    if cache_x.shape[2] < 2 and feat_cache[idx] is not None and feat_cache[idx] != "Rep":
                        # cache last frame of last two chunk
                        cache_x = torch.cat(
                            [feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2
                        )
                    if cache_x.shape[2] < 2 and feat_cache[idx] is not None and feat_cache[idx] == "Rep":
                        cache_x = torch.cat([torch.zeros_like(cache_x).to(cache_x.device), cache_x], dim=2)
                    if feat_cache[idx] == "Rep":
                        x = self.time_conv(x)
                    else:
                        x = self.time_conv(x, feat_cache[idx])
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1

                    x = x.reshape(b, 2, c, t, h, w)
                    x = torch.stack((x[:, 0, :, :, :, :], x[:, 1, :, :, :, :]), 3)
                    x = x.reshape(b, c, t * 2, h, w)
        t = x.shape[2]
        x = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)

        ori_dtype = x.dtype
        x = self.resample(x.to(torch.bfloat16)).to(ori_dtype)

        x = x.view(b, t, x.size(1), x.size(2), x.size(3)).permute(0, 2, 1, 3, 4)

        if self.mode == "downsample3d":
            if feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    feat_cache[idx] = x.clone()
                    feat_idx[0] += 1
                else:
                    cache_x = x[:, :, -1:, :, :].clone()
                    x = self.time_conv(torch.cat([feat_cache[idx][:, :, -1:, :, :], x], 2))
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1
        return x


autoencoder_kl_wan.WanResample = AscendWanResample

autoencoder_kl_wan.AutoencoderKLWan._encode_ascend = _encode_ascend
autoencoder_kl_wan.AutoencoderKLWan.encode = encode_ascend

autoencoder_kl_wan.AutoencoderKLWan._decode_ascend = _decode_ascend
autoencoder_kl_wan.AutoencoderKLWan.decode = decode_ascend
autoencoder_kl_wan.AutoencoderKLWan.enable_lightning = enable_lightning
autoencoder_kl_wan.AutoencoderKLWan = add_lightning_init(AutoencoderKLWan)

VideoProcessor.enable_lightning = enable_lightning
VideoProcessor.postprocess_video = postprocess_video_ascend
VideoProcessor = add_lightning_init(VideoProcessor)
