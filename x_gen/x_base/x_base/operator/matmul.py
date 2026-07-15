import gc
import logging

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch_npu
from hadamard_transform import hadamard_transform
from torch import nn

# 量化配置管理
from x_base.config import QuantLayerConfig, quant_config_manager

try:
    from fast_hadamard_npu import fast_hadamard_inplace  # noqa: F401
except ImportError:
    print("fast_hadamard_npu not install")
try:
    import turing_cloud_ops
except ImportError:
    print("turing_cloud_ops not install")

logging.basicConfig(level=logging.INFO)


class WeightQuantLinearModule(nn.Module):
    def __init__(self, weight, bias, name=None):
        super(WeightQuantLinearModule, self).__init__()  # noqa: UP008
        quant_weight, pertoken_weight_scale = torch_npu.npu_dynamic_quant(weight.npu())
        del weight
        self.register_parameter("weight", torch.nn.Parameter(quant_weight.cpu(), requires_grad=False))
        self.register_parameter("antiquant_scale", torch.nn.Parameter(pertoken_weight_scale.cpu(), requires_grad=False))
        self.register_parameter("bias", torch.nn.Parameter(bias.cpu(), requires_grad=False))
        self.out_features, self.in_features = quant_weight.shape

    def forward(self, x):
        ori_shape = x.shape
        ori_dtype = x.dtype
        x = x.reshape(-1, ori_shape[-1])
        x_int8, pertoken_scale = torch_npu.npu_dynamic_quant(x)
        if self.weight.dtype != torch.int8:
            self.weight.data = self.weight.data.to(torch.int8)
        out = torch_npu.npu_quant_matmul(
            x_int8,
            self.weight.t(),
            self.antiquant_scale,
            bias=self.bias,
            pertoken_scale=pertoken_scale,
            output_dtype=ori_dtype,
        )
        out = out.reshape(*ori_shape[:-1], -1)
        return out


class WeightQuantLinearModule_w4a4(nn.Module):
    def __init__(self, weight, bias, name=None):
        super(WeightQuantLinearModule_w4a4, self).__init__()  # noqa: UP008
        d0, d1 = weight.shape
        next_power = 1
        while next_power < d1:
            next_power <<= 1
        self.d1 = d1
        self.next_power = next_power

        weight = F.pad(weight, (0, self.next_power - self.d1))
        weight = hadamard_transform(weight)

        self.pad_size = self.next_power - self.d1

        self.register_parameter("weight", torch.nn.Parameter(weight.cpu(), requires_grad=False))
        self.register_parameter("bias", torch.nn.Parameter(bias.to(torch.float32).cpu(), requires_grad=False))

        self.out_features, self.in_features = weight.shape
        del weight

        self.dtype = torch.bfloat16
        self.name = name

        self.is_first_init = True

    def smoothquant(self, x, w, ori_dtype, alpha=0.575):
        x = F.pad(x, (0, self.pad_size))
        x = hadamard_transform(x)

        x_scale = torch.amax(torch.abs(x), dim=0, keepdim=True)
        w_scale = torch.amax(torch.abs(w), dim=0, keepdim=True)
        x_scale = torch.abs(x_scale).clamp(min=1e-3, max=1e3)
        w_scale = torch.abs(w_scale).clamp(min=1e-8, max=1e3)

        scale = (x_scale) ** alpha / (w_scale + 1e-5) ** (1 - alpha)
        self.smooth_scale = scale

        x_smooth = x / scale
        w_smooth = w * scale

        x_quant, pertoken_scale = torch_npu.npu_dynamic_quant(x_smooth, dst_type=torch.quint4x2)
        quant_weight, w4a4_pertoken_weight_scale = torch_npu.npu_dynamic_quant(w_smooth.npu(), dst_type=torch.quint4x2)

        self.weight = torch.nn.Parameter(quant_weight, requires_grad=False)
        self.weight_scale = w4a4_pertoken_weight_scale
        self.is_first_init = False

    def forward(self, x):
        ori_shape = x.shape
        ori_dtype = x.dtype
        x = x.reshape(-1, ori_shape[-1])

        if self.is_first_init:
            self.smoothquant(x, self.weight, ori_dtype)

        x = F.pad(x, (0, self.pad_size)).to(torch.float16)
        x = hadamard_transform(x)
        x_quant, pertoken_scale = torch_npu.npu_dynamic_quant(
            (x / self.smooth_scale).to(ori_dtype), dst_type=torch.quint4x2
        )

        if self.weight.dtype != torch.int32:
            self.weight.data = self.weight.data.to(torch.int32)

        out = torch_npu.npu_quant_matmul(
            x_quant,
            self.weight.t(),
            self.weight_scale,
            bias=self.bias,
            pertoken_scale=pertoken_scale.squeeze().to(torch.float32),
            output_dtype=ori_dtype,
        )
        out = out.reshape(*ori_shape[:-1], -1)

        return out


def enable_weight_quant_linear(model: nn.Module, quant_config: QuantLayerConfig, dtype: str = "w8a8") -> nn.Module:
    """启用权重量化

    根据量化配置对模型中的 Linear 层进行量化。

    Args:
        model: 待量化的模型
        quant_config: 量化层配置（指定白名单/黑名单/w4a4模式）
        dtype: 量化精度 ('w8a8' 或 'w4a4')

    Returns:
        量化后的模型
    """
    for name, layer in model.named_modules():
        # 使用配置判断是否量化该层
        if not quant_config.should_quantize(name, layer):
            continue

        # undo lora rename operation
        clean_name = name.replace(".base_layer", "")

        # 选择量化模块类型
        if "w4a4" in dtype and quant_config.should_use_w4a4(name):
            quant_module = WeightQuantLinearModule_w4a4(layer.weight, layer.bias, name)
        else:
            quant_module = WeightQuantLinearModule(layer.weight, layer.bias, name)

        # 替换模块
        if "." in clean_name:
            submodules, layer_name = clean_name.rsplit(".", 1)
            setattr(model.get_submodule(submodules), layer_name, quant_module)
        else:
            # 根层级层，直接设置在模型上
            setattr(model, clean_name, quant_module)
        del layer.weight, layer.bias, layer

    return model


class QuantWanCausalConv3d(nn.Conv3d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int, int],
        stride: int | tuple[int, int, int] = 1,
        padding: int | tuple[int, int, int] = 0,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self._padding = (self.padding[2], self.padding[2], self.padding[1], self.padding[1], 2 * self.padding[0], 0)
        self.padding = (0, 0, 0)

        self.quant_weight = True
        self.dilations = [1, 1, 1]

    def transform_5d_to_7d(self, weight_5d, n0=16, c0=32):
        n, c, d, h, w = weight_5d.shape
        n1 = n // n0
        c1 = c // c0
        weight_reshaped = weight_5d.reshape(n1, n0, c1, c0, d, h, w)
        weight_7d = weight_reshaped.permute(4, 2, 5, 6, 0, 1, 3).contiguous()

        return weight_7d

    def _quantize_on_init(self, re=True):
        original_weight = self.weight.data
        original_weight = original_weight.to(torch.bfloat16)

        weight_abs_max = torch.abs(original_weight).amax(dim=[1, 2, 3, 4])
        weight_scale = weight_abs_max / 127.0
        quant_weight = torch_npu.npu_quantize(
            original_weight, scales=weight_scale, zero_points=None, dtype=torch.qint8, axis=0, div_mode=True
        )
        quant_weight = self.transform_5d_to_7d(quant_weight)

        self.weight_quant = nn.Parameter(quant_weight, requires_grad=False)
        self.weight_scale = weight_scale

        self.quant_weight = False

    def forward(self, x, cache_x=None):
        # N C D H W
        padding = list(self._padding)
        if cache_x is not None and self._padding[4] > 0:
            cache_x = cache_x.to(x.device)
            x = torch.cat([cache_x, x], dim=2)
            padding[4] -= cache_x.shape[2]
        x = F.pad(x, padding)
        x = x.to(torch.bfloat16)

        x_int8, input_scale = turing_cloud_ops.cloud_dynamic_quant(
            x, smooth_scales=None, group_index=None, dst_type=2, is_symmetrical=True, quant_mode=1
        )

        if self.quant_weight:
            self._quantize_on_init()

        self.scale = (input_scale * self.weight_scale).to(torch.float32)
        output = turing_cloud_ops.cloud_conv3d_quant(
            x_int8,
            self.weight_quant,
            self.bias,
            self.scale,
            self.stride,
            self.padding,
            self.dilations,
        )
        return output


def enable_conv3d_quant(model):
    for name, layer in model.named_modules():
        if "WanCausalConv3d" in layer.__class__.__name__:
            if layer.weight.shape[0] > 16 and layer.weight.shape[1] > 32:
                quant_module = QuantWanCausalConv3d(
                    layer.in_channels, layer.out_channels, layer.kernel_size, layer.stride, layer.padding
                )

                quant_module.weight.data.copy_(layer.weight.data)

                if layer.bias is not None:
                    quant_module.bias.data.copy_(layer.bias.data)
                elif quant_module.bias is not None:
                    quant_module.bias.data.zero_()
                if hasattr(layer, "_padding"):
                    quant_module._padding = layer._padding
                del layer

                submodules, layer_name = name.split(".")[:-1], name.split(".")[-1]
                setattr(model.get_submodule(".".join(submodules)), layer_name, quant_module)
    return model


class MatmulManager:
    """量化管理器

    提供在线动态量化、VAE Conv3D 量化等功能。
    支持通过配置文件定制不同模型的量化策略。
    """

    def __init__(self):
        pass

    def enable_online_dynamic_quant(
        self, transformer, quant_config: QuantLayerConfig | None = None, dtype: str = "w8a8"
    ):
        """启用在线动态量化

        Args:
            transformer: 待量化的 transformer 模型
            quant_config: 量化配置，None 表示自动推断
            dtype: 量化精度 ('w8a8' 或 'w4a4')

        Returns:
            量化后的模型
        """
        try:
            cur_rank = dist.get_rank() if dist.is_initialized() else 0
            logging.info(f"Rank {cur_rank} start quantification.")  # noqa: G004

            # 自动推断配置
            if quant_config is None:
                config_name = quant_config_manager.detect_config_name(transformer)
                quant_config = quant_config_manager.get_config(config_name)
                logging.info(f"Rank {cur_rank} auto-detected quant config: {config_name}")  # noqa: G004

            model = enable_weight_quant_linear(transformer, quant_config, dtype=dtype)

            gc.collect()
            torch.cuda.empty_cache()
            logging.info(f"Rank {cur_rank} finish quantification.")  # noqa: G004

        except Exception as e:
            logging.error(f"Quantization failed: {e}")  # noqa: G004
            raise ValueError("Dynamic weight modify model failed. Please try running the program again.")

        return model

    def enable_vae_conv3d_quant(self, vae):
        """启用 VAE Conv3D 量化

        Args:
            vae: VAE 模型

        Returns:
            量化后的 VAE 模型
        """
        try:
            logging.info(f"Rank {dist.get_rank()} start conv3d quantification.")  # noqa: G004

            model = enable_conv3d_quant(vae)

            gc.collect()
            torch.cuda.empty_cache()
            logging.info(f"Rank {dist.get_rank()} finish conv3d quantification.")  # noqa: G004

        except Exception as e:
            logging.error(f"VAE Conv3D quantization failed: {e}")  # noqa: G004
            raise ValueError("Dynamic vae conv3d modify model failed. Please try running the program again.")

        return model


matmul_manager = MatmulManager()
