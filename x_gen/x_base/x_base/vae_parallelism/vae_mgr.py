import math

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from diffusers.models.autoencoders.vae import DecoderOutput
from functorch.einops import rearrange
from PIL import Image

from ..utils.infer_info import infer_info
from .save_video_stream import SaveVideoStream


class VAEManager:
    def __init__(
        self,
        input_shape,
        factor: int = 8,
        is_encode: bool = False,
        pad_mode: str = "all_sides_valid",
        pad_content: str = "constant",
        use_blend: bool = True,
    ):
        """
        Args:
            input_shape: (B, C, T, H, W) input tensor shape.
            factor: Downsample/Upsample factor (e.g., 8).
            is_encode: True for Image -> Latent, False for Latent -> Image.
            pad_mode:
                - "tail_only": All sliding windows stay inside the original region;
                  padding exists only on the right and bottom sides.
                - "all_sides": Padding is applied on all four sides. When using this mode,
                  `pad_content` takes effect.
                - "all_sides_valid": Padding is applied on all four sides. 超出输出边界的部分会被去掉。
            pad_content:
                - "constant": Pad with zeros.
                - "reflect": Reflect padding.
                - "replicate": Replicate padding.
            use_blend: Whether to blend overlapping regions when merging tiles.
        """
        if len(input_shape) != 5:
            raise ValueError("input_shape must have 5 dimensions: (B, C, T, H, W)")
        batch, channels, frames, total_in_h, total_in_w = input_shape
        self.total_in_h = total_in_h
        self.total_in_w = total_in_w

        self.factor = factor
        self.is_encode = is_encode
        self.pad_mode = pad_mode
        self.pad_content = pad_content
        self.use_blend = use_blend

        # 1. 计算分块数量 (Tile Grid)
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1

        # 逻辑分块所用的"有效 world_size"，16卡时退化为8卡的切分规格
        self.tile_world_size = infer_info.get_tile_world_size(self.world_size)
        # 当前 Rank 在逻辑分块中的身份（16卡时 rank 8 与 rank 0 处理同一块）
        self.tile_rank = self.rank % self.tile_world_size

        self.num_h_tiles, self.num_w_tiles = split_rectangle(total_in_h, total_in_w, self.tile_world_size)

        # 2. 定义 Padding 大小 (Input Padding)，决定了 Tile 为了消除边界效应需要多读多少像素
        if is_encode:
            # Image -> Latent
            self.in_pad_size = infer_info.vae_pad_latent_size * factor
            self.out_pad_size = infer_info.vae_pad_latent_size
        else:
            # Latent -> Image
            self.in_pad_size = infer_info.vae_pad_latent_size
            self.out_pad_size = infer_info.vae_pad_latent_size * factor

        # 3. 计算输入侧尺寸
        if pad_mode == "tail_only":
            # 步长 = (总长 - 单倍Padding) / 块数。
            self.in_stride_h = math.ceil((total_in_h - self.in_pad_size) / self.num_h_tiles)
            self.in_stride_w = math.ceil((total_in_w - self.in_pad_size) / self.num_w_tiles)
            # 窗口长度 = 步长 + 单倍Padding
            self.in_window_h = self.in_stride_h + self.in_pad_size
            self.in_window_w = self.in_stride_w + self.in_pad_size
        else:
            # 步长= 总长 / 块数。
            self.in_stride_h = math.ceil(total_in_h / self.num_h_tiles)
            self.in_stride_w = math.ceil(total_in_w / self.num_w_tiles)
            # 窗口长度 = 步长 + 双倍Padding
            self.in_window_h = self.in_stride_h + 2 * self.in_pad_size
            self.in_window_w = self.in_stride_w + 2 * self.in_pad_size

        # 4. 计算输出侧尺寸
        if is_encode:
            # Downsampling (Image -> Latent)
            self.out_stride_h = int(self.in_stride_h / factor)
            self.out_stride_w = int(self.in_stride_w / factor)
            self.out_window_h = int(self.in_window_h / factor)
            self.out_window_w = int(self.in_window_w / factor)
        else:
            # Upsampling (Latent -> Image)
            self.out_stride_h = self.in_stride_h * factor
            self.out_stride_w = self.in_stride_w * factor
            self.out_window_h = self.in_window_h * factor
            self.out_window_w = self.in_window_w * factor

        # Blend Size: 输出时用于混合的重叠区域大小。
        self.blend_h = self.out_window_h - self.out_stride_h
        self.blend_w = self.out_window_w - self.out_stride_w

        self.save_video_stream = None

        # [新增] 用于存储 all_sides_valid 模式下的补全信息
        # 格式: (pad_left, pad_right, pad_top, pad_bottom)
        self._curr_pad_info = (0, 0, 0, 0)

    def get_tile_from_x(self, x: torch.Tensor) -> torch.Tensor:
        if self.pad_mode == "all_sides_valid":
            return self._get_tile_valid_mode(x)

        # 1. 预处理
        B, C, T, H, W = x.shape
        x = x.view(-1, T, H, W)

        # 2. 计算几何对齐所需的额外 Padding (Extra Alignment Padding)
        extra_pad_w = -self.total_in_w % self.num_w_tiles
        extra_pad_h = -self.total_in_h % self.num_h_tiles

        # 3. 根据 padding_strategy (位置策略) 构建 pad_tuple
        if self.pad_mode == "tail_only":
            # [策略 A: Tail Only] 左/上不补，右/下补足
            pad_left = 0
            pad_top = 0
            pad_right = extra_pad_w
            pad_bottom = extra_pad_h
        elif self.pad_mode == "all_sides":
            # [策略 B: All Sides] (原 else 分支逻辑)
            # 四周都补 context (in_pad_size)，且右/下额外补足对齐量。
            pad_left = self.in_pad_size
            pad_top = self.in_pad_size
            pad_right = self.in_pad_size + extra_pad_w
            pad_bottom = self.in_pad_size + extra_pad_h

        pad_tuple = (pad_left, pad_right, pad_top, pad_bottom)

        # 4. 执行 Padding
        if self.pad_content == "constant":
            x = F.pad(x, pad_tuple, mode="constant", value=0)
        else:
            # ReplicationPad2d 在 NPU 上不支持 BF16（DT_BFLOAT16）
            need_cast = (x.dtype == torch.bfloat16) and (self.pad_content == "replicate")
            if need_cast:
                x = x.half()
            x = F.pad(x, pad_tuple, mode=self.pad_content)
            if need_cast:
                x = x.bfloat16()

        # 5. 还原维度
        # H, W 已经变了，需要获取新的尺寸
        H_padded, W_padded = x.shape[2], x.shape[3]
        x = x.view(B, C, T, H_padded, W_padded)

        # 6. 获得具体切片，根据当前 Rank 从完整输入 x 中切分出对应的 Tile
        h_idx = self.tile_rank // self.num_w_tiles
        w_idx = self.tile_rank % self.num_w_tiles

        # 计算切片起始点 (基于 Stride)
        h_start = h_idx * self.in_stride_h
        w_start = w_idx * self.in_stride_w

        # 裁剪：从 start 开始，切 window 大小
        return x[:, :, :, h_start : h_start + self.in_window_h, w_start : w_start + self.in_window_w]

    def _get_tile_valid_mode(self, x: torch.Tensor) -> torch.Tensor:
        """
        只读取原始图片中的有效区域。
        并记录因为超出边界而少读了多少 Padding，以便在 align_out 时补回。
        """
        B, C, T, H, W = x.shape

        h_idx = self.tile_rank // self.num_w_tiles
        w_idx = self.tile_rank % self.num_w_tiles

        # 1. 理论上的 Window 范围 (包含 Padding 的理想范围)
        # 中心对齐逻辑：Start = Index * Stride - Pad
        h_start_theory = h_idx * self.in_stride_h - self.in_pad_size
        w_start_theory = w_idx * self.in_stride_w - self.in_pad_size

        h_end_theory = h_start_theory + self.in_window_h
        w_end_theory = w_start_theory + self.in_window_w

        # 2. 实际有效的图片范围 (Clamp 到 [0, Total])
        h_start_valid = max(0, h_start_theory)
        w_start_valid = max(0, w_start_theory)
        h_end_valid = min(self.total_in_h, h_end_theory)
        w_end_valid = min(self.total_in_w, w_end_theory)

        # 3. 计算“缺失”的像素量 (Offset)
        # 如果理论起点是 -32，实际起点是 0，说明左边/上边少了 32 个像素 -> pad_top/left = 32
        pad_top = h_start_valid - h_start_theory
        pad_left = w_start_valid - w_start_theory

        # 如果理论终点是 Total+32，实际终点是 Total，说明右边/下边少了 32 个像素 -> pad_bottom/right = 32
        pad_bottom = h_end_theory - h_end_valid
        pad_right = w_end_theory - w_end_valid

        # [关键] 存下来给 align_out 用
        self._curr_pad_info = (pad_left, pad_right, pad_top, pad_bottom)

        # 4. 执行切片 (只切有效区域)
        # x shape: [B, C, T, H, W]
        # 注意：这里的 tile 尺寸可能小于 self.in_window_h/w
        tile = x[:, :, :, h_start_valid:h_end_valid, w_start_valid:w_end_valid]
        return tile

    def align_out(self, out: torch.Tensor) -> torch.Tensor:
        """
        根据输入时缺失的像素，对输出 Tensor 进行补全，使其恢复到 out_window 的大小。
        这样才能满足 all_gather 的形状要求。
        """
        if self.pad_mode == "all_sides_valid":
            # 1. 取出输入时的缺失量
            pl, pr, pt, pb = self._curr_pad_info

            # 2. 根据下采样/上采样 缩放 Padding 量
            if self.is_encode:
                # Downsampling (Image -> Latent)
                # 必须保证整除，通常 VAE 的 stride/pad 设置都是 factor 的倍数
                pl = pl // self.factor
                pr = pr // self.factor
                pt = pt // self.factor
                pb = pb // self.factor
            else:
                # Upsampling (Latent -> Image)
                pl = pl * self.factor
                pr = pr * self.factor
                pt = pt * self.factor
                pb = pb * self.factor

            # 3. 定向 Padding
            # F.pad 顺序: (Left, Right, Top, Bottom)
            # 补的值通常是 0 (constant)，因为这些区域在 merge 时会被切掉 (overlap)
            if pl > 0 or pr > 0 or pt > 0 or pb > 0:
                out = F.pad(out, (pl, pr, pt, pb), mode="constant", value=0)
            return out

        else:
            # 原有的 tail_only 逻辑 (只补右下)
            if out.shape[-2] < self.out_window_h:
                out = F.pad(out, (0, 0, 0, self.out_window_h - out.shape[-2]))
            if out.shape[-1] < self.out_window_w:
                out = F.pad(out, (0, self.out_window_w - out.shape[-1]))
            return out

    def collect_out(self, out: torch.Tensor) -> torch.Tensor:
        """
        收集所有 Rank 的输出并拼接。
        """
        # 1. Gather all tiles
        repeat_dims = [self.world_size] + [1] * out.dim()
        gathered_out = torch.zeros_like(out).unsqueeze(0).repeat(*repeat_dims).contiguous()
        torch.cuda.set_device(self.rank)
        dist.all_gather_into_tensor(gathered_out, out)

        # 16卡时 gathered_out[0..7] 与 [8..15] 内容相同，只取前 tile_world_size 份
        gathered_out = gathered_out[: self.tile_world_size]

        # 2. Organize linear gathered chunks into a 2D Grid
        # -----------------------------------------------------------
        # spatial_grid[row][col] -> Tensor
        spatial_grid = [[None for _ in range(self.num_w_tiles)] for _ in range(self.num_h_tiles)]
        for global_idx in range(gathered_out.shape[0]):
            h_idx = global_idx // self.num_w_tiles
            w_idx = global_idx % self.num_w_tiles
            spatial_grid[h_idx][w_idx] = gathered_out[global_idx]

        # 3. Merge Tiles (Blend & Crop) based on Padding Mode
        # -----------------------------------------------------------
        out = self._merge_tiles(spatial_grid)

        # 4. Crop
        final_h = int(self.total_in_h / self.factor) if self.is_encode else int(self.total_in_h * self.factor)
        final_w = int(self.total_in_w / self.factor) if self.is_encode else int(self.total_in_w * self.factor)

        out = out[:, :, :, :final_h, :final_w]
        return out

    def _blend_tile(self, i: int, j: int, spatial_grid: list, tile: torch.Tensor) -> torch.Tensor:
        """抽离：处理 tile 的混合（Blend）逻辑"""
        if not self.use_blend:
            return tile

        if i > 0:
            tile = blend_v(spatial_grid[i - 1][j], tile, self.blend_h)
        if j > 0:
            tile = blend_h(spatial_grid[i][j - 1], tile, self.blend_w)
        return tile

    def _crop_tile(self, i: int, j: int, n_rows: int, n_cols: int, tile: torch.Tensor) -> torch.Tensor:
        """抽离：处理 tile 的裁剪（Crop）逻辑"""
        if self.pad_mode == "tail_only":
            # 使用三元表达式替代 if/else 块，直接降低复杂度
            h_end = self.out_stride_h if i < n_rows - 1 else None
            w_end = self.out_stride_w if j < n_cols - 1 else None
            return tile[..., :h_end, :w_end]

        if "all_sides" in self.pad_mode:
            if not self.use_blend:
                pad = self.out_pad_size
                return tile[..., pad:-pad, pad:-pad]

            # All Sides 且 use_blend=True 时，使用三元表达式压平 4 个 if/else
            top_crop = self.blend_h // 2 if i == 0 else 0
            bottom_crop = self.blend_h // 2 if i == n_rows - 1 else self.blend_h
            left_crop = self.blend_w // 2 if j == 0 else 0
            right_crop = self.blend_w // 2 if j == n_cols - 1 else self.blend_w

            return tile[..., top_crop:-bottom_crop, left_crop:-right_crop]

        return tile

    def _merge_tiles(self, spatial_grid) -> torch.Tensor:
        """
        Internal helper to blend and merge the grid of tiles.
        """
        result_rows = []
        n_rows = len(spatial_grid)
        n_cols = len(spatial_grid[0])

        for i, row in enumerate(spatial_grid):
            result_row = []
            for j, tile in enumerate(row):
                if infer_info.vae_pad_latent_size > 0:
                    # 1. 混合
                    tile = self._blend_tile(i, j, spatial_grid, tile)
                    # 2. 裁剪
                    tile = self._crop_tile(i, j, n_rows, n_cols, tile)

                result_row.append(tile)
            # 按宽度拼接当前行
            result_rows.append(torch.cat(result_row, dim=-1))

        # 按高度拼接所有行
        return torch.cat(result_rows, dim=-2)

    def create_video_stream(self):
        if infer_info.task_type != "t2i" and self.rank == 0:
            self.save_video_stream = SaveVideoStream()

    def write_video_stream(self, out: torch.Tensor):
        if self.rank == 0:
            out = torch.clamp(out, min=-1.0, max=1.0)
            out = out.squeeze(0)
            out = rearrange(out, "c t h w -> t h w c")
            out = (out / 2 + 0.5).clamp(0, 1) * 255
            if infer_info.task_type == "t2i":
                out = out.squeeze(0)
                out = out.cpu().numpy().astype(np.uint8)
                im = Image.fromarray(out)
                im.save(infer_info.save_path)
            else:
                self.save_video_stream.save(out)

    def close_video_stream(self):
        if infer_info.task_type != "t2i" and self.rank == 0:
            self.save_video_stream.close()


def split_rectangle(total_height: int, total_width: int, world_size: int):
    """
    计算最佳的切分网格 (Rows, Cols)，使得每个 Tile 的形状尽可能接近正方形。

    原理:
        我们需要将图像切分为 world_size 个块。
        假设切分为 num_rows x num_cols 的网格 (其中 num_rows * num_cols = world_size)。
        每个 Tile 的尺寸为: (total_height / num_rows) x (total_width / num_cols)。
        为了处理效率和感受野的均匀性，我们希望 Tile 的长宽比接近 1:1。

    Args:
        total_height: 输入图像的总高度。
        total_width: 输入图像的总宽度。
        world_size: 并行计算的总数 (即需要切分的块数)。

    Returns:
        (num_rows, num_cols):
            - num_rows: 高度方向切几刀 (对应 H 维度)。
            - num_cols: 宽度方向切几刀 (对应 W 维度)。
    """
    # 1. 寻找 world_size 的所有因子对 (Factor Pairs)
    factor_pairs = []
    for num_rows in range(1, world_size + 1):
        if world_size % num_rows == 0:
            num_cols = world_size // num_rows
            factor_pairs.append((num_rows, num_cols))

    min_ratio_diff = float("inf")
    best_rows = 1
    best_cols = world_size

    # 2. 遍历所有可能的网格布局，寻找最优解
    for num_rows, num_cols in factor_pairs:
        # 计算当前切分下，单个 Tile 的高度和宽度
        tile_h = total_height / num_rows
        tile_w = total_width / num_cols

        # 计算宽高比 (Aspect Ratio)
        # tile_h / tile_w 越接近 1 越好
        ratio = tile_h / tile_w
        diff = abs(ratio - 1)

        if diff < min_ratio_diff:
            min_ratio_diff = diff
            best_rows = num_rows
            best_cols = num_cols

    return best_rows, best_cols


def blend_v(a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
    blend_extent = min(a.shape[3], b.shape[3], blend_extent)
    w = torch.linspace(0, 1, blend_extent, device=a.device, dtype=a.dtype).view(1, 1, 1, -1, 1)
    b[..., :blend_extent, :] = a[..., -blend_extent:, :] * (1 - w) + b[..., :blend_extent, :] * w
    return b


def blend_h(a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
    blend_extent = min(a.shape[4], b.shape[4], blend_extent)
    w = torch.linspace(0, 1, blend_extent, device=a.device, dtype=a.dtype).view(1, 1, 1, 1, -1)
    b[..., :blend_extent] = a[..., -blend_extent:] * (1 - w) + b[..., :blend_extent] * w
    return b


def blend_t(a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
    blend_extent = min(a.shape[-3], b.shape[-3], blend_extent)
    for x in range(blend_extent):
        b[:, :, x, :, :] = a[:, :, -blend_extent + x, :, :] * (1 - x / blend_extent) + b[:, :, x, :, :] * (
            x / blend_extent
        )
    return b


def merge_spatial_tiles_cog(spatial_rows, blend_extent, row_limit):
    """合并空间分块（与原实现保持一致）"""
    result_rows = []
    for i, row in enumerate(spatial_rows):
        result_row = []
        for j, tile in enumerate(row):
            if i > 0:
                tile = blend_v(spatial_rows[i - 1][j], tile, blend_extent)
            if j > 0:
                tile = blend_h(row[j - 1], tile, blend_extent)
            result_row.append(tile[:, :, :, :row_limit, :row_limit])
        result_rows.append(torch.cat(result_row, dim=-1))
    return torch.cat(result_rows, dim=-2)


def parallel_data_generator(gathered_results, gathered_dim_metadata):
    global_idx = 0
    for i, per_rank_metadata in enumerate(gathered_dim_metadata):
        _start_shape = 0
        for shape in per_rank_metadata:
            mul_shape = math.prod(shape)
            yield (gathered_results[i, _start_shape : _start_shape + mul_shape].reshape(shape), global_idx)
            _start_shape += mul_shape
            global_idx += 1


def parallel_spatial_tiled_decode(
    self,
    z: torch.FloatTensor,
    return_dict: bool = True,
    use_conv_cache: bool = True,
) -> DecoderOutput | torch.FloatTensor:
    """
    仅基于空间维度分块的并行解码，时间维度保持完整
    """
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    B, C, T, H, W = z.shape

    # 仅计算空间重叠参数
    s_overlap_size = int(self.tile_latent_min_size * (1 - self.tile_overlap_factor))
    s_blend_extent = int(self.tile_sample_min_size * self.tile_overlap_factor)
    s_row_limit = self.tile_sample_min_size - s_blend_extent

    # 计算空间分块数量
    num_h_tiles = (H + s_overlap_size - 1) // s_overlap_size
    num_w_tiles = (W + s_overlap_size - 1) // s_overlap_size
    total_spatial_tiles = num_h_tiles * num_w_tiles

    # 分配各GPU处理的分块范围
    tiles_per_rank = total_spatial_tiles // world_size
    extra_rank = total_spatial_tiles % world_size
    if rank < extra_rank:
        start_tile_idx = rank * (tiles_per_rank + 1)
        end_tile_idx = start_tile_idx + tiles_per_rank + 1
    else:
        start_tile_idx = extra_rank * (tiles_per_rank + 1) + (rank - extra_rank) * tiles_per_rank
        end_tile_idx = start_tile_idx + tiles_per_rank

    local_results = []
    local_dim_metadata = []
    # 处理分配的分块
    for global_idx in range(start_tile_idx, end_tile_idx):
        conv_cache = None
        h_idx = global_idx // num_w_tiles
        w_idx = global_idx % num_w_tiles

        # 空间位置计算
        h_start = h_idx * s_overlap_size
        w_start = w_idx * s_overlap_size

        # 提取完整时间维度的分块
        tile = z[
            :,
            :,
            :,  # 时间维度全保留
            h_start : h_start + self.tile_latent_min_size,
            w_start : w_start + self.tile_latent_min_size,
        ]

        # 处理分块
        if self.post_quant_conv is not None:
            tile = self.post_quant_conv(tile)
        if use_conv_cache:
            decoded, conv_cache = self.decoder(tile, conv_cache)
        else:
            decoded = self.decoder(tile)

        # 保存元数据和结果
        local_results.append(decoded.reshape(-1))
        local_dim_metadata.append(decoded.shape)

    # 结果收集与同步逻辑（与原函数保持一致）
    results = torch.cat(local_results, dim=0).contiguous()

    local_size = torch.tensor([results.size(0)], device=results.device, dtype=torch.int64)
    all_sizes = [torch.zeros(1, device=results.device, dtype=torch.int64) for _ in range(world_size)]
    dist.all_gather(all_sizes, local_size)

    max_size = max(size.item() for size in all_sizes)

    padded_results = torch.zeros(max_size, device=results.device)
    padded_results[: results.size(0)] = results

    gathered_dim_metadata = [None] * world_size
    gathered_results = torch.zeros_like(padded_results).repeat(world_size, 1).contiguous()
    dist.all_gather_into_tensor(gathered_results, padded_results)
    torch.cuda.set_device(rank)
    dist.all_gather_object(gathered_dim_metadata, local_dim_metadata)

    # 重组分块数据
    data = [[None for _ in range(num_w_tiles)] for _ in range(num_h_tiles)]
    for current_data, global_idx in parallel_data_generator(gathered_results, gathered_dim_metadata):
        h_idx = global_idx // num_w_tiles
        w_idx = global_idx % num_w_tiles
        data[h_idx][w_idx] = current_data

    # 合并空间分块
    merged = merge_spatial_tiles_cog(data, s_blend_extent, s_row_limit)

    if not return_dict:
        return (merged,)
    return DecoderOutput(sample=merged)


# vae3
def temporal_tiled_decode(
    self,
    z: torch.FloatTensor,
    return_dict: bool = True,
    sample_shape=None,
    save_path: str = None,
    use_conv_cache: bool = True,
) -> DecoderOutput | torch.FloatTensor:
    B, C, T, H, W = z.shape
    overlap_size = int(self.tile_latent_min_tsize * (1 - self.tile_overlap_factor))
    blend_extent_default = int(self.tile_sample_min_tsize * self.tile_overlap_factor)

    if dist.get_rank() == 0:
        save_video_stream = SaveVideoStream()

    last_decoded = None
    total_out = 0
    out_chunks = []

    for i in range(0, T, overlap_size):
        latent_end = min(i + self.tile_latent_min_tsize + 1, T)
        tile = z[:, :, i:latent_end, :, :]
        decoded = self.parallel_spatial_tiled_decode(tile, return_dict=True, use_conv_cache=use_conv_cache).sample

        actual_len = decoded.shape[2]

        curr_blend = min(blend_extent_default, actual_len - 1)
        is_first = i == 0
        is_last = latent_end == T

        if not is_first:
            decoded = decoded[:, :, 1:, :, :]
            actual_len -= 1
            decoded = blend_t(last_decoded, decoded, curr_blend)

        if is_last:
            video_chunk = decoded
        else:
            keep = actual_len - curr_blend
            video_chunk = decoded[:, :, :keep, :, :]

        out_len = video_chunk.shape[2]
        total_out += out_len
        if dist.get_rank() == 0:
            save_video_stream.save(video_chunk)

        out_chunks.append(video_chunk)
        last_decoded = decoded[:, :, -curr_blend:, :, :]

    if dist.get_rank() == 0:
        save_video_stream.close()

        # NOTE: This function works correctly for HunyuanVideo,
        #       where the total frame count always satisfies frames = 4 * k + 1.
        # TODO: Verify that the same logic holds for CogVideo,
        #       whose temporal-upsampling scheme may differ.
        exp_total = sample_shape[0] if sample_shape is not None else None
        if exp_total is not None and total_out != exp_total:
            print(f"[decode] FINISHED: produced {total_out} frames (expected {exp_total})")

    dec = torch.cat(out_chunks, dim=2)

    if not return_dict:
        return (dec,)
    return DecoderOutput(sample=dec)


def tiled_decode_parallel(
    self,
    z: torch.FloatTensor,
    return_dict: bool = True,
    use_conv_cache: bool = True,
) -> DecoderOutput | torch.FloatTensor:
    """
    Parallel version of tiled_decode that distributes both temporal and spatial computation across GPUs
    """
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    B, C, T, H, W = z.shape
    # Calculate parameters
    t_overlap_size = int(self.tile_latent_min_tsize * (1 - self.tile_overlap_factor))
    t_blend_extent = int(self.tile_sample_min_tsize * self.tile_overlap_factor)
    t_limit = self.tile_sample_min_tsize - t_blend_extent
    s_overlap_size = int(self.tile_latent_min_size * (1 - self.tile_overlap_factor))
    s_blend_extent = int(self.tile_sample_min_size * self.tile_overlap_factor)
    s_row_limit = self.tile_sample_min_size - s_blend_extent
    # Calculate tile dimensions
    num_t_tiles = (T + t_overlap_size - 1) // t_overlap_size
    num_h_tiles = (H + s_overlap_size - 1) // s_overlap_size
    num_w_tiles = (W + s_overlap_size - 1) // s_overlap_size
    total_spatial_tiles = num_h_tiles * num_w_tiles
    total_tiles = num_t_tiles * total_spatial_tiles
    # Calculate tiles per rank and padding
    tiles_per_rank = (total_tiles + world_size - 1) // world_size
    start_tile_idx = rank * tiles_per_rank
    end_tile_idx = min((rank + 1) * tiles_per_rank, total_tiles)
    local_results = []
    local_dim_metadata = []
    # Process assigned tiles
    for local_idx, global_idx in enumerate(range(start_tile_idx, end_tile_idx)):
        conv_cache = None
        # Convert flat index to 3D indices
        t_idx = global_idx // total_spatial_tiles
        spatial_idx = global_idx % total_spatial_tiles
        h_idx = spatial_idx // num_w_tiles
        w_idx = spatial_idx % num_w_tiles
        # Calculate positions
        t_start = t_idx * t_overlap_size
        h_start = h_idx * s_overlap_size
        w_start = w_idx * s_overlap_size
        # Extract and process tile
        tile = z[
            :,
            :,
            t_start : t_start + self.tile_latent_min_tsize + 1,
            h_start : h_start + self.tile_latent_min_size,
            w_start : w_start + self.tile_latent_min_size,
        ]
        # Process tile
        if self.post_quant_conv is not None:
            tile = self.post_quant_conv(tile)
        if use_conv_cache:
            decoded, conv_cache = self.decoder(tile, conv_cache)
        else:
            decoded = self.decoder(tile)

        if t_start > 0:
            decoded = decoded[:, :, 1:, :, :]
        # Store metadata
        shape = decoded.shape
        # Store decoded data (flattened)
        decoded_flat = decoded.reshape(-1)
        local_results.append(decoded_flat)
        local_dim_metadata.append(shape)

    results = torch.cat(local_results, dim=0).contiguous()
    # first gather size to pad the results
    local_size = torch.tensor([results.size(0)], device=results.device, dtype=torch.int64)
    all_sizes = [torch.zeros(1, device=results.device, dtype=torch.int64) for _ in range(world_size)]
    dist.all_gather(all_sizes, local_size)
    max_size = max(size.item() for size in all_sizes)
    padded_results = torch.zeros(max_size, device=results.device)
    padded_results[: results.size(0)] = results
    # Gather all results
    gathered_dim_metadata = [None] * world_size
    gathered_results = (
        torch.zeros_like(padded_results).repeat(world_size, *[1] * len(padded_results.shape)).contiguous()
    )  # use contiguous to make sure it won't copy data in the following operations
    dist.all_gather_into_tensor(gathered_results, padded_results)
    torch.cuda.set_device(rank)
    dist.all_gather_object(gathered_dim_metadata, local_dim_metadata)
    # Process gathered results
    data = [[[[] for _ in range(num_w_tiles)] for _ in range(num_h_tiles)] for _ in range(num_t_tiles)]
    for current_data, global_idx in parallel_data_generator(gathered_results, gathered_dim_metadata):
        t_idx = global_idx // total_spatial_tiles
        spatial_idx = global_idx % total_spatial_tiles
        h_idx = spatial_idx // num_w_tiles
        w_idx = spatial_idx % num_w_tiles
        data[t_idx][h_idx][w_idx] = current_data
    # Merge results
    result_slices = []
    last_slice_data = None
    for i, tem_data in enumerate(data):
        slice_data = merge_spatial_tiles_cog(tem_data, s_blend_extent, s_row_limit)
        if i > 0:
            slice_data = blend_t(last_slice_data, slice_data, t_blend_extent)
            result_slices.append(slice_data[:, :, :t_limit, :, :])
        else:
            result_slices.append(slice_data[:, :, : t_limit + 1, :, :])
        last_slice_data = slice_data
    dec = torch.cat(result_slices, dim=2)
    if not return_dict:
        return (dec,)
    return DecoderOutput(sample=dec)


def padded_tensor(tensor, target_shape):
    """Pad tensor to target shape and return the padding sizes."""
    padding = []
    for dim in range(len(tensor.shape)):
        pad_size = target_shape[dim] - tensor.shape[dim]
        if pad_size < 0:
            raise ValueError(
                f"Target shape {target_shape} is smaller than tensor shape {tensor.shape} in dimension {dim}."
            )

        # Ensure padding is split evenly
        pad_left = pad_size // 2
        pad_right = pad_size - pad_left  # Ensure total padding is equal
        padding.append((pad_left, pad_right))

    # Flatten padding for the pad function
    padding_flat = [p for pair in padding[::-1] for p in pair]  # Reverse and flatten

    padded = torch.nn.functional.pad(tensor, padding_flat)  # Apply padding
    padding_tensor = torch.tensor(padding, device=tensor.device)
    return padded, padding_tensor  # Return both padded tensor and padding sizes


def depadded_tensor(padded, padding):
    """Remove padding from the padded tensor based on the recorded padding sizes."""
    padding_flat = [p for pair in padding for p in pair]  # Flatten padding for the unpad function
    # Check tensor dimensions

    # Create slices to remove padding
    slices = []
    for i in range(0, len(padding_flat), 2):
        start = padding_flat[i]
        end = padded.shape[i // 2] - padding_flat[i + 1]
        if end < start:
            raise ValueError(f"Invalid slice: start {start} should not be greater than end {end}.")
        slices.append(slice(start, end))

    return padded[tuple(slices)]


def tiled_encode_parallel(
    self,
    x: torch.Tensor,
    *,
    overlap_height: int,
    overlap_width: int,
    blend_extent_height: int,
    blend_extent_width: int,
    row_limit_height: int,
    row_limit_width: int,
    num_tiles_height: int,
    num_tiles_width: int,
    use_conv_cache: bool = True,
) -> torch.Tensor:
    """
    Encode a 5-D video/latent tensor in a memory-friendly, tile-wise fashion
    and seamlessly stitch the individual tiles back together.

    Parameters
    ----------
    x
        Input tensor of shape ``(B, C, T, H, W)``.
    overlap_height / overlap_width
        Stride between adjacent tiles (in spatial pixels); determines
        the *sample*-space overlap.
    blend_extent_height / blend_extent_width
        Width (in latent pixels) of the vertical / horizontal band used
        to alpha-blend neighbouring tiles.
    row_limit_height / row_limit_width
        Portion of each tile that survives after blending
        (``tile_latent_size – blend_extent``).
    num_tiles_height / num_tiles_width
        Grid dimensions computed for the sample-space tiling.

    Returns
    -------
    torch.Tensor
        The full latent tensor (`B × C × T × latent_H × latent_W`)
        reconstructed from the tiled encoding.
    """

    # ------------------------------------------------------------------
    # 1) Distributed bookkeeping
    # ------------------------------------------------------------------
    batch_size, num_channels, num_frames, height, width = x.shape
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Flatten the 2-D tile grid into a list and shard it across ranks
    tile_indices = [(i, j) for i in range(num_tiles_height) for j in range(num_tiles_width)]
    total_tiles = len(tile_indices)
    tiles_per_rank = total_tiles // world_size
    start_index = rank * tiles_per_rank
    end_index = start_index + tiles_per_rank if rank != world_size - 1 else total_tiles

    # ------------------------------------------------------------------
    # 2) Encode the tiles assigned to this rank
    # ------------------------------------------------------------------
    decoded_tiles: list[torch.Tensor] = []
    padding_records: list[torch.Tensor] = []

    for index in range(start_index, end_index):
        i, j = tile_indices[index]

        # Sample-space boundaries of the current tile
        start_h = i * overlap_height
        end_h = min(start_h + self.tile_sample_min_height, height)
        start_w = j * overlap_width
        end_w = min(start_w + self.tile_sample_min_width, width)
        tile = x[:, :, :, start_h:end_h, start_w:end_w].to(f"cuda:{rank}")

        # Encode tile → latent, then (optionally) post-process
        conv_cache = None
        if use_conv_cache:
            enc_tile, conv_cache = self.encoder(tile, conv_cache=conv_cache)
        else:
            enc_tile = self.encoder(tile)
        if self.quant_conv is not None:
            enc_tile = self.quant_conv(enc_tile)

        # Right-pad to the uniform latent size expected by collective ops
        padded_tile, padding = padded_tensor(enc_tile, self.max_shape_encode)
        decoded_tiles.append(padded_tile)
        padding_records.append(padding)

    # ------------------------------------------------------------------
    # 3) All-gather paddings and tiles so every rank holds the full batch
    # ------------------------------------------------------------------
    # Gather padding records
    gathered_pads = [torch.empty_like(torch.cat(padding_records, dim=0)) for _ in range(world_size)]
    dist.all_gather(gathered_pads, torch.cat(padding_records, dim=0).contiguous())
    gathered_paddings_tensor = torch.cat(gathered_pads, dim=0)

    # Gather decoded tiles
    gathered_tiles = [torch.empty_like(torch.cat(decoded_tiles, dim=0)) for _ in range(world_size)]
    dist.all_gather(gathered_tiles, torch.cat(decoded_tiles, dim=0).contiguous())

    # ------------------------------------------------------------------
    # 4) Build a convenient structure for depadding
    # ------------------------------------------------------------------
    gathered_paddings1 = gathered_paddings_tensor.view(
        num_tiles_height, num_tiles_width, 5, *gathered_paddings_tensor.shape[1:]
    )

    gathered_paddings: list[list[list[tuple[int, int]]]] = []
    for i in range(num_tiles_height):
        row: list[list[tuple[int, int]]] = []
        for j in range(num_tiles_width):
            padding = gathered_paddings1[i, j]
            # `p` is shaped (2,) → convert to Python tuple
            padding_list = [tuple(p.cpu().numpy()) for p in padding]
            row.append(padding_list)
        gathered_paddings.append(row)

    # ------------------------------------------------------------------
    # 5) Depad every tile and place them in a 2-D latent grid
    # ------------------------------------------------------------------
    all_tiles = torch.cat(gathered_tiles, dim=0)
    latent_grid = all_tiles.view(num_tiles_height, num_tiles_width, *all_tiles.shape[1:])

    rows: list[list[torch.Tensor]] = [
        [depadded_tensor(latent_grid[i, j].unsqueeze(0), gathered_paddings[i][j]) for j in range(num_tiles_width)]
        for i in range(num_tiles_height)
    ]

    # ------------------------------------------------------------------
    # 6) Blend overlaps to avoid seams
    # ------------------------------------------------------------------
    blended_rows: list[torch.Tensor] = []
    for i, row in enumerate(rows):
        blended_row: list[torch.Tensor] = []
        for j, tile in enumerate(row):
            # Blend with tile above / left whenever those exist
            if i > 0:
                tile = blend_v(rows[i - 1][j], tile, blend_extent_height)
            if j > 0:
                tile = blend_h(row[j - 1], tile, blend_extent_width)

            # Retain only the valid interior (non-overlapping strip)
            tile = tile[:, :, :, :row_limit_height, :row_limit_width]
            blended_row.append(tile)

        # Concatenate tiles horizontally
        blended_rows.append(torch.cat(blended_row, dim=4))

    # ------------------------------------------------------------------
    # 7) Concatenate the rows vertically to obtain the full latent
    # ------------------------------------------------------------------
    enc = torch.cat(blended_rows, dim=3)

    return enc


def merge_spatial_tiles_blend(
    spatial_rows: list[list[torch.Tensor]],
    h_blend_extent: int,
    w_blend_extent: int,
):
    n_rows = len(spatial_rows)
    result_rows = []
    for i, row in enumerate(spatial_rows):
        result_row = []
        n_cols = len(row)
        for j, tile in enumerate(row):
            if i > 0:
                tile = blend_v(spatial_rows[i - 1][j], tile, h_blend_extent)
            if j > 0:
                tile = blend_h(row[j - 1], tile, w_blend_extent)

            # Decide how much of the blending padding to crop away.
            # sample_pad_size = h_blend_extent // 2  (same logic applies to width).
            #
            # – Interior tiles (not on the outer video border):
            #   • We discard the full padding on the *bottom* and *right* sides
            #     (h_blend_extent == 2 * sample_pad_size), because those pixels
            #     have already been blended with the neighbouring tiles.
            #   • We keep the padding on the *top* and *left* sides so the
            #     overlap used for blending is preserved for the tiles that come
            #     before them.
            #
            # – Border tiles:
            #   • First row / first column: crop only the outer half-padding
            #     (sample_pad_size) on the *top* / *left* edge so the frame keeps
            #     its blend with the interior while removing the extra canvas
            #     outside the video.
            #   • Last row / last column: same idea for the *bottom* / *right*
            #     edge.
            #
            # The variables below encode this logic.
            top_crop = h_blend_extent // 2 if i == 0 else 0
            bottom_crop = h_blend_extent // 2 if i == n_rows - 1 else h_blend_extent
            left_crop = w_blend_extent // 2 if j == 0 else 0
            right_crop = w_blend_extent // 2 if j == n_cols - 1 else w_blend_extent

            result_row.append(tile[..., top_crop:-bottom_crop, left_crop:-right_crop])

        result_rows.append(torch.cat(result_row, dim=-1))
    return torch.cat(result_rows, dim=-2)


def merge_spatial_tiles(
    spatial_rows: list[list[torch.Tensor]],
    sample_pad_size: int,
):
    result_rows = []
    for i, row in enumerate(spatial_rows):
        result_row = []
        for j, tile in enumerate(row):
            result_row.append(tile[:, :, :, sample_pad_size:-sample_pad_size, sample_pad_size:-sample_pad_size])
        result_rows.append(torch.cat(result_row, dim=-1))
    return torch.cat(result_rows, dim=-2)


def lcm(a: int, b: int) -> int:
    return abs(a * b) // math.gcd(a, b)


def floor_to_multiple(x: int, m: int) -> int:
    return (x // m) * m


def get_patch_hw(patch_size):
    if isinstance(patch_size, int):
        return patch_size, patch_size
    ps = list(patch_size)
    if len(ps) == 1:
        return int(ps[0]), int(ps[0])
    return int(ps[-2]), int(ps[-1])
