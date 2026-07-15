"""
Sequence Parallelism - Communication Module

通信原语和复合操作，提供完整的序列并行通信功能。

## 模块结构

1. Primitives - 基础通信原语 (Autograd)
   - AllGather: 收集所有进程的张量
   - ReduceScatter: 分块求和并散射
   - AllToAllAutograd: 通用 AllToAll
   - AllToAll4DAutograd: 4D AllToAll (序列并行专用)
   - AllGatherOverlapped: 通信计算重叠 (两卡专用)

2. Collective - 复合通信操作
   - gather_sequence: 收集完整序列
   - split_sequence: 切分序列
   - all_to_all_4d: 4D AllToAll 便捷函数
   - all_to_all_before_attn: Attention 前转换
   - all_to_all_after_attn: Attention 后转换

## 向后兼容

所有 API 与原 comm.py 保持兼容。
"""

from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor
from torch.distributed import ProcessGroup, Work

from .errors import IncompatibleDimensionError
from .padding import get_pad_manager

# =============================================================================
# AllGather - 沿维度 0 收集所有进程的张量
# =============================================================================


class AllGather(torch.autograd.Function):
    """AllGather 通信原语

    Forward: 收集所有进程的 tensor，沿维度 0 拼接
    Backward: ReduceScatter 梯度

    Input:  tensor of shape (*input_shape)
    Output: tensor of shape (world_size, *input_shape)

    Example:
        >>> # 单进程模式
        >>> AllGatter.apply(None, tensor, group, False)
        (tensor.unsqueeze(0), None)
        >>>
        >>> # 多进程模式
        >>> output, handle = AllGather.apply(None, tensor, group, True)
    """

    @staticmethod
    def forward(
        ctx: Any,
        tensor: Tensor,
        group: ProcessGroup | None = None,
        overlap: bool = False,
    ) -> tuple[Tensor, Work | None]:
        """AllGather 前向传播

        Args:
            ctx: Autograd context
            tensor: 输入张量
            group: ProcessGroup，None 表示全局
            overlap: 是否异步执行 (用于通信计算重叠)

        Returns:
            (output, handle)
            - output: shape=(world_size, *input_shape)
            - handle: 异步句柄，同步模式为 None
        """
        if ctx is None and overlap:
            raise ValueError("ctx cannot be None when overlap=True")

        if ctx is not None:
            ctx.comm_grp = group

        world_size = dist.get_world_size(group)

        # 单进程优化
        if world_size == 1:
            return tensor.unsqueeze(0), None

        # 创建输出缓冲区
        buffer_shape = (world_size,) + tensor.shape
        output = torch.empty(buffer_shape, dtype=tensor.dtype, device=tensor.device)
        buffer_list = list(torch.chunk(output, world_size, dim=0))

        if overlap:
            handle = dist.all_gather(buffer_list, tensor, group=group, async_op=True)
            return output, handle
        else:
            dist.all_gather(buffer_list, tensor, group=group)
            return output, None

    @staticmethod
    def backward(ctx: Any, *grad_outputs) -> tuple[Tensor, None, None]:
        """AllGather 反向传播 = ReduceScatter"""
        grad_input = ReduceScatter.forward(None, grad_outputs[0], ctx.comm_grp, False)[0]
        return grad_input, None, None


# =============================================================================
# ReduceScatter - 沿维度 0 分块求和并散射
# =============================================================================


class ReduceScatter(torch.autograd.Function):
    """ReduceScatter 通信原语

    Forward: 对 tensor 沿维度 0 分块求和并散射到各进程
    Backward: AllGather 梯度

    Input:  tensor of shape (world_size, *output_shape)
    Output: tensor of shape (*output_shape)

    Example:
        >>> output, handle = ReduceScatter.apply(None, tensor, group, False)
    """

    @staticmethod
    def forward(
        ctx: Any,
        tensor: Tensor,
        group: ProcessGroup,
        overlap: bool = False,
    ) -> tuple[Tensor, Work | None]:
        """ReduceScatter 前向传播

        Args:
            ctx: Autograd context
            tensor: 输入张量，shape[0] 必须等于 world_size
            group: ProcessGroup
            overlap: 是否异步执行

        Returns:
            (output, handle)
        """
        if ctx is None and overlap:
            raise ValueError("ctx cannot be None when overlap=True")

        if ctx is not None:
            ctx.comm_grp = group

        world_size = dist.get_world_size(group)

        # 单进程优化
        if world_size == 1:
            return tensor.squeeze(0), None

        if not tensor.is_contiguous():
            tensor = tensor.contiguous()

        # 创建输出缓冲区
        output_shape = tensor.shape[1:]
        output = torch.empty(output_shape, dtype=tensor.dtype, device=tensor.device)
        buffer_list = list(torch.chunk(tensor, world_size, dim=0))

        if overlap:
            handle = dist.reduce_scatter(output, buffer_list, group=group, async_op=True)
            return output, handle
        else:
            dist.reduce_scatter(output, buffer_list, group=group)
            return output, None

    @staticmethod
    def backward(ctx: Any, *grad_outputs) -> tuple[Tensor, None, None]:
        """ReduceScatter 反向传播 = AllGather"""
        grad_input = AllGather.forward(None, grad_outputs[0], ctx.comm_grp, False)[0]
        return grad_input, None, None


# =============================================================================
# AllToAll - 通用 AllToAll 操作
# =============================================================================


class AllToAllAutograd(torch.autograd.Function):
    """AllToAll 通信原语

    Forward: 沿 scatter_dim 分块，沿 gather_dim 拼接
    Backward: 交换 scatter_dim 和 gather_dim 的 AllToAll

    Example:
        >>> output = AllToAllAutograd.apply(tensor, group, scatter_dim, gather_dim)
    """

    @staticmethod
    def forward(ctx: Any, tensor: Tensor, group: ProcessGroup, scatter_dim: int, gather_dim: int) -> Tensor:
        """AllToAll 前向传播

        Args:
            ctx: Autograd context
            tensor: 输入张量
            group: ProcessGroup
            scatter_dim: 散射维度
            gather_dim: 收集维度

        Returns:
            转换后的张量
        """
        ctx.process_group = group
        ctx.scatter_dim = scatter_dim
        ctx.gather_dim = gather_dim

        world_size = dist.get_world_size(group)

        if world_size == 1:
            return tensor

        # 分块
        input_list = [t.contiguous() for t in torch.tensor_split(tensor, world_size, scatter_dim)]
        output_list = [torch.empty_like(input_list[0]) for _ in range(world_size)]

        # AllToAll
        dist.all_to_all(output_list, input_list, group=group)

        # 拼接
        return torch.cat(output_list, dim=gather_dim).contiguous()

    @staticmethod
    def backward(ctx: Any, *grad_outputs) -> tuple[Tensor, None, None, None]:
        """AllToAll 反向传播"""
        grad_input = AllToAllAutograd.apply(
            grad_outputs[0],
            ctx.process_group,
            ctx.gather_dim,  # 注意：交换 scatter 和 gather
            ctx.scatter_dim,
        )
        return grad_input, None, None, None


# =============================================================================
# AllToAll 4D - 序列并行专用
# =============================================================================


class AllToAll4DAutograd(torch.autograd.Function):
    """4D AllToAll 通信原语 (序列并行专用)

    专门优化序列并行中的维度转换。

    scatter_dim=2, gather_dim=1:
        Input:  (bs, seqlen/P, h)
        Output: (bs, seqlen, h/P)

    scatter_dim=1, gather_dim=2:
        Input:  (bs, seqlen, h/P)
        Output: (bs, seqlen/P, h)
    """

    @staticmethod
    def forward(
        ctx: Any,
        group: ProcessGroup,
        tensor: Tensor,
        scatter_dim: int,
        gather_dim: int,
        use_sync: bool = False,
    ) -> Tensor:
        """4D AllToAll 前向传播"""
        ctx.group = group
        ctx.scatter_dim = scatter_dim
        ctx.gather_dim = gather_dim
        ctx.use_sync = use_sync

        return _all_to_all_4d_impl(tensor, scatter_dim, gather_dim, group, use_sync)

    @staticmethod
    def backward(ctx: Any, *grad_output: Tensor) -> tuple[None, Tensor, None, None, None]:
        """4D AllToAll 反向传播"""
        return (
            None,
            AllToAll4DAutograd.apply(
                ctx.group,
                grad_output[0],
                ctx.gather_dim,  # 交换维度
                ctx.scatter_dim,
                ctx.use_sync,
            ),
            None,
            None,
            None,
        )


def _all_to_all_4d_impl(
    tensor: Tensor, scatter_dim: int, gather_dim: int, group: ProcessGroup, use_sync: bool
) -> Tensor:
    """4D AllToAll 实现细节"""
    world_size = dist.get_world_size(group)

    if world_size == 1:
        return tensor

    if scatter_dim == 2 and gather_dim == 1:
        # (bs, seqlen/P, h) -> (bs, seqlen, h/P)
        bs, shard_seqlen, h = tensor.shape
        seqlen = shard_seqlen * world_size
        shard_h = h // world_size

        input_t = tensor.reshape(bs, shard_seqlen, world_size, shard_h).transpose(0, 2).contiguous()

        output = torch.empty_like(input_t)
        dist.all_to_all_single(output, input_t, group=group)

        if use_sync:
            torch.cuda.synchronize()

        output = output.reshape(seqlen, bs, shard_h)
        output = output.transpose(0, 1).contiguous().reshape(bs, seqlen, shard_h)
        return output

    elif scatter_dim == 1 and gather_dim == 2:
        # (bs, seqlen, h/P) -> (bs, seqlen/P, h)
        bs, seqlen, shard_h = tensor.shape
        h = shard_h * world_size
        shard_seqlen = seqlen // world_size

        input_t = (
            tensor.reshape(bs, world_size, shard_seqlen, shard_h)
            .transpose(0, 3)
            .transpose(0, 1)
            .contiguous()
            .reshape(world_size, shard_h, shard_seqlen, bs)
        )

        output = torch.empty_like(input_t)
        dist.all_to_all_single(output, input_t, group=group)

        if use_sync:
            torch.cuda.synchronize()

        output = output.reshape(h, shard_seqlen, bs)
        output = output.transpose(0, 2).contiguous().reshape(bs, shard_seqlen, h)
        return output

    else:
        raise ValueError(
            f"Invalid dimensions: scatter_dim={scatter_dim}, gather_dim={gather_dim}. "
            f"Must be (scatter_dim=1, gather_dim=2) or (scatter_dim=2, gather_dim=1)."
        )


# =============================================================================
# AllGather Overlapped - 通信计算重叠 (两卡专用)
# =============================================================================


class AllGatherOverlapped(torch.autograd.Function):
    """AllGather 通信计算重叠 (两卡场景专用)

    在执行 AllGather 的同时计算本地 QKV，实现通信计算重叠。
    仅适用于 world_size=2 的场景。

    Example:
        >>> qkv = AllGatherOverlapped.apply(inputs, weight, bias, sp_rank, sp_size, group)
    """

    @staticmethod
    def forward(
        ctx: Any,
        inputs: Tensor,
        weight: Tensor,
        bias: Tensor,
        sp_rank: int,
        sp_size: int,
        group: ProcessGroup | None = None,
    ) -> Tensor:
        """通信计算重叠 AllGather

        Args:
            ctx: Autograd context
            inputs: 输入张量
            weight: 线性层权重
            bias: 线性层偏置
            sp_rank: 序列并行 rank
            sp_size: 序列并行 size
            group: ProcessGroup

        Returns:
            QKV 张量
        """
        from torch.distributed._functional_collectives import all_gather_tensor

        ctx.group = group
        ctx.sp_rank = sp_rank
        ctx.sp_size = sp_size

        # AllGather inputs
        all_inputs = all_gather_tensor(inputs.unsqueeze(0), 0, group)

        # 计算本地 QKV
        local_qkv = F.linear(inputs, weight, bias).unsqueeze(0)

        # 计算远程 QKV
        remote_inputs = all_inputs[1 - sp_rank].view(list(local_qkv.shape[:-1]) + [-1])
        remote_qkv = F.linear(remote_inputs, weight, bias)

        # 拼接
        if sp_rank == 0:
            qkv = torch.cat([local_qkv, remote_qkv], dim=0)
        else:
            qkv = torch.cat([remote_qkv, local_qkv], dim=0)
        qkv = rearrange(qkv, "sp b n c -> b (sp n) c")

        ctx.save_for_backward(inputs, weight, remote_inputs)
        return qkv

    @staticmethod
    def backward(ctx: Any, *grad_outputs) -> tuple[Tensor, Tensor, Tensor, None, None, None]:
        """反向传播"""
        from torch.distributed._functional_collectives import reduce_scatter_tensor

        group = ctx.group
        sp_rank = ctx.sp_rank
        sp_size = ctx.sp_size
        inputs, weight, remote_inputs = ctx.saved_tensors

        # 分割梯度
        qkv_grad = grad_outputs[0]
        qkv_grad = rearrange(qkv_grad, "b (sp n) c -> sp b n c", sp=sp_size)
        qkv_grad = torch.chunk(qkv_grad, 2, dim=0)

        if sp_rank == 0:
            local_qkv_grad, remote_qkv_grad = qkv_grad
        else:
            remote_qkv_grad, local_qkv_grad = qkv_grad

        # 计算远程梯度
        remote_inputs_grad = torch.matmul(remote_qkv_grad, weight).squeeze(0)
        weight_grad = torch.matmul(remote_qkv_grad.transpose(-1, -2), remote_inputs).squeeze(0).sum(0)
        bias_grad = remote_qkv_grad.squeeze(0).sum(0).sum(0)

        # 异步 ReduceScatter
        remote_inputs_grad_zero = torch.zeros_like(remote_inputs_grad)
        if sp_rank == 0:
            remote_inputs_grad = torch.cat([remote_inputs_grad_zero, remote_inputs_grad], dim=0)
        else:
            remote_inputs_grad = torch.cat([remote_inputs_grad, remote_inputs_grad_zero], dim=0)
        remote_inputs_grad = reduce_scatter_tensor(remote_inputs_grad, "sum", 0, group)

        # 计算本地梯度
        local_input_grad = torch.matmul(local_qkv_grad, weight).squeeze(0)
        weight_grad += torch.matmul(local_qkv_grad.transpose(-1, -2), inputs).squeeze(0).sum(0)
        bias_grad += local_qkv_grad.squeeze(0).sum(0).sum(0)

        inputs_grad = remote_inputs_grad + local_input_grad
        return inputs_grad, weight_grad, bias_grad, None, None, None


# =============================================================================
# 便捷函数
# =============================================================================


def all_gather(tensor: Tensor, group: ProcessGroup | None = None, overlap: bool = False) -> tuple[Tensor, Work | None]:
    """AllGather 便捷函数

    Args:
        tensor: 输入张量
        group: ProcessGroup
        overlap: 是否异步

    Returns:
        (output, handle)
    """
    return AllGather.apply(None, tensor, group, overlap)


def reduce_scatter(tensor: Tensor, group: ProcessGroup, overlap: bool = False) -> tuple[Tensor, Work | None]:
    """ReduceScatter 便捷函数

    Args:
        tensor: 输入张量
        group: ProcessGroup
        overlap: 是否异步

    Returns:
        (output, handle)
    """
    return ReduceScatter.apply(None, tensor, group, overlap)


def all_to_all(tensor: Tensor, group: ProcessGroup, scatter_dim: int = 2, gather_dim: int = 1) -> Tensor:
    """AllToAll 便捷函数

    Args:
        tensor: 输入张量
        group: ProcessGroup
        scatter_dim: 散射维度
        gather_dim: 收集维度

    Returns:
        转换后的张量
    """
    return AllToAllAutograd.apply(tensor, group, scatter_dim, gather_dim)


# =============================================================================
# Tensor Padding
# =============================================================================


def pad_tensor(tensor: Tensor, dim: int, pad: int) -> Tensor:
    """沿指定维度 padding 张量

    Args:
        tensor: 输入张量
        dim: padding 维度
        pad: padding 大小

    Returns:
        Padding 后的张量

    Example:
        >>> x = torch.randn(2, 10, 64)
        >>> padded = pad_tensor(x, dim=1, pad=2)
        >>> padded.shape
        torch.Size([2, 12, 64])
    """
    if pad <= 0:
        return tensor

    pad_size = list(tensor.shape)
    pad_size[dim] = pad
    padding = torch.zeros(pad_size, dtype=tensor.dtype, device=tensor.device)

    return torch.cat([tensor, padding], dim=dim)


# =============================================================================
# Sequence Gather & Split (带 Autograd)
# =============================================================================


class _GatherForwardSplitBackward(torch.autograd.Function):
    """Gather 序列 (前向) / Split 序列 (反向)

    前向: AllGather 收集所有分片
    反向: Split 切分梯度到各进程
    """

    @staticmethod
    def symbolic(graph, tensor: Tensor) -> Tensor:
        return tensor

    @staticmethod
    def forward(ctx, tensor: Tensor, process_group: ProcessGroup, dim: int, grad_scale: str, pad: int) -> Tensor:
        ctx.process_group = process_group
        ctx.dim = dim
        ctx.grad_scale = grad_scale
        ctx.pad = pad

        return _gather_sequence_impl(tensor, process_group, dim, pad)

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        # 梯度缩放
        if ctx.grad_scale == "up":
            grad_output = grad_output * dist.get_world_size(ctx.process_group)
        elif ctx.grad_scale == "down":
            grad_output = grad_output / dist.get_world_size(ctx.process_group)

        grad_input = _split_sequence_impl(grad_output, ctx.process_group, ctx.dim, ctx.pad)
        return grad_input, None, None, None, None


class _SplitForwardGatherBackward(torch.autograd.Function):
    """Split 序列 (前向) / Gather 序列 (反向)

    前向: Split 切分到各进程
    反向: AllGather 收集梯度
    """

    @staticmethod
    def symbolic(graph, tensor: Tensor) -> Tensor:
        return tensor

    @staticmethod
    def forward(ctx, tensor: Tensor, process_group: ProcessGroup, dim: int, grad_scale: str, pad: int) -> Tensor:
        ctx.process_group = process_group
        ctx.dim = dim
        ctx.grad_scale = grad_scale
        ctx.pad = pad

        return _split_sequence_impl(tensor, process_group, dim, pad)

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        # 梯度缩放
        if ctx.grad_scale == "up":
            grad_output = grad_output * dist.get_world_size(ctx.process_group)
        elif ctx.grad_scale == "down":
            grad_output = grad_output / dist.get_world_size(ctx.process_group)

        grad_input = _gather_sequence_impl(grad_output, ctx.process_group, ctx.dim, ctx.pad)
        return grad_input, None, None, None, None


def _split_sequence_impl(tensor: Tensor, group: ProcessGroup, dim: int, pad: int) -> Tensor:
    """Split 序列实现

    Args:
        tensor: 输入张量
        group: ProcessGroup
        dim: 切分维度
        pad: padding 大小

    Returns:
        切分后的张量
    """
    world_size = dist.get_world_size(group)
    rank = dist.get_rank(group)

    if world_size == 1:
        return tensor

    # Padding
    if pad > 0:
        tensor = pad_tensor(tensor, dim, pad)

    dim_size = tensor.size(dim)
    if dim_size % world_size != 0:
        raise IncompatibleDimensionError(dim, dim_size, world_size, operation="split")

    # 切分
    chunk_size = dim_size // world_size
    tensor_list = torch.split(tensor, chunk_size, dim=dim)

    return tensor_list[rank].contiguous()


def _gather_sequence_impl(tensor: Tensor, group: ProcessGroup, dim: int, pad: int) -> Tensor:
    """Gather 序列实现

    Args:
        tensor: 输入张量
        group: ProcessGroup
        dim: 收集维度
        pad: padding 大小

    Returns:
        收集后的张量
    """
    tensor = tensor.contiguous()
    world_size = dist.get_world_size(group)

    if world_size == 1:
        return tensor

    # AllGather
    tensor_list = [torch.empty_like(tensor) for _ in range(world_size)]
    dist.all_gather(tensor_list, tensor, group=group)

    # 拼接
    output = torch.cat(tensor_list, dim=dim)

    # 去除 padding
    if pad > 0:
        output = output.narrow(dim, 0, output.size(dim) - pad)

    return output


def gather_sequence(
    tensor: Tensor, process_group: ProcessGroup, dim: int, grad_scale: str = "", pad: int = 0
) -> Tensor:
    """收集序列分片

    前向传播时 AllGather 收集所有分片，反向传播时 Split 切分梯度。

    Args:
        tensor: 输入张量 (本地分片)
        process_group: ProcessGroup
        dim: 收集维度
        grad_scale: 梯度缩放模式 ("up", "down", "")
        pad: padding 大小

    Returns:
        完整序列张量

    Example:
        >>> # 本地张量 shape=(bs, seqlen/P, hidden)
        >>> full_seq = gather_sequence(local_tensor, sp_group, dim=1)
        >>> # full_seq shape=(bs, seqlen, hidden)
    """
    return _GatherForwardSplitBackward.apply(tensor, process_group, dim, grad_scale, pad)


def split_sequence(tensor: Tensor, process_group: ProcessGroup, dim: int, grad_scale: str = "", pad: int = 0) -> Tensor:
    """切分序列

    前向传播时 Split 切分到各进程，反向传播时 AllGather 收集梯度。

    Args:
        tensor: 输入张量 (完整序列)
        process_group: ProcessGroup
        dim: 切分维度
        grad_scale: 梯度缩放模式
        pad: padding 大小

    Returns:
        本地分片张量

    Example:
        >>> # 完整张量 shape=(bs, seqlen, hidden)
        >>> local_seq = split_sequence(full_tensor, sp_group, dim=1)
        >>> # local_seq shape=(bs, seqlen/P, hidden)
    """
    return _SplitForwardGatherBackward.apply(tensor, process_group, dim, grad_scale, pad)


# =============================================================================
# AllToAll 4D (序列并行专用)
# =============================================================================


def all_to_all_4d(
    tensor: Tensor,
    scatter_dim: int = 2,
    gather_dim: int = 1,
    group: ProcessGroup | None = None,
    use_sync: bool = False,
) -> Tensor:
    """4D AllToAll 操作 (序列并行专用)

    用于序列并行中的 head 和 sequence 维度转换。

    scatter_dim=2, gather_dim=1:
        Input:  (bs, seqlen/P, h)
        Output: (bs, seqlen, h/P)
        说明: 从序列切分转换为 head 切分

    scatter_dim=1, gather_dim=2:
        Input:  (bs, seqlen, h/P)
        Output: (bs, seqlen/P, h)
        说明: 从 head 切分转换为序列切分

    Args:
        tensor: 输入张量
        scatter_dim: 散射维度 (1 或 2)
        gather_dim: 收集维度 (1 或 2)
        group: ProcessGroup
        use_sync: 是否同步等待

    Returns:
        转换后的张量

    Example:
        >>> # Attention 前: (bs, seqlen/P, h) -> (bs, seqlen, h/P)
        >>> x = all_to_all_4d(x, scatter_dim=2, gather_dim=1, group=sp_group)
        >>>
        >>> # Attention 后: (bs, seqlen, h/P) -> (bs, seqlen/P, h)
        >>> y = all_to_all_4d(y, scatter_dim=1, gather_dim=2, group=sp_group)
    """
    return AllToAll4DAutograd.apply(group, tensor, scatter_dim, gather_dim, use_sync)


# =============================================================================
# Attention 前后的 AllToAll (带 Padding)
# =============================================================================


def all_to_all_before_attn(
    tensor: Tensor, process_group: ProcessGroup | None = None, scatter_dim: int = 2, gather_dim: int = 1
) -> Tensor:
    """Attention 前的 AllToAll

    将序列切分的张量转换为 head 切分，供 Attention 计算。
    自动处理 padding。

    Args:
        tensor: 输入张量 (bs, seqlen/P, h)
        process_group: ProcessGroup
        scatter_dim: 散射维度
        gather_dim: 收集维度

    Returns:
        转换后的张量 (bs, seqlen, h/P)
    """
    pad = get_pad_manager().get_or_default("pad", 0)

    # AllToAll
    output = AllToAllAutograd.apply(tensor, process_group, scatter_dim, gather_dim)

    # 去除 padding
    if pad > 0:
        output = output.narrow(gather_dim, 0, output.size(gather_dim) - pad)

    return output


def all_to_all_after_attn(
    tensor: Tensor, process_group: ProcessGroup | None = None, scatter_dim: int = 1, gather_dim: int = 2
) -> Tensor:
    """Attention 后的 AllToAll

    将 head 切分的张量转换回序列切分。
    自动处理 padding。

    Args:
        tensor: 输入张量 (bs, seqlen, h/P)
        process_group: ProcessGroup
        scatter_dim: 散射维度
        gather_dim: 收集维度

    Returns:
        转换后的张量 (bs, seqlen/P, h)
    """
    pad = get_pad_manager().get_or_default("pad", 0)

    # 添加 padding
    if pad > 0:
        tensor = pad_tensor(tensor, scatter_dim, pad)

    # AllToAll
    return AllToAllAutograd.apply(tensor, process_group, scatter_dim, gather_dim)


# =============================================================================
# Batch Function Helper
# =============================================================================


def batch_func(func, *args):
    """对 batch_size=2 的张量应用函数

    仅处理 shape[0]==2 的张量，其他参数原样返回。
    用于处理 CFG (Classifier-Free Guidance) 场景。

    Args:
        func: 要应用的函数
        *args: 参数列表

    Returns:
        处理后的参数列表

    Example:
        >>> def double(x):
        ...     return x * 2
        >>>
        >>> tensor = torch.randn(2, 10)
        >>> scalar = 5
        >>> result = batch_func(double, tensor, scalar)
        >>> # result[0] = tensor * 2
        >>> # result[1] = 5 (原样)
    """
    batch = []
    for arg in args:
        if isinstance(arg, torch.Tensor) and arg.shape[0] == 2:
            batch.append(func(arg))
        else:
            batch.append(arg)

    return batch
