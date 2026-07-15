"""
Cache 加速模块基础组件

提供缓存策略抽象基类、TeaCache/MagCache 实现、以及通用工具类。
"""

import contextlib
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from diffusers.utils import logging

# 从统一配置模块导入
from x_base.config import load_cache_config

logger = logging.get_logger(__name__)

# ============ 分布式通信工具 ============
if dist.is_available():
    import torch.distributed._functional_collectives as ft_c
    import torch.distributed.distributed_c10d as c10d
else:
    ft_c = None
    c10d = None


def get_group(group=None):
    """获取分布式进程组"""
    if group is None:
        group = c10d._get_default_group()
    if isinstance(group, dist.ProcessGroup):
        pg: dist.ProcessGroup | list[dist.ProcessGroup] = group
    else:
        pg = group.get_group()
    return pg


def _maybe_wait(tensor: torch.Tensor) -> torch.Tensor:
    """异步集体通信结果等待（兼容 tracing 场景）"""
    if isinstance(tensor, ft_c.AsyncCollectiveTensor):
        return tensor.wait()
    return tensor


def all_reduce_sync(x, *args, group=None, **kwargs):
    """同步 all_reduce 操作"""
    group = get_group(group)
    x = ft_c.all_reduce(x, *args, group=group, **kwargs)
    world_size = dist.get_world_size()
    x /= world_size
    return x


# ============ 缓存上下文管理 ============
@dataclass
class CacheContext:
    """缓存上下文，管理中间计算结果的存储"""

    buffers: dict[str, torch.Tensor] = field(default_factory=dict)
    incremental_name_counters: defaultdict[str, int] = field(default_factory=lambda: defaultdict(int))

    def get_incremental_name(self, name=None):
        if name is None:
            name = "default"
        idx = self.incremental_name_counters[name]
        self.incremental_name_counters[name] += 1
        return f"{name}_{idx}"

    def reset_incremental_names(self):
        self.incremental_name_counters.clear()

    @torch.compiler.disable()
    def get_buffer(self, name):
        return self.buffers.get(name)

    @torch.compiler.disable()
    def set_buffer(self, name, buffer):
        self.buffers[name] = buffer

    def clear_buffers(self):
        self.buffers.clear()


_current_cache_context: CacheContext | None = None


def create_cache_context() -> CacheContext:
    return CacheContext()


def get_current_cache_context() -> CacheContext | None:
    return _current_cache_context


def set_current_cache_context(cache_context_value=None):
    global _current_cache_context
    _current_cache_context = cache_context_value


@contextlib.contextmanager
def cache_context(cache_context_value):
    global _current_cache_context
    old_cache_context = _current_cache_context
    _current_cache_context = cache_context_value
    try:
        yield
    finally:
        _current_cache_context = old_cache_context


@torch.compiler.disable()
def get_buffer(name):
    cache_context_value = get_current_cache_context()
    if cache_context_value is None:
        raise ValueError("cache_context must be set before")
    return cache_context_value.get_buffer(name)


@torch.compiler.disable()
def set_buffer(name, buffer):
    cache_context_value = get_current_cache_context()
    if cache_context_value is None:
        raise ValueError("cache_context must be set before")
    cache_context_value.set_buffer(name, buffer)


# ============ 缓存策略抽象基类 ============
class CacheStrategy(ABC):
    """缓存策略抽象基类

    所有缓存策略（TeaCache、MagCache等）都需要实现此接口。
    """

    @abstractmethod
    def should_skip(self, step: int, modulated_input: torch.Tensor | None = None) -> bool:
        """判断当前步骤是否可以跳过计算，使用缓存结果

        Args:
            step: 当前推理步骤
            modulated_input: 调制后的输入张量（用于计算相似度）

        Returns:
            True 表示可以跳过，使用缓存；False 表示需要计算
        """
        pass

    @abstractmethod
    def get_residual(self, step: int) -> torch.Tensor | None:
        """获取缓存的残差

        Args:
            step: 当前推理步骤

        Returns:
            缓存的残差张量，若无缓存返回 None
        """
        pass

    @abstractmethod
    def update_cache(self, step: int, residual: torch.Tensor):
        """更新缓存

        Args:
            step: 当前推理步骤
            residual: 新计算的残差
        """
        pass

    @abstractmethod
    def reset(self):
        """重置缓存状态，准备新的推理序列"""
        pass

    def on_step_complete(self, step: int):  # noqa: B027
        """步骤完成后的回调（可选实现）"""
        pass


# ============ TeaCache 策略实现 ============
class TeaCacheStrategy(CacheStrategy):
    """TeaCache 缓存策略

    基于输入调制张量的相对变化来判断是否可以跳过计算。
    支持条件/非条件步骤的独立缓存（even/odd）。
    """

    def __init__(
        self,
        threshold: float,
        coefficients: list[float],
        num_steps: int,
        ret_steps: int = 2,
        cutoff_steps: int | None = None,
        use_ref_steps: bool = True,
        parallelized: bool = False,
    ):
        """
        Args:
            threshold: 缓存跳过阈值
            coefficients: 多项式系数，用于缩放相对差异
            num_steps: 总推理步骤数
            ret_steps: 保留的初始步骤数（不跳过）
            cutoff_steps: 截断步骤数（之后不跳过）
            use_ref_steps: 是否使用参考步骤模式
            parallelized: 是否启用分布式
        """
        self.threshold = threshold
        self.coefficients = np.poly1d(coefficients)
        self.num_steps = num_steps
        self.ret_steps = ret_steps
        self.cutoff_steps = cutoff_steps or num_steps
        self.use_ref_steps = use_ref_steps
        self.parallelized = parallelized

        # 状态：even/odd 独立跟踪
        self._accumulated_distance = [0.0, 0.0]
        self._previous_input = [None, None]
        self._previous_residual = [None, None]
        self._should_calc = [True, True]

    def _get_idx(self, step: int) -> int:
        """获取步骤索引（0=even, 1=odd）"""
        return step % 2

    def should_skip(self, step: int, modulated_input: torch.Tensor | None = None) -> bool:
        idx = self._get_idx(step)

        # warmup 或 cooldown 阶段不跳过
        if step < self.ret_steps or step >= self.cutoff_steps:
            self._should_calc[idx] = True
            self._accumulated_distance[idx] = 0.0
            if modulated_input is not None:
                self._previous_input[idx] = modulated_input.clone()
            return False

        # 计算相对差异
        prev_input = self._previous_input[idx]
        if prev_input is None or modulated_input is None:
            self._should_calc[idx] = True
            return False

        relative_diff = (modulated_input - prev_input).abs().mean() / prev_input.abs().mean()

        # 累积缩放后的差异
        scaled_diff = self.coefficients(relative_diff.cpu().item())
        self._accumulated_distance[idx] += scaled_diff

        # 判断是否超过阈值
        should_calc = self._accumulated_distance[idx] >= self.threshold
        self._should_calc[idx] = should_calc

        if should_calc:
            self._accumulated_distance[idx] = 0.0

        # 更新前一次输入
        self._previous_input[idx] = modulated_input.clone()

        return not should_calc

    def get_residual(self, step: int) -> torch.Tensor | None:
        idx = self._get_idx(step)
        return self._previous_residual[idx]

    def update_cache(self, step: int, residual: torch.Tensor):
        idx = self._get_idx(step)
        self._previous_residual[idx] = residual

    def reset(self):
        self._accumulated_distance = [0.0, 0.0]
        self._previous_input = [None, None]
        self._previous_residual = [None, None]
        self._should_calc = [True, True]

    def on_step_complete(self, step: int):
        """步骤计数器递增（可选）"""
        pass


class TeaCacheSimpleStrategy(CacheStrategy):
    """简化版 TeaCache 策略（无 even/odd 分离）

    用于 Hunyuan、CogVideoX 等不需要区分条件/非条件步骤的模型。
    """

    def __init__(
        self,
        threshold: float,
        coefficients: list[float],
        num_steps: int,
    ):
        self.threshold = threshold
        self.coefficients = np.poly1d(coefficients)
        self.num_steps = num_steps

        self._accumulated_distance = 0.0
        self._previous_input: torch.Tensor | None = None
        self._previous_residual: torch.Tensor | None = None

    def should_skip(self, step: int, modulated_input: torch.Tensor | None = None) -> bool:
        # 首步和末步不跳过
        if step == 0 or step == self.num_steps - 1:
            self._accumulated_distance = 0.0
            if modulated_input is not None:
                self._previous_input = modulated_input.clone()
            return False

        if self._previous_input is None or modulated_input is None:
            self._previous_input = modulated_input.clone() if modulated_input else None
            return False

        relative_diff = (modulated_input - self._previous_input).abs().mean() / self._previous_input.abs().mean()
        scaled_diff = self.coefficients(relative_diff.cpu().item())
        self._accumulated_distance += scaled_diff

        should_calc = self._accumulated_distance >= self.threshold
        if should_calc:
            self._accumulated_distance = 0.0

        self._previous_input = modulated_input.clone()
        return not should_calc

    def get_residual(self, step: int) -> torch.Tensor | None:
        return self._previous_residual

    def update_cache(self, step: int, residual: torch.Tensor):
        self._previous_residual = residual

    def reset(self):
        self._accumulated_distance = 0.0
        self._previous_input = None
        self._previous_residual = None


# ============ MagCache 策略实现 ============
class MagCacheStrategy(CacheStrategy):
    """MagCache 缓存策略

    基于残差的幅度比（magnitude ratio）和累积误差来判断是否跳过。
    支持 distill 模式（单缓存）和标准模式（even/odd 双缓存）。
    """

    def __init__(
        self,
        threshold: float,
        K: int,
        num_steps: int,
        mag_ratios: np.ndarray,
        retention_ratio: float = 0.2,
        is_distill: bool = False,
        split_step: int | None = None,
    ):
        """
        Args:
            threshold: 累积误差阈值
            K: 最大连续跳过步数
            num_steps: 总推理步骤数
            mag_ratios: 预计算的幅度比数组
            retention_ratio: 保留比例（前N步不跳过）
            is_distill: 是否为蒸馏模式
            split_step: 分界步骤（用于 Wan2.2）
        """
        self.threshold = threshold
        self.K = K
        self.num_steps = num_steps
        self.mag_ratios = mag_ratios
        self.retention_ratio = retention_ratio
        self.is_distill = is_distill
        self.split_step = split_step

        # 状态初始化
        num_caches = 1 if is_distill else 2
        self._accumulated_err = [0.0] * num_caches
        self._accumulated_steps = [0] * num_caches
        self._accumulated_ratio = [1.0] * num_caches
        self._residual_cache: list[torch.Tensor | None] = [None] * num_caches

        # Wan2.2 模式标志
        self._wan2_mode: str | None = None

    def set_wan2_mode(self, mode: str):
        """设置 Wan2.2 模式（t2v/i2v）"""
        self._wan2_mode = mode

    def _get_idx(self, step: int) -> int:
        if self.is_distill:
            return 0
        return step % 2

    def _should_use_magcache(self, step: int) -> bool:
        """判断当前步骤是否应该使用 magcache 逻辑"""
        if self.split_step is not None and self._wan2_mode is not None:
            if self._wan2_mode == "i2v":
                return step >= int(self.split_step + (self.num_steps - self.split_step) * self.retention_ratio)
            else:  # t2v
                retention = self.retention_ratio
                if step < int(self.split_step * retention):
                    return False
                if self.split_step <= step <= int((self.num_steps - self.split_step) * retention + self.split_step):  # noqa: SIM103
                    return False
                return True
        else:
            return step >= int(self.num_steps * self.retention_ratio)

    def should_skip(self, step: int, modulated_input: torch.Tensor | None = None) -> bool:
        if not self._should_use_magcache(step):
            return False

        idx = self._get_idx(step)
        cur_mag_ratio = self.mag_ratios[step]

        # 更新累积比例和误差
        self._accumulated_ratio[idx] *= cur_mag_ratio
        self._accumulated_steps[idx] += 1
        cur_skip_err = abs(1.0 - self._accumulated_ratio[idx])
        self._accumulated_err[idx] += cur_skip_err

        # 判断是否满足跳过条件
        should_skip = self._accumulated_err[idx] < self.threshold and self._accumulated_steps[idx] <= self.K

        if not should_skip:
            # 重置累积状态
            self._accumulated_err[idx] = 0.0
            self._accumulated_steps[idx] = 0
            self._accumulated_ratio[idx] = 1.0

        return should_skip

    def get_residual(self, step: int) -> torch.Tensor | None:
        idx = self._get_idx(step)
        return self._residual_cache[idx]

    def update_cache(self, step: int, residual: torch.Tensor):
        idx = self._get_idx(step)
        self._residual_cache[idx] = residual

    def reset(self):
        num_caches = 1 if self.is_distill else 2
        self._accumulated_err = [0.0] * num_caches
        self._accumulated_steps = [0] * num_caches
        self._accumulated_ratio = [1.0] * num_caches
        self._residual_cache = [None] * num_caches


# ============ 遗留支持：CachedTransformerBlocks ============
class CachedTransformerBlocks(torch.nn.Module):
    """缓存 Transformer 块（遗留接口，保持向后兼容）

    用于 faiz 模式的缓存实现。
    """

    def __init__(
        self,
        transformer_blocks,
        single_transformer_blocks=None,
        *,
        transformer=None,
        residual_diff_threshold,
        return_hidden_states_first=True,
    ):
        super().__init__()
        self.transformer = transformer
        self.transformer_blocks = transformer_blocks
        self.single_transformer_blocks = single_transformer_blocks
        self.residual_diff_threshold = residual_diff_threshold
        self.return_hidden_states_first = return_hidden_states_first

    def forward(self, hidden_states, encoder_hidden_states, *args, **kwargs):
        if self.residual_diff_threshold <= 0.0:
            return self._forward_no_cache(hidden_states, encoder_hidden_states, *args, **kwargs)

        original_hidden_states = hidden_states
        first_transformer_block = self.transformer_blocks[0]
        hidden_states, encoder_hidden_states = first_transformer_block(
            hidden_states, encoder_hidden_states, *args, **kwargs
        )
        if not self.return_hidden_states_first:
            hidden_states, encoder_hidden_states = encoder_hidden_states, hidden_states
        first_hidden_states_residual = hidden_states - original_hidden_states
        del original_hidden_states

        can_use_cache = self._get_can_use_cache(
            first_hidden_states_residual,
            threshold=self.residual_diff_threshold,
            parallelized=self.transformer is not None,
        )

        torch._dynamo.graph_break()
        if can_use_cache:
            del first_hidden_states_residual
            hidden_states, encoder_hidden_states = self._apply_prev_residual(hidden_states, encoder_hidden_states)
        else:
            set_buffer("first_hidden_states_residual", first_hidden_states_residual)
            del first_hidden_states_residual
            (
                hidden_states,
                encoder_hidden_states,
                hidden_states_residual,
                encoder_hidden_states_residual,
            ) = self._call_remaining_blocks(hidden_states, encoder_hidden_states, *args, **kwargs)
            set_buffer("hidden_states_residual", hidden_states_residual)
            set_buffer("encoder_hidden_states_residual", encoder_hidden_states_residual)
        torch._dynamo.graph_break()

        return (
            (hidden_states, encoder_hidden_states)
            if self.return_hidden_states_first
            else (encoder_hidden_states, hidden_states)
        )

    def _forward_no_cache(self, hidden_states, encoder_hidden_states, *args, **kwargs):
        for block in self.transformer_blocks:
            hidden_states, encoder_hidden_states = block(hidden_states, encoder_hidden_states, *args, **kwargs)
            if not self.return_hidden_states_first:
                hidden_states, encoder_hidden_states = encoder_hidden_states, hidden_states
        if self.single_transformer_blocks is not None:
            hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
            for block in self.single_transformer_blocks:
                hidden_states = block(hidden_states, *args, **kwargs)
            hidden_states = hidden_states[:, encoder_hidden_states.shape[1] :]
        return (
            (hidden_states, encoder_hidden_states)
            if self.return_hidden_states_first
            else (encoder_hidden_states, hidden_states)
        )

    def _call_remaining_blocks(self, hidden_states, encoder_hidden_states, *args, **kwargs):
        original_hidden_states = hidden_states
        original_encoder_hidden_states = encoder_hidden_states
        for block in self.transformer_blocks[1:]:
            hidden_states, encoder_hidden_states = block(hidden_states, encoder_hidden_states, *args, **kwargs)
            if not self.return_hidden_states_first:
                hidden_states, encoder_hidden_states = encoder_hidden_states, hidden_states
        if self.single_transformer_blocks is not None:
            hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
            for block in self.single_transformer_blocks:
                hidden_states = block(hidden_states, *args, **kwargs)
            encoder_hidden_states, hidden_states = hidden_states.split(
                [encoder_hidden_states.shape[1], hidden_states.shape[1] - encoder_hidden_states.shape[1]],
                dim=1,
            )

        hidden_states_shape = hidden_states.shape
        encoder_hidden_states_shape = encoder_hidden_states.shape

        hidden_states = hidden_states.flatten().contiguous().reshape(hidden_states_shape)
        encoder_hidden_states = encoder_hidden_states.flatten().contiguous().reshape(encoder_hidden_states_shape)

        hidden_states_residual = hidden_states - original_hidden_states
        encoder_hidden_states_residual = encoder_hidden_states - original_encoder_hidden_states
        return hidden_states, encoder_hidden_states, hidden_states_residual, encoder_hidden_states_residual

    @staticmethod
    @torch.compiler.disable()
    def _get_can_use_cache(first_hidden_states_residual, *, threshold, parallelized=False):
        prev_first_hidden_states_residual = get_buffer("first_hidden_states_residual")
        can_use_cache = prev_first_hidden_states_residual is not None and _are_two_tensors_similar(
            prev_first_hidden_states_residual,
            first_hidden_states_residual,
            threshold=threshold,
            parallelized=parallelized,
        )
        return can_use_cache

    @staticmethod
    @torch.compiler.disable()
    def _apply_prev_residual(hidden_states, encoder_hidden_states):
        hidden_states_residual = get_buffer("hidden_states_residual")
        if hidden_states_residual is None:
            raise ValueError("hidden_states_residual must be set before")
        hidden_states = hidden_states_residual + hidden_states

        encoder_hidden_states_residual = get_buffer("encoder_hidden_states_residual")
        if encoder_hidden_states_residual is None:
            raise ValueError("encoder_hidden_states_residual must be set before")
        encoder_hidden_states = encoder_hidden_states_residual + encoder_hidden_states

        hidden_states = hidden_states.contiguous()
        encoder_hidden_states = encoder_hidden_states.contiguous()

        return hidden_states, encoder_hidden_states


@torch.compiler.disable()
def _are_two_tensors_similar(t1, t2, *, threshold, parallelized=False):
    mean_diff = (t1 - t2).abs().mean()
    mean_t1 = t1.abs().mean()
    if parallelized:
        mean_diff = all_reduce_sync(mean_diff, "sum")
        mean_t1 = all_reduce_sync(mean_t1, "sum")
    diff = mean_diff / mean_t1
    return diff.item() < threshold


# ============ 工具函数 ============
def nearest_interp(src_array: np.ndarray, target_length: int) -> np.ndarray:
    """最近邻插值"""
    src_length = len(src_array)
    if target_length == 1:
        return np.array([src_array[-1]])

    scale = (src_length - 1) / (target_length - 1)
    mapped_indices = np.round(np.arange(target_length) * scale).astype(int)
    return src_array[mapped_indices]


# ============ TeaCache 配置辅助函数 ============
def get_teacache_config(model_type: str, model_key: str, use_ref_steps: bool = True) -> dict[str, Any]:
    """获取 TeaCache 配置

    Args:
        model_type: 模型类型，如 "wan", "hunyuan", "cogvideox"
        model_key: 模型键，如 "2.1-T2V-1.3B", "HunyuanVideo-13B"
        use_ref_steps: 是否使用 ref_steps 模式

    Returns:
        包含 coefficients, threshold, ret_steps 等的配置字典
    """
    config = load_cache_config()

    # 获取模型特定配置
    teacache_config = config.get("teacache", {})
    model_config = teacache_config.get(model_type, {}).get(model_key, {})

    if not model_config:
        logger.warning(f"No teacache config found for {model_type}/{model_key}, using defaults")  # noqa: G004
        return {}

    result = {}

    # Wan 模型有 use_ref_steps/standard 两套配置
    if model_type == "wan":
        mode_key = "use_ref_steps" if use_ref_steps else "standard"
        mode_config = model_config.get(mode_key, {})
        result["coefficients"] = mode_config.get("coefficients", [])
        result["ret_steps"] = mode_config.get("ret_steps", 2 if not use_ref_steps else 10)
        result["cutoff_mode"] = mode_config.get("cutoff_mode", "early")
        # threshold 可能在顶层
        result["threshold"] = model_config.get("threshold")
    else:
        # Hunyuan、CogVideoX 等模型配置更简单
        result["coefficients"] = model_config.get("coefficients", [])
        result["threshold"] = model_config.get("threshold", 0.1)

    return result


def get_teacache_coefficients(model_type: str, model_key: str, use_ref_steps: bool = True) -> list[float] | None:
    """获取 TeaCache 多项式系数

    Args:
        model_type: 模型类型
        model_key: 模型键
        use_ref_steps: 是否使用 ref_steps 模式

    Returns:
        系数列表，若未找到返回 None
    """
    config = get_teacache_config(model_type, model_key, use_ref_steps)
    return config.get("coefficients") or None
