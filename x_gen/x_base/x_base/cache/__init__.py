"""
X_BASE Cache 加速模块

提供 TeaCache 和 MagCache 两种推理加速策略，支持 Wan、Hunyuan、CogVideoX 等模型。

## 加速策略

| 策略 | 说明 | 适用场景 |
|------|------|----------|
| TeaCache | 基于输入调制张量的相对变化判断是否跳过 | 高质量、低延迟 |
| MagCache | 基于残差幅度比和累积误差判断是否跳过 | 高压缩率 |

## 使用方式

```python
from x_base.cache import teacache_wan_init, magcache_wan_init

# TeaCache 初始化
teacache_wan_init(pipe, args)

# MagCache 初始化
magcache_wan_init(pipe, args)
```
"""

# 基础组件
from .base import (
    # 缓存上下文
    CacheContext,
    # 工具类
    CachedTransformerBlocks,
    # 缓存策略
    CacheStrategy,
    MagCacheStrategy,
    TeaCacheSimpleStrategy,
    TeaCacheStrategy,
    # 分布式通信
    all_reduce_sync,
    cache_context,
    create_cache_context,
    get_buffer,
    get_current_cache_context,
    load_cache_config,
    nearest_interp,
    set_buffer,
    set_current_cache_context,
)

# 模型实现
from .models import (
    magcache_wan_calibration,
    magcache_wan_calibration_init,
    magcache_wan_forward,
    magcache_wan_init,
    # CogVideoX
    teacache_cogvideox_forward,
    teacache_cogvideox_init,
    # Hunyuan
    teacache_hunyuan_forward,
    teacache_hunyuan_init,
    # Wan
    teacache_wan_forward,
    teacache_wan_init,
    teacache_wan_vace_forward,
)

# 工具函数
from .utils import (
    cogvideox_post_forward,
    cogvideox_pre_forward,
    hunyuan_post_forward,
    hunyuan_pre_forward,
    post_forward_lora,
    pre_forward,
    wan_post_forward,
    wan_pre_forward,
)


# ============ 高层封装函数 ============
def cache_on_pipe(pipe, args):
    """将 Cache 加速应用到 pipeline

    根据 turbo_mode 选择合适的加速策略：
    - "faiz" / "teacache": TeaCache 加速
    - "next_faiz" / "magcache": MagCache 加速
    - None: 不加速，返回原 pipe

    Args:
        pipe: DiffusionPipeline 实例
        args: 参数对象，需包含 turbo_mode, model 等属性

    Returns:
        应用了加速策略的 pipe（同一实例）
    """
    import importlib

    # 检查 turbo_mode
    if args.turbo_mode is None:
        return pipe

    # 获取 pipeline 类型
    pipeline_name = pipe.__class__.__name__

    # 映射到对应的适配器模块
    adapter_map = {
        "WanPipeline": "wan",
        "WanImageToVideoPipeline": "wan",
        "HunyuanVideoPipeline": "hunyuan",
        "CogVideoXPipeline": "cogvideox",
    }

    adapter_name = adapter_map.get(pipeline_name)
    if adapter_name is None:
        raise ValueError(f"Unknown pipeline class name: {pipeline_name}")

    # 动态导入对应的模型模块
    try:
        adapter_module = importlib.import_module(f"x_base.cache.models.{adapter_name}")
    except ModuleNotFoundError as e:
        raise ValueError(f"Failed to import adapter module for {adapter_name}") from e

    # 选择初始化函数
    if args.turbo_mode in ("faiz", "teacache"):
        init_func = getattr(adapter_module, "teacache_init", None)
    elif args.turbo_mode in ("next_faiz", "magcache"):
        init_func = getattr(adapter_module, "magcache_init", None)
    else:
        # 未知模式，返回原 pipe
        return pipe

    if init_func is None:
        return pipe

    init_func(pipe, args)
    return pipe


# 向后兼容别名
turbo_on_pipe = cache_on_pipe


__all__ = [
    # 基础组件 - 缓存上下文
    "CacheContext",
    "create_cache_context",
    "get_current_cache_context",
    "set_current_cache_context",
    "cache_context",
    "get_buffer",
    "set_buffer",
    # 基础组件 - 缓存策略
    "CacheStrategy",
    "TeaCacheStrategy",
    "TeaCacheSimpleStrategy",
    "MagCacheStrategy",
    # 基础组件 - 工具类
    "CachedTransformerBlocks",
    "nearest_interp",
    "load_cache_config",
    "all_reduce_sync",
    # 工具函数
    "pre_forward",
    "post_forward_lora",
    "wan_pre_forward",
    "wan_post_forward",
    "hunyuan_pre_forward",
    "hunyuan_post_forward",
    "cogvideox_pre_forward",
    "cogvideox_post_forward",
    # 高层封装
    "cache_on_pipe",
    "turbo_on_pipe",  # 向后兼容别名
    # Wan 模型
    "teacache_wan_forward",
    "teacache_wan_vace_forward",
    "magcache_wan_forward",
    "magcache_wan_calibration",
    "teacache_wan_init",
    "magcache_wan_init",
    "magcache_wan_calibration_init",
    # Hunyuan 模型
    "teacache_hunyuan_forward",
    "teacache_hunyuan_init",
    # CogVideoX 模型
    "teacache_cogvideox_forward",
    "teacache_cogvideox_init",
]
