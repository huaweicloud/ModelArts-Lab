"""
X_BASE 配置模块

提供统一的配置加载和管理接口。

## 功能

- cache_config.yaml: Cache 加速配置（TeaCache/MagCache）
- quant_config.yaml: 量化层配置

## 使用方式

```python
from x_base.config import load_cache_config, load_quant_config

# 加载配置
cache_cfg = load_cache_config()
quant_cfg = load_quant_config()

# 使用量化配置管理器
from x_base.config import quant_config_manager

config = quant_config_manager.get_config("qwen_image")
```
"""

from .config_loader import (
    CACHE_DIT_CONFIG,
    QWEN_LORA_SCHEDULER_CONFIG,
    ConfigLoadError,
    QuantConfigManager,
    # 量化配置
    QuantLayerConfig,
    # 配置加载
    get_config,
    load_cache_config,
    load_quant_config,
    offload_config_manager,
    quant_config_manager,
)

__all__ = [
    # 配置加载
    "get_config",
    "load_cache_config",
    "load_quant_config",
    "ConfigLoadError",
    # 量化配置
    "QuantLayerConfig",
    "QuantConfigManager",
    "quant_config_manager",
    "offload_config_manager",
    "CACHE_DIT_CONFIG",
    "QWEN_LORA_SCHEDULER_CONFIG",
]
