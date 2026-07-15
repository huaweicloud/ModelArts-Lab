"""
统一配置加载器

提供 YAML 配置文件的统一加载、缓存和访问接口。
"""

import fnmatch
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from torch import nn
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP

logger = logging.getLogger(__name__)


class ConfigLoadError(Exception):
    """配置加载异常"""

    pass


def _load_yaml_config(config_filename: str) -> dict[str, Any]:
    """加载 YAML 配置文件（内部函数）

    Args:
        config_filename: 配置文件名（不含路径）

    Returns:
        配置字典

    Raises:
        ConfigLoadError: 配置文件不存在或解析失败
    """
    from importlib import resources

    import yaml

    try:
        # Python 3.9+
        config_package = resources.files("x_base.config")
        with config_package.joinpath(config_filename).open("r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except (AttributeError, TypeError):
        # Python 3.7-3.8 兼容
        import os

        import x_base.config as config_module

        config_path = os.path.join(os.path.dirname(config_module.__file__), config_filename)
        with open(config_path, encoding="utf-8") as f:
            return yaml.safe_load(f)
    except FileNotFoundError as e:
        raise ConfigLoadError(f"Config file not found: {config_filename}") from e
    except Exception as e:
        raise ConfigLoadError(f"Failed to load config {config_filename}: {e}") from e


# ============ 配置缓存 ============
_config_cache: dict[str, dict[str, Any]] = {}


def get_config(config_name: str, reload: bool = False) -> dict[str, Any]:
    """获取配置（带缓存）

    Args:
        config_name: 配置名称（不含 .yaml 后缀）
        reload: 是否强制重新加载

    Returns:
        配置字典
    """
    cache_key = f"{config_name}.yaml"

    if not reload and cache_key in _config_cache:
        return _config_cache[cache_key]

    config = _load_yaml_config(cache_key)
    _config_cache[cache_key] = config
    return config


def load_cache_config() -> dict[str, Any]:
    """加载 Cache 配置（向后兼容接口）

    Returns:
        Cache 配置字典
    """
    return get_config("cache_config")


def load_quant_config() -> dict[str, Any]:
    """加载量化配置

    Returns:
        量化配置字典
    """
    return get_config("quant_config")


# ============ 量化层配置类 ============
@dataclass
class QuantLayerConfig:
    """量化层配置

    支持白名单/黑名单模式，以及 glob 模式匹配。

    Attributes:
        include_patterns: 白名单模式列表（None 表示不限制）
        exclude_patterns: 黑名单模式列表
        layer_types: 量化的层类型列表
        w4a4_patterns: w4a4 量化模块选择模式（匹配这些层名使用 w4a4 模块）
        w4a4_block_patterns: w4a4 模块的 block 限制（必须同时匹配）
    """

    include_patterns: list[str] | None = None
    exclude_patterns: list[str] = field(default_factory=list)
    layer_types: list[str] = field(default_factory=lambda: ["Linear"])
    # w4a4 量化模块选择配置
    w4a4_patterns: list[str] = field(default_factory=lambda: ["to_k", "to_v", "to_q", "to_out", "proj"])
    w4a4_block_patterns: list[str] = field(default_factory=lambda: ["blocks"])

    # 层类型映射（不参与 dataclass 的 __init__、__repr__、__eq__）
    _layer_type_map: dict[str, type[nn.Module]] = field(
        default_factory=lambda: {
            "Linear": nn.Linear,
            "Conv2d": nn.Conv2d,
            "Conv3d": nn.Conv3d,
        },
        repr=False,
        compare=False,
    )

    _parsed_layer_types: tuple[type[nn.Module], ...] = field(
        default_factory=tuple,
        repr=False,
        compare=False,
        init=False,
    )

    def __post_init__(self):
        """后处理：解析层类型"""
        self._parsed_layer_types = tuple(self._layer_type_map.get(lt, nn.Linear) for lt in self.layer_types)

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "QuantLayerConfig":
        """从字典创建配置实例

        Args:
            config_dict: 配置字典

        Returns:
            QuantLayerConfig 实例
        """
        return cls(
            include_patterns=config_dict.get("include_patterns"),
            exclude_patterns=config_dict.get("exclude_patterns", []),
            layer_types=config_dict.get("layer_types", ["Linear"]),
            w4a4_patterns=config_dict.get("w4a4_patterns", ["to_k", "to_v", "to_q", "to_out", "proj"]),
            w4a4_block_patterns=config_dict.get("w4a4_block_patterns", ["blocks"]),
        )

    def should_quantize(self, layer_name: str, layer: nn.Module) -> bool:
        """判断该层是否需要量化

        Args:
            layer_name: 层名称
            layer: 层实例

        Returns:
            True 表示需要量化，False 表示跳过
        """
        # 1. 类型检查
        if not isinstance(layer, self._parsed_layer_types):
            return False

        # 2. 黑名单检查（优先级最高）
        if self._matches_patterns(layer_name, self.exclude_patterns):
            return False

        # 3. 白名单检查
        if self.include_patterns is None or len(self.include_patterns) == 0:
            return True  # 无白名单 = 全量化

        return self._matches_patterns(layer_name, self.include_patterns)

    def should_use_w4a4(self, layer_name: str) -> bool:
        """判断该层是否应该使用 w4a4 量化模块

        w4a4 模块需要同时满足两个条件：
        1. 层名匹配 w4a4_patterns 中的至少一个（如 to_k, to_v, proj 等）
        2. 层名匹配 w4a4_block_patterns 中的至少一个（如 blocks）

        Args:
            layer_name: 层名称

        Returns:
            True 表示使用 w4a4 模块，False 表示使用普通 w8a8 模块
        """
        # 必须同时匹配 w4a4_patterns 和 w4a4_block_patterns
        matches_w4a4 = self._matches_patterns(layer_name, self.w4a4_patterns)
        matches_block = self._matches_patterns(layer_name, self.w4a4_block_patterns)
        return matches_w4a4 and matches_block

    @staticmethod
    def _matches_patterns(name: str, patterns: list[str]) -> bool:
        """检查名称是否匹配任一模式

        支持：
        - glob 模式（如 "blocks.*.attn"）
        - 子串匹配（如 "mlp"）

        Args:
            name: 层名称
            patterns: 模式列表

        Returns:
            True 表示匹配
        """
        for pattern in patterns:  # noqa: SIM110
            if fnmatch.fnmatch(name, pattern) or pattern in name:
                return True
        return False

    def __repr__(self) -> str:
        return (
            f"QuantLayerConfig("
            f"include={self.include_patterns}, "
            f"exclude={self.exclude_patterns}, "
            f"types={self.layer_types}, "
            f"w4a4_patterns={self.w4a4_patterns})"
        )


class QuantConfigManager:
    """量化配置管理器

    管理不同模型的量化配置，支持自动推断和手动指定。
    """

    _instance: Optional["QuantConfigManager"] = None

    def __new__(cls):
        """单例模式"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """初始化配置管理器"""
        if self._initialized:
            return

        self._config: dict[str, Any] = load_quant_config()
        self._model_mapping: dict[str, str] = self._config.get("model_type_mapping", {})
        self._configs: dict[str, QuantLayerConfig] = {}
        self._initialized = True

    def reload(self):
        """重新加载配置"""
        self._config = get_config("quant_config", reload=True)
        self._model_mapping = self._config.get("model_type_mapping", {})
        self._configs.clear()

    def get_config(self, config_name: str) -> QuantLayerConfig:
        """获取指定名称的量化配置

        Args:
            config_name: 配置名称（如 "qwen_image", "default"）

        Returns:
            QuantLayerConfig 实例
        """
        if config_name not in self._configs:
            config_dict = self._config.get(config_name)
            if config_dict is None:
                logger.warning(f"Quant config '{config_name}' not found, using default")  # noqa: G004
                config_dict = self._config.get("default", {})
            self._configs[config_name] = QuantLayerConfig.from_dict(config_dict)

        return self._configs[config_name]

    def detect_config_name(self, transformer) -> str:
        """根据模型实例自动推断配置名称

        Args:
            transformer: Transformer 模型实例

        Returns:
            配置名称
        """
        # 1. 通过类名精确匹配
        class_name = transformer.__class__.__name__
        if class_name in self._model_mapping:
            return self._model_mapping[class_name]

        # 2. 通过基类链匹配
        for base_class in transformer.__class__.__mro__:
            base_name = base_class.__name__
            if base_name in self._model_mapping:
                return self._model_mapping[base_name]

        # 3. 模糊匹配（类名包含关键特征）
        class_name_lower = class_name.lower()
        if "qwen" in class_name_lower and "image" in class_name_lower:
            return "qwen_image"
        elif "wan" in class_name_lower:
            return "wan"
        elif "hunyuan" in class_name_lower:
            return "hunyuan"
        elif "cogvideo" in class_name_lower:
            return "cogvideox"

        return "default"

    def get_config_for_model(self, transformer) -> QuantLayerConfig:
        """获取模型的量化配置（自动推断）

        Args:
            transformer: Transformer 模型实例

        Returns:
            QuantLayerConfig 实例
        """
        config_name = self.detect_config_name(transformer)
        return self.get_config(config_name)


class OffloadConfigManager:
    """Offload配置管理器

    管理offload配置，支持自动推断和手动指定。
    """

    _instance: Optional["OffloadConfigManager"] = None

    def __new__(cls):
        """单例模式"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """初始化配置管理器"""
        if self._initialized:
            return

        self._config: dict[str, Any] = get_config("offload_config")
        self._block_mapping: dict[str, str] = self._config.get("transformer_block_mapping", {})
        self._initialized = True

    def reload(self):
        """重新加载配置"""
        self._config = get_config("offload_config", reload=True)
        self._block_mapping = self._config.get("transformer_block_mapping", {})

    def get_block_name(self, transformer) -> str:
        """根据transformer实例获取对应的transformer_block名称

        Args:
            transformer: Transformer 模型实例

        Returns:
            transformer_block名称
        """
        # 检查并解包 FullyShardedDataParallel 模块
        if isinstance(transformer, FSDP):
            transformer = transformer.module

        # 1. 通过类名精确匹配
        class_name = transformer.__class__.__name__
        if class_name in self._block_mapping:
            return self._block_mapping[class_name]

        # 2. 通过基类链匹配
        for base_class in transformer.__class__.__mro__:
            base_name = base_class.__name__
            if base_name in self._block_mapping:
                return self._block_mapping[base_name]

        # 3. 模糊匹配（类名包含关键特征）
        class_name_lower = class_name.lower()
        transformer_name = "default"
        if "qwen" in class_name_lower and "image" in class_name_lower:
            transformer_name = "QwenImageTransformer2DModel"
        elif "wan" in class_name_lower:
            transformer_name = "WanTransformer3DModel"
        elif "hunyuan" in class_name_lower:
            transformer_name = "HunyuanVideoTransformer3DModel"
        elif "cogvideo" in class_name_lower:
            transformer_name = "CogVideoXTransformer3DModel"

        # 4. 返回默认值
        return self._block_mapping.get(transformer_name, "transformer_blocks")


# 全局单例
quant_config_manager = QuantConfigManager()

# OffloadConfigManager全局单例
offload_config_manager = OffloadConfigManager()

# cache_dit配置
CACHE_DIT_CONFIG = get_config(config_name="cache_dit_config", reload=True)["DBCache_config"]

# qwen lora scheduler配置
QWEN_LORA_SCHEDULER_CONFIG = get_config(config_name="pipeline_scheduler_config", reload=True)[
    "qwen_lora_scheduler_config"
]
