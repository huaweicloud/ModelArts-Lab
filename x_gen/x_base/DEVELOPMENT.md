# x_base 开发文档

> 基于代码结构和测试单元反向生成的开发文档

## 目录

- [模块概述](#模块概述)
- [项目结构](#项目结构)
- [配置模块 (config)](#配置模块-config)
- [推理信息模块 (utils)](#推理信息模块-utils)
- [缓存加速模块 (cache)](#缓存加速模块-cache)
- [序列并行模块 (sequence_parallelism)](#序列并行模块-sequence_parallelism)
- [VAE 并行模块 (vae_parallelism)](#vae-并行模块-vae_parallelism)
- [PHAA 模块 (phaa)](#phaa-模块-phaa)
- [算子优化模块 (operator)](#算子优化模块-operator)
- [无限显存模块 (infinite_vram)](#无限显存模块-infinite_vram)
- [CFG 并行模块 (cfg_parallelism)](#cfg-并行模块-cfg_parallelism)
- [预处理模块 (preprocess)](#预处理模块-preprocess)
- [支持的模型列表](#支持的模型列表)
- [快速开始](#快速开始)

---

## 模块概述

`x_base` 是一个视频生成推理加速框架，提供以下功能：

1. **缓存加速 (cache)** - 支持 TeaCache 和 MagCache 两种推理加速策略
2. **序列并行 (sequence_parallelism)** - 分布式序列并行，跨卡切分序列维度
3. **VAE 并行 (vae_parallelism)** - VAE 编解码分布式并行
4. **PHAA (phaa)** - 通算并行，计算与通信重叠
5. **算子优化 (operator)** - Attention、Matmul、RoPE 等算子量化与融合
6. **无限显存 (infinite_vram)** - 显存优化策略，权重动态 offload
7. **配置管理 (config)** - 基于 YAML 的配置加载
8. **推理信息管理 (utils)** - 统一管理推理参数和模型信息

---

## 项目结构

```
x_base/
├── __init__.py                    # 顶层导出
├── x_base/                        # 源码目录
│   ├── __init__.py               # 模块导出
│   ├── cache/                    # 缓存加速模块 (原 turbo)
│   │   ├── __init__.py           # 主入口，导出 cache_on_pipe, turbo_on_pipe
│   │   ├── base.py               # CacheStrategy, TeaCacheStrategy, MagCacheStrategy
│   │   ├── utils.py              # 工具函数
│   │   └── models/               # 模型适配器
│   │       ├── wan.py            # Wan Pipeline
│   │       ├── hunyuan.py        # HunyuanVideo Pipeline
│   │       └── cogvideox.py      # CogVideoX Pipeline
│   ├── sequence_parallelism/     # 序列并行模块
│   │   ├── __init__.py           # 顶层导出
│   │   ├── errors.py             # 错误类型
│   │   ├── comm.py               # 通信原语
│   │   ├── padding.py            # Padding 管理
│   │   ├── mesh.py               # ProcessGroupMesh
│   │   ├── backend.py            # 通信后端抽象
│   │   └── pipeline.py           # enable_sp
│   ├── vae_parallelism/          # VAE 并行模块
│   │   ├── __init__.py           # 导出 VAEManager, enable_vae_lightning
│   │   ├── vae_mgr.py            # VAE 管理器
│   │   ├── utils.py              # 工具函数
│   │   └── save_video_stream.py  # 视频流保存
│   ├── phaa/                     # PHAA 模块
│   │   ├── __init__.py           # 导出 enable_phaa, phaa_on_pipe
│   │   ├── globals.py            # 全局状态
│   │   └── utils.py              # 工具函数
│   ├── fsdp/                     # FSDP 模块
│   │   ├── __init__.py           # (空)
│   │   └── fsdp.py               # fsdp_init
│   ├── operator/                 # 算子优化模块
│   │   ├── __init__.py           # 导出 attention_manager, matmul_manager, rope_manager
│   │   ├── attention.py          # Attention A8W8 量化
│   │   ├── matmul.py             # Matmul 动态量化
│   │   ├── rope.py               # RoPE 融合算子
│   │   └── rainfusion.py         # Rainfusion 算子
│   ├── infinite_vram/            # 无限显存模块
│   │   ├── __init__.py           # 导出 OffloadManager
│   │   ├── offload_manager.py    # Offload 管理器
│   │   └── utils.py              # 工具函数
│   ├── cfg_parallelism/          # CFG 并行模块
│   │   └── __init__.py           # get_cfg_group
│   ├── preprocess/               # 预处理模块
│   │   └── vace_preprocess.py    # 视频预处理
│   ├── config/                   # 配置模块
│   │   ├── __init__.py           # 配置加载器
│   │   ├── config_loader.py      # 统一配置加载
│   │   ├── cache_config.yaml     # 缓存配置
│   │   ├── quant_config.yaml     # 量化配置
│   │   └── offload_config.yaml   # Offload 配置
│   └── utils/                    # 工具函数模块
│       ├── __init__.py           # 导出 infer_info, read_text, list_cases
│       └── infer_info.py         # 推理信息管理
└── tests/                        # 测试目录
    ├── conftest.py               # pytest 配置
    ├── test_config.py            # 配置测试
    ├── test_infer_info.py        # 推理信息测试
    ├── test_utils.py             # 工具函数测试
    ├── test_cache.py             # 缓存模块测试
    └── test_matmul.py            # Matmul 测试
```

---

## 配置模块 (config)

### 配置文件

| 文件 | 说明 |
|------|------|
| `cache_config.yaml` | 缓存加速配置（TeaCache/MagCache 参数） |
| `quant_config.yaml` | 量化配置（Attention/Matmul 量化参数） |
| `offload_config.yaml` | 显存 offload 配置 |

### 配置加载

```python
from x_base.config import load_cache_config, offload_config_manager

# 加载缓存配置
cache_config = load_cache_config()

# Offload 配置管理器
offload_config = offload_config_manager
```

---

## 推理信息模块 (utils)

### InferInfo 类

管理推理配置信息的核心类。

```python
from x_base.utils import infer_info
from x_base.utils.infer_info import InferInfo

# 使用全局实例
print(infer_info.model)

# 创建新实例
info = InferInfo()
info.update_info(args)
```

---

## 缓存加速模块 (cache)

### 概述

Cache 模块提供两种加速策略：

| 策略 | 名称 | 说明 | 适用场景 |
|------|------|------|----------|
| `faiz` / `teacache` | TeaCache | 基于输入调制张量的相对变化判断是否跳过 | 高质量、低延迟 |
| `next_faiz` / `magcache` | MagCache | 基于残差幅度比和累积误差判断是否跳过 | 高压缩率 |

### cache_on_pipe (主入口)

自动根据 Pipeline 类型选择对应适配器。

```python
from x_base.cache import cache_on_pipe
# 向后兼容别名
from x_base.cache import turbo_on_pipe

pipe = cache_on_pipe(pipe, args)
# 或
pipe = turbo_on_pipe(pipe, args)  # 向后兼容
```

**参数：**
- `pipe`: DiffusionPipeline 实例
- `args`: 包含 `turbo_mode` 属性的 Namespace

**Pipeline 路由：**

| Pipeline 类名 | 适配器模块 |
|--------------|-----------|
| `WanPipeline` | `x_base.cache.models.wan` |
| `WanI2VPipeline` | `x_base.cache.models.wan` |
| `HunyuanVideoPipeline` | `x_base.cache.models.hunyuan` |
| `CogVideoXPipeline` | `x_base.cache.models.cogvideox` |

### 缓存策略类

```python
from x_base.cache import (
    CacheStrategy,           # 抽象基类
    TeaCacheStrategy,        # TeaCache 策略
    TeaCacheSimpleStrategy,  # 简化版 TeaCache
    MagCacheStrategy,        # MagCache 策略
)
```

### 缓存上下文

```python
from x_base.cache import (
    CacheContext,
    create_cache_context,
    get_current_cache_context,
    cache_context,
    get_buffer,
    set_buffer,
)

ctx = create_cache_context()
with cache_context(ctx):
    set_buffer("key", tensor)
    retrieved = get_buffer("key")
```

---

## 序列并行模块 (sequence_parallelism)

### 概述

分布式序列并行，跨卡切分序列维度。

### API 使用

#### Padding 管理

```python
from x_base.sequence_parallelism import PadManager, get_pad_manager

manager = get_pad_manager()
pad = manager.set("attention", seq_len, sp_size)
```

#### 并行管理

```python
from x_base.sequence_parallelism import ParallelManager, ParallelConfig

config = ParallelConfig(dp_size=2, sp_size=4)
manager = ParallelManager(config)
```

#### 通信原语

```python
from x_base.sequence_parallelism import (
    all_gather,
    reduce_scatter,
    all_to_all,
    gather_sequence,
    split_sequence,
    all_to_all_before_attn,
    all_to_all_after_attn,
    pad_tensor,
)
```

#### 启用序列并行

```python
from x_base.sequence_parallelism import enable_sp

enable_sp(parallel_manager, yunchang_backend)
```

---

## VAE 并行模块 (vae_parallelism)

### 概述

VAE 编解码分布式并行，支持 VAE Lightning 加速。

### API 使用

```python
from x_base.vae_parallelism import (
    VAEManager,
    enable_vae_lightning,
    enable_lightning,
    add_lightning_init,
    parallel_spatial_tiled_decode,
    SaveVideoStream,
)

# 启用 VAE Lightning 加速
enable_vae_lightning(vae)

# VAE 管理器
vae_manager = VAEManager(...)
```

---

## PHAA 模块 (phaa)

### 概述

通算并行，计算与通信重叠。

### API 使用

```python
from x_base.phaa import (
    enable_phaa,
    is_phaa_enabled,
    set_phaa_split_num,
    get_phaa_split_num,
    phaa_on_pipe,
)

# 启用 PHAA
phaa_on_pipe(pipe, args)
```

---

## 算子优化模块 (operator)

### 概述

提供 Attention、Matmul、RoPE 等算子的量化与融合优化。

### API 使用

```python
from x_base.operator import (
    attention_manager,       # Attention A8W8 量化
    matmul_manager,          # Matmul 动态量化
    rope_manager,            # RoPE 融合算子
    WeightQuantLinearModule, # 量化线性模块
)
```

| 算子 | 说明 |
|------|------|
| `attention_manager` | Attention A8W8 量化 |
| `matmul_manager` | Matmul 动态量化 |
| `rope_manager` | 旋转位置编码融合算子 |

---

## 无限显存模块 (infinite_vram)

### 概述

显存优化策略，通过权重动态 offload 降低显存占用。

### 原理

- 将部分层权重驻留于 RAM
- 在推理时通过 PCIe 动态加载到 VRAM
- 适用于 DiT 架构的长推理场景

### API 使用

```python
from x_base.infinite_vram import (
    OffloadManager,
    OffloadManager_For_Save_Memory,
    foreach_copy_,
    print_gpu_memory,
)

# 初始化 Offload 管理器
offloader = OffloadManager(
    transformer,
    module_groups={"blocks": transformer.blocks},
    keep_n={"blocks": 1},
    device=torch.device("cuda:0"),
    dist_group=group,
    sync_at_layer=True,
)
offloader.enable()
```

### 局限

- 无法和 FSDP 搭配使用
- 当前只有视频生成模型可用
- 无法与量化 matmul 一起用

---

## CFG 并行模块 (cfg_parallelism)

### 概述

Classifier-Free Guidance 并行。

### API 使用

```python
from x_base.cfg_parallelism import get_cfg_group

cfg_group = get_cfg_group()
```

---

## 预处理模块 (preprocess)

### 概述

输入数据预处理工具。

### API 使用

```python
from x_base.preprocess.vace_preprocess import prepare_video_and_mask

video, mask = prepare_video_and_mask(video_path, mask_path)
```

---

## 支持的模型列表

### 必需模型

| 模型名 | 类型 | 帧数约束 |
|--------|------|---------|
| `Wan2.1-T2V-1.3B` | Text-to-Video | 无 |
| `Wan2.1-T2V-14B` | Text-to-Video | 无 |
| `Wan2.1-I2V-14B` (480p) | Image-to-Video | 无 |
| `Wan2.1-I2V-14B` (720p) | Image-to-Video | 无 |
| `Wan2.2-T2V-A14B` | Text-to-Video | 无 |
| `Wan2.2-I2V-A14B` | Image-to-Video | 无 |

### 特殊约束模型

| 模型名 | 帧数约束 |
|--------|---------|
| `HunyuanVideo-T2V-13B` | frames = 4*k + 1 |
| `HunyuanVideo-I2V-13B` | frames = 4*k + 1 |

---

## 快速开始

### 基本使用

```python
from argparse import Namespace
from x_base.cache import cache_on_pipe
from x_base.utils import infer_info

# 1. 配置推理参数
args = Namespace(
    model="Wan2.1-T2V-1.3B",
    task_type="t2v",
    width=832,
    height=480,
    frames=121,
    save_fps=16,
    save_path="./output.mp4",
    num_inference_steps=50,
    turbo_mode="faiz",  # 使用 TeaCache 加速
    # ... 其他参数
)

# 2. 更新推理信息
infer_info.update_info(args)

# 3. 加载 Pipeline (示例)
# pipe = load_your_pipeline(args)

# 4. 应用 Cache 加速
pipe = cache_on_pipe(pipe, args)

# 5. 执行推理
# result = pipe(prompt="...")
```

### 启用多项加速特性

```python
# 通过 x_diffusers 参数自动启用相应特性

# 序列并行
--sp 8

# FSDP
--fsdp

# 通算并行
--phaa

# 量化
--atten_a8w8 --matmul_a8w8

# RoPE 融合
--rope_fused

# VAE 加速
--vae_lightning

# 无限显存
--infinite_vram
```

---

## 测试

### 运行测试

```bash
# 运行所有测试
pytest x_base/tests/

# 跳过需要 torch 的测试
pytest -m "not requires_torch" x_base/tests/

# 仅运行集成测试
pytest -m "integration" x_base/tests/

# 运行特定测试文件
pytest x_base/tests/test_cache.py -v
```

### 测试标记

| 标记 | 说明 |
|------|------|
| `@pytest.mark.requires_torch` | 需要 torch 环境 |
| `@pytest.mark.requires_diffusers` | 需要 diffusers 环境 |
| `@pytest.mark.integration` | 集成测试 |

---

## 版本兼容性

- **Python 3.9+**: 使用 `importlib.resources.files()`
- **Python 3.7-3.8**: 使用传统文件路径方式

---

## 迁移说明 (turbo → cache)

旧代码中的 `turbo` 模块已重构为 `cache` 模块：

```python
# 旧导入 (已废弃，但仍可用)
from x_base.turbo import turbo_on_pipe  # ⚠️ 废弃

# 新导入 (推荐)
from x_base.cache import cache_on_pipe  # ✅ 推荐
from x_base.cache import turbo_on_pipe  # ✅ 向后兼容别名
```

---

*文档生成时间: 2026-03-31*
*基于代码结构: x_base/x_base/*
