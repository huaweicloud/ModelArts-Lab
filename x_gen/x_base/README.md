# X_BASE

X_BASE 是 X_GEN 的基础库，实现各类高性能加速特性。

## 加速特性

### 分布式并行

| 特性 | 目录 | 说明 |
|------|------|------|
| 序列并行 (SP) | `sequence_parallelism/` | 分布式序列并行，跨卡切分序列维度 |
| FSDP | `fsdp/` | Fully Sharded Data Parallel，降低显存占用 |
| PHAA | `phaa/` | 通算并行，计算与通信重叠 |
| VAE 并行 | `vae_parallelism/` | VAE 编解码分布式并行 |
| CFG 并行 | `cfg_parallelism/` | Classifier-Free Guidance 并行 |

### 计算优化

| 特性 | 目录 | 说明 |
|------|------|------|
| 无限显存 | `infinite_vram/` | 显存优化策略 |

### 辅助模块

| 模块 | 目录 | 说明 |
|------|------|------|
| 缓存管理 | `cache/` | 推理缓存管理 |
| 工具函数 | `utils/` | 通用工具函数 |

### 算子优化

| 算子 | 文件 | 说明 |
|------|------|------|
| Attention | `operator/attention.py` | Attention A8W8 量化 |
| Matmul | `operator/matmul.py` | Matmul 动态量化 |
| RoPE | `operator/rope.py` | 旋转位置编码融合算子 |
| Rainfusion | `operator/rainfusion.py` | Rainfusion 算子 |

### 预处理

| 功能 | 目录 | 说明 |
|------|------|------|
| 预处理 | `preprocess/` | 输入数据预处理工具 |

## 配置

| 文件 | 说明                     |
|------|------------------------|
| `config/cache_config.yaml` | 缓存配置                   |
| `config/offload_config.yaml` | 显存加载配置                 |
| `config/quant_config.yaml` | 量化配置                   |
| `requirements.txt` | 依赖：torch, transformers |

## 使用方式

X_BASE 作为底层库，通常不直接使用。通过 x_diffusers 的参数自动启用相应特性：

```shell
# 启用序列并行
--sp 8

# 启用 FSDP
--fsdp

# 启用通算并行
--phaa

# 启用量化
--atten_a8w8 --matmul_a8w8

# 启用 RoPE 融合
--rope_fused

# 启用 VAE 加速
--vae_lightning
```

## 开发者文档

详细的 API 说明、配置规范、代码示例和测试说明，请参考 [DEVELOPMENT.md](DEVELOPMENT.md)。
