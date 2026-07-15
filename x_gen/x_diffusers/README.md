# X_DIFFUSERS

X_DIFFUSERS 是 X_GEN 的 Diffusers 适配层，提供模型推理的 Pipeline 和 API。

## 模块结构

```
x_diffusers/
├── x_diffusers/
│   ├── adaptor/          # 适配器
│   │   ├── config.py     # 配置解析
│   │   ├── load_pipe.py  # Pipeline 加载
│   │   ├── infer_tools.py # 推理工具
│   │   ├── distillation.py # 蒸馏支持
│   │   └── utils.py      # 工具函数
│   └── framework/        # 推理框架
│       ├── pipeline/     # 推理 Pipeline
│       ├── transformer/  # Transformer 模型适配
│       ├── vae/          # VAE 模型适配
│       ├── schedulers/   # 调度器
│       └── lora/         # LoRA 支持
└── requirements.txt
```

## 支持模型

| 模型 | Transformer 模块 | VAE 模块 |
|------|-------------------|----------|
| Wan | `transformer/wan.py` | `vae/wan.py` |
| Wan VACE | `transformer/wan_vace.py` | `vae/wan.py` |
| HunyuanVideo | `transformer/hunyuan.py` | `vae/hunyuan.py` |
| CogVideoX | `transformer/cogvideox.py` | `vae/cogvideox.py` |

## VACE 任务类型

| 任务 | 参数值 | 说明 |
|------|--------|------|
| 文生视频 | `t2v` | 纯文本生成视频 |
| 图生视频 | `i2v` | 图片生成视频 |
| 首尾帧生成 | `flf2v` | 首帧+尾帧生成中间视频 |
| 视频延伸 | `v2lf` | 视频延伸到最后帧 |
| 视频修补 | `inpaint` | 视频区域修复 |
| 视频延展 | `outpaint` | 视频边界扩展 |
| 随机生成 | `random2v` | 随机帧引导生成 |
| 姿态控制 | `openpose` | OpenPose 姿态引导生成 |
| 绘画扩展 | `iwri` | 图像绘制扩展 |

## API 使用

### 基本用法

```python
from x_diffusers import init_env, InferenceManager, parse_args

# 初始化环境
init_env()

# 创建推理管理器
inference_manager = InferenceManager()

# 解析参数并执行推理
args = parse_args()
inference_manager.infer(args)
```

### 命令行参数

| 参数 | 类型 | 说明 |
|------|------|------|
| `--model` | str | 模型名称，如 `Wan2.2-T2V-A14B` |
| `--pretrained_model_name_or_path` | str | 模型权重路径 |
| `--task_type` | str | 任务类型：`t2v`, `i2v` |
| `--vace_task` | str | VACE 任务类型 |
| `--save_path` | str | 输出视频路径 |
| `--prompt` | str | 正向提示词 |
| `--negative_prompt` | str | 负向提示词 |
| `--width` | int | 视频宽度 |
| `--height` | int | 视频高度 |
| `--frames` | int | 帧数 |
| `--num_inference_steps` | int | 推理步数 |
| `--guidance_scale` | float | 引导系数 |
| `--seed` | int | 随机种子 |

### 加速参数

| 参数 | 说明 |
|------|------|
| `--sp` | 序列并行度 |
| `--fsdp` | 启用 FSDP |
| `--phaa` | 启用通算并行 |
| `--turbo_mode` | Turbo 模式：`faiz`, `next_faiz` |
| `--vae_lightning` | VAE 加速 |
| `--atten_a8w8` | Attention 量化 |
| `--matmul_a8w8` | Matmul 量化 |
| `--rope_fused` | RoPE 融合算子 |

## Pipeline 类型

| Pipeline | 说明 |
|----------|------|
| `pipeline.py` | 标准推理 Pipeline |
| `joint.py` | 联合推理 Pipeline |
| `ten_second.py` | 10 秒长视频推理 |

## LoRA 支持

```python
from x_diffusers import lora

# 加载 LoRA
lora.load_lora_weights(pipeline, lora_path)
```
