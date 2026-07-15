# X_GEN

X_GEN 是基于昇腾 NPU 的高性能视频生成推理引擎，支持 Wan、HunyuanVideo、CogVideoX 等主流视频生成模型。

## 支持模型

| 模型 | 任务类型 | 说明 |
|------|----------|------|
| Wan2.1-T2V-1.3B | 文生视频 | 1.3B 轻量版 |
| Wan2.1-T2V-14B | 文生视频 | 14B 标准版 |
| Wan2.1-I2V-14B | 图生视频 | 图像引导生成 |
| Wan2.1-VACE-1.3B | 多任务 | 视频编辑多任务 |
| Wan2.1-VACE-14B | 多任务 | 视频编辑多任务 |
| Wan2.2-T2V-A14B | 文生视频 | Wan2.2 加速版 |
| Wan2.2-I2V-A14B | 图生视频 | Wan2.2 加速版 |
| HunyuanVideo-T2V-13B | 文生视频 | 腾讯混元 |
| CogVideoX-5B | 文生视频 | 智谱 CogVideoX |

## 项目结构

```
x_gen/
├── x_base/              # 基础库 - 加速特性实现
├── x_diffusers/         # Diffusers 适配层 - 推理框架
├── example/             # 使用示例与脚本
├── comfyui/             # ComfyUI 集成
├── tools/               # 模型转换、超分等工具
├── x_vllm_omni/         # vLLM 集成（开发中）
└── build.sh             # 构建脚本
```

## 环境安装

### 1. Docker 环境

使用昇腾云 PyTorch 2.5.1 版本 Docker 镜像创建容器。

### 2. 安装依赖

```shell
# 安装昇腾加速包
pip install x_base-*-linux_aarch64.whl
pip install x_diffusers*-none-any.whl

```

### 3. 构建 Whl 包（可选）

```shell
bash build.sh
```

## 权重下载

| 模型 | 下载地址 |
|------|----------|
| Wan2.1-T2V-14B | https://huggingface.co/Wan-AI/Wan2.1-T2V-14B-Diffusers |
| Wan2.1-I2V-14B | https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-720P-Diffusers |
| Wan2.1-T2V-1.3B | https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers |
| Wan2.1-VACE-1.3B | https://huggingface.co/Wan-AI/Wan2.1-VACE-1.3B-diffusers |
| Wan2.1-VACE-14B | https://huggingface.co/Wan-AI/Wan2.1-VACE-14B-diffusers |
| Wan2.2-T2V-A14B | https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B-Diffusers |
| Wan2.2-I2V-A14B | https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B-Diffusers |
| HunyuanVideo | https://huggingface.co/tencent/HunyuanVideo |
| IFRNet_S_Vimeo90K | https://github.com/Fannovel16/ComfyUI-Frame-Interpolation/releases/tag/models |

### 使用 huggingface-cli

```shell
pip install "huggingface_hub[cli]"
huggingface-cli download Wan-AI/Wan2.2-T2V-A14B-Diffusers \
    --local-dir ./weights/Wan-AI/Wan2.2-T2V-A14B-Diffusers
```

## 快速开始

```shell
cd example/diffusers/scripts
bash infer_wan2.2_14b_t2v_720p.sh
```

## 详细文档

| 模块 | 说明 | 文档 |
|------|------|------|
| x_base | 加速特性（SP、FSDP、PHAA、量化等） | [x_base/README.md](x_base/README.md) |
| x_diffusers | API 使用、Pipeline 配置 | [x_diffusers/README.md](x_diffusers/README.md) |
| example | 示例脚本说明 | [example/README.md](example/README.md) |
| comfyui | ComfyUI 集成 | [comfyui/README.md](comfyui/README.md) |
| tools | 模型转换、超分工具 | [tools/README.md](tools/README.md) |
