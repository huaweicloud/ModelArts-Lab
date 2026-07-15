# 工具集

本目录包含模型转换、环境配置和第三方模型支持等工具。

## 目录结构

```
tools/
├── conversion/           # 模型转换
├── setup/                # 环境配置
│   ├── ffmpeg/          # FFmpeg 配置
│   └── usp/             # USP 配置
└── thirdparty_model/    # 第三方模型
    └── seedvr2/         # SeedVR2 超分模型
```

## 模型转换

### Wan2.2 权重转换

```shell
python tools/conversion/wan2.2_convert_safetensors.py \
    --input_path /path/to/original_weights \
    --output_path /path/to/converted_weights
```

用于将原始 safetensors 格式转换为 X_GEN 兼容格式。

## 环境配置

### FFmpeg

```shell
bash tools/setup/ffmpeg/install.sh
```

### USP

```shell
bash tools/setup/usp/install.sh
```

## 第三方模型

### SeedVR2 超分模型

SeedVR2 用于视频超分辨率增强。

```shell
bash tools/thirdparty_model/seedvr2/seedvr2.sh
```

#### Patch 说明

| 文件 | 说明 |
|------|------|
| `seedvr2.patch` | 主 patch |
| `seedvr2_frame.patch` | 帧处理 patch |
| `seedvr2_server.patch` | 服务端 patch |

#### 使用方式

1. 克隆 SeedVR2 仓库
2. 应用对应 patch
3. 运行推理脚本

```shell
# 示例
git clone https://github.com/xxx/seedvr2.git
cd seedvr2
git apply /path/to/seedvr2.patch
bash seedvr2.sh
```

## 插帧模型

插帧功能使用 IFRNet_S_Vimeo90K 模型，下载地址：

https://github.com/Fannovel16/ComfyUI-Frame-Interpolation/releases/tag/models

下载 `IFRNet_S_Vimeo90K.pth`，通过 `--frame_model_path` 指定路径。
