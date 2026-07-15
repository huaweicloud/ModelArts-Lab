# 图像生成服务

基于 x_gen 框架的同步图像生成在线服务。

## 功能特性

- 同步 HTTP API 接口
- 支持图像生成（基于 ImageInferenceManager）
- 支持单卡和多卡并行（cfg_parallel_size=1/2）
- 支持 base64 编码返回
- 支持批量生成
- 支持配置管理
- 支持图像下载接口
- 支持定时清理历史图像

## 目录结构

```
server/
├── image_server.py           # 服务主程序
├── start_image_server.sh     # 启动脚本
├── config_.json              # 配置
├── README.md                 # 本文档
```

## 快速开始

### 1. 启动服务

在服务器容器内执行：

```
# 单卡模式
bash start_server.sh single

# 多卡模式（双卡并行）
bash start_server.sh multi

# 带额外参数
bash start_server.sh single 0.0.0.0 5000 false --model Qwen-Image-Edit
```

### 2. 启动参数说明

```bash
bash start_server.sh [mode] [host] [port] [warmup] [extra_args...]
```

| 参数 | 说明 | 默认值 |
|------|------|--------|
| mode | single(单卡) / multi(多卡) | single |
| host | 服务监听地址 | 0.0.0.0 |
| port | 服务端口 | 5000 |
| warmup | 是否预热 (true/false) | false |
| extra_args | 额外参数传递给 image_server.py | - |

### 3. 额外参数示例

```bash
# 指定模型
bash start_server.sh single 0.0.0.0 5000 false --model Qwen-Image-Edit

# 禁用 cache-dit
bash start_server.sh multi 0.0.0.0 5000 true --no-cache-dit

# 添加 LoRA
bash start_server.sh single 0.0.0.0 5000 false --lora-path /path/to/lora --lora-weight 0.8
```

## API 接口

### 1. 健康检查

```http
GET /health
```

响应：
```json
{
    "status": "healthy",
    "initialized": true,
    "current_model": "Qwen-Image",
    "rank": 0,
    "world_size": 1,
    "supported_models": ["Qwen-Image", "Qwen-Image-Edit", ...],
    "cleanup_config": {
        "enabled": true,
        "interval_hours": 6,
        "retain_days": 7,
        "output_dir": "/home/ma-user/generated_images"
    },
    "timestamp": "2024-01-01T12:00:00"
}
```

### 2. 生成图像

```http
POST /generate
Content-Type: application/json

{
    "prompt": "A beautiful landscape",
    "negative_prompt": " ",
    "height": 512,
    "width": 512,
    "num_inference_steps": 20,
    "true_cfg_scale": 4.0,
    "seed": 42,
    "return_base64": true,
    "save_path": "/tmp/output.png"
}
```

**参数说明：**

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| prompt | string | 是 | - | 生成提示词（最大1000字符） |
| negative_prompt | string | 否 | " " | 负面提示词 |
| height | int | 否 | 512 | 图像高度（64-4096，8的倍数） |
| width | int | 否 | 512 | 图像宽度（64-4096，8的倍数） |
| num_inference_steps | int | 否 | 20 | 推理步数（1-200） |
| true_cfg_scale | float | 否 | 4.0 | CFG缩放系数（0-20） |
| seed | int | 否 | 42 | 随机种子 |
| return_base64 | bool | 否 | true | 是否返回base64编码 |
| save_path | string | 否 | 自动生成 | 保存路径 |
| image_base64 | string | 否 | - | 输入图像base64编码（Edit模型） |
| image_path | string | 否 | - | 输入图像文件路径（Edit模型） |

> **注意：** `image_base64` 和 `image_path` 用于图像编辑模型（如 Qwen-Image-Edit）。两者同时提供时，优先使用 `image_base64`。

**响应：**

```json
{
    "status": "success",
    "infer_time": 18.09,
    "prompt": "A beautiful landscape",
    "filename": "image_20240101_120000.png",
    "download_url": "http://127.0.0.1:5000/download/image_20240101_120000.png",
    "save_path": "/home/ma-user/generated_images/image_20240101_120000.png",
    "image_size": 197730,
    "image_base64": "iVBORw0KGgoAAAANSUhEUgAA...",
    "image_format": "png"
}
```

### 3. 下载图像

```http
GET /download/<filename>
```

直接访问返回的 `download_url` 即可下载图像文件。

示例：
```bash
curl -O http://127.0.0.1:5000/download/image_20240101_120000.png
```

或在网页中直接访问该URL。

### 4. 使用 base64 输入图像（Edit 模型）

```http
POST /generate
Content-Type: application/json

{
    "prompt": "change to blue color",
    "image_base64": "iVBORw0KGgoAAAANSUhEUgAAAIAAAACACAIAAABMXPac...",
    "width": 512,
    "height": 512,
    "num_inference_steps": 20,
    "return_base64": false
}
```

> `image_base64` 支持 data URI 前缀格式：`data:image/png;base64,iVBORw0...`

### 5. 手动触发清理

```http
POST /cleanup
```

响应：
```json
{
    "status": "success",
    "message": "Cleanup completed",
    "current_files": 10,
    "total_size_mb": 2.45
}
```

### 6. 手动触发心跳

```http
POST /heartbeat
```

响应：
```json
{
    "status": "success",
    "message": "Heartbeat sent",
    "timestamp": "2024-01-01T12:00:00"
}
```

**说明**：
- 仅在多卡模式下有效
- 手动发送心跳信号保持分布式通信活跃
- 正常情况下服务会自动发送心跳（每 30 秒）

### 7. 批量生成

```http
POST /generate/batch
Content-Type: application/json

{
    "prompts": [
        "A sunset over the ocean",
        "A cat on a blanket",
        "A futuristic city"
    ],
    "height": 512,
    "width": 512
}
```

### 8. 配置管理

```http
GET /config
```

```http
POST /config
Content-Type: application/json

{
    "num_inference_steps": 30,
    "true_cfg_scale": 5.0
}
```

### 9. 查看参数规范

```http
GET /params
```

返回所有生成参数的详细规范说明。

### 10. 查看支持的模型

```http
GET /models
```

响应：
```json
{
    "current_model": "Qwen-Image",
    "supported_models": [
        "Qwen-Image",
        "Qwen-Image-Edit",
        "qwen-image-edit-2509",
        "qwen-image-edit-2511",
        "longcat-image",
        "longcat-image-edit",
        "z-image"
    ],
    "model_paths": {
        "Qwen-Image": "/home/models/Qwen-Image",
        "Qwen-Image-Edit": "/home/models/Qwen-Image-Edit",
        ...
    }
}
```

## 配置说明

### 单卡配置 (config_single.json)

```json
{
    "model": "Qwen-Image",
    "pretrained_model_name_or_path": "/home/models/Qwen-Image",
    "dtype": "bf16",
    "cfg_parallel_size": 1,
    "matmul_a8w8": true,
    "atten_laser": true,
    "cache_dit": true
}
```

### 多卡配置 (config_multi.json)

```json
{
    "model": "Qwen-Image",
    "pretrained_model_name_or_path": "/home/models/Qwen-Image",
    "dtype": "bf16",
    "cfg_parallel_size": 2,
    "matmul_a8w8": true,
    "atten_laser": true,
    "cache_dit": true
}
```

### 配置项说明

| 配置项 | 说明 |
|--------|------|
| model | 模型名称 |
| pretrained_model_name_or_path | 模型路径 |
| dtype | 数据类型 (fp16/bf16/fp32) |
| cfg_parallel_size | CFG并行数 (1=单卡, 2=双卡) |
| matmul_a8w8 | 矩阵乘法量化加速 |
| atten_laser | 注意力机制优化 |
| cache_dit | DiT缓存优化 |

## 定时清理功能

### 配置参数

服务内置定时清理功能，在 `image_server.py` 中配置：

```python
CLEANUP_INTERVAL_HOURS = 6                            # 清理间隔（小时）
CLEANUP_RETAIN_DAYS = 7                               # 保留天数
CLEANUP_ENABLED = True                                # 是否启用清理
```

### 工作机制

- 服务启动时自动启动清理线程（仅 rank 0）
- 每 6 小时自动清理一次
- 删除超过 7 天的图像文件
- 支持手动触发 `POST /cleanup`

## 心跳机制

### 配置参数

服务内置心跳机制，用于保持多卡模式下分布式通信活跃，防止长时间无请求导致服务挂掉。在 `image_server.py` 中配置：

```python
HEARTBEAT_INTERVAL_SECONDS = 30  # 心跳间隔（秒）
HEARTBEAT_ENABLED = True          # 是否启用心跳
```

### 工作机制

- **仅在多卡模式下生效**：单卡模式不需要心跳
- **自动发送心跳**：rank 0 每 30 秒向所有 worker 广播心跳信号
- **worker 响应**：worker 收到心跳信号后继续等待，保持连接活跃
- **支持手动触发**：`POST /heartbeat` 接口可手动发送心跳

### 手动触发心跳

```http
POST /heartbeat
```

响应：
```json
{
    "status": "success",
    "message": "Heartbeat sent",
    "timestamp": "2024-01-01T12:00:00"
}
```

### 健康检查接口增强

`GET /health` 接口现在包含心跳信息：

```json
{
    "status": "healthy",
    "initialized": true,
    "current_model": "Qwen-Image",
    "rank": 0,
    "world_size": 2,
    "heartbeat_config": {
        "enabled": true,
        "interval_seconds": 30,
        "last_heartbeat": "2024-01-01T12:00:00"
    }
}
```

## 支持的模型

| 模型名称 | 模型路径 | 说明 |
|----------|----------|------|
| Qwen-Image | /home/models/Qwen-Image | 文生图模型 |
| Qwen-Image-Edit | /home/models/Qwen-Image-Edit | 图像编辑模型 |
| qwen-image-edit-2509 | /home/models/qwen-image-edit-2509 | 图像编辑模型 |
| qwen-image-edit-2511 | /home/models/qwen-image-edit-2511 | 图像编辑模型 |
| Qwen-Image-2512 | /home/models/Qwen-Image-2512 | 文生图模型 |
| longcat-image | /home/models/longcat-image | 文生图模型 |
| longcat-image-edit | /home/models/longcat-image-edit | 图像编辑模型 |
| z-image | /home/models/z-image | 文生图模型 |

**注意：** 切换模型需要重启服务。

## 客户端示例

```python
from client_example import ImageGenerationClient

# 创建客户端
client = ImageGenerationClient(host='http://127.0.0.1', port=5000)

# 检查服务状态
health = client.health_check()
print(health)

# 生成图像
result = client.generate(
    prompt='A beautiful sunset over the ocean',
    height=512,
    width=512,
    num_inference_steps=20,
    return_base64=True
)

# 保存图像
if result.get('status') == 'success' and 'image_base64' in result:
    client.save_image_from_base64(result['image_base64'], 'output.png')

# 或直接下载
if 'download_url' in result:
    print(f"Download URL: {result['download_url']}")
```

## 性能优化

- **matmul_a8w8**: 矩阵乘法量化加速
- **atten_laser**: 注意力机制优化
- **cache_dit**: DiT 缓存优化
- **多卡并行**: cfg_parallel_size=2 双卡 CFG 并行

## 注意事项

1. **多卡并行**: `cfg_parallel_size=2` 时，会使用 `init_cfg_env` 初始化多卡环境
2. **设备配置**: 通过环境变量 `ASCEND_RT_VISIBLE_DEVICES` 指定使用的 NPU 设备
3. **模型切换**: 需要重启服务，使用 `--model` 参数指定不同模型
4. **图像保存**: 默认保存到 `/home/ma-user/generated_images/`
5. **定时清理**: 自动清理超过 7 天的图像，可通过配置修改
6. **依赖安装**: 需要安装 Flask 和 flask-cors
   ```bash
   pip install flask flask-cors
   ```

### 检查服务状态

```bash
curl http://127.0.0.1:5000/health
```
