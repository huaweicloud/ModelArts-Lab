# 视频生成服务

基于 Wan2.2-T2V/I2V-14B 模型的视频生成服务，支持文生视频(t2v)和图生视频(i2v)。

## 功能特性

- 异步 HTTP API 接口
- 支持文生视频 (T2V) 和图生视频 (I2V)
- 支持单卡和多卡并行推理
- 支持 5s 和 10s 视频生成
- 支持超分辨率 (SeedVR2)
- 支持帧插值 (平滑视频)
- 支持视频下载接口
- 支持任务状态查询
- 支持配置管理
- 支持推理指标统计
- 支持定时清理历史视频

## 目录结构

```
server/
├── video_server.py           # 服务主程序
├── start_video_service.sh    # 启动脚本
├── video_service_ctl.sh      # 服务管理脚本
├── config_wan22_t2v.json     # T2V 模型配置
├── config_wan22_i2v.json     # I2V 模型配置
├── Video_Readme.md           # 本文档
├── VIDEO_API.md              # API 详细文档
└── client_example.py         # 客户端示例
```

## 快速开始

### 1. 启动服务

在服务器容器内执行：

```bash
# 进入项目目录
cd /Ascend_cloud_aigc_poc/aigc_inference/torch_npu/x_gen/server

# 单卡 T2V
bash start_video_service.sh wan2.2-t2v-14b 1 5001

# 4卡 T2V (推荐)
bash start_video_service.sh wan2.2-t2v-14b 4 5006 --warmup

# 8卡 T2V (最高性能)
bash start_video_service.sh wan2.2-t2v-14b 8 5006 --warmup

# 8卡 I2V
bash start_video_service.sh wan2.2-i2v-14b 8 5006 --warmup

# 8卡 + 10s 模式 + 预热
bash start_video_service.sh wan2.2-i2v-14b 8 5006 --ten-second --warmup
```

### 2. 启动参数说明

| 参数 | 说明 | 可选值 |
|------|------|--------|
| model | 模型名称 | wan2.2-t2v-14b, wan2.2-i2v-14b |
| num_gpus | GPU 数量 | 1, 2, 4, 8 |
| port | 服务端口 | 如 5001, 5006 |

**选项参数：**

| 选项 | 说明 | 默认值 |
|------|------|--------|
| --devices \<ids\> | 指定 NPU 设备 | 自动选择 |
| --steps \<n\> | 默认推理步数 | 单卡 40, 多卡 20 |
| --ten-second | 启用 10s 视频模式 | 关闭 |
| --no-init | 跳过预初始化 | 预初始化 |
| --warmup | 启动时预热 | 关闭 |

### 3. 启动示例

```bash
# 单卡 T2V，端口 5001
bash start_video_service.sh wan2.2-t2v-14b 1 5001

# 4卡 T2V，带预热 (默认 NPU 4,5,6,7)
bash start_video_service.sh wan2.2-t2v-14b 4 5006 --warmup

# 8卡 T2V，带预热
bash start_video_service.sh wan2.2-t2v-14b 8 5006 --warmup

# 8卡 I2V，启用 10s 模式
bash start_video_service.sh wan2.2-i2v-14b 8 5006 --ten-second --warmup
```

### 4. 检查服务状态

```bash
curl http://127.0.0.1:5006/health
```

### 5. 停止服务

```bash
pkill -f "video_server.py.*port 5006"
```

## API 接口

### 1. 健康检查

GET /health

### 2. 提交视频生成任务

POST /generate
Content-Type: application/json

**T2V 请求示例**：
```json
{
  "prompt": "A cat walking on a sunny beach",
  "width": 832,
  "height": 480,
  "frames": 81,
  "num_inference_steps": 40,
  "seed": 42
}
```

**I2V 请求示例**：
```json
{
  "prompt": "The person starts walking forward",
  "i2v_image_path": "/path/to/input_image.jpg",
  "width": 832,
  "height": 480,
  "frames": 81,
  "num_inference_steps": 20
}
```

**请求参数**：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| prompt | string | **必填** | 文本提示词 |
| width | int | 832 | 视频宽度 |
| height | int | 480 | 视频高度 |
| frames | int | 81 | 帧数 |
| num_inference_steps | int | 40 | 推理步数 |
| seed | int | 随机 | 随机种子 |
| duration | int | 5 | 视频时长 (5 或 10) |
| save_fps | int | 16 | 输出帧率 |
| i2v_image_path | string | - | I2V 输入图片路径 |
| ten_second | bool | false | 启用 10s 视频生成 |
| adopt_sr | bool | false | 启用超分辨率 |
| resolution | int | 480 | SR 目标分辨率 |

### 3. 查询任务状态

GET /status/\<task_id\>

### 4. 下载视频

GET /download/\<task_id\>

### 5. 获取推理指标

GET /metrics

## 客户端示例

### Python 示例

```python
import requests
import time

BASE_URL = "http://127.0.0.1:5006"

# 提交任务
response = requests.post(
    f"{BASE_URL}/generate",
    json={
        "prompt": "A beautiful sunset over the ocean",
        "width": 832,
        "height": 480,
        "frames": 81,
        "num_inference_steps": 20,
        "seed": 42
    }
)
task_id = response.json()["task_id"]

# 轮询状态
while True:
    status = requests.get(f"{BASE_URL}/status/{task_id}").json()
    if status["status"] == "completed":
        print(f"Video: {status['result']['video_path']}")
        break
    elif status["status"] == "failed":
        print(f"Error: {status['error']}")
        break
    time.sleep(10)

# 下载视频
video_data = requests.get(f"{BASE_URL}/download/{task_id}").content
with open("output.mp4", "wb") as f:
    f.write(video_data)
```


# Video Generation Service API

## 服务端点

| 端点 | 方法 | 描述 |
|------|------|------|
| `/health` | GET | 健康检查 |
| `/metrics` | GET | 推理统计指标 |
| `/generate` | POST | 提交视频生成任务 |
| `/status/<task_id>` | GET | 查询任务状态 |
| `/result/<task_id>` | GET | 获取任务结果 |
| `/tasks` | GET | 列出所有任务 |
| `/cancel/<task_id>` | POST | 取消任务 |
| `/params` | GET | 查看参数规范 |
| `/config` | GET/POST | 查看/更新配置 |

## API 使用示例

### 健康检查
```bash
curl -s http://127.0.0.1:5001/health | python -m json.tool
```

响应:
```json
{
  "status": "healthy",
  "model": "Wan2.2-T2V-A14B",
  "task_type": "t2v",
  "rank": 0,
  "world_size": 1,
  "is_initialized": true,
  "tasks": {
    "total": 10,
    "running": 1,
    "pending": 0,
    "completed": 8,
    "failed": 1
  }
}
```

### 提交 T2V 任务
```bash
curl -X POST http://127.0.0.1:5001/generate \
  -H 'Content-Type: application/json' \
  -d '{
    "prompt": "A cat walking on a sunny beach",
    "width": 832,
    "height": 480,
    "frames": 81,
    "num_inference_steps": 40,
    "seed": 42
  }'
```

响应:
```json
{
  "message": "Task submitted successfully",
  "status": "accepted",
  "task_id": "92cd8423-f828-4abe-92cd-17dcd46de1f2"
}
```

### 提交 I2V 任务
```bash
curl -X POST http://127.0.0.1:5002/generate \
  -H 'Content-Type: application/json' \
  -d '{
    "prompt": "The person starts walking forward",
    "i2v_image_path": "/path/to/input_image.jpg",
    "width": 832,
    "height": 480,
    "frames": 81,
    "num_inference_steps": 20,
    "seed": 42
  }'
```

### 提交 10s 视频任务 (I2V + 10s + SR)
```bash
curl -X POST http://127.0.0.1:5001/generate \
  -H 'Content-Type: application/json' \
  -d '{
    "prompt": "A cat walking gracefully on a sunny beach",
    "i2v_image_path": "/path/to/input_image.jpg",
    "width": 832,
    "height": 480,
    "frames": 81,
    "num_inference_steps": 6,
    "seed": 42,
    "ten_second": true,
    "adopt_sr": true,
    "resolution": 1080
  }'
```

> **10s 视频生成说明**:
> - `ten_second=true`: 启用两阶段生成 (5s base + 5s extension)
> - `adopt_sr=true`: 启用超分辨率 (SeedVR2)
> - `resolution`: SR 目标分辨率 (480/720/1080)
> - 推荐: 8 GPUs + 6 steps + 10s + SR 1080p
> - 自动启用: x=True, joint=True, inf_vram_blocks_num=1

### 查询任务状态
```bash
curl -s http://127.0.0.1:5001/status/<task_id> | python -m json.tool
```

响应:
```json
{
  "task_id": "92cd8423-...",
  "task_type": "video_generation",
  "status": "completed",
  "progress": 100.0,
  "result": {
    "video_path": "/tmp/generated_videos/video_xxx.mp4",
    "infer_time": 261.82
  },
  "created_at": "2026-04-09T16:11:12.960996",
  "started_at": "2026-04-09T16:11:12.961216",
  "completed_at": "2026-04-09T16:16:41.616901",
  "infer_time": 261.82,
  "error": null
}
```

### 获取推理指标
```bash
curl -s http://127.0.0.1:5001/metrics | python -m json.tool
```

响应:
```json
{
  "completed_tasks": 8,
  "avg_infer_time": 250.5,
  "min_infer_time": 180.2,
  "max_infer_time": 320.8,
  "model": "Wan2.2-T2V-A14B",
  "world_size": 1
}
```

## 生成参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| prompt | string | **必填** | 文本提示词 (≤2000字符) |
| negative_prompt | string | "" | 负向提示词 (≤2000字符) |
| width | int | 832 | 视频宽度 (64-2048, **必须为偶数**) |
| height | int | 480 | 视频高度 (64-2048, **必须为偶数**) |
| frames | int | 81 | 帧数 (1-200) |
| num_inference_steps | int | 40 | 推理步数 (1-200) |
| seed | int | 42 | 随机种子 (≥0) |
| guidance_scale | float | 模型默认 | 引导强度 1 (0-100) |
| guidance_scale_2 | float | 模型默认 | 引导强度 2 (0-100) |
| i2v_image_path | string | None | I2V 输入图片路径 (I2V模型必填) |
| ten_second | bool | false | 启用 10s 视频生成 (两阶段: 5s+5s) |
| adopt_sr | bool | false | 启用超分辨率 (SeedVR2) |
| resolution | int | 480 | SR 目标分辨率 (480/720/1080) |
| frame_interpolation | bool | false | 启用帧插值 (平滑视频) |

### 参数校验

服务会对所有参数进行校验：

**错误示例** (参数无效，请求被拒绝):
```json
{
  "status": "error",
  "message": "Parameter validation failed",
  "details": {
    "errors": [
      "prompt is required and cannot be empty",
      "width must be even, got 833"
    ],
    "warnings": []
  }
}
```

**警告示例** (参数有效，但有潜在问题):
```json
{
  "status": "accepted",
  "task_id": "xxx",
  "message": "Task submitted successfully",
  "warnings": [
    "High resolution (1920x1080) may cause OOM, recommended: 832x480 or 480p/720p",
    "Large frame count (150) will significantly increase generation time"
  ]
}
```

### 类型自动转换

字符串形式的数字会自动转换为对应类型：
```json
{
  "prompt": "test",
  "width": "832",    // 自动转为 int
  "seed": "42"       // 自动转为 int
}
```

## 任务状态

| 状态 | 说明 |
|------|------|
| pending | 等待处理 |
| running | 正在推理 |
| completed | 完成 |
| failed | 失败 |
| cancelled | 已取消 |

## Python 调用示例

```python
import requests
import time

# 提交任务
response = requests.post(
    "http://127.0.0.1:5001/generate",
    json={
        "prompt": "A beautiful sunset over the ocean",
        "width": 832,
        "height": 480,
        "frames": 81,
        "num_inference_steps": 40,
        "seed": 42
    }
)
task_id = response.json()["task_id"]
print(f"Task submitted: {task_id}")

# 轮询状态
while True:
    status = requests.get(f"http://127.0.0.1:5001/status/{task_id}").json()
    print(f"Status: {status['status']}")

    if status["status"] == "completed":
        print(f"Video saved to: {status['result']['video_path']}")
        print(f"Inference time: {status['infer_time']:.2f}s")
        break
    elif status["status"] == "failed":
        print(f"Error: {status['error']}")
        break

    time.sleep(10)
```

## 性能参考

| GPU 数量 | 推理时间 (20步) | 加速比 |
|----------|-----------------|--------|
| 1 x 910B3 | ~260s | 1.0x |
| 2 x 910B3 | ~130s | 2.0x |
| 8 x 910B3 | ~60s | 4.3x |

> 实际性能取决于分辨率、帧数、模型等参数
>
> 默认开启: `atten_laser`, `matmul_a8w8`, `rope_fused`
