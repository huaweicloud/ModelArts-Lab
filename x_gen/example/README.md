# 示例脚本

本目录包含各模型的推理示例脚本。

## 目录结构

```
example/
└── diffusers/
    ├── infer.py              # 推理入口
    ├── infer_batch.py        # 批量推理
    ├── infer_image_gen.py    # 图像生成
    ├── scripts/              # 标准示例脚本
    │   ├── wan_vace/         # VACE 任务示例
    │   └── scripts_self/     # 自定义示例
    ├── setup/                # 环境配置
    ├── tools/                # 工具脚本
    └── weights/              # 权重目录（需自行下载）
```

## 示例脚本列表

### Wan2.2 系列

| 脚本 | 模型 | 分辨率 |
|------|------|--------|
| `infer_wan2.2_14b_t2v_720p.sh` | Wan2.2-T2V-A14B | 1280x720 |
| `infer_wan2.2_14b_t2v_720p_minimum.sh` | Wan2.2-T2V-A14B | 1280x720 (最小配置) |
| `infer_wan2.2_14b_t2v_480p.sh` | Wan2.2-T2V-A14B | 832x480 |
| `infer_wan2.2_14b_t2v_480p_minimum.sh` | Wan2.2-T2V-A14B | 832x480 (最小配置) |
| `infer_wan2.2_14b_i2v_720p.sh` | Wan2.2-I2V-A14B | 1280x720 |
| `infer_wan2.2_14b_i2v_480p.sh` | Wan2.2-I2V-A14B | 832x480 |
| `infer_wan2.2_14b_t2i_720p.sh` | Wan2.2-T2V-A14B | 1280x720 (T2I) |
| `infer_wan2.2_14b_t2i_480p.sh` | Wan2.2-T2V-A14B | 832x480 (T2I) |

### Wan2.1 系列

| 脚本 | 模型 | 说明 |
|------|------|------|
| `infer_wan2.1_1.3b_t2v.sh` | Wan2.1-T2V-1.3B | 轻量版 T2V |
| `infer_wan2.1_14b_t2v_720p.sh` | Wan2.1-T2V-14B | 720P T2V |
| `infer_wan2.1_14b_t2v_480p.sh` | Wan2.1-T2V-14B | 480P T2V |
| `infer_wan2.1_14b_i2v_720p.sh` | Wan2.1-I2V-14B | 720P I2V |
| `infer_wan2.1_14b_i2v_480p.sh` | Wan2.1-I2V-14B | 480P I2V |
| `infer_wan2.1_14b_t2i_720p.sh` | Wan2.1-T2V-14B | 720P T2I |
| `infer_wan2.1_14b_t2i_480p.sh` | Wan2.1-T2V-14B | 480P T2I |

### HunyuanVideo

| 脚本 | 说明 |
|------|------|
| `infer_hunyuan_video_13b_t2v_sp.sh` | HunyuanVideo-13B 序列并行 |

### CogVideoX

| 脚本 | 说明 |
|------|------|
| `infer_cogvideo_5b.sh` | CogVideoX-5B |

### VACE 任务

| 脚本 | 任务 |
|------|------|
| `wan_vace/infer_wan_vace_14b_t2v.sh` | 文生视频 |
| `wan_vace/infer_wan_vace_14b_i2v.sh` | 图生视频 |
| `wan_vace/infer_wan_vace_14b_inpaint.sh` | 视频修补 |
| `wan_vace/infer_wan_vace_14b_outpaint.sh` | 视频延展 |
| `wan_vace/infer_wan_vace_14b_flf2v.sh` | 首尾帧生成 |
| `wan_vace/infer_wan_vace_14b_v2lf.sh` | 视频延伸 |
| `wan_vace/infer_wan_vace_14b_random2v.sh` | 随机生成 |
| `wan_vace/infer_wan_vace_14b_openpose.sh` | 姿态控制 |
| `wan_vace/infer_wan_vace_14b_iwri.sh` | 绘画扩展 |
| `wan_vace/openpose_preprocess.sh` | OpenPose 预处理 |

## 环境变量

运行前需设置：

```shell
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export MEMORY_FRAGMENTATION=1
export COMBINED_ENABLE=1
export TASK_QUEUE_ENABLE=2
export TOKENIZERS_PARALLELISM=false
```

## 快速运行

```shell
cd example/diffusers/scripts

# Wan2.2-T2V 720P (8卡)
bash infer_wan2.2_14b_t2v_720p.sh

# Wan2.1-T2V-1.3B (单卡)
bash infer_wan2.1_1.3b_t2v.sh
```

## 自定义示例

`scripts_self/` 目录存放自定义脚本，参考现有脚本修改参数即可。
