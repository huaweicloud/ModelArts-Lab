
# Infinity VRAM // 无限显存

## 背景

- **带宽对比**：
  - PCIe：32 GB/s
  - HBM：0.8–1.6 TB/s
- **以往场景**（语言 / 图片 / 视频）：
  - 每个推理 step 很短。
  - 权重和激活数据必须驻留在 HBM 中。
- **当前场景**：基于 DiT 架构的视频推理，每 step 推理时长更久（例如单个 FlashAttention 可达 100ms以上）。
- **优化点**：可利用较长的推理时间，通过 PCIe 在推理前从 RAM 加载权重（约 1.6GB~3.2GB）到 VRAM。

**结论**：
权重数据可驻留于 RAM，仅在需要时传输至 HBM，从而缓解显存压力。

---

## 思路

- 将部分层（如 DiT 的前若干层）常驻于 HBM。
- 在推理到第 n 层时：
  - 加载第 n+1 层的权重。
  - 同时释放之前的层权重。
- 以此动态 offload 权重，降低显存占用。

---

## 局限

- 无法和FSDP搭配使用，作用都是降低显存，原理上有些区别
- 当前只有视频生成模型可用
- 无法与量化matmul一起用，因为涉及到权重改造
- 进一步offload掉 text_encoder 需要到fiffuser或wan源码中将使用完毕的text_encoder 卸载，当前版本为集成此功能


---



## 方案设计

1. 引入一个 **管理器 `OffloadManager`**：
   - 初始化时传入模型和需要控制的层类型。
2. 使用 **三条 CUDA Stream**：
   - 默认计算流（default）
   - `H2D`（Host to Device）
   - `D2H`（Device to Host）
3. 使用 **两个 forward hook**：
   - `register_forward_pre_hook`：在 forward 前加载权重
   - `register_forward_hook`：forward 后释放权重
4. **内存优化**：
   - 启用 `pin_memory` 加快数据搬运
5. **模块控制**：
   - 所有模块需标注 `index` 用于排序与识别
   - 若没有，需要用户自行添加

---

## 多卡 Sequence Parallelism (SP) 问题

- **问题现象**：
  - SP 多卡通信过程中，每个 block 内 FA 前后在 seq 维度进行切分 + alltoall。
  - 各卡计算快慢不一，导致 **hook 不同步触发**，引发错误。

- **解决方案**：
  - 在关键位置添加 `torch.distributed.barrier()`，用于不同 GPU 间同步。

---

## 算子下发问题

- **问题现象**：
  - 层数过多时参数下发频繁，ARM 上面临带宽瓶颈。

- **解决方案**：
  - 通过 **一次性整体下发**，减少频繁通信造成的延迟。

---

## 使用方法

### 1. 对于 `Wan` 系列模型：

只需配置：

```python
infinite_VRAM = True
```

### 2. 自定义使用示例

```python
# 环境准备
self.pipe.to("cpu")
self.pipe.vae.to("npu")
self.pipe.text_encoder.to("npu")

# 模块分配
for name, module in self.pipe.transformer.named_children():
    if name == "blocks":
        module.to("cpu")
    else:
        module.to("npu")

# 参数转移（非递归）
for name, param in self.pipe.transformer.named_parameters(recurse=False):
    if name != "blocks":
        param.data = param.data.to("npu", non_blocking=True)

# 多卡设置
import torch.distributed as dist
group = dist.group.WORLD

from diffusers.pipelines.wan.off_mana import OffloadManager

# 初始化 Offload 管理器
offloader = OffloadManager(
    self.pipe.transformer,
    module_groups={"blocks": self.pipe.transformer.blocks},
    keep_n={"blocks": 1},
    device=torch.device(f"cuda:{dist.get_rank()}"),
    dist_group=group,
    sync_at_layer=True  # 开启层间同步
)

offloader.enable()
```

---

## TODO

- [√] 支持单卡
- [√] 支持多卡
- [√] 支持模型内多类blocks
- [√] 支持sp切分
- [√] 支持Wan2.1Model
- [ ] 支持Wan2.2Model
- [ ] 进一步拉平和原始推理性能差别
- [ ] 支持更多model



