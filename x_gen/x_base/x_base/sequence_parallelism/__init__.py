"""
Sequence Parallelism Module

序列并行模块，提供分布式序列并行功能。

## 文件结构

```
sequence_parallelism/
├── __init__.py           # 顶层导出
├── errors.py             # 统一错误类型 (~50行)
├── comm.py               # 通信原语 (~970行)
├── padding.py            # Padding 管理 (~100行)
├── mesh.py               # ProcessGroupMesh + ParallelManager (~350行)
├── backend.py            # 通信后端抽象 (~300行)
└── pipeline.py           # enable_sp (~100行)
```

## API 使用

### Padding 管理
```python
from x_base.sequence_parallelism import PadManager, get_pad_manager

manager = get_pad_manager()
pad = manager.set("attention", seq_len, sp_size)
```

### 并行管理
```python
from x_base.sequence_parallelism import ParallelManager, ParallelConfig

config = ParallelConfig(dp_size=2, sp_size=4)
manager = ParallelManager(config)
```

### 通信原语
```python
from x_base.sequence_parallelism import (
    all_gather,
    reduce_scatter,
    all_to_all,
    gather_sequence,
    split_sequence,
)
```

### 启用序列并行
```python
from x_base.sequence_parallelism import enable_sp

enable_sp(parallel_manager, yunchang_backend)
```
"""

# =============================================================================
# 错误类型
# =============================================================================
# =============================================================================
# 通信后端
# =============================================================================
from .backend import (
    BaseCommBackend,
    CollectiveBackend,
    TorchDistBackend,
)

# =============================================================================
# 核心通信原语
# =============================================================================
from .comm import (
    # Primitives
    AllGather,
    AllGatherOverlapped,
    AllToAllAutograd,
    ReduceScatter,
    all_gather,
    all_to_all,
    all_to_all_4d,
    all_to_all_after_attn,
    all_to_all_before_attn,
    batch_func,
    # Collective
    gather_sequence,
    pad_tensor,
    reduce_scatter,
    split_sequence,
)
from .errors import (
    AllToAllDimensionError,
    BackendNotAvailableError,
    IncompatibleDimensionError,
    PadNotSetError,
    ParallelConfigError,
    ProcessGroupError,
    SequenceParallelError,
)

# =============================================================================
# ProcessGroup Mesh
# =============================================================================
from .mesh import (
    ParallelConfig,
    ParallelManager,
    ProcessGroupMesh,
    initialize,
)

# =============================================================================
# Padding 管理
# =============================================================================
from .padding import (
    PadManager,
    get_pad,  # 向后兼容 (deprecated)
    get_pad_manager,
    set_pad,  # 向后兼容 (deprecated)
)

# =============================================================================
# Pipeline 集成
# =============================================================================
from .pipeline import enable_sp

# =============================================================================
# __all__
# =============================================================================

__all__ = [
    # Errors
    "SequenceParallelError",
    "PadNotSetError",
    "ProcessGroupError",
    "ParallelConfigError",
    "IncompatibleDimensionError",
    "BackendNotAvailableError",
    "AllToAllDimensionError",
    # Padding
    "PadManager",
    "get_pad_manager",
    "set_pad",  # deprecated
    "get_pad",  # deprecated
    # Mesh
    "ProcessGroupMesh",
    "ParallelManager",
    "ParallelConfig",
    "initialize",
    # Backends
    "CollectiveBackend",
    "BaseCommBackend",
    "TorchDistBackend",
    # Primitives
    "AllGather",
    "ReduceScatter",
    "AllToAllAutograd",
    "AllGatherOverlapped",
    "all_gather",
    "reduce_scatter",
    "all_to_all",
    # Collective
    "gather_sequence",
    "split_sequence",
    "all_to_all_4d",
    "all_to_all_before_attn",
    "all_to_all_after_attn",
    "pad_tensor",
    "batch_func",
    # Pipeline
    "enable_sp",
]
