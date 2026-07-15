"""
Sequence Parallelism - 统一错误类型

定义序列并行模块所有异常类型，提供清晰的错误信息。
"""


class SequenceParallelError(Exception):
    """序列并行基础错误

    所有序列并行相关异常的基类。
    """

    pass


class PadNotSetError(SequenceParallelError):
    """Padding 值未设置错误

    当尝试获取未设置的 padding 值时抛出。
    """

    def __init__(self, name: str):
        super().__init__(f"Padding '{name}' has not been set. Call PadManager.set() or set_pad() first.")
        self.name = name


class ProcessGroupError(SequenceParallelError):
    """ProcessGroup 相关错误

    当 ProcessGroup 初始化或使用出现问题时抛出。
    """

    def __init__(self, message: str, group: str | None = None):
        if group:
            message = f"[Group: {group}] {message}"
        super().__init__(message)


class ParallelConfigError(SequenceParallelError):
    """并行配置错误

    当并行配置参数不合法或不兼容时抛出。
    """

    def __init__(self, message: str):
        super().__init__(message)


class IncompatibleDimensionError(SequenceParallelError):
    """维度不兼容错误

    当张量维度不能被并行度整除时抛出。
    """

    def __init__(self, dim: int, size: int, divisor: int, operation: str = "split"):
        suggested_size = ((size // divisor) + 1) * divisor if size % divisor != 0 else size
        super().__init__(
            f"Dimension {dim} has size {size}, which is not divisible by {divisor} "
            f"for {operation} operation. "
            f"Consider padding to {suggested_size} or using PadManager."
        )
        self.dim = dim
        self.size = size
        self.divisor = divisor


class BackendNotAvailableError(SequenceParallelError):
    """通信后端不可用错误

    当请求的通信后端未安装或不可用时抛出。
    """

    def __init__(self, backend_name: str, install_hint: str | None = None):
        message = f"Communication backend '{backend_name}' is not available."
        if install_hint:
            message += f" Install with: {install_hint}"
        super().__init__(message)
        self.backend_name = backend_name


class AllToAllDimensionError(SequenceParallelError):
    """AllToAll 维度参数错误

    当 scatter_idx 或 gather_idx 参数不合法时抛出。
    """

    def __init__(self, scatter_idx: int, gather_idx: int):
        super().__init__(
            f"Invalid AllToAll dimensions: scatter_idx={scatter_idx}, gather_idx={gather_idx}. "
            f"Both must be 1 or 2, and scatter_idx != gather_idx."
        )
        self.scatter_idx = scatter_idx
        self.gather_idx = gather_idx
