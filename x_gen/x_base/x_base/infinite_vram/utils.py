import torch

from x_base.utils.infer_info import infer_info

"""
用途：减少h2d拷贝算子的下发时间
原理：10个算子下发10次，我这里只需要下发一次即可
foreach copy 兼容包装（PyTorch < 2.1 fallback 到逐 tensor）
"""


def foreach_copy_(dests: list[torch.Tensor], srcs: list[torch.Tensor], non_blocking: bool = True):
    if infer_info.use_matmul_a4w4 or infer_info.use_matmul_a8w8:
        for dest, src in zip(dests, srcs):
            dest.copy_(src, non_blocking=False)
    else:
        try:
            torch._foreach_copy_(dests, srcs, non_blocking=non_blocking)
        except AttributeError:
            for d, s in zip(dests, srcs):
                d.copy_(s, non_blocking=non_blocking)


"""
打印gpu memory信息 定位显存使用
"""


def print_gpu_memory(message=""):
    allocated = torch.cuda.memory_allocated() / 1024**2
    reserved = torch.cuda.memory_reserved() / 1024**2
    print(f"[显存监控] {message}")
    print(f"  已分配显存: {allocated:.2f} MB")
    print(f"  已保留显存: {reserved:.2f} MB")
