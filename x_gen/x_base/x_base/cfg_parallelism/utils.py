import torch
import torch.distributed as dist
import torch_npu
from torch.distributed import ProcessGroup


class GroupCoordinator:
    # difference between `local_rank` and `rank_in_group`:
    # if we have a group of size 4 across two nodes:
    # Process | Node | Rank | Local Rank | Rank in Group
    #   0     |   0  |  0   |     0      |       0
    #   1     |   0  |  1   |     1      |       1
    #   2     |   1  |  2   |     0      |       2
    #   3     |   1  |  3   |     1      |       3
    cpu_group: ProcessGroup  # group for CPU communication
    device_group: ProcessGroup  # group for device communication

    def __init__(
        self,
        group_ranks: list[list[int]],
        local_rank: int,
    ):
        self.rank = torch.distributed.get_rank()
        self.local_rank = local_rank
        self.device_group = None
        self.cpu_group = None

        for ranks in group_ranks:
            device_group = torch.distributed.new_group(ranks, backend="hccl")
            # a group with `gloo` backend, to allow direct coordination between
            # processes through the CPU.
            cpu_group = torch.distributed.new_group(ranks, backend="gloo")
            if self.rank in ranks:
                self.ranks = ranks
                self.world_size = len(ranks)
                self.rank_in_group = ranks.index(self.rank)
                self.device_group = device_group
                self.cpu_group = cpu_group

        if self.cpu_group is None:
            raise ValueError("cpu通信组初始化失败")
        if self.device_group is None:
            raise ValueError("通信组初始化失败")

        self.device = torch.device("npu", torch_npu.npu.current_device())

    def all_reduce(self, input_: torch.Tensor, op=torch.distributed.ReduceOp.SUM) -> torch.Tensor:
        """
        NOTE: This operation will be applied in-place or out-of-place.
        Always assume this function modifies its input, but use the return
        value as the output.
        """
        # Bypass the function if we are using only 1 GPU.
        if self.world_size == 1:
            return input_
        else:
            torch.distributed.all_reduce(input_, op=op, group=self.device_group)
        return input_

    def all_to_all_single(self, output_, input_, send_sizes, recv_sizes):
        if self.world_size == 1:
            return input_
        else:
            torch.distributed.all_to_all_single(
                output_, input_, output_split_sizes=recv_sizes, input_split_sizes=send_sizes, group=self.device_group
            )
            return output_

    def all_gather(
        self, input_: torch.Tensor, dim: int = 0, separate_tensors: bool = False
    ) -> torch.Tensor | list[torch.Tensor]:
        world_size = self.world_size
        # Bypass the function if we are using only 1 GPU.
        if world_size == 1:
            return input_
        if not (-input_.dim() <= dim < input_.dim()):
            raise ValueError(f"Invalid dim ({dim}) for input tensor with shape {input_.size()}")
        if dim < 0:
            # Convert negative dim to positive.
            dim += input_.dim()

        # Allocate output tensor.
        input_size = list(input_.size())
        input_size[0] *= world_size
        output_tensor = torch.empty(input_size, dtype=input_.dtype, device=input_.device)

        # All-gather.
        torch.distributed.all_gather_into_tensor(output_tensor, input_, group=self.device_group)
        if dim != 0:
            input_size[0] //= world_size
            output_tensor = output_tensor.reshape(
                [
                    world_size,
                ]
                + input_size
            )
            output_tensor = output_tensor.movedim(0, dim)

        if separate_tensors:
            tensor_list = [
                output_tensor.view(-1).narrow(0, input_.numel() * i, input_.numel()).view_as(input_)
                for i in range(world_size)
            ]
            return tensor_list
        else:
            input_size = list(input_.size())
            input_size[dim] = input_size[dim] * world_size
            # Reshape
            output_tensor = output_tensor.reshape(input_size)
            return output_tensor


_TP: GroupCoordinator | None = None
_EP: GroupCoordinator | None = None
_CFG: GroupCoordinator | None = None


def init_TP(tp_size):
    global _TP
    if _TP is not None:
        raise ValueError("Tensor parallel group is already initialized")

    rank = dist.get_rank()

    tp_group_size = tp_size
    tp_group_id = rank // tp_group_size
    tp_ranks = list(range(tp_group_id * tp_group_size, (tp_group_id + 1) * tp_group_size))
    # TP 组内的 local rank
    tp_rank = tp_ranks.index(rank)

    _TP = GroupCoordinator(
        group_ranks=[tp_ranks],
        local_rank=tp_rank,
    )


def get_tp_group() -> GroupCoordinator:
    if _TP is None:
        raise ValueError("tensor model parallel group is not initialized")
    return _TP


def get_tensor_model_parallel_world_size():
    """Return world size for the tensor model parallel group."""
    return get_tp_group().world_size


def get_tensor_model_parallel_rank():
    """Return my rank for the tensor model parallel group."""
    return get_tp_group().rank_in_group


def init_EP(ep_size):
    global _EP
    if _EP is not None:
        raise ValueError("Expert parallel group is already initialized")

    rank = dist.get_rank()

    ep_group_size = ep_size
    ep_group_id = rank // ep_group_size
    ep_ranks = list(range(ep_group_id * ep_group_size, (ep_group_id + 1) * ep_group_size))
    # EP 组内的 local rank
    ep_rank = ep_ranks.index(rank)

    _EP = GroupCoordinator(
        group_ranks=[ep_ranks],
        local_rank=ep_rank,
    )


def get_ep_group() -> GroupCoordinator:
    if _EP is None:
        raise ValueError("Expert model parallel group is not initialized")
    return _EP


def get_expert_parallel_world_size():
    """Return world size for the expert model parallel group."""
    return get_ep_group().world_size


def get_expert_parallel_rank():
    """Return my rank for the export model parallel group."""
    return get_ep_group().rank_in_group


def init_CFG_Parallel(cfg_size):
    global _CFG
    if _CFG is not None:
        raise ValueError("CFG parallel group is already initialized")

    rank = dist.get_rank()

    cfg_group_size = cfg_size
    cfg_group_id = rank // cfg_group_size
    cfg_ranks = list(range(cfg_group_id * cfg_group_size, (cfg_group_id + 1) * cfg_group_size))
    # CFG 组内的 local rank
    cfg_rank = cfg_ranks.index(rank)

    _CFG = GroupCoordinator(
        group_ranks=[cfg_ranks],
        local_rank=cfg_rank,
    )


def get_cfg_group() -> GroupCoordinator:
    if _CFG is None:
        raise ValueError("cfg model parallel group is not initialized")
    return _CFG


def get_cfg_parallel_world_size():
    """Return world size for the cfg model parallel group."""
    return get_cfg_group().world_size


def get_cfg_parallel_rank():
    """Return my rank for the cfg model parallel group."""
    return get_cfg_group().rank_in_group
