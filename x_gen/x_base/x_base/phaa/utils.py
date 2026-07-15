from diffusers import DiffusionPipeline
from torch.npu import Stream

from .globals import enable_phaa, set_phaa_split_num


def phaa_on_pipe(pipe: DiffusionPipeline, transformer_2=False, transformer_3=False, phaa_split_num=1, *args, **kwargs):
    pipe_cls_name = pipe.__class__.__name__
    if pipe_cls_name.startswith("Wan"):
        inject_phaa_for_wan(pipe.transformer, phaa_split_num)
        if transformer_2:
            inject_phaa_for_wan(pipe.transformer_2, phaa_split_num)
        if transformer_3:
            inject_phaa_for_wan(pipe.transformer_3, phaa_split_num)
    set_phaa_split_num(phaa_split_num)
    enable_phaa()


def check_split_num_correctness(attn, split_num):
    world_size = attn.parallel_manager.sp_size
    if split_num is not None and (attn.heads / world_size) % split_num != 0 and not attn.parallel_manager.enable_usp:
        raise ValueError(
            f"Number of per-device attn heads {attn.heads / world_size} must be divisible by PHAA split num {split_num}"
        )


def inject_phaa_for_wan(transformer, split_num):
    comm_stream = Stream()
    for block in transformer.blocks:
        check_split_num_correctness(block.attn1, split_num)
        check_split_num_correctness(block.attn2, split_num)
        block.attn1.processor.comm_stream = comm_stream
        block.attn2.processor.comm_stream = comm_stream
