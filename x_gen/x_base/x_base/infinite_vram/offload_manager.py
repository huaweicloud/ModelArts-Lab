import weakref
from collections.abc import Sequence

import torch
import torch.distributed as dist

from .utils import foreach_copy_


class OffloadManager:
    def __init__(
        self,
        root: torch.nn.Module,
        module_groups: dict[str, Sequence[torch.nn.Module]],
        keep_n: dict[str, int] | int,
        device: torch.device | None = None,
        *,
        dist_group: dist.ProcessGroup | None = None,
        sync_at_layer: bool = False,
    ):
        self.root = weakref.proxy(root)
        self.device = device or next(root.parameters()).device
        self.groups = module_groups
        self.keep_n = {k: keep_n for k in module_groups} if isinstance(keep_n, int) else keep_n

        self.layer_params: dict[str, list[list[torch.Tensor]]] = {
            tag: [list(self._iter_tensors(m)) for m in grp] for tag, grp in self.groups.items()
        }

        self.h2d_stream = torch.cuda.Stream()
        self.d2h_stream = torch.cuda.Stream()

        self.dist_group = dist_group or (dist.group.WORLD if sync_at_layer else None)
        self.sync_at_layer = sync_at_layer
        if self.sync_at_layer and not dist.is_initialized():
            raise RuntimeError("sync_at_layer=True 但未初始化 torch.distributed")

        self.enabled = False
        self.handles: list[torch.utils.hooks.RemovableHandle] = []

        # 填充 index / depth
        for tag, grp in self.groups.items():
            for i, m in enumerate(grp):
                m.index = getattr(m, "index", i)
                m.depth = getattr(m, "depth", len(grp))

    def enable(self):
        if self.enabled:
            return
        # 只加载前keep_n层到显存
        for tag, grp in self.groups.items():
            for i in range(min(self.keep_n[tag], len(grp))):
                grp[i].to(self.device, non_blocking=False)

        # offload 后部分层到CPU
        for tag, grp in self.groups.items():
            if len(grp) > self.keep_n[tag]:
                self._setup_parameter_offload(tag, grp, self.keep_n[tag])

        torch.cuda.empty_cache()  # 立即回收

        # 注册 hooks
        for tag, grp in self.groups.items():
            for m in grp:
                self.handles.append(m.register_forward_pre_hook(self._prefetch_factory(tag), with_kwargs=False))
                # 为所有非常驻层注册释放hook
                if m.index >= self.keep_n[tag]:
                    self.handles.append(m.register_forward_hook(self._release_factory(tag), with_kwargs=False))
        self.enabled = True

    def disable(self):
        if not self.enabled:
            return
        self._restore_all()
        self._remove_hooks()
        self.enabled = False

    # 新增四个内部方法，对之前的hook()进行拆分，降低圈复杂度
    def _maybe_free_prev_layer(
        self, prev_idx: int, keep_n: int, grp: Sequence[torch.nn.Module], params: list[list[torch.Tensor]]
    ) -> None:
        if prev_idx < keep_n:
            return
        prev_mod = grp[prev_idx]
        needs_free = getattr(prev_mod, "_so_needs_free", False)
        already_free = getattr(prev_mod, "_so_freed", False)
        compute_evt = getattr(prev_mod, "_so_compute_evt", None)

        if not (needs_free and not already_free and compute_evt is not None):
            return

        with torch.cuda.stream(self.h2d_stream):
            self.h2d_stream.wait_event(compute_evt)
            for p in params[prev_idx]:
                self._release_tensor(p)
        prev_mod._so_freed = True

    # === 新增：把“等待本层预取完成”的逻辑封装 ===
    def _wait_prefetch_if_needed(self, module, keep_n: int) -> None:
        if module.index < keep_n:
            return
        evt = getattr(module, "_so_prefetch_evt", None)
        if evt is None:
            return
        torch.cuda.current_stream().wait_event(evt)
        module._so_prefetch_evt = None

    # === 新增：是否应该预取下一层（把边界与状态判断打包） ===
    def _should_prefetch_next(self, next_idx: int, keep_n: int, grp: Sequence[torch.nn.Module]) -> bool:
        if not (keep_n <= next_idx < len(grp)):
            return False
        # getattr 一步到位，少一次 and 判断
        return getattr(grp[next_idx], "_so_prefetch_evt", None) is None

    # === 新增：执行真正的预取（包含分配、foreach_copy_、记录事件）===
    def _do_prefetch(self, next_idx: int, params: list[list[torch.Tensor]], next_mod: torch.nn.Module) -> None:
        with torch.cuda.stream(self.h2d_stream):
            src_list, dst_list = [], []
            for p in params[next_idx]:
                need_alloc = (p.data.device != self.device) or (p.data.untyped_storage().size() == 0)
                if need_alloc:
                    p.data = torch.empty_like(getattr(p, "p_cpu", p.data), device=self.device)
                dst_list.append(p.data)
                src_list.append(getattr(p, "p_cpu", p.data))

            foreach_copy_(dst_list, src_list, non_blocking=True)
            evt = torch.cuda.Event()
            self.h2d_stream.record_event(evt)
            next_mod._so_prefetch_evt = evt

    # === 重写：把复杂度从 hook 中“搬走”，让它只做流程编排 ===
    def _prefetch_factory(self, tag: str):
        keep_n = self.keep_n[tag]
        grp = self.groups[tag]
        params = self.layer_params[tag]

        def hook(module, _inputs):
            # A. 释放上一层（若需要）
            self._maybe_free_prev_layer(module.index - 1, keep_n, grp, params)

            # B. 等待本层预取完成（若需要）
            self._wait_prefetch_if_needed(module, keep_n)

            # C. 可选的分布式同步
            if self.sync_at_layer and self.dist_group is not None:
                dist.barrier(group=self.dist_group)

            # D. 仅预取“紧邻”的下一层（若需要）
            next_idx = module.index + 1
            if self._should_prefetch_next(next_idx, keep_n, grp):
                self._do_prefetch(next_idx, params, grp[next_idx])

        return hook

    def _release_factory(self, tag: str):
        keep_n = self.keep_n[tag]

        def hook(module, _inp, _out):
            if module.index < keep_n:
                return

            # 记录本层计算完成事件，但不立即释放显存
            # 显存释放将在下一层的prefetch_hook中进行
            evt = torch.cuda.Event()
            torch.cuda.current_stream().record_event(evt)
            module._so_compute_evt = evt
            module._so_needs_free = True
            module._so_freed = False

        return hook

    @staticmethod
    def _iter_tensors(m):
        yield from m.parameters(recurse=True)
        yield from m.buffers(recurse=True)

    def _setup_parameter_offload(self, tag, grp, keep_n):
        """将非常驻层的参数offload到CPU"""
        for idx in range(keep_n, len(grp)):
            with torch.cuda.stream(self.d2h_stream):
                all_size = 0
                for p in self.layer_params[tag][idx]:
                    tmp_size = 1
                    for s in p.shape:
                        tmp_size *= s
                    all_size += tmp_size

                all_tmp_cpu = torch.empty([all_size], dtype=torch.bfloat16, pin_memory=True, device="cpu")
                offset = 0
                for p in self.layer_params[tag][idx]:
                    if not hasattr(p, "p_cpu"):
                        p.p_cpu = all_tmp_cpu[offset : offset + p.numel()].view(p.shape)
                        offset += p.numel()

                        # p.p_cpu = torch.empty_like(
                        #     p.data, pin_memory=True, device="cpu")
                        p.orig_shape = p.data.shape
                    p.p_cpu.copy_(p.data, non_blocking=True)

        # 等待拷贝完成
        torch.cuda.current_stream().wait_stream(self.d2h_stream)

        # 释放非常驻层的显存
        for idx in range(keep_n, len(grp)):
            for p in self.layer_params[tag][idx]:
                self._release_tensor(p)

    @staticmethod
    def _release_tensor(p: torch.Tensor):
        """释放张量的显存"""
        try:
            p.data.untyped_storage().resize_(0)
            p.data.resize_(0)
        except RuntimeError:
            p.data = torch.empty(0, dtype=p.dtype, device=p.device)
        p._released = True

    def _restore_all(self):
        """恢复所有参数到GPU"""
        for tag, grp in self.groups.items():
            for idx, m in enumerate(grp):
                for p in self.layer_params[tag][idx]:
                    if p.data.untyped_storage().size() == 0:
                        restored = torch.empty(p.orig_shape, dtype=p.dtype, device=self.device)
                        restored.copy_(p.p_cpu, non_blocking=False)
                        p.data = restored
                    elif p.data.device != self.device:
                        p.data = p.data.to(self.device, non_blocking=False)
                    p._released = False

    def _remove_hooks(self):
        """移除所有hooks"""
        for h in self.handles:
            h.remove()
        self.handles.clear()
