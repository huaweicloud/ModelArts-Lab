from collections.abc import Sequence

import torch


class OffloadCommonMixin:
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

    def _release_factory(self, tag: str):
        keep_n = self.keep_n[tag]

        def hook(module, _inp, _out):
            if module.index < keep_n:
                return

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
