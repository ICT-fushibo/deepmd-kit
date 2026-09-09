"""Compare rotation VJPs using the same native indexed accumulation on both sides.

Only the comparison intercepts index_add. Production uses the explicit sorted
index_put path in rotation_vjp; baseline/Opt3 and global flags are untouched.
"""
import torch
from torch.utils._python_dispatch import TorchDispatchMode


class RotationVJPComparison(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func in (torch.ops.aten.index_add_.default, torch.ops.aten.index_add.default):
            target, dim, index, source = args[:4]
            if dim == 0 and target.is_cuda:
                alpha = kwargs.get("alpha", args[4] if len(args) > 4 else 1)
                out = target if func == torch.ops.aten.index_add_.default else target.clone()
                value = source if alpha == 1 else source * alpha
                return torch.ops.aten.index_put_.default(out, [index], value, True)
        return func(*args, **kwargs)
