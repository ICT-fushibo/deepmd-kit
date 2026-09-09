"""Opt4-only fused rotation forward with the native reference VJP sequence.

The general SeZM Triton switches and their backward kernels are not changed.
Float32 edge-atomic reduction had failed the frozen full-checkpoint VJP gate;
retain bmm + index_select_backward for that reduction, without widening dtype.
"""
from typing import Tuple

import torch
from torch import Tensor


def rotation_reference(x, index, wigner, indices, mask, back=False):
    dim = mask.shape[0]
    matrix = wigner[:, :dim, :dim] * mask
    if back:
        flat = x.transpose(1, 2).reshape(x.shape[0], x.shape[2], -1)
        return torch.bmm(matrix.index_select(2, indices), flat)
    return torch.bmm(matrix.index_select(1, indices), x.index_select(0, index))


@torch.library.custom_op("dpa4_opt4::rotation", mutates_args=())
def rotation(x: Tensor, index: Tensor, wigner: Tensor, indices: Tensor,
             mask: Tensor, lmax: int, back: bool) -> Tensor:
    if not x.is_cuda:
        return rotation_reference(x, index, wigner, indices, mask, back)
    from deepmd.kernels.triton.sezm.so2_rotation import rotate_back_block_so2, rotate_to_local_block
    if back:
        return rotate_back_block_so2(x, wigner, lmax)
    return rotate_to_local_block(x, index, wigner, lmax)


@rotation.register_fake
def _rotation_fake(x, index, wigner, indices, mask, lmax, back):
    if back:
        return x.new_empty((x.shape[0], mask.shape[0], x.shape[1] * x.shape[3]))
    return x.new_empty((index.shape[0], indices.shape[0], x.shape[2]))


@torch.library.custom_op("dpa4_opt4::rotation_vjp", mutates_args=())
def rotation_vjp(g: Tensor, x: Tensor, index: Tensor, wigner: Tensor, indices: Tensor,
                 mask: Tensor, back: bool) -> Tuple[Tensor, Tensor]:
    dim = mask.shape[0]
    matrix = wigner[:, :dim, :dim] * mask
    if back:
        flat = x.transpose(1, 2).reshape(x.shape[0], x.shape[2], -1)
        gx_flat = torch.bmm(matrix.index_select(2, indices).transpose(1, 2), g)
        gx = gx_flat.reshape(x.transpose(1, 2).shape).transpose(1, 2).contiguous()
        gw_reduced = torch.bmm(g, flat.transpose(1, 2))
        gw_full = torch.index_add(torch.zeros_like(matrix), 2, indices, gw_reduced)
    else:
        rows = matrix.index_select(1, indices)
        gx_edge = torch.bmm(rows.transpose(1, 2), g)
        # Same ATen primitive as the original index_select autograd, reading
        # CURRENT GPU endpoints. Never cache a setup-time edge permutation.
        gx = torch.ops.aten.index_select_backward.default(gx_edge, list(x.shape), 0, index)
        gw_reduced = torch.bmm(g, x.index_select(0, index).transpose(1, 2))
        gw_full = torch.index_add(torch.zeros_like(matrix), 1, indices, gw_reduced)
    gw = torch.zeros_like(wigner)
    gw[:, :dim, :dim].copy_(gw_full * mask)
    return gx, gw


@rotation_vjp.register_fake
def _rotation_vjp_fake(g, x, index, wigner, indices, mask, back):
    return x.new_empty(x.shape), wigner.new_empty(wigner.shape)


def _setup(ctx, inputs, output):
    x, index, wigner, indices, mask, _, back = inputs
    ctx.save_for_backward(x, index, wigner, indices, mask)
    ctx.back = back


def _backward(ctx, g):
    x, index, wigner, indices, mask = ctx.saved_tensors
    gx, gw = rotation_vjp(g, x, index, wigner, indices, mask, ctx.back)
    return gx, None, gw, None, None, None, None


rotation.register_autograd(_backward, setup_context=_setup)
