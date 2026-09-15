"""DPA4 Opt4: fixed-slot weighted SO2 destination reduction."""
from __future__ import annotations

import torch
from torch import nn

from md_benchmark.opt4_fx import CheckedRegion
from md_benchmark.opt4_ops import csr_weighted_segment_sum
from md_benchmark.opt4_registry import fixed_csr_layout, record


class _NativeWeightedReduce(nn.Module):
    def __init__(self, edge_rows: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("edge_rows", edge_rows, persistent=False)

    def forward(self, values, weights, row_scale):
        out = values.new_zeros((row_scale.shape[0], *values.shape[1:]))
        out.index_add_(
            0,
            self.edge_rows,
            values * weights.reshape(-1, *((1,) * (values.ndim - 1))),
        )
        return out * row_scale.reshape(-1, *((1,) * (values.ndim - 1)))


class _FixedWeightedCSR(nn.Module):
    def __init__(self, row_ptr, edge_rows, max_row) -> None:
        super().__init__()
        self.register_buffer("row_ptr", row_ptr, persistent=False)
        self.register_buffer("edge_rows", edge_rows, persistent=False)
        self.max_row = int(max_row)

    def set_layout(self, row_ptr, edge_rows, max_row) -> None:
        self.row_ptr = row_ptr
        self.edge_rows = edge_rows
        self.max_row = int(max_row)

    def forward(self, values, weights, row_scale):
        return csr_weighted_segment_sum(
            values.contiguous(),
            weights.reshape(-1).contiguous(),
            row_scale.reshape(-1).contiguous(),
            self.row_ptr,
            self.edge_rows,
            self.max_row,
        )


def refresh(model, options) -> None:
    parameter = next(model.parameters())
    row_ptr, edge_rows, max_row = fixed_csr_layout(options, parameter)
    for module in model.modules():
        region = getattr(module, "_opt4_weighted_csr", None)
        if isinstance(region, CheckedRegion):
            region.reference.edge_rows = edge_rows
            region.compiled.set_layout(row_ptr, edge_rows, max_row)
            region.signatures.clear()


def install(model, passes, report, options):
    modules = []
    if "so2_weighted_csr_reduce" not in passes:
        return
    parameter = next(model.parameters())
    row_ptr, edge_rows, max_row = fixed_csr_layout(options, parameter)
    for path, module in list(model.named_modules()):
        if type(module).__name__ != "SO2Convolution":
            continue
        detail = {
            "module": path,
            "validated_shapes": 0,
            "benchmark_requested": report.get("benchmark_boundaries", False),
        }
        module._opt4_weighted_csr = CheckedRegion(
            _NativeWeightedReduce(edge_rows),
            detail,
            _FixedWeightedCSR(row_ptr, edge_rows, max_row),
        )
        modules.append(detail)
    record(
        report,
        "so2_weighted_csr_reduce",
        len(modules),
        "triton-fixed-csr-explicit-vjp",
        modules=modules,
        fused_boundaries=["edge-weight", "destination-reduce", "degree-normalize"],
        gemm="unchanged",
        rotation="unchanged",
        radial_mixer="unchanged",
        reverse_edge=False,
        fusion_scope="forward-and-backward",
    )
