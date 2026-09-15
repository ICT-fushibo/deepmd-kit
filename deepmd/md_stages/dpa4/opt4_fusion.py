"""DPA4 Opt4: fixed-slot weighted SO2 destination reduction."""
from __future__ import annotations

import torch
from torch import nn

from md_benchmark.opt4_fx import CheckedRegion
from md_benchmark.opt4_ops import (
    csr_group_weighted_segment_sum,
    csr_weighted_segment_sum,
)
from md_benchmark.opt4_registry import FusionSetupError, fixed_csr_layout, record


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


class _NativeAttentionReduce(nn.Module):
    def __init__(self, edge_rows: torch.Tensor, rows: int) -> None:
        super().__init__()
        self.register_buffer("edge_rows", edge_rows, persistent=False)
        self.rows = int(rows)

    def forward(self, values, weights):
        out = values.new_zeros((self.rows, *values.shape[1:]))
        out.index_add_(0, self.edge_rows, values * weights[:, None, :, None])
        return out


class _FixedAttentionCSR(nn.Module):
    def __init__(self, row_ptr, edge_rows, max_row) -> None:
        super().__init__()
        self.register_buffer("row_ptr", row_ptr, persistent=False)
        self.register_buffer("edge_rows", edge_rows, persistent=False)
        self.max_row = int(max_row)

    def set_layout(self, row_ptr, edge_rows, max_row) -> None:
        self.row_ptr = row_ptr
        self.edge_rows = edge_rows
        self.max_row = int(max_row)

    def forward(self, values, weights):
        return csr_group_weighted_segment_sum(
            values.contiguous(),
            weights.contiguous(),
            self.row_ptr,
            self.edge_rows,
            self.max_row,
        )


def refresh(model, options) -> None:
    parameter = next(model.parameters())
    row_ptr, edge_rows, max_row = fixed_csr_layout(options, parameter)
    for module in model.modules():
        if hasattr(module, "_opt4_weighted_csr") or hasattr(
            module, "_opt4_attention_csr"
        ):
            module._opt4_edge_capacity = int(edge_rows.numel())
        region = getattr(module, "_opt4_weighted_csr", None)
        if isinstance(region, CheckedRegion):
            region.reference.edge_rows = edge_rows
            region.compiled.set_layout(row_ptr, edge_rows, max_row)
            region.signatures.clear()
        attention = getattr(module, "_opt4_attention_csr", None)
        if isinstance(attention, CheckedRegion):
            attention.reference.edge_rows = edge_rows
            attention.reference.rows = row_ptr.shape[0] - 1
            attention.compiled.set_layout(row_ptr, edge_rows, max_row)
            attention.signatures.clear()


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
        module._opt4_edge_capacity = int(edge_rows.numel())
        if int(getattr(module, "n_atten_head", 0)) > 0:
            if bool(getattr(module, "use_flash_atten", False)):
                raise FusionSetupError(
                    "so2_weighted_csr_reduce cannot replace an active DPA4 flash-attention path"
                )
            module._opt4_attention_csr = CheckedRegion(
                _NativeAttentionReduce(edge_rows, row_ptr.shape[0] - 1),
                detail,
                _FixedAttentionCSR(row_ptr, edge_rows, max_row),
            )
            detail["aggregation_mode"] = "attention-focus-head"
        else:
            module._opt4_weighted_csr = CheckedRegion(
                _NativeWeightedReduce(edge_rows),
                detail,
                _FixedWeightedCSR(row_ptr, edge_rows, max_row),
            )
            detail["aggregation_mode"] = "envelope-degree"
        modules.append(detail)
    record(
        report,
        "so2_weighted_csr_reduce",
        len(modules),
        "triton-fixed-csr-forward-value-vjp-aten-weight-vjp",
        modules=modules,
        fused_boundaries=["attention-or-edge-weight", "destination-reduce"],
        gemm="unchanged",
        rotation="unchanged",
        radial_mixer="unchanged",
        reverse_edge=False,
        fusion_scope="forward-and-value-vjp; native-order explicit weight-vjp",
    )
