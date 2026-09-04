"""DPA4 Opt4 route.

Opt4 currently reuses DPA4's fixed-slot Opt3 graph.  The model's edge and
angle representations are direction dependent, so no reverse-edge shortcut or
generic CSR kernel is enabled in this first implementation.  Keeping the
single Opt3 runner here also preserves its hard-overflow and sink-padding
semantics while the fusion work is evaluated independently.
"""

from __future__ import annotations

from md_benchmark.md_route import MDRunRequest, MDRunResult, validate_result
from md_benchmark.opt4_policy import run_opt4_with_opt3

from . import opt3


def run_md(request: MDRunRequest) -> MDRunResult:
    if request.model != "dpa4" or request.stage != "opt4":
        raise ValueError(f"DPA4 Opt4 route received {request.model}/{request.stage}")
    result, policy = run_opt4_with_opt3(request, opt3.run_md, model="dpa4")
    result.stage = "opt4"
    result.metadata.update(policy.metadata)
    result.metadata.update(
        {
            "opt4_model_strategy": "fixed-slot-edge-angle-layout",
            "opt4_optimization_targets": [
                "edge-geometry-radial-switch-layout",
                "angle-mask-weight-layout",
                "fixed-slot-reduction-workspace",
            ],
            "opt4_fixed_address_buffers": True,
            "opt4_custom_kernel": False,
            "opt4_reverse_edge_verified": False,
        }
    )
    validate_result(request, result)
    return result


__all__ = ["run_md"]
