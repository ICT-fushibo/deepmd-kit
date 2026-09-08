"""Opt4 explicit, validated candidate dispatch; Opt3 remains the off path."""

from __future__ import annotations

from md_benchmark.md_route import MDRunRequest, MDRunResult, validate_result
from md_benchmark.opt4_policy import run_opt4_with_opt3

from . import opt3


def run_md(request: MDRunRequest) -> MDRunResult:
    if request.model != "dpa4" or request.stage != "opt4":
        raise ValueError(f"DPA4 Opt4 route received {request.model}/{request.stage}")
    result, _policy = run_opt4_with_opt3(request, opt3.run_md, model="dpa4")
    result.stage = "opt4"
    validate_result(request, result)
    return result


__all__ = ["run_md"]
