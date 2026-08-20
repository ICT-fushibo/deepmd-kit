"""Stable DPA3/DPA4 MD route for the shared acceleration benchmark.

``baseline/eager`` is intentionally a scientific-correctness reference: it
disables all opt-in reduced-precision, compiled, and fused inference switches.
DPA3 uses DeepMD's native dense neighbor-list builder.  DPA4/SeZM necessarily
keeps its model-owned Toolkit-Ops edge-list builder, which is the only supported
CUDA eager path and is reported explicitly rather than being mistaken for an
Opt1 implementation.  ``baseline`` with ``auto``, ``vesin``, or ``nv`` remains
an explicit existing-optimization control for models that honor the external
neighbor-list selector.
"""

from __future__ import annotations

import os
from pathlib import Path

from md_benchmark.md_route import (
    MDRunRequest,
    MDRunResult,
    configure_torch_baseline,
    run_ase_baseline,
    run_optimized_stage,
)


_DEEPMD_BASELINE_ENV = {
    # DeepMD reads this once when its backend is first imported.  Pin the
    # scientific baseline to the documented high-precision interface so a
    # caller's shell cannot change baseline/Opt1 parity.  Checkpoint component
    # precision (for example FP32 descriptor weights) remains unchanged.
    "DP_INTERFACE_PREC": "high",
    "DP_ACT_INFER": "0",
    "DP_COMPILE_INFER": "0",
    "DP_CUDA_INFER": "0",
    "DP_CUTE_INFER": "0",
    "DP_TRITON_INFER": "0",
    "DP_TF32_INFER": "0",
    "DP_AMP_INFER": "0",
}


def _configure_deepmd_baseline() -> None:
    """Make the eager/native reference independent of the caller's shell."""
    configure_torch_baseline()
    # DeepMD reads these variables while constructing/loading the model, so
    # set them before importing and instantiating the ASE calculator.
    os.environ.update(_DEEPMD_BASELINE_ENV)


def run_md(request: MDRunRequest) -> MDRunResult:
    if request.model not in {"dpa3", "dpa4"}:
        raise ValueError(f"deepmd.md_route does not own model {request.model!r}")
    if request.stage != "baseline":
        return run_optimized_stage(
            request,
            module_prefix=f"deepmd.md_stages.{request.model}",
        )

    _configure_deepmd_baseline()
    if request.model == "dpa4":
        artifact_suffix = Path(request.model_path).suffix.lower()
        if artifact_suffix != ".pt":
            raise ValueError(
                "The DPA4 scientific baseline requires the raw '.pt' training "
                "checkpoint. A '.pt2' artifact has already captured AOTInductor "
                "and kernel policy, while '.pth' is interpreted as TorchScript. "
                f"Got {request.model_path!r}."
            )
    from deepmd.calculator import DP

    nlist_backend = request.backend
    if nlist_backend == "eager":
        nlist_backend = "native"
    if nlist_backend not in {"native", "auto", "vesin", "nv"}:
        raise ValueError(
            "DeepMD backend must be eager/native, auto, vesin, or nv; "
            f"got {request.backend!r}"
        )
    calculator = DP(model=request.model_path, nlist_backend=nlist_backend)
    if request.model == "dpa4":
        # SeZM owns an O(N) Toolkit-Ops edge-list builder and deliberately
        # ignores DeepEval's external nlist strategy.  This is the supported
        # eager inference path, not an opt-in baseline choice.  State that
        # explicitly so a report never mislabels it as a dense native list.
        baseline_profile = "scientific-eager-dpa4-internal-nv-nlist"
        effective_nlist_backend = "dpa4-internal-nvalchemiops"
    else:
        baseline_profile = (
            "scientific-eager-native"
            if nlist_backend == "native"
            else "existing-neighborlist-optimization-control"
        )
        effective_nlist_backend = nlist_backend
    metadata = {
        "baseline_profile": baseline_profile,
        "requested_neighborlist_backend": nlist_backend,
        "neighborlist_backend": effective_nlist_backend,
        "deepmd_inference_env": dict(_DEEPMD_BASELINE_ENV),
        "torch_float32_matmul_precision": "highest",
        "tf32": False,
    }
    if request.model == "dpa4":
        metadata |= {
            "model_artifact": "raw-pt-training-checkpoint",
            "intrinsic_model_optimization": "gpu-o-n-nvalchemiops-edge-builder",
            "captured_aotinductor": False,
            "legacy_fitting_type_compatibility_policy": (
                "ener-is-normalized-in-memory-to-dpa4_ener; weights-unchanged"
            ),
        }
    return run_ase_baseline(
        request,
        calculator,
        metadata=metadata,
    )
