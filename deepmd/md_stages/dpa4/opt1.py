"""DPA4 Opt1: eager GPU-resident molecular dynamics.

The FP64 positions, momenta, forces, thermostat variables, and integration
state live on one CUDA device for the full trajectory.  The DPA4/SeZM ``.pt``
checkpoint is loaded eagerly with the same high-precision DeepMD interface as
the scientific baseline; checkpoint-defined neural-network precision remains
unchanged.  The model keeps its model-owned nvalchemiops edge builder.  CUDA
Graph, ``torch.compile``, Triton/CuTe kernels, AMP, TF32, and model-specific
fusion are deliberately disabled at this stage.

Only requested observations/trajectory frames and the final state cross the
device boundary.  The per-step force call consumes and returns CUDA tensors;
it never enters the numpy/ASE calculator API.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from torch import Tensor

from deepmd.md_stages.dpa3.opt1 import (
    GPUMDState,
    GPUNoseHooverChain,
    GPUVelocityVerletBerendsen,
    _build_integrator,
    _evaluate_state,
    _run_measured_loop,
    _state_to_atoms,
)
from md_benchmark.md_route import (
    MDRunRequest,
    MDRunResult,
    configure_torch_baseline,
    validate_result,
)


_DEEPMD_OPT1_ENV = {
    # This process-wide setting is read when deepmd.pt is first imported.
    # Match the scientific baseline explicitly.  It controls the model
    # interface/environment-matrix dtype, not the checkpoint's descriptor and
    # fitting-network precision settings.
    "DP_INTERFACE_PREC": "high",
    "DP_ACT_INFER": "0",
    "DP_COMPILE_INFER": "0",
    "DP_CUDA_INFER": "0",
    "DP_CUTE_INFER": "0",
    "DP_TRITON_INFER": "0",
    "DP_TF32_INFER": "0",
    "DP_AMP_INFER": "0",
}


def _configure_opt1() -> None:
    """Pin eager execution and the scientific-baseline interface policy."""
    configure_torch_baseline()
    os.environ.update(_DEEPMD_OPT1_ENV)


def _require_raw_pt(model_path: str | Path) -> Path:
    path = Path(model_path)
    if path.suffix.lower() != ".pt":
        raise ValueError(
            "DPA4 Opt1 requires the raw '.pt' training checkpoint. A '.pt2' "
            "package already captures AOTInductor policy and '.pth' is treated "
            f"as TorchScript; got {path!s}"
        )
    return path.resolve()


class DPA4EnergyForceEvaluator:
    """Direct CUDA-tensor evaluator for an eager DPA4/SeZM checkpoint."""

    def __init__(
        self,
        atoms: Any,
        model_path: str | Path,
        *,
        device: str | torch.device,
    ) -> None:
        self.device = torch.device(device)
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("DPA4 Opt1 requires an available CUDA device")
        checkpoint_path = _require_raw_pt(model_path)

        # DPA4Wrapper is the checkpoint compatibility authority.  It loads the
        # raw weights into an eager SeZMModel.  In particular, the released
        # legacy fitting wire value ``ener`` is normalized to ``dpa4_ener`` by
        # get_sezm_model only in memory; neither checkpoint nor weights change.
        try:
            from deepmd.pt.nvalchemi import DPA4Wrapper
        except ImportError as exc:
            raise ImportError(
                "DPA4 Opt1 requires nvalchemi-toolkit and "
                "nvalchemi-toolkit-ops; install deepmd-kit[nvalchemi]"
            ) from exc
        from deepmd.pt.model.model.sezm_model import SeZMModel
        from deepmd.pt.utils.nv_nlist import is_nv_available

        if not is_nv_available():
            raise RuntimeError(
                "DPA4 Opt1 requires the nvalchemiops CUDA neighbor-list backend"
            )
        with torch.cuda.device(self.device):
            wrapper = DPA4Wrapper.from_checkpoint(
                checkpoint_path,
                device=self.device,
            )
        model = wrapper.model
        if not isinstance(model, SeZMModel):
            raise TypeError(
                "DPA4 Opt1 expected an eager SeZMModel from the raw checkpoint"
            )
        if model.get_active_mode() != "ener":
            raise ValueError(
                "DPA4 Opt1 supports only the conservative energy/force head"
            )
        model_dtype = model.global_pt_float_precision
        if model_dtype not in (torch.float32, torch.float64):
            raise ValueError(
                "DPA4 Opt1 supports float32/float64 DeepMD interfaces; "
                f"loaded interface precision is {model_dtype}"
            )
        if any(parameter.device != self.device for parameter in model.parameters()):
            raise RuntimeError("DPA4 model parameters did not all load on CUDA")
        if any(buffer.device != self.device for buffer in model.buffers()):
            raise RuntimeError("DPA4 model buffers did not all load on CUDA")

        self._model = model.eval()
        for parameter in self._model.parameters():
            parameter.requires_grad_(False)
        self.model_dtype = model_dtype
        self.neighbor_backend = "dpa4-internal-nvalchemiops-eager"

        definition = json.loads(model.model_def_script)
        descriptor_definition = definition.get("descriptor", {})
        fitting_definition = definition.get("fitting_net", {})
        descriptor_type = str(descriptor_definition.get("type", ""))
        model_type = str(definition.get("type", ""))
        if "dpa4" not in {descriptor_type.lower(), model_type.lower()}:
            raise ValueError(
                "The requested checkpoint does not identify as DPA4/SeZM: "
                f"model type={model_type!r}, descriptor type={descriptor_type!r}"
            )
        fitting_type = str(fitting_definition.get("type", ""))
        self.legacy_fitting_type_normalized = fitting_type == "ener"
        self.descriptor_precision = str(
            descriptor_definition.get("precision", "checkpoint-default")
        )
        self.fitting_precision = str(
            fitting_definition.get("precision", "checkpoint-default")
        )

        type_index = {
            symbol: index for index, symbol in enumerate(model.get_type_map())
        }
        try:
            atom_types = [type_index[symbol] for symbol in atoms.get_chemical_symbols()]
        except KeyError as exc:
            raise ValueError(
                f"Element {exc.args[0]!r} is absent from the checkpoint type map"
            ) from exc
        self.atom_types = torch.tensor(
            atom_types, dtype=torch.long, device=self.device
        ).unsqueeze(0)
        self.cell = torch.as_tensor(
            np.asarray(atoms.cell), dtype=self.model_dtype, device=self.device
        ).reshape(1, 3, 3)
        self.fparam = self._optional_input(
            atoms.info.get("fparam"),
            size=model.get_dim_fparam(),
            name="fparam",
            atomwise=False,
        )
        self.aparam = self._optional_input(
            atoms.info.get("aparam"),
            size=model.get_dim_aparam(),
            name="aparam",
            atomwise=True,
        )
        charge_spin = atoms.info.get("charge_spin")
        if charge_spin is None:
            if model.has_chg_spin_ebd() and not model.has_default_chg_spin():
                raise ValueError(
                    "Checkpoint requires charge_spin but the Atoms object does "
                    "not provide it"
                )
            self.charge_spin = None
        else:
            tensor = torch.as_tensor(
                charge_spin, dtype=self.model_dtype, device=self.device
            )
            if tensor.numel() != 2:
                raise ValueError("charge_spin must contain [charge, spin]")
            self.charge_spin = tensor.reshape(1, 2)

    def _optional_input(
        self,
        value: Any,
        *,
        size: int,
        name: str,
        atomwise: bool,
    ) -> Tensor | None:
        if value is None:
            if size and name == "aparam":
                raise ValueError(
                    f"Checkpoint requires {name} with {size} values per atom"
                )
            # The model owns any configured default fparam.
            return None
        tensor = torch.as_tensor(value, dtype=self.model_dtype, device=self.device)
        expected = size * self.atom_types.shape[1] if atomwise else size
        if tensor.numel() != expected:
            raise ValueError(
                f"{name} has {tensor.numel()} values, expected {expected}"
            )
        if atomwise:
            return tensor.reshape(1, self.atom_types.shape[1], size)
        return tensor.reshape(1, size)

    def __call__(self, positions: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Evaluate on device; the hot path contains no host/numpy conversion."""
        if positions.device != self.device:
            raise ValueError(
                f"positions must remain on {self.device}, got {positions.device}"
            )
        if positions.dtype is not torch.float64:
            raise ValueError("DPA4 Opt1 MD positions must remain FP64")
        expected_shape = (self.atom_types.shape[1], 3)
        if positions.shape != expected_shape:
            raise ValueError(
                f"positions shape {tuple(positions.shape)} != {expected_shape}"
            )

        # SeZM builds its compact COO edge schema through nvalchemiops on the
        # same CUDA device.  ``DP_COMPILE_INFER=0`` selects eager core_compute;
        # the edge-list path does not use NvNeighborList's optional compiled
        # dense-list truncation helper.
        model_positions = positions.to(dtype=self.model_dtype).unsqueeze(0)
        with torch.enable_grad(), torch.cuda.device(self.device):
            output = self._model(
                model_positions,
                self.atom_types,
                box=self.cell,
                fparam=self.fparam,
                aparam=self.aparam,
                do_atomic_virial=True,
                charge_spin=self.charge_spin,
            )
        try:
            force = output["force"].reshape(-1, 3).detach()
            energy = output["energy"].reshape(-1)[0].detach()
            virial = output["virial"].reshape(3, 3).detach()
        except KeyError as exc:
            raise RuntimeError(
                f"DPA4 tensor evaluator did not produce {exc.args[0]!r}; "
                f"available outputs are {sorted(output)}"
            ) from exc
        return force, energy, virial


def run_md(request: MDRunRequest) -> MDRunResult:
    """Run DPA4 Opt1; unsupported modes fail instead of falling back."""
    if request.model != "dpa4" or request.stage != "opt1":
        raise ValueError(f"DPA4 Opt1 route received {request.model}/{request.stage}")
    if request.backend != "gpu-resident":
        raise ValueError(
            "DPA4 Opt1 requires backend='gpu-resident'; "
            f"got {request.backend!r}"
        )
    if request.config.dtype != "float64":
        raise ValueError("DPA4 Opt1 requires --dtype float64 for the MD state")
    if request.config.dtype != "float64":
        raise ValueError(
            "DPA4 Opt1 fixes the physical MD state to float64; "
            f"got dtype={request.config.dtype!r}"
        )
    if request.config.ensemble.lower() != "nvt":
        raise ValueError("DPA4 Opt1 supports only NVT")
    if request.atoms.constraints:
        raise NotImplementedError("DPA4 Opt1 does not support ASE constraints")
    if not bool(np.asarray(request.atoms.pbc).all()):
        raise NotImplementedError(
            "DPA4 Opt1 currently requires fully periodic structures"
        )

    _configure_opt1()
    device = torch.device(request.config.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("DPA4 Opt1 requires config.device to name a CUDA device")

    atoms = request.atoms.copy()
    MaxwellBoltzmannDistribution(
        atoms,
        temperature_K=request.config.temperature_k,
        rng=np.random.default_rng(request.config.seed),
    )
    positions = torch.as_tensor(
        atoms.get_positions(), dtype=torch.float64, device=device
    ).clone()
    momenta = torch.as_tensor(
        atoms.get_momenta(), dtype=torch.float64, device=device
    ).clone()
    masses = torch.as_tensor(
        atoms.get_masses(), dtype=torch.float64, device=device
    )
    state = GPUMDState(positions=positions, momenta=momenta)
    initial_state = state.initial_clone()
    evaluator = DPA4EnergyForceEvaluator(atoms, request.model_path, device=device)

    if request.config.warmup_steps:
        warmup_integrator = _build_integrator(request, masses)
        _evaluate_state(state, evaluator)
        for _ in range(request.config.warmup_steps):
            warmup_integrator.step(state, evaluator)
        torch.cuda.synchronize(device)
        state.restore_initial_(initial_state)

    integrator = _build_integrator(request, masses)
    elapsed, observations, trajectory, trajectory_path = _run_measured_loop(
        request, state, evaluator, integrator, masses
    )
    final_atoms = _state_to_atoms(atoms, state)
    metadata = {
        "engine": "gpu_resident",
        "backend": "gpu-resident",
        "model_path": str(_require_raw_pt(request.model_path)),
        "model_artifact": "raw-pt-training-checkpoint",
        "integrator": request.config.integrator,
        "neighborlist_backend": evaluator.neighbor_backend,
        "neighbor_rebuilt_each_force_evaluation": True,
        "md_state_precision": "float64",
        "model_precision": str(evaluator.model_dtype).removeprefix("torch."),
        "model_interface_precision": str(evaluator.model_dtype).removeprefix(
            "torch."
        ),
        "checkpoint_descriptor_precision": evaluator.descriptor_precision,
        "checkpoint_fitting_precision": evaluator.fitting_precision,
        "stress_convention": "ase-tensile=-sym(deepmd-virial)/volume",
        "legacy_fitting_type_compatibility": (
            "ener-normalized-in-memory-to-dpa4_ener"
            if evaluator.legacy_fitting_type_normalized
            else "not-required"
        ),
        "checkpoint_modified": False,
        "warmup_steps": request.config.warmup_steps,
        "torch_compile": False,
        "cuda_graph": False,
        "kernel_fusion": False,
        "triton": False,
        "cute": False,
        "amp": False,
        "tf32": False,
        "hot_loop_numpy_roundtrip": False,
        "trajectory_frame_semantics": "step-0-plus-record-interval",
        "deepmd_inference_env": dict(_DEEPMD_OPT1_ENV),
    }
    if request.config.integrator == "nose_hoover_chain":
        metadata["nose_hoover_chain"] = {
            "tchain": int(request.options.get("tchain", 3)),
            "tloop": int(request.options.get("tloop", 1)),
        }
    result = MDRunResult(
        model=request.model,
        stage=request.stage,
        completed_steps=request.config.steps,
        elapsed_s=elapsed,
        peak_cuda_memory_gb=torch.cuda.max_memory_allocated(device) / 1e9,
        final_atoms=final_atoms,
        observations=observations,
        trajectory=trajectory,
        trajectory_path=trajectory_path,
        metadata=metadata,
    )
    validate_result(request, result)
    return result


__all__ = [
    "DPA4EnergyForceEvaluator",
    "GPUMDState",
    "GPUNoseHooverChain",
    "GPUVelocityVerletBerendsen",
    "run_md",
]
