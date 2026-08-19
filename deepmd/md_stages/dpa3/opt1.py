"""DPA3 Opt1: eager GPU-resident molecular dynamics.

This stage deliberately contains no CUDA Graph, ``torch.compile``, Triton,
custom fusion, AMP, or TF32.  The FP64 MD state and both supported NVT
integrators remain on one CUDA device.  The DPA3 TorchScript model is called
directly on CUDA tensors, and its neighbor list is rebuilt on the same device
with nvalchemiops for every force evaluation.  Host transfers occur only at
requested observation/trajectory boundaries and once for the final state.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import ase.io
import numpy as np
import torch
from ase import Atoms, units
from ase.calculators.singlepoint import SinglePointCalculator
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from torch import Tensor

from md_benchmark.md_route import (
    MDObservation,
    MDRunRequest,
    MDRunResult,
    configure_torch_baseline,
    validate_result,
)


_DEEPMD_OPT1_ENV = {
    "DP_ACT_INFER": "0",
    "DP_COMPILE_INFER": "0",
    "DP_CUDA_INFER": "0",
    "DP_CUTE_INFER": "0",
    "DP_TRITON_INFER": "0",
    "DP_TF32_INFER": "0",
    "DP_AMP_INFER": "0",
}

_FOURTH_ORDER_COEFFS = (
    1.0 / (2.0 - 2.0 ** (1.0 / 3.0)),
    -(2.0 ** (1.0 / 3.0)) / (2.0 - 2.0 ** (1.0 / 3.0)),
    1.0 / (2.0 - 2.0 ** (1.0 / 3.0)),
)


def _configure_opt1() -> None:
    """Pin the eager/full-precision policy before loading DeePMD."""
    configure_torch_baseline()
    os.environ.update(_DEEPMD_OPT1_ENV)


class EnergyForceEvaluator(Protocol):
    device: torch.device
    model_dtype: torch.dtype
    neighbor_backend: str

    def __call__(self, positions: Tensor) -> tuple[Tensor, Tensor, Tensor]: ...


class DPA3EnergyForceEvaluator:
    """Direct tensor evaluator for a frozen DPA3 TorchScript checkpoint."""

    def __init__(
        self,
        atoms: Atoms,
        model_path: str | Path,
        *,
        device: str | torch.device,
    ) -> None:
        self.device = torch.device(device)
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("DPA3 Opt1 requires an available CUDA device")
        if Path(model_path).suffix.lower() != ".pth":
            raise ValueError(
                "DPA3 Opt1 requires the frozen TorchScript '.pth' checkpoint; "
                f"got {model_path!s}"
            )

        # Import after the inference environment is pinned.  DeepPot remains
        # the checkpoint/type-map authority; its numpy eval API is never used.
        from deepmd.infer import DeepPot
        from deepmd.pt.utils.nv_nlist import NvNeighborList

        with torch.cuda.device(self.device):
            self.potential = DeepPot(
                str(Path(model_path).resolve()),
                auto_batch_size=False,
                nlist_backend="nv",
            )
        backend = self.potential.deep_eval
        if not isinstance(backend._nlist_builder, NvNeighborList):
            raise RuntimeError(
                "DPA3 Opt1 requested nvalchemiops, but DeepMD did not select "
                "NvNeighborList"
            )
        # The eager helper is numerically identical and keeps torch.compile out
        # of Opt1.  The default=True path remains unchanged for all other users.
        backend._nlist_builder.compile_truncation = False
        self._backend = backend
        self._module = backend.dp.to(self.device)
        self._module.eval()
        for parameter in self._module.parameters():
            parameter.requires_grad_(False)

        model_definition = backend.get_model_def_script()
        descriptor = model_definition.get("descriptor", {})
        descriptor_type = str(descriptor.get("type", "")).lower()
        model_type = str(model_definition.get("type", "")).lower()
        if "dpa3" not in {descriptor_type, model_type}:
            raise ValueError(
                "The requested DPA3 Opt1 checkpoint does not identify as DPA3: "
                f"model type={model_type!r}, descriptor type={descriptor_type!r}"
            )
        if backend.get_has_spin() or backend.get_has_hessian():
            raise NotImplementedError("DPA3 Opt1 does not support spin/Hessian models")

        model_dtype = getattr(
            backend.dp.model["Default"],
            "global_pt_float_precision",
            None,
        )
        if not isinstance(model_dtype, torch.dtype):
            model_dtype = next(
                (
                    value.dtype
                    for value in self._module.parameters()
                    if value.is_floating_point()
                ),
                torch.float64,
            )
        self.model_dtype = model_dtype
        self.neighbor_backend = "nvalchemiops-eager"

        type_index = dict(
            zip(
                self.potential.get_type_map(),
                range(self.potential.get_ntypes()),
                strict=True,
            )
        )
        try:
            atom_types = [type_index[symbol] for symbol in atoms.get_chemical_symbols()]
        except KeyError as exc:
            raise ValueError(
                f"Element {exc.args[0]!r} is absent from the checkpoint type map"
            ) from exc
        self.atom_types = torch.tensor(
            atom_types, dtype=torch.long, device=self.device
        ).unsqueeze(0)
        self.cell = (
            torch.as_tensor(
                np.asarray(atoms.cell), dtype=self.model_dtype, device=self.device
            ).reshape(1, 3, 3)
            if bool(np.asarray(atoms.pbc).any())
            else None
        )
        self.fparam = self._optional_input(
            atoms.info.get("fparam"),
            expected_size=self.potential.get_dim_fparam(),
            name="fparam",
            allow_missing=self.potential.has_default_fparam(),
        )
        self.aparam = self._optional_input(
            atoms.info.get("aparam"),
            expected_size=self.potential.get_dim_aparam() * len(atoms),
            name="aparam",
            atomwise=True,
        )
        charge_spin = atoms.info.get("charge_spin")
        if charge_spin is None:
            if (
                self.potential.has_chg_spin_ebd()
                and not self.potential.has_default_chg_spin()
            ):
                raise ValueError(
                    "Checkpoint requires charge_spin, but the Atoms object does "
                    "not provide it"
                )
            self.charge_spin = None
        else:
            self.charge_spin = self._optional_input(
                charge_spin, expected_size=2, name="charge_spin"
            )

    def _optional_input(
        self,
        value: Any,
        *,
        expected_size: int,
        name: str,
        atomwise: bool = False,
        allow_missing: bool = False,
    ) -> Tensor | None:
        if value is None:
            if expected_size and not allow_missing:
                raise ValueError(
                    f"Checkpoint requires {name} with size {expected_size}, but the "
                    "Atoms object does not provide it"
                )
            return None
        tensor = torch.as_tensor(value, dtype=self.model_dtype, device=self.device)
        if tensor.numel() != expected_size:
            raise ValueError(
                f"{name} has {tensor.numel()} values, expected {expected_size}"
            )
        if atomwise:
            return tensor.reshape(1, len(self.atom_types[0]), -1)
        return tensor.reshape(1, -1)

    def __call__(self, positions: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if positions.device != self.device:
            raise ValueError(
                f"positions must remain on {self.device}, got {positions.device}"
            )
        if positions.shape != (self.atom_types.shape[1], 3):
            raise ValueError(
                f"positions shape {tuple(positions.shape)} does not match "
                f"({self.atom_types.shape[1]}, 3)"
            )
        model_positions = positions.to(dtype=self.model_dtype).unsqueeze(0)
        # Conservative force/virial generation uses autograd inside the DPA3
        # model.  Model parameters are frozen by inference-only ModelWrapper.
        with torch.enable_grad(), torch.cuda.device(self.device):
            output = self._backend._eval_lower_strategy(
                model_positions,
                self.atom_types,
                self.cell,
                self.fparam,
                self.aparam,
                self.charge_spin,
                False,
            )
        try:
            force = output["force"].reshape(-1, 3).detach()
            energy = output["energy"].reshape(-1)[0].detach()
            virial = output["virial"].reshape(3, 3).detach()
        except KeyError as exc:
            raise RuntimeError(
                f"DPA3 tensor evaluator did not produce {exc.args[0]!r}; "
                f"available outputs are {sorted(output)}"
            ) from exc
        return force, energy, virial


@dataclass
class GPUMDState:
    """All mutable physical MD state on one device in FP64."""

    positions: Tensor
    momenta: Tensor
    forces: Tensor | None = None
    potential_energy: Tensor | None = None
    virial: Tensor | None = None

    def initial_clone(self) -> "GPUMDState":
        return GPUMDState(self.positions.clone(), self.momenta.clone())

    def restore_initial_(self, other: "GPUMDState") -> None:
        self.positions.copy_(other.positions)
        self.momenta.copy_(other.momenta)
        self.forces = None
        self.potential_energy = None
        self.virial = None


class _Integrator(Protocol):
    name: str

    def step(self, state: GPUMDState, evaluator: EnergyForceEvaluator) -> None: ...


class GPUVelocityVerletBerendsen:
    """Unconstrained ASE NVTBerendsen equations on CUDA tensors."""

    name = "berendsen"

    def __init__(
        self,
        masses: Tensor,
        *,
        timestep_fs: float,
        temperature_k: float,
        thermostat_time_fs: float,
    ) -> None:
        self.masses = masses.reshape(-1, 1)
        self.dt = float(timestep_fs) * units.fs
        self.temperature_k = float(temperature_k)
        self.taut = float(thermostat_time_fs) * units.fs
        self.degrees_of_freedom = 3 * self.masses.shape[0]

    def _scale_momenta(self, momenta: Tensor) -> Tensor:
        kinetic = (0.5 * momenta.square() / self.masses).sum()
        temperature = 2.0 * kinetic / (self.degrees_of_freedom * units.kB)
        scale = torch.sqrt(
            1.0
            + (self.temperature_k / temperature.clamp_min(1.0e-12) - 1.0)
            * (self.dt / self.taut)
        ).clamp(0.9, 1.1)
        return momenta * scale

    def step(self, state: GPUMDState, evaluator: EnergyForceEvaluator) -> None:
        if state.forces is None:
            _evaluate_state(state, evaluator)
        assert state.forces is not None
        momenta = self._scale_momenta(state.momenta)
        momenta = momenta + 0.5 * self.dt * state.forces
        # Match ASE NVTBerendsen(fixcm=True): subtract mean momentum rather
        # than mass-weighted center-of-mass velocity.
        momenta = momenta - momenta.mean(dim=0, keepdim=True)
        positions = state.positions + self.dt * momenta / self.masses
        force, energy, virial = evaluator(positions)
        force64 = force.to(dtype=torch.float64)
        state.positions = positions
        state.momenta = momenta + 0.5 * self.dt * force64
        state.forces = force64
        state.potential_energy = energy
        state.virial = virial


class GPUNoseHooverChain:
    """ASE NoseHooverChainNVT(tchain=3,tloop=1) equations on CUDA."""

    name = "nose_hoover_chain"

    def __init__(
        self,
        masses: Tensor,
        *,
        timestep_fs: float,
        temperature_k: float,
        thermostat_time_fs: float,
        tchain: int = 3,
        tloop: int = 1,
    ) -> None:
        if tchain < 1 or tloop < 1:
            raise ValueError("tchain and tloop must be positive")
        self.masses = masses.reshape(-1, 1)
        self.dt = float(timestep_fs) * units.fs
        self.tchain = int(tchain)
        self.tloop = int(tloop)
        k_t = units.kB * float(temperature_k)
        tdamp = float(thermostat_time_fs) * units.fs
        q = torch.full(
            (self.tchain,),
            k_t * tdamp**2,
            dtype=torch.float64,
            device=masses.device,
        )
        q[0] = 3 * self.masses.shape[0] * k_t * tdamp**2
        self.q_mass = q
        self.k_t = float(k_t)
        self.eta = torch.zeros_like(q)
        self.p_eta = torch.zeros_like(q)

    def _integrate_p_eta_j(
        self, momenta: Tensor, j: int, delta2: float, delta4: float
    ) -> None:
        if j < self.tchain - 1:
            self.p_eta[j].mul_(
                torch.exp(-delta4 * self.p_eta[j + 1] / self.q_mass[j + 1])
            )
        if j == 0:
            g_j = (momenta.square() / self.masses).sum() - (
                3 * self.masses.shape[0] * self.k_t
            )
        else:
            g_j = self.p_eta[j - 1].square() / self.q_mass[j - 1] - self.k_t
        self.p_eta[j].add_(delta2 * g_j)
        if j < self.tchain - 1:
            self.p_eta[j].mul_(
                torch.exp(-delta4 * self.p_eta[j + 1] / self.q_mass[j + 1])
            )

    def _integrate_loop(self, momenta: Tensor, delta: float) -> Tensor:
        delta2 = delta / 2.0
        delta4 = delta / 4.0
        for j in reversed(range(self.tchain)):
            self._integrate_p_eta_j(momenta, j, delta2, delta4)
        self.eta.add_(delta * self.p_eta / self.q_mass)
        momenta = momenta * torch.exp(-delta * self.p_eta[0] / self.q_mass[0])
        for j in range(self.tchain):
            self._integrate_p_eta_j(momenta, j, delta2, delta4)
        return momenta

    def _integrate_thermostat(self, momenta: Tensor, delta: float) -> Tensor:
        for _ in range(self.tloop):
            for coefficient in _FOURTH_ORDER_COEFFS:
                momenta = self._integrate_loop(
                    momenta, coefficient * delta / self.tloop
                )
        return momenta

    def step(self, state: GPUMDState, evaluator: EnergyForceEvaluator) -> None:
        if state.forces is None:
            _evaluate_state(state, evaluator)
        assert state.forces is not None
        half_dt = self.dt / 2.0
        momenta = self._integrate_thermostat(state.momenta, half_dt)
        momenta = momenta + half_dt * state.forces
        positions = state.positions + self.dt * momenta / self.masses
        force, energy, virial = evaluator(positions)
        force64 = force.to(dtype=torch.float64)
        momenta = momenta + half_dt * force64
        momenta = self._integrate_thermostat(momenta, half_dt)
        state.positions = positions
        state.momenta = momenta
        state.forces = force64
        state.potential_energy = energy
        state.virial = virial


def _evaluate_state(state: GPUMDState, evaluator: EnergyForceEvaluator) -> None:
    force, energy, virial = evaluator(state.positions)
    state.forces = force.to(dtype=torch.float64)
    state.potential_energy = energy
    state.virial = virial


def _build_integrator(request: MDRunRequest, masses: Tensor) -> _Integrator:
    config = request.config
    common = {
        "masses": masses,
        "timestep_fs": config.timestep_fs,
        "temperature_k": config.temperature_k,
        "thermostat_time_fs": config.thermostat_time_fs,
    }
    if config.integrator == "berendsen":
        return GPUVelocityVerletBerendsen(**common)
    if config.integrator == "nose_hoover_chain":
        return GPUNoseHooverChain(
            **common,
            tchain=int(request.options.get("tchain", 3)),
            tloop=int(request.options.get("tloop", 1)),
        )
    raise ValueError(f"Unsupported integrator {config.integrator!r}")


def _kinetic_energy(state: GPUMDState, masses: Tensor) -> Tensor:
    return (0.5 * state.momenta.square() / masses.reshape(-1, 1)).sum()


def _state_to_atoms(
    template: Atoms,
    state: GPUMDState,
    *,
    step: int | None = None,
) -> Atoms:
    if state.forces is None or state.potential_energy is None or state.virial is None:
        raise RuntimeError("Cannot snapshot an unevaluated GPU MD state")
    positions = state.positions.detach().cpu().numpy()
    momenta = state.momenta.detach().cpu().numpy()
    forces = state.forces.detach().cpu().numpy()
    energy = float(state.potential_energy.detach().cpu())
    virial = state.virial.detach().to(torch.float64)
    virial = 0.5 * (virial + virial.transpose(0, 1))
    volume = float(template.get_volume())
    stress_matrix = (-virial / volume).cpu().numpy()
    stress = stress_matrix.flat[[0, 4, 8, 5, 2, 1]]

    frame = template.copy()
    frame.set_positions(positions)
    frame.set_momenta(momenta)
    if step is not None:
        frame.info["md_step"] = int(step)
    frame.calc = SinglePointCalculator(
        frame,
        energy=energy,
        forces=forces,
        stress=stress,
    )
    return frame


def _observation(
    step: int, state: GPUMDState, masses: Tensor
) -> MDObservation:
    if state.forces is None or state.potential_energy is None:
        raise RuntimeError("Cannot observe an unevaluated GPU MD state")
    return MDObservation(
        step=step,
        potential_energy_ev=float(state.potential_energy.detach().cpu()),
        kinetic_energy_ev=float(_kinetic_energy(state, masses).detach().cpu()),
        forces_ev_per_a=state.forces.detach().cpu().numpy().copy(),
        positions_a=state.positions.detach().cpu().numpy().copy(),
    )


def _prepare_trajectory(request: MDRunRequest) -> tuple[Path | None, Path | None]:
    if not request.config.collect_trajectory:
        return None, None
    if request.config.record_interval < 1:
        raise ValueError("collect_trajectory requires record_interval >= 1")
    if request.output_path is None:
        return None, None
    final_path = Path(request.output_path)
    partial_path = final_path.with_name(f"{final_path.stem}.part.extxyz")
    final_path.parent.mkdir(parents=True, exist_ok=True)
    if final_path.exists() and not request.options.get("overwrite", False):
        raise FileExistsError(f"Refusing to overwrite {final_path}")
    for stale in (final_path, partial_path):
        if stale.exists():
            stale.unlink()
    return final_path, partial_path


def _append_trajectory(
    frame: Atoms,
    *,
    memory_frames: list[Atoms] | None,
    partial_path: Path | None,
) -> None:
    if partial_path is not None:
        ase.io.write(partial_path, frame, append=True, format="extxyz")
    else:
        assert memory_frames is not None
        memory_frames.append(frame)


def _run_measured_loop(
    request: MDRunRequest,
    state: GPUMDState,
    evaluator: EnergyForceEvaluator,
    integrator: _Integrator,
    masses: Tensor,
) -> tuple[float, list[MDObservation], list[Atoms] | None, str | None]:
    config = request.config
    observations: list[MDObservation] = []
    memory_frames: list[Atoms] | None = (
        [] if config.collect_trajectory and request.output_path is None else None
    )
    final_path, partial_path = _prepare_trajectory(request)
    observation_steps = set(config.observation_steps)

    torch.cuda.reset_peak_memory_stats(evaluator.device)
    torch.cuda.synchronize(evaluator.device)
    started = time.perf_counter()
    _evaluate_state(state, evaluator)
    if config.collect_trajectory:
        _append_trajectory(
            _state_to_atoms(request.atoms, state, step=0),
            memory_frames=memory_frames,
            partial_path=partial_path,
        )

    for step in range(1, config.steps + 1):
        integrator.step(state, evaluator)
        if config.collect_statistics and step in observation_steps:
            observations.append(_observation(step, state, masses))
        if config.collect_trajectory and step % config.record_interval == 0:
            _append_trajectory(
                _state_to_atoms(request.atoms, state, step=step),
                memory_frames=memory_frames,
                partial_path=partial_path,
            )
    torch.cuda.synchronize(evaluator.device)
    elapsed = time.perf_counter() - started
    if final_path is not None and partial_path is not None:
        os.replace(partial_path, final_path)
    return (
        elapsed,
        observations,
        memory_frames,
        str(final_path) if final_path is not None else None,
    )


def run_md(request: MDRunRequest) -> MDRunResult:
    """Run the DPA3 Opt1 route; unsupported modes fail instead of falling back."""
    if request.model != "dpa3" or request.stage != "opt1":
        raise ValueError(
            f"DPA3 Opt1 route received {request.model}/{request.stage}"
        )
    if request.backend != "gpu-resident":
        raise ValueError(
            "DPA3 Opt1 requires backend='gpu-resident'; "
            f"got {request.backend!r}"
        )
    if request.config.dtype != "float64":
        raise ValueError("DPA3 Opt1 requires --dtype float64 for the MD state")
    if request.config.ensemble.lower() != "nvt":
        raise ValueError("DPA3 Opt1 supports only NVT")
    if request.atoms.constraints:
        raise NotImplementedError("DPA3 Opt1 does not support ASE constraints")
    if not bool(np.asarray(request.atoms.pbc).all()):
        raise NotImplementedError(
            "DPA3 Opt1 currently requires fully periodic structures"
        )

    _configure_opt1()
    device = torch.device(request.config.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("DPA3 Opt1 requires config.device to name a CUDA device")

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
    evaluator = DPA3EnergyForceEvaluator(
        atoms, request.model_path, device=device
    )

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
        "model_path": str(Path(request.model_path).resolve()),
        "integrator": request.config.integrator,
        "neighborlist_backend": evaluator.neighbor_backend,
        "neighbor_rebuilt_each_force_evaluation": True,
        "md_state_precision": "float64",
        "model_precision": str(evaluator.model_dtype).removeprefix("torch."),
        "warmup_steps": request.config.warmup_steps,
        "torch_compile": False,
        "cuda_graph": False,
        "kernel_fusion": False,
        "triton": False,
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
    "DPA3EnergyForceEvaluator",
    "GPUMDState",
    "GPUNoseHooverChain",
    "GPUVelocityVerletBerendsen",
    "run_md",
]
