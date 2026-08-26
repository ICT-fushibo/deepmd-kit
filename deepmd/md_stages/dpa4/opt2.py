"""DPA4 Opt2: strict model-only CUDA Graph molecular dynamics.

The GPU-resident Opt1 integrator and nvalchemiops neighbour search remain
eager.  Each force evaluation builds the compact edge schema outside the
CUDA Graph and copies it into fixed-address, fixed-capacity buffers.  The
captured region starts at :meth:`SeZMModel.forward_lower` and contains only
the DPA4 lower model, conservative edge-force autograd, and virial assembly.

There is no eager fallback.  A topology that exceeds the preflight edge or
search capacity raises explicitly, as does any unsupported CUDA Graph
capture.  ``torch.compile``, TF32, AMP, Triton/CuTe and model-specific fusion
stay disabled so this stage measures CUDA Graph independently.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import torch  # noqa: TID253
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from torch import Tensor  # noqa: TID253

if TYPE_CHECKING:
    from pathlib import Path

from deepmd.dpmodel.utils.neighbor_list import EdgeNeighborList
from deepmd.md_stages.dpa3.opt1 import (
    GPUMDState,
    _build_integrator,
    _evaluate_state,
    _run_measured_loop,
    _state_to_atoms,
)
from deepmd.md_stages.dpa4.opt1 import (
    DPA4EnergyForceEvaluator,
    _DEEPMD_OPT1_ENV,
    _configure_opt1,
    _evaluator_metadata,
    _require_raw_pt,
)
from md_benchmark.md_route import MDRunRequest, MDRunResult, validate_result
from md_benchmark.performance import (
    CudaPhaseProfiler,
    performance_profile_requested,
)


_CANONICAL_BACKEND = "model-only-cuda-graph"
_COMPATIBLE_BACKENDS = {_CANONICAL_BACKEND, "gpu-resident"}


def _resolve_edge_capacity(
    initial_edges: int,
    *,
    capacity_factor: float,
    capacity_headroom: int,
    explicit_capacity: int | None,
) -> int:
    """Resolve a fixed graph edge axis without silently truncating edges."""
    if initial_edges < 1:
        raise ValueError("initial edge schema must contain at least one edge slot")
    if capacity_factor < 1.0:
        raise ValueError("graph_edge_capacity_factor must be at least 1.0")
    if capacity_headroom < 0:
        raise ValueError("graph_edge_capacity_headroom must be non-negative")
    if explicit_capacity is not None:
        if explicit_capacity < initial_edges:
            raise ValueError(
                "graph_edge_capacity is smaller than the initial edge count: "
                f"{explicit_capacity} < {initial_edges}"
            )
        return int(explicit_capacity)
    return max(
        initial_edges + int(capacity_headroom),
        math.ceil(initial_edges * float(capacity_factor)),
    )


@dataclass
class _StaticEdgeGraphInputs:
    """Fixed-address tensors consumed by the captured SeZM lower graph."""

    coord: Tensor
    atype: Tensor
    edge_index: Tensor
    edge_vec: Tensor
    edge_scatter_index: Tensor
    edge_mask: Tensor

    @classmethod
    def allocate(
        cls,
        schema: EdgeNeighborList,
        capacity: int,
    ) -> _StaticEdgeGraphInputs:
        if capacity < schema.edge_vec.shape[0]:
            raise ValueError(
                f"edge capacity {capacity} < initial edges {schema.edge_vec.shape[0]}"
            )
        buffers = cls(
            coord=torch.empty_like(schema.coord),
            atype=torch.empty_like(schema.atype),
            edge_index=torch.zeros(
                (2, capacity),
                dtype=schema.edge_index.dtype,
                device=schema.edge_index.device,
            ),
            edge_vec=torch.zeros(
                (capacity, 3),
                dtype=schema.edge_vec.dtype,
                device=schema.edge_vec.device,
            ),
            edge_scatter_index=torch.zeros(
                (2, capacity),
                dtype=schema.edge_scatter_index.dtype,
                device=schema.edge_scatter_index.device,
            ),
            edge_mask=torch.zeros(
                capacity,
                dtype=torch.bool,
                device=schema.edge_mask.device,
            ),
        )
        buffers.copy_schema_(schema)
        return buffers

    @property
    def capacity(self) -> int:
        return int(self.edge_vec.shape[0])

    def addresses(self) -> tuple[int, ...]:
        """Return stable storage addresses for tests and runtime assertions."""
        return tuple(
            tensor.data_ptr()
            for tensor in (
                self.coord,
                self.atype,
                self.edge_index,
                self.edge_vec,
                self.edge_scatter_index,
                self.edge_mask,
            )
        )

    def copy_schema_(self, schema: EdgeNeighborList) -> int:
        """Copy one dynamic eager schema into the fixed graph buffers."""
        edge_count = int(schema.edge_vec.shape[0])
        if edge_count > self.capacity:
            raise RuntimeError(
                "DPA4 Opt2 edge capacity exceeded: "
                f"required {edge_count}, fixed capacity {self.capacity}. "
                "Increase route option 'graph_edge_capacity' or its "
                "factor/headroom and restart the run."
            )
        if schema.coord.shape != self.coord.shape:
            raise RuntimeError(
                f"DPA4 Opt2 coord shape changed from {tuple(self.coord.shape)} "
                f"to {tuple(schema.coord.shape)}"
            )
        if schema.atype.shape != self.atype.shape:
            raise RuntimeError(
                f"DPA4 Opt2 atype shape changed from {tuple(self.atype.shape)} "
                f"to {tuple(schema.atype.shape)}"
            )
        self.coord.copy_(schema.coord)
        self.atype.copy_(schema.atype)
        self.edge_mask.zero_()
        self.edge_index[:, :edge_count].copy_(schema.edge_index)
        self.edge_vec[:edge_count].copy_(schema.edge_vec)
        self.edge_scatter_index[:, :edge_count].copy_(
            schema.edge_scatter_index
        )
        self.edge_mask[:edge_count].copy_(schema.edge_mask)
        return edge_count


class DPA4ModelOnlyGraphEvaluator(DPA4EnergyForceEvaluator):
    """DPA4 evaluator with eager geometry and captured lower-model compute."""

    def __init__(
        self,
        atoms: Any,
        model_path: str | Path,
        *,
        device: str | torch.device,
        graph_edge_capacity_factor: float = 1.25,
        graph_edge_capacity_headroom: int = 64,
        graph_edge_capacity: int | None = None,
        neighbor_capacity_factor: float = 1.25,
        neighbor_capacity_headroom: int = 16,
        neighbor_search_capacity: int | None = None,
        capture_warmup_replays: int = 3,
        validation_energy_atol: float = 1.0e-6,
        validation_force_atol: float = 1.0e-6,
        validation_virial_atol: float = 1.0e-5,
        profiler: CudaPhaseProfiler | None = None,
    ) -> None:
        super().__init__(
            atoms,
            model_path,
            device=device,
            profiler=profiler,
        )
        if capture_warmup_replays < 1:
            raise ValueError("cuda_graph_capture_warmup_replays must be positive")
        if min(
            validation_energy_atol,
            validation_force_atol,
            validation_virial_atol,
        ) < 0:
            raise ValueError("CUDA Graph validation tolerances must be non-negative")
        if not hasattr(torch.cuda, "CUDAGraph"):
            raise RuntimeError("This PyTorch build does not provide CUDA Graph")

        from deepmd.pt.utils.nv_nlist import NvNeighborList

        self._neighbor_builder = NvNeighborList(compile_truncation=False)
        initial_md_positions = torch.as_tensor(
            atoms.get_positions(),
            dtype=torch.float64,
            device=self.device,
        )
        ragged_eager_reference = DPA4EnergyForceEvaluator.__call__(
            self,
            initial_md_positions,
        )
        initial_positions = initial_md_positions.to(
            dtype=self.model_dtype
        ).reshape(1, -1, 3)
        self.neighbor_shape_metadata = self._neighbor_builder.prepare_fixed_shape(
            initial_positions,
            self.cell,
            float(self._model.get_rcut()),
            list(self._model.get_sel()),
            capacity_factor=float(neighbor_capacity_factor),
            capacity_headroom=int(neighbor_capacity_headroom),
            search_capacity=neighbor_search_capacity,
        )
        initial_schema = self._build_edge_schema(initial_positions)
        initial_edge_count = int(initial_schema.edge_vec.shape[0])
        edge_capacity = _resolve_edge_capacity(
            initial_edge_count,
            capacity_factor=float(graph_edge_capacity_factor),
            capacity_headroom=int(graph_edge_capacity_headroom),
            explicit_capacity=graph_edge_capacity,
        )
        self._static = _StaticEdgeGraphInputs.allocate(
            initial_schema,
            edge_capacity,
        )
        self.initial_edge_count = initial_edge_count
        self.last_edge_count = initial_edge_count
        self.max_edge_count = initial_edge_count
        self.graph_edge_capacity = edge_capacity
        self._input_addresses = self._static.addresses()
        self.neighbor_backend = "dpa4-internal-nvalchemiops-eager-outside-graph"
        self.validation_energy_atol = float(validation_energy_atol)
        self.validation_force_atol = float(validation_force_atol)
        self.validation_virial_atol = float(validation_virial_atol)
        self.validation_energy_abs_error = math.inf
        self.validation_force_max_abs_error = math.inf
        self.validation_virial_max_abs_error = math.inf
        self.replay_stability_energy_abs_error = math.inf
        self.replay_stability_force_max_abs_error = math.inf
        self.replay_stability_virial_max_abs_error = math.inf
        self.validation_passed = False
        self.numerical_validation_within_tolerance = False
        self.numerical_validation_diagnostics: dict[str, dict[str, float | bool]] = {}
        self.capture_count = 0
        self.validation_replays = 0
        self.production_replays = 0
        self._captured_force: Tensor
        self._captured_energy: Tensor
        self._captured_virial: Tensor
        with torch.cuda.device(self.device):
            self._graph = torch.cuda.CUDAGraph()
            self._capture_model_graph(
                int(capture_warmup_replays),
                ragged_eager_reference,
            )
        self._output_addresses = tuple(
            value.data_ptr()
            for value in (
                self._captured_force,
                self._captured_energy,
                self._captured_virial,
            )
        )

    def _build_edge_schema(self, model_positions: Tensor) -> EdgeNeighborList:
        schema = self._neighbor_builder.build(
            model_positions,
            self.atom_types,
            self.cell,
            float(self._model.get_rcut()),
            list(self._model.get_sel()),
            return_mode="edges",
        )
        if not isinstance(schema, EdgeNeighborList):
            raise TypeError("DPA4 Opt2 neighbor builder did not return edge schema")
        return schema

    def _run_lower_model(self) -> dict[str, Tensor]:
        with torch.enable_grad():
            return self._model.forward_lower(
                self._static.coord,
                self._static.atype,
                self._static.edge_index,
                self._static.edge_vec,
                self._static.edge_scatter_index,
                self._static.edge_mask,
                fparam=self.fparam,
                aparam=self.aparam,
                do_atomic_virial=self.do_atomic_virial,
                charge_spin=self.charge_spin,
            )

    @staticmethod
    def _extract_outputs(output: dict[str, Tensor]) -> tuple[Tensor, Tensor, Tensor]:
        try:
            force = output["extended_force"].reshape(-1, 3).detach()
            energy = output["energy"].reshape(-1)[0].detach()
            virial = output["virial"].reshape(3, 3).detach()
        except KeyError as exc:
            raise RuntimeError(
                f"DPA4 lower graph did not produce {exc.args[0]!r}; "
                f"available outputs are {sorted(output)}"
            ) from exc
        return force, energy, virial

    def _capture_model_graph(
        self,
        warmup_replays: int,
        ragged_eager_reference: tuple[Tensor, Tensor, Tensor],
    ) -> None:
        current_stream = torch.cuda.current_stream(self.device)
        capture_stream = torch.cuda.Stream(device=self.device)
        capture_stream.wait_stream(current_stream)
        try:
            fixed_eager_reference: tuple[Tensor, Tensor, Tensor] | None = None
            with torch.cuda.stream(capture_stream):
                for _ in range(warmup_replays):
                    fixed_eager_reference = tuple(
                        value.clone()
                        for value in self._extract_outputs(
                            self._run_lower_model()
                        )
                    )
            assert fixed_eager_reference is not None
            capture_stream.synchronize()
            with torch.cuda.graph(self._graph, stream=capture_stream):
                output = self._run_lower_model()
            self._captured_force, self._captured_energy, self._captured_virial = (
                self._extract_outputs(output)
            )
            current_stream.wait_stream(capture_stream)
            torch.cuda.synchronize(self.device)

            self._graph.replay()
            first_replay = tuple(
                value.clone()
                for value in (
                    self._captured_force,
                    self._captured_energy,
                    self._captured_virial,
                )
            )
            self._graph.replay()
            second_replay = tuple(
                value.clone()
                for value in (
                    self._captured_force,
                    self._captured_energy,
                    self._captured_virial,
                )
            )
            torch.cuda.synchronize(self.device)

            replay_errors = tuple(
                float((second - first).abs().max().item())
                for first, second in zip(
                    first_replay,
                    second_replay,
                    strict=True,
                )
            )
            fixed_eager_errors = tuple(
                float((replay - eager).abs().max().item())
                for replay, eager in zip(
                    second_replay,
                    fixed_eager_reference,
                    strict=True,
                )
            )
            ragged_eager_errors = tuple(
                float((replay - eager).abs().max().item())
                for replay, eager in zip(
                    second_replay,
                    ragged_eager_reference,
                    strict=True,
                )
            )
            validation_errors = tuple(
                max(fixed_error, ragged_error)
                for fixed_error, ragged_error in zip(
                    fixed_eager_errors,
                    ragged_eager_errors,
                    strict=True,
                )
            )
            (
                self.replay_stability_force_max_abs_error,
                self.replay_stability_energy_abs_error,
                self.replay_stability_virial_max_abs_error,
            ) = replay_errors
            (
                self.validation_force_max_abs_error,
                self.validation_energy_abs_error,
                self.validation_virial_max_abs_error,
            ) = validation_errors
            all_errors = (*replay_errors, *fixed_eager_errors, *ragged_eager_errors)
            if not all(math.isfinite(error) for error in all_errors):
                raise FloatingPointError(
                    "DPA4 Opt2 numerical validation produced non-finite errors"
                )
            atom_count = int(self.atom_types.shape[1])
            for (
                name,
                replay_error,
                fixed_error,
                ragged_error,
                tolerance,
            ) in (
                (
                    "force",
                    self.replay_stability_force_max_abs_error,
                    fixed_eager_errors[0],
                    ragged_eager_errors[0],
                    self.validation_force_atol,
                ),
                (
                    "energy",
                    self.replay_stability_energy_abs_error,
                    fixed_eager_errors[1],
                    ragged_eager_errors[1],
                    self.validation_energy_atol,
                ),
                (
                    "virial",
                    self.replay_stability_virial_max_abs_error,
                    fixed_eager_errors[2],
                    ragged_eager_errors[2],
                    self.validation_virial_atol,
                ),
            ):
                # Fixed-capacity and ragged reductions can differ by roundoff.
                # Keep force/virial absolute tolerances unchanged, but express
                # total-energy parity as the configured per-atom tolerance.
                ragged_tolerance = (
                    tolerance * atom_count if name == "energy" else tolerance
                )
                self.numerical_validation_diagnostics[name] = {
                    "replay_abs_error": replay_error,
                    "fixed_eager_abs_error": fixed_error,
                    "ragged_eager_abs_error": ragged_error,
                    "replay_within_tolerance": replay_error <= tolerance,
                    "fixed_eager_within_tolerance": fixed_error <= tolerance,
                    "ragged_eager_within_tolerance": (
                        ragged_error <= ragged_tolerance
                    ),
                    "absolute_tolerance": tolerance,
                    "ragged_tolerance": ragged_tolerance,
                }
                # All finite numerical differences are scientific report fields;
                # tolerance exceedance never rejects a performance experiment.
            self.numerical_validation_within_tolerance = all(
                bool(value["replay_within_tolerance"])
                and bool(value["fixed_eager_within_tolerance"])
                and bool(value["ragged_eager_within_tolerance"])
                for value in self.numerical_validation_diagnostics.values()
            )
            self.capture_count = 1
            self.validation_replays = 2
            self.validation_passed = True
        except Exception as exc:
            raise RuntimeError(
                "DPA4 Opt2 model-only CUDA Graph capture failed. There is no "
                "eager fallback; use Opt1 for eager execution or fix the "
                "reported unsupported operation."
            ) from exc

    def _assert_static_addresses(self) -> None:
        if self._static.addresses() != self._input_addresses:
            raise RuntimeError("DPA4 Opt2 graph input storage address changed")
        output_addresses = tuple(
            value.data_ptr()
            for value in (
                self._captured_force,
                self._captured_energy,
                self._captured_virial,
            )
        )
        if output_addresses != self._output_addresses:
            raise RuntimeError("DPA4 Opt2 graph output storage address changed")

    def __call__(self, positions: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Build geometry eagerly, copy static inputs, and replay the model."""
        if positions.device != self.device:
            raise ValueError(
                f"positions must remain on {self.device}, got {positions.device}"
            )
        if positions.dtype is not torch.float64:
            raise ValueError("DPA4 Opt2 MD positions must remain FP64")
        expected_shape = (self.atom_types.shape[1], 3)
        if positions.shape != expected_shape:
            raise ValueError(
                f"positions shape {tuple(positions.shape)} != {expected_shape}"
            )

        with torch.cuda.device(self.device):
            with self.profiler.phase("model_input"):
                model_positions = positions.to(dtype=self.model_dtype).unsqueeze(0)
            with self.profiler.phase("neighbor_list"):
                schema = self._build_edge_schema(model_positions)
            self.last_edge_count = self._static.copy_schema_(schema)
            self.max_edge_count = max(self.max_edge_count, self.last_edge_count)
            self._assert_static_addresses()
            with self.profiler.phase("model_energy_force"):
                self._graph.replay()
            self.production_replays += 1
        return (
            self._captured_force,
            self._captured_energy,
            self._captured_virial,
        )


def run_md(request: MDRunRequest) -> MDRunResult:
    """Run strict DPA4 model-only CUDA Graph MD without eager fallback."""
    if request.model != "dpa4" or request.stage != "opt2":
        raise ValueError(f"DPA4 Opt2 route received {request.model}/{request.stage}")
    if request.backend not in _COMPATIBLE_BACKENDS:
        raise ValueError(
            "DPA4 Opt2 requires backend='model-only-cuda-graph' "
            "('gpu-resident' is accepted as a compatibility alias); "
            f"got {request.backend!r}"
        )
    if request.config.dtype != "float64":
        raise ValueError(
            "DPA4 Opt2 fixes the physical MD state to float64; "
            f"got dtype={request.config.dtype!r}"
        )
    if request.config.ensemble.lower() != "nvt":
        raise ValueError("DPA4 Opt2 supports only NVT")
    if request.atoms.constraints:
        raise NotImplementedError("DPA4 Opt2 does not support ASE constraints")
    if not bool(np.asarray(request.atoms.pbc).all()):
        raise NotImplementedError(
            "DPA4 Opt2 currently requires fully periodic structures"
        )

    _configure_opt1()
    device = torch.device(request.config.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("DPA4 Opt2 requires config.device to name a CUDA device")

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
    profiler = CudaPhaseProfiler(
        enabled=performance_profile_requested(request.options),
        device=device,
    )
    configured_graph_capacity = request.options.get("graph_edge_capacity")
    configured_search_capacity = request.options.get("neighbor_search_capacity")
    evaluator = DPA4ModelOnlyGraphEvaluator(
        atoms,
        request.model_path,
        device=device,
        graph_edge_capacity_factor=float(
            request.options.get("graph_edge_capacity_factor", 1.25)
        ),
        graph_edge_capacity_headroom=int(
            request.options.get("graph_edge_capacity_headroom", 64)
        ),
        graph_edge_capacity=(
            None
            if configured_graph_capacity is None
            else int(configured_graph_capacity)
        ),
        neighbor_capacity_factor=float(
            request.options.get("neighbor_capacity_factor", 1.25)
        ),
        neighbor_capacity_headroom=int(
            request.options.get("neighbor_capacity_headroom", 16)
        ),
        neighbor_search_capacity=(
            None
            if configured_search_capacity is None
            else int(configured_search_capacity)
        ),
        capture_warmup_replays=int(
            request.options.get("cuda_graph_capture_warmup_replays", 3)
        ),
        validation_energy_atol=float(
            request.options.get("cuda_graph_energy_atol", 1.0e-6)
        ),
        validation_force_atol=float(
            request.options.get("cuda_graph_force_atol", 1.0e-6)
        ),
        validation_virial_atol=float(
            request.options.get("cuda_graph_virial_atol", 1.0e-5)
        ),
        profiler=profiler,
    )

    if request.config.warmup_steps:
        warmup_integrator = _build_integrator(request, masses)
        _evaluate_state(state, evaluator)
        for _ in range(request.config.warmup_steps):
            warmup_integrator.step(state, evaluator)
        torch.cuda.synchronize(device)
        state.restore_initial_(initial_state)

    evaluator.production_replays = 0
    integrator = _build_integrator(request, masses)
    elapsed, observations, trajectory, trajectory_path = _run_measured_loop(
        request,
        state,
        evaluator,
        integrator,
        masses,
        profiler,
    )
    final_atoms = _state_to_atoms(atoms, state)
    expected_replays = request.config.steps + 1
    if evaluator.production_replays != expected_replays:
        raise RuntimeError(
            "DPA4 Opt2 production replay count mismatch: "
            f"expected={expected_replays}, actual={evaluator.production_replays}"
        )
    metadata = {
        "engine": "gpu_resident",
        "backend": _CANONICAL_BACKEND,
        "requested_backend": request.backend,
        "model_path": str(_require_raw_pt(request.model_path)),
        "model_artifact": "raw-pt-training-checkpoint",
        "integrator": request.config.integrator,
        "neighborlist_backend": evaluator.neighbor_backend,
        "neighborlist_fixed_shape": dict(evaluator.neighbor_shape_metadata),
        "neighbor_rebuilt_each_force_evaluation": True,
        "neighbor_list_inside_cuda_graph": False,
        "graph_capture_scope": "model-only-sezm-forward-lower",
        "graph_lower_entry": "SeZMModel.forward_lower",
        "graph_edge_capacity": evaluator.graph_edge_capacity,
        "graph_initial_edge_count": evaluator.initial_edge_count,
        "graph_final_edge_count": evaluator.last_edge_count,
        "graph_max_edge_count": evaluator.max_edge_count,
        "graph_overflow_policy": "explicit-error-no-truncation-no-fallback",
        "graph_input_addresses_fixed": True,
        "graph_output_addresses_fixed": True,
        "graph_capture_count": evaluator.capture_count,
        "graph_validation_replays": evaluator.validation_replays,
        "graph_production_replays": evaluator.production_replays,
        "graph_validation_passed": evaluator.validation_passed,
        "graph_numerical_validation_failure_policy": "report_only",
        "graph_numerical_validation_within_tolerance": (
            evaluator.numerical_validation_within_tolerance
        ),
        "graph_numerical_validation_diagnostics": (
            evaluator.numerical_validation_diagnostics
        ),
        "graph_validation_energy_abs_error": (
            evaluator.validation_energy_abs_error
        ),
        "graph_validation_force_max_abs_error": (
            evaluator.validation_force_max_abs_error
        ),
        "graph_validation_virial_max_abs_error": (
            evaluator.validation_virial_max_abs_error
        ),
        "graph_replay_stability_energy_abs_error": (
            evaluator.replay_stability_energy_abs_error
        ),
        "graph_replay_stability_force_max_abs_error": (
            evaluator.replay_stability_force_max_abs_error
        ),
        "graph_replay_stability_virial_max_abs_error": (
            evaluator.replay_stability_virial_max_abs_error
        ),
        "md_state_precision": "float64",
        **_evaluator_metadata(evaluator),
        "stress_convention": "ase-tensile=-sym(deepmd-virial)/volume",
        "legacy_fitting_type_compatibility": (
            "ener-normalized-in-memory-to-dpa4_ener"
            if evaluator.legacy_fitting_type_normalized
            else "not-required"
        ),
        "checkpoint_modified": False,
        "warmup_steps": request.config.warmup_steps,
        "torch_compile": False,
        "cuda_graph": True,
        "cuda_graph_scope": "model-only",
        "kernel_fusion": False,
        "triton": False,
        "cute": False,
        "amp": False,
        "tf32": False,
        "hot_loop_numpy_roundtrip": False,
        "trajectory_frame_semantics": "step-0-plus-record-interval",
        "deepmd_inference_env": dict(_DEEPMD_OPT1_ENV),
        "performance_profile": profiler.summary(synchronize=False),
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


__all__ = ["DPA4ModelOnlyGraphEvaluator", "run_md"]
