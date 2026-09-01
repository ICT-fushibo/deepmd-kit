"""DPA4 Opt3: one-capacity whole-step CUDA Graph molecular dynamics.

The complete unconstrained NVT step is captured: Berendsen or Nose-Hoover
Chain integration, fixed-capacity nvalchemiops neighbour construction, the
SeZM forward/force path, and the persistent GPU-state update.  Capacity uses
the eSEN-style uniform per-atom CAP policy.  Real edge counts may vary while
the edge tensor shape stays fixed; unused slots are masked, non-zero-distance
self sinks distributed over the atoms.

This development stage deliberately has no transactional rollback, graph
bucket cache, eager fallback, compilation, reduced precision, or custom
fusion.  A neighbour-capacity overflow raises asynchronously on CUDA and the
run must be restarted with a larger ``graph_edge_capacity``.
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
    GPUNoseHooverChain,
    GPUVelocityVerletBerendsen,
    _build_integrator,
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


_CANONICAL_BACKEND = "whole-step-cuda-graph"


def _round_up(value: int, alignment: int) -> int:
    if alignment < 1:
        raise ValueError("graph_edge_capacity_alignment must be positive")
    return ((value + alignment - 1) // alignment) * alignment


@dataclass(frozen=True)
class _CapacityPlan:
    """One whole-step graph capacity and its per-atom search width."""

    edge_capacity: int
    neighbors_per_atom: int
    candidate_slots: int


def _resolve_capacity_plan(
    *,
    atom_count: int,
    initial_max_neighbors: int,
    capacity_factor: float,
    capacity_headroom: int,
    capacity_alignment: int,
    explicit_edge_capacity: int | None,
    explicit_search_capacity: int | None,
) -> _CapacityPlan:
    """Resolve eSEN-style uniform per-atom CAP without silent truncation."""
    if atom_count < 1:
        raise ValueError("DPA4 Opt3 requires at least one atom")
    if initial_max_neighbors < 0:
        raise ValueError("initial_max_neighbors must be non-negative")
    if capacity_factor < 1.0:
        raise ValueError("graph_edge_capacity_factor must be at least 1.0")
    if capacity_headroom < 0:
        raise ValueError("graph_edge_capacity_headroom must be non-negative")
    if capacity_alignment < 1:
        raise ValueError("graph_edge_capacity_alignment must be positive")

    if explicit_edge_capacity is not None:
        requested_edge_capacity = int(explicit_edge_capacity)
        if requested_edge_capacity < atom_count:
            raise ValueError(
                "graph_edge_capacity must provide at least one slot per atom"
            )
        if explicit_search_capacity is None:
            # A total-edge probe does not bound an individual high-coordinate
            # center.  Convert it conservatively to a uniform row width, align
            # to the eSEN slot quantum, then add one complete guard bucket.
            inferred_search_capacity = math.ceil(
                requested_edge_capacity / atom_count
            )
            search_capacity = (
                _round_up(inferred_search_capacity, capacity_alignment)
                + capacity_alignment
            )
        else:
            # An explicit per-center value is authoritative and intentionally
            # overrides the inferred guard policy.
            search_capacity = int(explicit_search_capacity)
        if search_capacity < 1:
            raise ValueError("neighbor_search_capacity must be positive")
        edge_capacity = max(
            requested_edge_capacity,
            atom_count * search_capacity,
        )
    else:
        requested = max(
            initial_max_neighbors + int(capacity_headroom),
            math.ceil(initial_max_neighbors * float(capacity_factor)),
            1,
        )
        search_capacity = (
            _round_up(requested, int(capacity_alignment))
            if explicit_search_capacity is None
            else int(explicit_search_capacity)
        )
        if search_capacity < 1:
            raise ValueError("neighbor_search_capacity must be positive")
        edge_capacity = atom_count * search_capacity

    if search_capacity < initial_max_neighbors:
        raise ValueError(
            "whole-step capacity is smaller than the initial per-atom "
            f"neighbor count: {search_capacity} < {initial_max_neighbors}"
        )
    candidate_slots = atom_count * search_capacity
    return _CapacityPlan(
        edge_capacity=edge_capacity,
        neighbors_per_atom=search_capacity,
        candidate_slots=candidate_slots,
    )


def _fixed_edge_schema_from_neighbor_matrix(
    *,
    coord: Tensor,
    atype: Tensor,
    cell: Tensor | None,
    neighbor_matrix: Tensor,
    num_neighbors: Tensor,
    shifts: Tensor,
    rcut: float,
    edge_capacity: int,
) -> EdgeNeighborList:
    """Create a fixed edge axis with distributed, masked self-sink padding.

    Unlike the ordinary edge converter this function has no data-dependent
    ``nonzero`` or compact allocation, so it can execute inside CUDA capture.
    nvalchemiops supplies a fixed-width row per center atom.  Invalid candidate
    slots and any extra tail capacity are represented by masked self edges with
    a finite non-zero vector beyond the cutoff.  Distributing those sinks over
    centers avoids concentrating zero-valued scatter traffic on atom zero.
    """
    nf, nloc = atype.shape[:2]
    if nf != 1:
        raise ValueError("DPA4 Opt3 currently supports one MD frame per graph")
    total_atoms, slots_per_atom = neighbor_matrix.shape
    if total_atoms != nloc:
        raise ValueError(
            "neighbor matrix atom dimension does not match the MD frame: "
            f"{total_atoms} != {nloc}"
        )
    candidate_slots = total_atoms * slots_per_atom
    if edge_capacity < candidate_slots:
        raise ValueError(
            "graph edge capacity is smaller than fixed neighbor slots: "
            f"{edge_capacity} < {candidate_slots}"
        )

    device = coord.device
    slot = torch.arange(slots_per_atom, dtype=torch.long, device=device)
    slot = slot.expand(total_atoms, slots_per_atom)
    dst = torch.arange(total_atoms, dtype=torch.long, device=device)
    dst = dst.unsqueeze(1).expand(-1, slots_per_atom).reshape(-1)
    valid_search = (slot < num_neighbors.reshape(-1, 1)).reshape(-1)
    src_raw = neighbor_matrix.reshape(-1).to(dtype=torch.long)
    src = torch.where(valid_search, src_raw, dst)
    shift = shifts.reshape(-1, 3).to(dtype=coord.dtype)
    shift = torch.where(valid_search.unsqueeze(1), shift, torch.zeros_like(shift))

    coord_flat = coord.reshape(total_atoms, 3)
    edge_vec = coord_flat.index_select(0, src) - coord_flat.index_select(0, dst)
    if cell is not None:
        edge_vec = edge_vec + (shift[:, :, None] * cell[0]).sum(dim=1)
    edge_len2 = torch.sum(edge_vec.square(), dim=-1)
    edge_mask = (
        valid_search
        & (edge_len2 > 1.0e-10)
        & (edge_len2 <= float(rcut) * float(rcut))
    )

    # A finite far vector avoids undefined direction normalization even in
    # implementations that form geometry before applying ``edge_mask``.
    far_vec = torch.zeros_like(edge_vec)
    far_vec[:, 0] = float(rcut) + 1.0
    sink = dst
    safe_src = torch.where(edge_mask, src, sink)
    safe_vec = torch.where(edge_mask.unsqueeze(1), edge_vec, far_vec)

    tail = edge_capacity - candidate_slots
    if tail:
        tail_sink = torch.remainder(
            torch.arange(tail, dtype=torch.long, device=device), nloc
        )
        tail_vec = torch.zeros((tail, 3), dtype=coord.dtype, device=device)
        tail_vec[:, 0] = float(rcut) + 1.0
        safe_src = torch.cat((safe_src, tail_sink), dim=0)
        dst = torch.cat((dst, tail_sink), dim=0)
        safe_vec = torch.cat((safe_vec, tail_vec), dim=0)
        edge_mask = torch.cat(
            (edge_mask, torch.zeros(tail, dtype=torch.bool, device=device)),
            dim=0,
        )

    edge_index = torch.stack((safe_src, dst), dim=0)
    return EdgeNeighborList(
        coord=coord,
        atype=atype,
        edge_index=edge_index,
        edge_vec=safe_vec,
        edge_scatter_index=edge_index,
        edge_mask=edge_mask,
    )


class DPA4WholeStepGraph(DPA4EnergyForceEvaluator):
    """Persistent DPA4 MD state and one captured complete NVT step."""

    def __init__(
        self,
        atoms: Any,
        model_path: str | Path,
        *,
        state: GPUMDState,
        masses: Tensor,
        request: MDRunRequest,
        graph_edge_capacity_factor: float = 1.10,
        graph_edge_capacity_headroom: int = 1,
        graph_edge_capacity_alignment: int = 8,
        graph_edge_capacity: int | None = None,
        neighbor_search_capacity: int | None = None,
        capture_warmup_replays: int = 3,
        validation_state_atol: float = 1.0e-10,
        validation_force_atol: float = 1.0e-6,
        validation_energy_atol: float = 1.0e-6,
        validation_virial_atol: float = 1.0e-5,
        validation_thermostat_atol: float = 1.0e-10,
        profiler: CudaPhaseProfiler | None = None,
    ) -> None:
        super().__init__(
            atoms,
            model_path,
            device=request.config.device,
            profiler=profiler,
        )
        if capture_warmup_replays < 1:
            raise ValueError("cuda_graph_capture_warmup_replays must be positive")
        if min(
            validation_state_atol,
            validation_force_atol,
            validation_energy_atol,
            validation_virial_atol,
            validation_thermostat_atol,
        ) < 0:
            raise ValueError("CUDA Graph validation tolerances must be non-negative")
        if not hasattr(torch.cuda, "CUDAGraph"):
            raise RuntimeError("This PyTorch build does not provide CUDA Graph")
        if state.positions.data_ptr() == state.momenta.data_ptr():
            raise RuntimeError("positions and momenta must have distinct storage")

        from deepmd.pt.utils.nv_nlist import NvNeighborList

        self.state = state
        self.masses = masses.reshape(-1, 1)
        self.request = request
        self.validation_tolerances = {
            "positions": float(validation_state_atol),
            "momenta": float(validation_state_atol),
            "forces": float(validation_force_atol),
            "energy": float(validation_energy_atol),
            "virial": float(validation_virial_atol),
            "eta": float(validation_thermostat_atol),
            "p_eta": float(validation_thermostat_atol),
        }
        self._integrator = _build_integrator(request, masses)
        self._neighbor_builder = NvNeighborList(compile_truncation=False)
        n_atoms = int(self.atom_types.shape[1])
        initial_model_positions = state.positions.to(dtype=self.model_dtype).reshape(
            1, n_atoms, 3
        )

        # This initial search is the only topology-dependent host read.  Keep
        # it separate from ``prepare_fixed_shape``: that general DeepMD helper
        # never chooses fewer than sum(sel)=384 slots, whereas eSEN CAP should
        # follow the observed per-center maximum (plus guard/alignment).
        from deepmd.pt_expt.utils.nv_graph_builder import nv_search_matrix

        _, _, _, initial_num_neighbors, _ = nv_search_matrix(
            initial_model_positions,
            self.cell,
            float(self._model.get_rcut()),
            start_capacity=int(sum(self._model.get_sel())),
        )
        initial_max_neighbors = int(initial_num_neighbors.max().detach().cpu())
        self.capacity_plan = _resolve_capacity_plan(
            atom_count=n_atoms,
            initial_max_neighbors=initial_max_neighbors,
            capacity_factor=float(graph_edge_capacity_factor),
            capacity_headroom=int(graph_edge_capacity_headroom),
            capacity_alignment=int(graph_edge_capacity_alignment),
            explicit_edge_capacity=graph_edge_capacity,
            explicit_search_capacity=neighbor_search_capacity,
        )
        self._neighbor_builder.prepare_fixed_shape(
            initial_model_positions,
            self.cell,
            float(self._model.get_rcut()),
            list(self._model.get_sel()),
            search_capacity=self.capacity_plan.neighbors_per_atom,
        )
        self.neighbor_shape_metadata = self._neighbor_builder.fixed_shape_metadata()
        self.neighbor_backend = "dpa4-nvalchemiops-fixed-cap-inside-graph"

        self._last_required_neighbors = torch.tensor(
            initial_max_neighbors, dtype=torch.int64, device=self.device
        )
        self._max_required_neighbors = self._last_required_neighbors.clone()
        self._overflow_flag = torch.zeros(
            (), dtype=torch.bool, device=self.device
        )
        self._overflow_count = torch.zeros(
            (), dtype=torch.int64, device=self.device
        )

        initial_force, initial_energy, initial_virial, initial_edge_count = (
            self._evaluate_positions(state.positions)
        )
        state.forces = initial_force.to(dtype=torch.float64).clone()
        state.potential_energy = initial_energy.clone()
        state.virial = initial_virial.clone()
        self.initial_edge_count = int(initial_edge_count.detach().cpu())
        self._last_edge_count = initial_edge_count.to(dtype=torch.int64).clone()
        self._max_edge_count = self._last_edge_count.clone()

        self._initial_positions = state.positions.clone()
        self._initial_momenta = state.momenta.clone()
        self._initial_forces = state.forces.clone()
        self._initial_energy = state.potential_energy.clone()
        self._initial_virial = state.virial.clone()
        self._initial_eta: Tensor | None = None
        self._initial_p_eta: Tensor | None = None
        if isinstance(self._integrator, GPUNoseHooverChain):
            self._initial_eta = self._integrator.eta.clone()
            self._initial_p_eta = self._integrator.p_eta.clone()

        self.capture_count = 0
        self.validation_replays = 0
        self.production_replays = 0
        self.validation_completed = False
        self.validation_finite = False
        self.validation_passed = False
        self.validation_within_tolerance = False
        self.validation_errors: dict[str, float] = {}
        self.validation_diagnostics: dict[str, dict[str, float | bool]] = {}
        self._state_addresses = self._addresses()
        with torch.cuda.device(self.device):
            self._graph = torch.cuda.CUDAGraph()
            self._capture_whole_step_graph(int(capture_warmup_replays))

    def _addresses(self) -> tuple[int, ...]:
        state = self.state
        assert state.forces is not None
        assert state.potential_energy is not None
        assert state.virial is not None
        tensors = [
            state.positions,
            state.momenta,
            state.forces,
            state.potential_energy,
            state.virial,
            self._last_edge_count,
            self._max_edge_count,
            self._last_required_neighbors,
            self._max_required_neighbors,
            self._overflow_flag,
            self._overflow_count,
        ]
        if isinstance(self._integrator, GPUNoseHooverChain):
            tensors.extend((self._integrator.eta, self._integrator.p_eta))
        return tuple(tensor.data_ptr() for tensor in tensors)

    def _build_fixed_edge_schema(self, positions: Tensor) -> EdgeNeighborList:
        from deepmd.pt_expt.utils.nv_graph_builder import nv_search_matrix_fixed

        coord = positions.to(dtype=self.model_dtype).reshape(1, -1, 3)
        coord, cell, neighbor_matrix, num_neighbors, shifts = (
            nv_search_matrix_fixed(
                coord,
                self.cell,
                float(self._model.get_rcut()),
                capacity=self.capacity_plan.neighbors_per_atom,
            )
        )
        required = num_neighbors.max().to(dtype=torch.int64)
        overflow = required > self.capacity_plan.neighbors_per_atom
        self._last_required_neighbors.copy_(required)
        self._max_required_neighbors.copy_(
            torch.maximum(self._max_required_neighbors, required)
        )
        self._overflow_flag.logical_or_(overflow)
        self._overflow_count.add_(overflow.to(dtype=torch.int64))
        # Keep this local assertion in addition to nv_search_matrix_fixed's
        # assertion.  The persistent status above makes the requirement visible
        # in metadata and tests; the assertion prevents a truncated replay from
        # ever being accepted as an MD step.
        torch._assert_async(
            ~overflow,
            "DPA4 Opt3 per-atom neighbor capacity exceeded; increase "
            "graph_edge_capacity or graph_neighbors_per_atom and restart",
        )
        return _fixed_edge_schema_from_neighbor_matrix(
            coord=coord,
            atype=self.atom_types,
            cell=cell,
            neighbor_matrix=neighbor_matrix,
            num_neighbors=num_neighbors,
            shifts=shifts,
            rcut=float(self._model.get_rcut()),
            edge_capacity=self.capacity_plan.edge_capacity,
        )

    def _run_model(self, schema: EdgeNeighborList) -> tuple[Tensor, Tensor, Tensor]:
        with torch.enable_grad():
            output = self._model.forward_lower(
                schema.coord,
                schema.atype,
                schema.edge_index,
                schema.edge_vec,
                schema.edge_scatter_index,
                schema.edge_mask,
                fparam=self.fparam,
                aparam=self.aparam,
                do_atomic_virial=self.do_atomic_virial,
                charge_spin=self.charge_spin,
            )
        try:
            force = output["extended_force"].reshape(-1, 3).detach()
            energy = output["energy"].reshape(-1)[0].detach()
            virial = output["virial"].reshape(3, 3).detach()
        except KeyError as exc:
            raise RuntimeError(
                f"DPA4 whole-step graph did not produce {exc.args[0]!r}; "
                f"available outputs are {sorted(output)}"
            ) from exc
        return force, energy, virial

    def _evaluate_positions(
        self, positions: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        schema = self._build_fixed_edge_schema(positions)
        force, energy, virial = self._run_model(schema)
        return force, energy, virial, schema.edge_mask.sum(dtype=torch.int64)

    def _step_body(self) -> None:
        state = self.state
        assert state.forces is not None
        assert state.potential_energy is not None
        assert state.virial is not None

        if isinstance(self._integrator, GPUVelocityVerletBerendsen):
            momenta = self._integrator._scale_momenta(state.momenta)
            momenta = momenta + 0.5 * self._integrator.dt * state.forces
            momenta = momenta - momenta.mean(dim=0, keepdim=True)
            positions = (
                state.positions
                + self._integrator.dt * momenta / self._integrator.masses
            )
            force, energy, virial, edge_count = self._evaluate_positions(positions)
            force64 = force.to(dtype=torch.float64)
            momenta = momenta + 0.5 * self._integrator.dt * force64
        elif isinstance(self._integrator, GPUNoseHooverChain):
            half_dt = self._integrator.dt / 2.0
            momenta = self._integrator._integrate_thermostat(
                state.momenta, half_dt
            )
            momenta = momenta + half_dt * state.forces
            positions = (
                state.positions
                + self._integrator.dt * momenta / self._integrator.masses
            )
            force, energy, virial, edge_count = self._evaluate_positions(positions)
            force64 = force.to(dtype=torch.float64)
            momenta = momenta + half_dt * force64
            momenta = self._integrator._integrate_thermostat(momenta, half_dt)
        else:  # pragma: no cover - _build_integrator rejects other modes
            raise TypeError(f"unsupported whole-step integrator {self._integrator!r}")

        # These in-place writes are the loop-carried dependencies.  Replays
        # read exactly the storage written by the preceding replay.
        state.positions.copy_(positions)
        state.momenta.copy_(momenta)
        state.forces.copy_(force64)
        state.potential_energy.copy_(energy)
        state.virial.copy_(virial)
        self._last_edge_count.copy_(edge_count)
        self._max_edge_count.copy_(torch.maximum(self._max_edge_count, edge_count))

    def restore_initial_(self) -> None:
        state = self.state
        assert state.forces is not None
        assert state.potential_energy is not None
        assert state.virial is not None
        state.positions.copy_(self._initial_positions)
        state.momenta.copy_(self._initial_momenta)
        state.forces.copy_(self._initial_forces)
        state.potential_energy.copy_(self._initial_energy)
        state.virial.copy_(self._initial_virial)
        self._last_edge_count.fill_(self.initial_edge_count)
        self._max_edge_count.fill_(self.initial_edge_count)
        self._last_required_neighbors.fill_(
            int(self.neighbor_shape_metadata["initial_max_neighbors"])
        )
        self._max_required_neighbors.copy_(self._last_required_neighbors)
        self._overflow_flag.zero_()
        self._overflow_count.zero_()
        if isinstance(self._integrator, GPUNoseHooverChain):
            assert self._initial_eta is not None
            assert self._initial_p_eta is not None
            self._integrator.eta.copy_(self._initial_eta)
            self._integrator.p_eta.copy_(self._initial_p_eta)

    def _state_snapshot(self) -> dict[str, Tensor]:
        state = self.state
        assert state.forces is not None
        assert state.potential_energy is not None
        assert state.virial is not None
        snapshot = {
            "positions": state.positions.clone(),
            "momenta": state.momenta.clone(),
            "forces": state.forces.clone(),
            "energy": state.potential_energy.clone(),
            "virial": state.virial.clone(),
        }
        if isinstance(self._integrator, GPUNoseHooverChain):
            snapshot["eta"] = self._integrator.eta.clone()
            snapshot["p_eta"] = self._integrator.p_eta.clone()
        return snapshot

    def _capture_whole_step_graph(self, warmup_replays: int) -> None:
        current_stream = torch.cuda.current_stream(self.device)
        capture_stream = torch.cuda.Stream(device=self.device)
        capture_stream.wait_stream(current_stream)
        try:
            with torch.cuda.stream(capture_stream):
                for _ in range(warmup_replays):
                    self._step_body()
            capture_stream.synchronize()
            self.restore_initial_()
            torch.cuda.synchronize(self.device)
            capture_stream.wait_stream(current_stream)
            with torch.cuda.graph(self._graph, stream=capture_stream):
                self._step_body()
            capture_stream.synchronize()
            self.restore_initial_()
            torch.cuda.synchronize(self.device)

            self._graph.replay()
            torch.cuda.synchronize(self.device)
            graph_state = self._state_snapshot()
            self.restore_initial_()
            self._step_body()
            torch.cuda.synchronize(self.device)
            eager_state = self._state_snapshot()
            if graph_state.keys() != eager_state.keys():
                raise RuntimeError(
                    "DPA4 Opt3 graph/eager validation state fields differ"
                )
            for name, eager in eager_state.items():
                graph_value = graph_state[name]
                if (
                    graph_value.shape != eager.shape
                    or graph_value.dtype != eager.dtype
                    or graph_value.device != eager.device
                ):
                    raise RuntimeError(
                        "DPA4 Opt3 graph/eager state contract mismatch for "
                        f"{name}: graph=({tuple(graph_value.shape)}, "
                        f"{graph_value.dtype}, {graph_value.device}), "
                        f"eager=({tuple(eager.shape)}, {eager.dtype}, "
                        f"{eager.device})"
                    )
            self.validation_errors = {
                name: float((graph_state[name] - eager).abs().max().detach().cpu())
                for name, eager in eager_state.items()
            }
            if not all(
                math.isfinite(error)
                for error in self.validation_errors.values()
            ):
                raise FloatingPointError(
                    "DPA4 Opt3 capture validation produced non-finite errors"
                )
            self.validation_finite = True
            self.validation_diagnostics = {
                name: {
                    "max_abs_error": error,
                    "absolute_tolerance": self.validation_tolerances[name],
                    "within_tolerance": (
                        error <= self.validation_tolerances[name]
                    ),
                }
                for name, error in self.validation_errors.items()
            }
            self.validation_within_tolerance = all(
                bool(diagnostic["within_tolerance"])
                for diagnostic in self.validation_diagnostics.values()
            )
            self.restore_initial_()
            torch.cuda.synchronize(self.device)
            if self._addresses() != self._state_addresses:
                raise RuntimeError("DPA4 Opt3 persistent state address changed")
            self.capture_count = 1
            self.validation_replays = 1
            self.validation_completed = True
            # Preserve this familiar field with strict numerical semantics.
            # A finite out-of-tolerance comparison remains report-only below.
            self.validation_passed = self.validation_within_tolerance
        except Exception as exc:
            raise RuntimeError(
                "DPA4 Opt3 whole-step CUDA Graph capture failed. There is no "
                "rollback or eager fallback; fix the unsupported operation or "
                "increase the fixed capacity."
            ) from exc

    def __call__(self, positions: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Eager fixed-capacity initial force used by the shared timer."""
        if positions.data_ptr() != self.state.positions.data_ptr():
            raise RuntimeError("DPA4 Opt3 initial evaluator requires static positions")
        force, energy, virial, edge_count = self._evaluate_positions(positions)
        state = self.state
        assert state.forces is not None
        assert state.potential_energy is not None
        assert state.virial is not None
        state.forces.copy_(force.to(dtype=torch.float64))
        state.potential_energy.copy_(energy)
        state.virial.copy_(virial)
        self._last_edge_count.copy_(edge_count)
        self._max_edge_count.copy_(torch.maximum(self._max_edge_count, edge_count))
        return state.forces, state.potential_energy, state.virial

    def step(self, state: GPUMDState, evaluator: Any) -> None:
        """Replay one captured step; evaluator is intentionally unused."""
        if state is not self.state or evaluator is not self:
            raise RuntimeError("DPA4 Opt3 requires its persistent state/evaluator")
        self._graph.replay()
        self.production_replays += 1

    @property
    def last_edge_count(self) -> int:
        return int(self._last_edge_count.detach().cpu())

    @property
    def max_edge_count(self) -> int:
        return int(self._max_edge_count.detach().cpu())

    @property
    def max_required_neighbors(self) -> int:
        return int(self._max_required_neighbors.detach().cpu())

    @property
    def overflow_count(self) -> int:
        return int(self._overflow_count.detach().cpu())

    def raise_if_overflow(self) -> None:
        """Reject a capacity-truncated trajectory at a synchronized boundary."""
        if bool(self._overflow_flag.detach().cpu()):
            raise RuntimeError(
                "DPA4 Opt3 neighbor capacity exceeded during production: "
                f"required up to {self.max_required_neighbors} neighbors/atom, "
                f"capacity={self.capacity_plan.neighbors_per_atom}. Increase "
                "graph_edge_capacity or graph_neighbors_per_atom and restart."
            )


def run_md(request: MDRunRequest) -> MDRunResult:
    """Run DPA4 whole-step CUDA Graph MD with strict capacity errors."""
    if request.model != "dpa4" or request.stage != "opt3":
        raise ValueError(f"DPA4 Opt3 route received {request.model}/{request.stage}")
    if request.backend != _CANONICAL_BACKEND:
        raise ValueError(
            "DPA4 Opt3 requires backend='whole-step-cuda-graph'; "
            f"got {request.backend!r}"
        )
    if request.config.dtype != "float64":
        raise ValueError("DPA4 Opt3 fixes the physical MD state to float64")
    if request.config.ensemble.lower() != "nvt":
        raise ValueError("DPA4 Opt3 supports only NVT")
    if request.config.integrator not in {"berendsen", "nose_hoover_chain"}:
        raise ValueError(
            "DPA4 Opt3 supports Berendsen and Nose-Hoover Chain NVT"
        )
    if request.atoms.constraints:
        raise NotImplementedError("DPA4 Opt3 does not support ASE constraints")
    if not bool(np.asarray(request.atoms.pbc).all()):
        raise NotImplementedError(
            "DPA4 Opt3 currently requires fully periodic structures"
        )

    _configure_opt1()
    device = torch.device(request.config.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("DPA4 Opt3 requires config.device to name a CUDA device")

    atoms = request.atoms.copy()
    MaxwellBoltzmannDistribution(
        atoms,
        temperature_K=request.config.temperature_k,
        rng=np.random.default_rng(request.config.seed),
    )
    state = GPUMDState(
        positions=torch.as_tensor(
            atoms.get_positions(), dtype=torch.float64, device=device
        ).clone(),
        momenta=torch.as_tensor(
            atoms.get_momenta(), dtype=torch.float64, device=device
        ).clone(),
    )
    masses = torch.as_tensor(
        atoms.get_masses(), dtype=torch.float64, device=device
    )
    profiler = CudaPhaseProfiler(
        enabled=performance_profile_requested(request.options),
        device=device,
    )
    configured_edge_capacity = request.options.get("graph_edge_capacity")
    configured_capacity_alignment = int(
        request.options.get("graph_edge_capacity_alignment", 8)
    )
    configured_search_capacity = request.options.get("graph_neighbors_per_atom")
    legacy_search_capacity = request.options.get("neighbor_search_capacity")
    if (
        configured_search_capacity is not None
        and legacy_search_capacity is not None
    ):
        if int(configured_search_capacity) != int(legacy_search_capacity):
            raise ValueError(
                "graph_neighbors_per_atom and neighbor_search_capacity disagree"
            )
    if configured_search_capacity is None:
        configured_search_capacity = legacy_search_capacity
    runner = DPA4WholeStepGraph(
        atoms,
        request.model_path,
        state=state,
        masses=masses,
        request=request,
        graph_edge_capacity_factor=float(
            request.options.get("graph_edge_capacity_factor", 1.10)
        ),
        graph_edge_capacity_headroom=int(
            request.options.get("graph_edge_capacity_headroom", 1)
        ),
        graph_edge_capacity_alignment=configured_capacity_alignment,
        graph_edge_capacity=(
            None
            if configured_edge_capacity is None
            else int(configured_edge_capacity)
        ),
        neighbor_search_capacity=(
            None
            if configured_search_capacity is None
            else int(configured_search_capacity)
        ),
        capture_warmup_replays=int(
            request.options.get("cuda_graph_capture_warmup_replays", 3)
        ),
        validation_state_atol=float(
            request.options.get("cuda_graph_state_atol", 1.0e-10)
        ),
        validation_force_atol=float(
            request.options.get("cuda_graph_force_atol", 1.0e-6)
        ),
        validation_energy_atol=float(
            request.options.get("cuda_graph_energy_atol", 1.0e-6)
        ),
        validation_virial_atol=float(
            request.options.get("cuda_graph_virial_atol", 1.0e-5)
        ),
        validation_thermostat_atol=float(
            request.options.get("cuda_graph_thermostat_atol", 1.0e-10)
        ),
        profiler=profiler,
    )

    for _ in range(request.config.warmup_steps):
        runner.step(state, runner)
    torch.cuda.synchronize(device)
    runner.restore_initial_()
    runner.production_replays = 0

    elapsed, observations, trajectory, trajectory_path = _run_measured_loop(
        request,
        state,
        runner,
        runner,
        masses,
        profiler,
    )
    runner.raise_if_overflow()
    if runner.production_replays != request.config.steps:
        raise RuntimeError(
            "DPA4 Opt3 production replay count mismatch: "
            f"expected={request.config.steps}, actual={runner.production_replays}"
        )
    final_atoms = _state_to_atoms(atoms, state)
    metadata = {
        "engine": "gpu_resident",
        "backend": _CANONICAL_BACKEND,
        "model_path": str(_require_raw_pt(request.model_path)),
        "model_artifact": "raw-pt-training-checkpoint",
        "integrator": request.config.integrator,
        "neighborlist_backend": runner.neighbor_backend,
        "neighborlist_fixed_shape": dict(runner.neighbor_shape_metadata),
        "neighbor_rebuilt_each_force_evaluation": True,
        "neighbor_list_inside_cuda_graph": True,
        "graph_capture_scope": "whole-md-step",
        "initial_force_inside_cuda_graph": False,
        "initial_force_evaluations": 1,
        "total_force_evaluations": request.config.steps + 1,
        "graph_requested_edge_capacity": (
            None
            if configured_edge_capacity is None
            else int(configured_edge_capacity)
        ),
        "graph_edge_capacity": runner.capacity_plan.edge_capacity,
        "graph_candidate_slots": runner.capacity_plan.candidate_slots,
        "graph_neighbors_per_atom": runner.capacity_plan.neighbors_per_atom,
        "graph_max_required_neighbors": runner.max_required_neighbors,
        "graph_overflow_count": runner.overflow_count,
        "graph_initial_edge_count": runner.initial_edge_count,
        "graph_final_edge_count": runner.last_edge_count,
        "graph_max_edge_count": runner.max_edge_count,
        "graph_capacity_policy": "esen-cap-uniform-per-atom-single-graph",
        "graph_capacity_guard_neighbors": (
            configured_capacity_alignment
            if configured_edge_capacity is not None
            and configured_search_capacity is None
            else 0
        ),
        "graph_padding_policy": "distributed-masked-self-sink-far-vector",
        "graph_overflow_policy": "explicit-error-no-rollback-no-fallback",
        "graph_input_addresses_fixed": True,
        "graph_output_addresses_fixed": True,
        "graph_capture_count": runner.capture_count,
        "graph_validation_replays": runner.validation_replays,
        "graph_production_replays": runner.production_replays,
        "graph_validation_completed": runner.validation_completed,
        "graph_validation_finite": runner.validation_finite,
        "graph_validation_passed": runner.validation_passed,
        "graph_validation_errors": dict(runner.validation_errors),
        "graph_validation_state_fields": sorted(runner.validation_errors),
        "graph_validation_tolerances": dict(runner.validation_tolerances),
        "graph_validation_diagnostics": dict(runner.validation_diagnostics),
        "graph_numerical_validation_within_tolerance": (
            runner.validation_within_tolerance
        ),
        "graph_numerical_validation_failure_policy": "warning-only",
        "md_state_precision": "float64",
        **_evaluator_metadata(runner),
        "stress_convention": "ase-tensile=-sym(deepmd-virial)/volume",
        "checkpoint_modified": False,
        "warmup_steps": request.config.warmup_steps,
        "torch_compile": False,
        "cuda_graph": True,
        "cuda_graph_scope": "whole-step",
        "cuda_graph_buckets": False,
        "transactional_rollback": False,
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
            "state_inside_cuda_graph": True,
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
    "DPA4WholeStepGraph",
    "_fixed_edge_schema_from_neighbor_matrix",
    "_resolve_capacity_plan",
    "run_md",
]
