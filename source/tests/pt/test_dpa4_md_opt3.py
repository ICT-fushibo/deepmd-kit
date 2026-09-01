"""CPU contracts for DPA4 one-capacity whole-step CUDA Graph Opt3."""

from __future__ import annotations

import inspect

import numpy as np
import pytest
import torch
from ase import Atoms

from deepmd.md_stages.dpa4 import opt3
from md_benchmark.md_route import MDConfig, MDRunRequest


def _atoms() -> Atoms:
    return Atoms(
        "Cu2",
        positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        cell=np.eye(3) * 8.0,
        pbc=True,
    )


def test_route_rejects_non_whole_step_backend_before_cuda() -> None:
    request = MDRunRequest(
        model="dpa4",
        stage="opt3",
        model_path="model.pt",
        atoms=_atoms(),
        config=MDConfig(steps=1, observation_steps=(1,), dtype="float64"),
        backend="model-only-cuda-graph",
    )
    with pytest.raises(ValueError, match="whole-step-cuda-graph"):
        opt3.run_md(request)


@pytest.mark.parametrize("integrator", ["berendsen", "nose_hoover_chain"])
def test_route_accepts_both_nvt_integrators_until_cuda(
    integrator: str, monkeypatch
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    request = MDRunRequest(
        model="dpa4",
        stage="opt3",
        model_path="model.pt",
        atoms=_atoms(),
        config=MDConfig(
            steps=1,
            observation_steps=(1,),
            dtype="float64",
            integrator=integrator,
        ),
        backend="whole-step-cuda-graph",
    )
    with pytest.raises(RuntimeError, match=r"requires config\.device"):
        opt3.run_md(request)


def test_route_dispatches_dpa4_opt3(monkeypatch) -> None:
    import deepmd.md_route as route

    request = MDRunRequest(
        model="dpa4",
        stage="opt3",
        model_path="model.pt",
        atoms=_atoms(),
        config=MDConfig(steps=1, observation_steps=(1,), dtype="float64"),
        backend="whole-step-cuda-graph",
    )
    called = {}

    def fake_dispatch(received, *, module_prefix):
        called["request"] = received
        called["module_prefix"] = module_prefix
        return object()

    monkeypatch.setattr(route, "run_optimized_stage", fake_dispatch)
    assert route.run_md(request) is not None
    assert called == {
        "request": request,
        "module_prefix": "deepmd.md_stages.dpa4",
    }


def test_capacity_plan_uses_esen_uniform_cap_and_alignment() -> None:
    plan = opt3._resolve_capacity_plan(
        atom_count=32,
        initial_max_neighbors=71,
        capacity_factor=1.10,
        capacity_headroom=1,
        capacity_alignment=8,
        explicit_edge_capacity=None,
        explicit_search_capacity=None,
    )
    assert plan.neighbors_per_atom == 80
    assert plan.candidate_slots == 2560
    assert plan.edge_capacity == 2560


def test_total_capacity_from_probe_is_converted_without_truncation() -> None:
    plan = opt3._resolve_capacity_plan(
        atom_count=32,
        initial_max_neighbors=85,
        capacity_factor=1.10,
        capacity_headroom=1,
        capacity_alignment=8,
        explicit_edge_capacity=2816,
        explicit_search_capacity=None,
    )
    assert plan.neighbors_per_atom == 96
    assert plan.candidate_slots == plan.edge_capacity == 3072
    assert plan.source == "total-edge-probe-plus-initial-per-atom"
    assert plan.guarded_initial_neighbors == 96
    assert plan.total_edge_derived_neighbors == 88

    explicit = opt3._resolve_capacity_plan(
        atom_count=32,
        initial_max_neighbors=89,
        capacity_factor=1.10,
        capacity_headroom=1,
        capacity_alignment=8,
        explicit_edge_capacity=2816,
        explicit_search_capacity=104,
    )
    assert explicit.neighbors_per_atom == 104
    assert explicit.edge_capacity == 3328
    assert explicit.source == "total-edge-and-per-atom-probe"

    with pytest.raises(ValueError, match="initial per-atom neighbor count"):
        opt3._resolve_capacity_plan(
            atom_count=32,
            initial_max_neighbors=97,
            capacity_factor=1.10,
            capacity_headroom=1,
            capacity_alignment=8,
            explicit_edge_capacity=2816,
            explicit_search_capacity=96,
        )


def test_total_probe_does_not_apply_a_second_alignment_guard() -> None:
    plan = opt3._resolve_capacity_plan(
        atom_count=108,
        initial_max_neighbors=80,
        capacity_factor=1.10,
        capacity_headroom=1,
        capacity_alignment=8,
        explicit_edge_capacity=8960,
        explicit_search_capacity=None,
    )

    assert plan.total_edge_derived_neighbors == 88
    assert plan.guarded_initial_neighbors == 88
    assert plan.neighbors_per_atom == 88
    assert plan.edge_capacity == 9504


def test_per_atom_probe_is_authoritative_for_dense_heterogeneous_system() -> None:
    plan = opt3._resolve_capacity_plan(
        atom_count=108,
        initial_max_neighbors=80,
        capacity_factor=1.10,
        capacity_headroom=1,
        capacity_alignment=8,
        explicit_edge_capacity=8960,
        explicit_search_capacity=112,
    )

    assert plan.neighbors_per_atom == 112
    assert plan.edge_capacity == 12096
    assert plan.source == "total-edge-and-per-atom-probe"


def test_fixed_edge_schema_uses_distributed_safe_sinks() -> None:
    coord = torch.tensor(
        [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]],
        dtype=torch.float64,
        device="cpu",
    )
    atype = torch.zeros((1, 2), dtype=torch.long, device="cpu")
    neighbor_matrix = torch.tensor(
        [[1, 0], [0, 1]], dtype=torch.long, device="cpu"
    )
    num_neighbors = torch.tensor([1, 2], dtype=torch.long, device="cpu")
    shifts = torch.zeros((2, 2, 3), dtype=torch.long, device="cpu")

    schema = opt3._fixed_edge_schema_from_neighbor_matrix(
        coord=coord,
        atype=atype,
        cell=torch.eye(3, dtype=torch.float64, device="cpu").reshape(1, 3, 3)
        * 8.0,
        neighbor_matrix=neighbor_matrix,
        num_neighbors=num_neighbors,
        shifts=shifts,
        rcut=3.0,
        edge_capacity=6,
    )

    assert schema.edge_index.shape == (2, 6)
    assert schema.edge_vec.shape == (6, 3)
    assert schema.edge_mask.tolist() == [True, False, True, False, False, False]
    # Invalid candidate slots remain distributed over their center atoms; the
    # explicit tail rotates over both atoms rather than concentrating on zero.
    assert schema.edge_index[:, 1].tolist() == [0, 0]
    assert schema.edge_index[:, 3].tolist() == [1, 1]
    assert schema.edge_index[:, 4:].tolist() == [[0, 1], [0, 1]]
    assert torch.all(schema.edge_vec[~schema.edge_mask, 0] > 3.0)
    assert torch.all(torch.isfinite(schema.edge_vec))


def test_capture_scope_contains_builder_model_integrator_and_state_update() -> None:
    capture_source = inspect.getsource(
        opt3.DPA4WholeStepGraph._capture_whole_step_graph
    )
    step_source = inspect.getsource(opt3.DPA4WholeStepGraph._step_body)
    build_source = inspect.getsource(
        opt3.DPA4WholeStepGraph._build_fixed_edge_schema
    )
    replay_source = inspect.getsource(opt3.DPA4WholeStepGraph.step)

    assert "torch.cuda.graph" in capture_source
    assert "self._step_body()" in capture_source
    builder_source = inspect.getsource(opt3._FixedShapeDPA4NeighborBuilder.build)
    assert "self._fixed_builder.build" in build_source
    assert "nv_search_matrix_fixed" not in build_source
    assert "torch.topk" in builder_source
    assert "nvalchemi" not in builder_source
    assert "_evaluate_positions(positions)" in step_source
    assert "GPUVelocityVerletBerendsen" in step_source
    assert "GPUNoseHooverChain" in step_source
    assert "state.positions.copy_" in step_source
    assert "state.momenta.copy_" in step_source
    assert "self._graph.replay()" in replay_source
    assert "fallback" not in replay_source.lower()


def test_device_overflow_status_is_recorded_and_rejected() -> None:
    build_source = inspect.getsource(
        opt3.DPA4WholeStepGraph._build_fixed_edge_schema
    )
    check_source = inspect.getsource(opt3.DPA4WholeStepGraph.raise_if_overflow)

    assert "num_neighbors.max()" in build_source
    assert "self._overflow_flag.logical_or_" in build_source
    assert "self._overflow_count.add_" in build_source
    assert "torch._assert_async" in build_source
    assert "increase" in check_source.lower()


def test_fixed_candidate_builder_matches_simple_periodic_topology() -> None:
    builder = opt3._FixedShapeDPA4NeighborBuilder(
        num_atoms=2,
        cell=torch.eye(3, dtype=torch.float64, device="cpu") * 8.0,
        cutoff=3.0,
        neighbors_per_atom=2,
    )
    coord, sources, counts, shifts = builder.build(
        torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            dtype=torch.float64,
            device="cpu",
        )
    )

    assert coord.shape == (1, 2, 3)
    assert counts.tolist() == [1, 1]
    assert sources.tolist() == [[1, 0], [0, 1]]
    assert torch.equal(shifts, torch.zeros_like(shifts))


def test_validation_distinguishes_hard_state_contract_and_warning_tolerance() -> None:
    capture_source = inspect.getsource(
        opt3.DPA4WholeStepGraph._capture_whole_step_graph
    )
    snapshot_source = inspect.getsource(opt3.DPA4WholeStepGraph._state_snapshot)
    route_source = inspect.getsource(opt3.run_md)

    assert 'snapshot["eta"]' in snapshot_source
    assert 'snapshot["p_eta"]' in snapshot_source
    assert "state contract mismatch" in capture_source
    assert "FloatingPointError" in capture_source
    assert "validation_within_tolerance" in capture_source
    assert '"graph_numerical_validation_failure_policy": "warning-only"' in (
        route_source
    )


def test_timer_counts_initial_force_and_one_replay_per_md_step() -> None:
    route_source = inspect.getsource(opt3.run_md)

    assert "_run_measured_loop" in route_source
    assert '"initial_force_evaluations": 1' in route_source
    assert '"total_force_evaluations": request.config.steps + 1' in route_source
    assert "runner.production_replays != request.config.steps" in route_source


def test_whole_step_hot_replay_has_no_host_or_numpy_conversion() -> None:
    source = inspect.getsource(opt3.DPA4WholeStepGraph.step)
    assert ".cpu(" not in source
    assert ".numpy(" not in source
    assert ".item(" not in source


def test_opt3_keeps_compile_amp_tf32_and_fusion_disabled() -> None:
    assert opt3._DEEPMD_OPT1_ENV["DP_COMPILE_INFER"] == "0"
    assert opt3._DEEPMD_OPT1_ENV["DP_TF32_INFER"] == "0"
    assert opt3._DEEPMD_OPT1_ENV["DP_AMP_INFER"] == "0"
    assert opt3._DEEPMD_OPT1_ENV["DP_TRITON_INFER"] == "0"
    assert opt3._DEEPMD_OPT1_ENV["DP_CUTE_INFER"] == "0"
