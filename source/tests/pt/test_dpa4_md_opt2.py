"""Contracts for DPA4 strict model-only CUDA Graph Opt2 MD."""

from __future__ import annotations

import inspect
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from ase import Atoms

from deepmd.dpmodel.utils.neighbor_list import EdgeNeighborList
from deepmd.md_stages.dpa4 import opt2
from md_benchmark.md_route import MDConfig, MDRunRequest


_CPU = torch.device("cpu")


def _atoms() -> Atoms:
    return Atoms(
        "Cu2",
        positions=[[0.0, 0.0, 0.0], [1.8, 1.8, 1.8]],
        cell=np.eye(3) * 7.2,
        pbc=True,
    )


def _schema(edge_count: int) -> EdgeNeighborList:
    return EdgeNeighborList(
        coord=torch.arange(6, dtype=torch.float64, device=_CPU).reshape(1, 2, 3),
        atype=torch.zeros((1, 2), dtype=torch.long, device=_CPU),
        edge_index=torch.zeros((2, edge_count), dtype=torch.long, device=_CPU),
        edge_vec=torch.arange(
            edge_count * 3, dtype=torch.float64, device=_CPU
        ).reshape(edge_count, 3),
        edge_scatter_index=torch.zeros(
            (2, edge_count), dtype=torch.long, device=_CPU
        ),
        edge_mask=torch.ones(edge_count, dtype=torch.bool, device=_CPU),
    )


def test_route_rejects_non_model_graph_backend_before_cuda() -> None:
    request = MDRunRequest(
        model="dpa4",
        stage="opt2",
        model_path="model.pt",
        atoms=_atoms(),
        config=MDConfig(steps=1, observation_steps=(1,), dtype="float64"),
        backend="native",
    )
    with pytest.raises(ValueError, match="model-only-cuda-graph"):
        opt2.run_md(request)


@pytest.mark.parametrize("backend", ["model-only-cuda-graph", "gpu-resident"])
def test_route_accepts_canonical_and_compatibility_backend_until_cuda(
    backend: str,
    monkeypatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    request = MDRunRequest(
        model="dpa4",
        stage="opt2",
        model_path="model.pt",
        atoms=_atoms(),
        config=MDConfig(steps=1, observation_steps=(1,), dtype="float64"),
        backend=backend,
    )
    with pytest.raises(RuntimeError, match=r"requires config\.device"):
        opt2.run_md(request)


def test_route_dispatches_dpa4_opt2(monkeypatch) -> None:
    import deepmd.md_route as route

    request = MDRunRequest(
        model="dpa4",
        stage="opt2",
        model_path="model.pt",
        atoms=_atoms(),
        config=MDConfig(steps=1, observation_steps=(1,), dtype="float64"),
        backend="model-only-cuda-graph",
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


def test_edge_capacity_is_preflighted_without_silent_truncation() -> None:
    assert (
        opt2._resolve_edge_capacity(
            100,
            capacity_factor=1.25,
            capacity_headroom=10,
            explicit_capacity=None,
        )
        == 125
    )
    assert (
        opt2._resolve_edge_capacity(
            100,
            capacity_factor=1.0,
            capacity_headroom=16,
            explicit_capacity=160,
        )
        == 160
    )
    with pytest.raises(ValueError, match="smaller than the initial edge count"):
        opt2._resolve_edge_capacity(
            100,
            capacity_factor=1.0,
            capacity_headroom=0,
            explicit_capacity=99,
        )


def test_max_real_neighbors_ignores_masked_sink_edges() -> None:
    schema = _schema(6)
    schema.edge_index[1] = torch.tensor([0, 0, 1, 1, 1, 0])
    schema.edge_mask[:] = torch.tensor([True, True, True, True, True, False])

    required = opt2._max_real_neighbors(schema, atom_count=2)

    assert required.dtype is torch.int64
    assert required.item() == 3


def test_static_edge_buffers_keep_addresses_and_mask_unused_tail() -> None:
    buffers = opt2._StaticEdgeGraphInputs.allocate(_schema(5), capacity=8)
    initial_addresses = buffers.addresses()

    buffers.copy_schema_(_schema(3))

    assert buffers.addresses() == initial_addresses
    assert buffers.edge_mask.tolist() == [
        True,
        True,
        True,
        False,
        False,
        False,
        False,
        False,
    ]
    torch.testing.assert_close(
        buffers.edge_vec[:3],
        _schema(3).edge_vec,
    )


def test_static_edge_buffers_raise_on_topology_overflow() -> None:
    buffers = opt2._StaticEdgeGraphInputs.allocate(_schema(4), capacity=5)
    with pytest.raises(RuntimeError, match="edge capacity exceeded"):
        buffers.copy_schema_(_schema(6))


def test_capture_boundary_is_lower_model_only_and_has_no_fallback() -> None:
    init_source = inspect.getsource(opt2.DPA4ModelOnlyGraphEvaluator.__init__)
    capture_source = inspect.getsource(
        opt2.DPA4ModelOnlyGraphEvaluator._capture_model_graph
    )
    lower_source = inspect.getsource(opt2.DPA4ModelOnlyGraphEvaluator._run_lower_model)
    hot_source = inspect.getsource(opt2.DPA4ModelOnlyGraphEvaluator.__call__)

    assert "torch.cuda.CUDAGraph" in init_source
    assert "torch.cuda.graph" in capture_source
    assert "forward_lower" in lower_source
    assert "_build_edge_schema" not in capture_source
    assert "_build_edge_schema" in hot_source
    assert hot_source.index("_build_edge_schema") < hot_source.index("_graph.replay")
    assert "fallback" not in hot_source.lower()


def test_capture_requires_eager_and_replay_numerical_validation() -> None:
    source = inspect.getsource(
        opt2.DPA4ModelOnlyGraphEvaluator._capture_model_graph
    )

    assert "eager_reference" in source
    assert source.count("self._graph.replay()") == 2
    assert "replay_errors" in source
    assert "validation_errors" in source
    assert "self.validation_passed = True" in source


def test_hot_path_builds_and_copies_edges_before_graph_replay(monkeypatch) -> None:
    events = []

    class FakeProfiler:
        def phase(self, _name):
            return nullcontext()

    class FakeStatic:
        def addresses(self):
            return (11, 12, 13, 14, 15, 16)

        def copy_schema_(self, _schema):
            events.append("copy")
            return 4

    evaluator = object.__new__(opt2.DPA4ModelOnlyGraphEvaluator)
    evaluator.device = _CPU
    evaluator.model_dtype = torch.float64
    evaluator.atom_types = torch.zeros((1, 2), dtype=torch.long, device=_CPU)
    evaluator.profiler = FakeProfiler()
    evaluator._static = FakeStatic()
    evaluator._input_addresses = evaluator._static.addresses()
    evaluator._captured_force = torch.ones(
        (2, 3), dtype=torch.float64, device=_CPU
    )
    evaluator._captured_energy = torch.tensor(
        -1.25, dtype=torch.float64, device=_CPU
    )
    evaluator._captured_virial = torch.eye(3, dtype=torch.float64, device=_CPU)
    evaluator._output_addresses = tuple(
        value.data_ptr()
        for value in (
            evaluator._captured_force,
            evaluator._captured_energy,
            evaluator._captured_virial,
        )
    )
    evaluator._graph = SimpleNamespace(replay=lambda: events.append("replay"))
    evaluator.production_replays = 0
    evaluator._track_neighbor_capacity = False
    evaluator._build_edge_schema = lambda _positions: (
        events.append("build") or _schema(4)
    )
    monkeypatch.setattr(torch.cuda, "device", lambda _device: nullcontext())

    force, energy, virial = evaluator(
        torch.zeros((2, 3), dtype=torch.float64, device=_CPU)
    )

    assert events == ["build", "copy", "replay"]
    assert evaluator.production_replays == 1
    torch.testing.assert_close(force, evaluator._captured_force)
    torch.testing.assert_close(energy, evaluator._captured_energy)
    torch.testing.assert_close(virial, evaluator._captured_virial)


def test_opt2_hot_force_path_has_no_host_or_numpy_conversion() -> None:
    source = inspect.getsource(opt2.DPA4ModelOnlyGraphEvaluator.__call__)
    assert ".cpu(" not in source
    assert ".numpy(" not in source
    assert ".item(" not in source


def test_opt2_keeps_compile_amp_tf32_and_fusion_disabled() -> None:
    assert opt2._DEEPMD_OPT1_ENV["DP_COMPILE_INFER"] == "0"
    assert opt2._DEEPMD_OPT1_ENV["DP_TF32_INFER"] == "0"
    assert opt2._DEEPMD_OPT1_ENV["DP_AMP_INFER"] == "0"
    assert opt2._DEEPMD_OPT1_ENV["DP_TRITON_INFER"] == "0"
    assert opt2._DEEPMD_OPT1_ENV["DP_CUTE_INFER"] == "0"
