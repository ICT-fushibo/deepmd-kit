"""Lightweight contracts for DPA3 model-only CUDA Graph Opt2."""

from __future__ import annotations

import inspect

import numpy as np
import pytest
import torch
from ase import Atoms

from deepmd.md_stages.dpa3 import opt2
from md_benchmark.md_route import MDConfig, MDRunRequest


_CPU = torch.device("cpu")


def _atoms() -> Atoms:
    return Atoms(
        "H2",
        positions=[[0.2, 0.1, 0.3], [0.7, 0.4, 0.2]],
        cell=np.eye(3) * 8.0,
        pbc=True,
    )


def _request(*, backend: str = "model-only-cuda-graph") -> MDRunRequest:
    return MDRunRequest(
        model="dpa3",
        stage="opt2",
        model_path="model.pth",
        atoms=_atoms(),
        config=MDConfig(steps=1, observation_steps=(1,)),
        backend=backend,
    )


def test_static_lower_inputs_reuse_addresses_and_copy_values() -> None:
    dynamic = (
        torch.randn(1, 6, 3, device=_CPU),
        torch.arange(6, device=_CPU).reshape(1, 6),
        torch.arange(8, device=_CPU).reshape(1, 2, 4),
        torch.arange(6, device=_CPU).reshape(1, 6),
    )
    static = opt2.StaticLowerInputs.from_dynamic(*dynamic)
    addresses = static.addresses()
    replacements = tuple(value + 1 for value in dynamic)

    static.copy_from_(*replacements)

    assert static.addresses() == addresses
    for name, expected in zip(
        ("extended_coord", "extended_atype", "nlist", "mapping"),
        replacements,
        strict=True,
    ):
        torch.testing.assert_close(getattr(static, name), expected)


def test_static_lower_inputs_reject_shape_change() -> None:
    dynamic = (
        torch.randn(1, 6, 3, device=_CPU),
        torch.arange(6, device=_CPU).reshape(1, 6),
        torch.arange(8, device=_CPU).reshape(1, 2, 4),
        torch.arange(6, device=_CPU).reshape(1, 6),
    )
    static = opt2.StaticLowerInputs.from_dynamic(*dynamic)
    with pytest.raises(opt2.CUDAGraphInputError, match="nlist shape changed"):
        static.copy_from_(
            dynamic[0],
            dynamic[1],
            torch.arange(10, device=_CPU).reshape(1, 2, 5),
            dynamic[3],
        )


def test_rewrite_capture_unsafe_scalar_zero_uses_device_allocation() -> None:
    function = torch.jit.CompilationUnit(
        """
        def scalar_zero_like(x: Tensor) -> Tensor:
            return torch.tensor(0, dtype=x.dtype, device=x.device)
        """
    ).scalar_zero_like
    assert any(node.kind() == "aten::tensor" for node in function.graph.nodes())

    assert opt2._rewrite_capture_unsafe_scalar_zeros_(function.graph) == 1
    assert not any(node.kind() == "aten::tensor" for node in function.graph.nodes())
    actual = function(torch.ones(3, dtype=torch.int64, device=_CPU))
    torch.testing.assert_close(
        actual,
        torch.zeros((), dtype=torch.int64, device=_CPU),
    )


def test_rewrite_capture_unsafe_boolean_index_put_uses_masked_fill() -> None:
    function = torch.jit.CompilationUnit(
        """
        def sanitize_padding(x: Tensor) -> Tensor:
            x[x == -1] = 0
            return x
        """
    ).sanitize_padding
    assert any(node.kind() == "aten::index_put_" for node in function.graph.nodes())

    opt2._rewrite_capture_unsafe_scalar_zeros_(function.graph)
    assert opt2._rewrite_capture_unsafe_index_put_zeros_(function.graph) == 1
    function._debug_flush_compilation_cache()
    assert not any(node.kind() == "aten::index_put_" for node in function.graph.nodes())
    assert any(node.kind() == "aten::masked_fill_" for node in function.graph.nodes())
    values = torch.tensor([-1, 4, -1], dtype=torch.int64)
    torch.testing.assert_close(
        function(values),
        torch.tensor([0, 4, 0], dtype=torch.int64),
    )


def test_replay_mock_reuses_static_input_and_output_addresses() -> None:
    dynamic = (
        torch.randn(1, 6, 3, device=_CPU),
        torch.arange(6, device=_CPU).reshape(1, 6),
        torch.arange(8, device=_CPU).reshape(1, 2, 4),
        torch.arange(6, device=_CPU).reshape(1, 6),
    )
    evaluator = object.__new__(opt2.DPA3ModelCUDAGraphEvaluator)
    evaluator.captured = True
    evaluator.static_inputs = opt2.StaticLowerInputs.from_dynamic(*dynamic)
    evaluator._initial_input_addresses = evaluator.static_inputs.addresses()
    evaluator.static_force = torch.tensor([[1.0, 2.0, 3.0]], device=_CPU)
    evaluator.static_energy = torch.tensor(4.0, device=_CPU)
    evaluator.static_virial = torch.eye(3, device=_CPU)
    evaluator.production_replays = 0
    evaluator.profiler = opt2.CudaPhaseProfiler(
        enabled=False, device=_CPU
    )
    evaluator._build_neighbor_inputs = lambda _positions: tuple(
        value + 1 for value in dynamic
    )

    class FakeGraph:
        calls = 0

        def replay(self) -> None:
            self.calls += 1

    evaluator.graph = FakeGraph()
    input_addresses = evaluator.static_inputs.addresses()
    output_addresses = evaluator.output_addresses()

    force, energy, virial = evaluator(torch.zeros(2, 3, device=_CPU))

    assert evaluator.graph.calls == 1
    assert evaluator.production_replays == 1
    assert evaluator.static_inputs.addresses() == input_addresses
    assert evaluator.output_addresses() == output_addresses
    torch.testing.assert_close(
        force,
        torch.tensor([[1.0, 2.0, 3.0]], device=_CPU),
    )
    torch.testing.assert_close(energy, torch.tensor(4.0, device=_CPU))
    torch.testing.assert_close(virial, torch.eye(3, device=_CPU))


def test_route_rejects_wrong_backend_before_cuda() -> None:
    with pytest.raises(ValueError, match="model-only-cuda-graph"):
        opt2.run_md(_request(backend="native"))


def test_route_dispatches_dpa3_opt2(monkeypatch) -> None:
    import deepmd.md_route as route

    request = _request()
    called = {}

    def fake_dispatch(received, *, module_prefix):
        called["request"] = received
        called["module_prefix"] = module_prefix
        return object()

    monkeypatch.setattr(route, "run_optimized_stage", fake_dispatch)
    result = route.run_md(request)
    assert result is not None
    assert called == {
        "request": request,
        "module_prefix": "deepmd.md_stages.dpa3",
    }


def test_capture_scope_keeps_neighbor_builder_outside_graph() -> None:
    build_source = inspect.getsource(
        opt2.DPA3ModelCUDAGraphEvaluator._build_neighbor_inputs
    )
    capture_source = inspect.getsource(opt2.DPA3ModelCUDAGraphEvaluator.capture)
    model_source = inspect.getsource(
        opt2.DPA3ModelCUDAGraphEvaluator._static_model_forward
    )
    call_source = inspect.getsource(opt2.DPA3ModelCUDAGraphEvaluator.__call__)

    assert "_nlist_builder.build" in build_source
    assert "torch.cuda.graph" in capture_source
    assert "_nlist_builder.build" not in capture_source
    assert "forward_common_lower" in model_source
    assert "torch.enable_grad" in model_source
    assert call_source.index("_build_neighbor_inputs") < call_source.index(
        "graph.replay"
    )
    assert "eager_evaluator(positions)" not in call_source


def test_opt2_source_disables_fallback_compile_amp_tf32_and_fusion() -> None:
    source = inspect.getsource(opt2.run_md)
    assert '"cuda_graph": True' in source
    assert '"cuda_graph_scope": "model-only"' in source
    assert '"torch_compile": False' in source
    assert '"kernel_fusion": False' in source
    assert '"amp": False' in source
    assert '"tf32": False' in source
    assert "DPA3EnergyForceEvaluator.__call__" not in source
