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


class _ObscuredActivation(torch.nn.Module):
    def forward(self, x: torch.Tensor, threshold: float) -> torch.Tensor:
        silu = torch.nn.functional.silu(x)
        mask = x > threshold
        if torch.any(mask):
            tail = torch.tanh(x - threshold)
            return torch.where(x < threshold, silu, tail)
        return silu


class _ObscuredContainer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.opaque = _ObscuredActivation()

    def forward(self, x: torch.Tensor, threshold: float) -> torch.Tensor:
        return self.opaque(x, threshold)


class _UnhandledActivation(torch.nn.Module):
    """Independent JIT type for the fail-closed rewrite test."""

    def forward(self, x: torch.Tensor, threshold: float) -> torch.Tensor:
        silu = torch.nn.functional.silu(x)
        mask = x > threshold
        if torch.any(mask):
            tail = torch.tanh(x - threshold)
            return torch.where(x < threshold, silu, tail)
        return silu


class _UnhandledContainer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.opaque = _UnhandledActivation()

    def forward(self, x: torch.Tensor, threshold: float) -> torch.Tensor:
        return self.opaque(x, threshold)


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
    values = torch.tensor([-1, 4, -1], dtype=torch.int64, device=_CPU)
    torch.testing.assert_close(
        function(values),
        torch.tensor([0, 4, 0], dtype=torch.int64, device=_CPU),
    )


@pytest.mark.parametrize(
    "assignment",
    [
        "2",
        "torch.zeros(2, device=x.device)",
    ],
)
def test_rewrite_boolean_index_put_rejects_non_scalar_zero(assignment) -> None:
    function = torch.jit.CompilationUnit(
        f"""
        def assign_value(x: Tensor) -> Tensor:
            x[x == -1] = {assignment}
            return x
        """
    ).assign_value
    assert any(node.kind() == "aten::index_put_" for node in function.graph.nodes())

    assert opt2._rewrite_capture_unsafe_index_put_zeros_(function.graph) == 0
    assert any(node.kind() == "aten::index_put_" for node in function.graph.nodes())
    assert not any(
        node.kind() == "aten::masked_fill_" for node in function.graph.nodes()
    )


def test_rewrite_boolean_index_put_accepts_scalar_zeros() -> None:
    function = torch.jit.CompilationUnit(
        """
        def assign_zero(x: Tensor) -> Tensor:
            x[x == -1] = torch.zeros((), device=x.device)
            return x
        """
    ).assign_zero

    assert opt2._rewrite_capture_unsafe_index_put_zeros_(function.graph) == 1
    assert not any(node.kind() == "aten::index_put_" for node in function.graph.nodes())
    assert any(
        node.kind() == "aten::masked_fill_" for node in function.graph.nodes()
    )


def test_rewrite_capture_unsafe_tensor_conditional_inlines_where_branch() -> None:
    function = torch.jit.CompilationUnit(
        """
        def custom_activation(x: Tensor, threshold: float) -> Tensor:
            silu = torch.nn.functional.silu(x)
            mask = x > threshold
            if torch.any(mask):
                tail = torch.tanh(x - threshold)
                return torch.where(x < threshold, silu, tail)
            return silu
        """
    ).custom_activation
    assert any(node.kind() == "prim::If" for node in function.graph.nodes())

    assert opt2._rewrite_capture_unsafe_tensor_conditionals_(function.graph) == 1
    function._debug_flush_compilation_cache()
    assert not any(node.kind() == "prim::If" for node in function.graph.nodes())
    values = torch.tensor([-2.0, 1.0], device=_CPU)
    expected = torch.where(
        values < 0.0,
        torch.nn.functional.silu(values),
        torch.tanh(values),
    )
    torch.testing.assert_close(function(values, 0.0), expected)


def test_model_patch_does_not_depend_on_archive_module_names() -> None:
    model = torch.jit.script(_ObscuredContainer())
    # Materialize the parent execution plan before mutating its nested child.
    model(torch.tensor([-1.0, 1.0], device=_CPU), 0.0)

    stats = opt2._patch_released_dpa3_jit_for_capture_(model)

    assert stats.methods_scanned == 2
    assert stats.tensor_conditionals == 1
    assert stats.total == 1
    assert stats.caches_flushed == 2
    child_graph = model.opaque._c._get_method("forward").graph
    assert not any(node.kind() == "prim::If" for node in child_graph.nodes())
    values = torch.tensor([-2.0, 1.0], device=_CPU)
    expected = torch.where(
        values < 0.0,
        torch.nn.functional.silu(values),
        torch.tanh(values),
    )
    torch.testing.assert_close(model(values, 0.0), expected)


def test_model_patch_rejects_unhandled_tensor_conditional(monkeypatch) -> None:
    # TorchScript method graphs are shared by instances of the same scripted
    # Python type.  The preceding success-path test intentionally mutates the
    # _ObscuredContainer graph, so this fail-closed test needs an independent
    # scripted type instead of depending on pytest execution order.
    model = torch.jit.script(_UnhandledContainer())
    monkeypatch.setattr(
        opt2,
        "_rewrite_capture_unsafe_tensor_conditionals_",
        lambda _graph: 0,
    )

    with pytest.raises(RuntimeError, match="left 1 capture-unsafe"):
        opt2._patch_released_dpa3_jit_for_capture_(model)


def test_fixed_neighbor_slot_configuration() -> None:
    kwargs = {"capacity_factor": 1.25, "capacity_headroom": 16}
    assert opt2._resolve_fixed_neighbor_slots(None, 1200, 100, **kwargs) == 125
    assert opt2._resolve_fixed_neighbor_slots(None, 1200, 40, **kwargs) == 56
    assert opt2._resolve_fixed_neighbor_slots(128, 1200, 100, **kwargs) == 128
    with pytest.raises(ValueError, match="must be positive"):
        opt2._resolve_fixed_neighbor_slots(0, 1200, 0, **kwargs)
    with pytest.raises(ValueError, match="smaller than the initial"):
        opt2._resolve_fixed_neighbor_slots(99, 1200, 100, **kwargs)
    with pytest.raises(ValueError, match="exceeds checkpoint"):
        opt2._resolve_fixed_neighbor_slots(1201, 1200, 100, **kwargs)


def test_static_dynamic_indices_and_edge_reduction_match_compact() -> None:
    from deepmd.pt.model.descriptor.repflows import _get_static_graph_index
    from deepmd.pt.model.network.utils import aggregate, get_graph_index

    # Padding is already sanitized to atom index 0, matching repflows.forward.
    nlist = torch.tensor([[[1, 2, 0], [0, 0, 0]]], device=_CPU)
    nlist_mask = torch.tensor(
        [[[True, True, False], [True, False, False]]], device=_CPU
    )
    a_mask = nlist_mask[:, :, :2]
    compact_edge, compact_angle = get_graph_index(
        nlist,
        nlist_mask,
        a_mask,
        3,
        True,
    )
    static_edge, static_angle = _get_static_graph_index(
        nlist,
        a_mask,
        3,
        True,
    )

    assert static_edge.shape == (2, 6)
    assert static_angle.shape == (3, 8)
    torch.testing.assert_close(
        static_edge[:, nlist_mask.reshape(-1)], compact_edge
    )

    dense_message = torch.tensor(
        [[1.0], [2.0], [50.0], [3.0], [60.0], [70.0]], device=_CPU
    )
    dense_message = dense_message * nlist_mask.reshape(-1, 1)
    compact_message = dense_message[nlist_mask.reshape(-1)]
    compact_reduced = aggregate(
        compact_message,
        compact_edge[0],
        average=False,
        num_owner=2,
    )
    static_reduced = aggregate(
        dense_message,
        static_edge[0],
        average=False,
        num_owner=2,
    )
    torch.testing.assert_close(static_reduced, compact_reduced)

    pair_mask = (
        a_mask[:, :, :, None] & a_mask[:, :, None, :]
    ).reshape(-1)
    dense_angle_message = torch.arange(
        1, 9, dtype=torch.float32, device=_CPU
    ).reshape(-1, 1)
    dense_angle_message = dense_angle_message * pair_mask.reshape(-1, 1)
    compact_angle_reduced = aggregate(
        dense_angle_message[pair_mask],
        compact_angle[1],
        average=False,
        num_owner=compact_edge.shape[1],
    )
    static_angle_reduced = aggregate(
        dense_angle_message,
        static_angle[1],
        average=False,
        num_owner=static_edge.shape[1],
    )
    torch.testing.assert_close(
        static_angle_reduced[nlist_mask.reshape(-1)],
        compact_angle_reduced,
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
