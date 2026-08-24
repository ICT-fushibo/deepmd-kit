"""DPA3 Opt2: model-only CUDA Graph on top of GPU-resident MD.

The nvalchemiops neighbor search remains eager and runs before every replay.
Its fixed-shape lower-interface tensors are copied into persistent buffers;
only ``forward_common_lower`` (including conservative force/virial autograd)
and extended-output communication are captured.  The MD integrator and all
neighbor-list work are deliberately outside the graph.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch  # noqa: TID253
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from torch import Tensor  # noqa: TID253

if TYPE_CHECKING:
    from collections.abc import Iterator

from deepmd.md_stages.dpa3.opt1 import (
    _DEEPMD_OPT1_ENV,
    DPA3EnergyForceEvaluator,
    GPUMDState,
    _build_integrator,
    _configure_opt1,
    _evaluate_state,
    _run_measured_loop,
    _state_to_atoms,
)
from md_benchmark.md_route import (
    MDRunRequest,
    MDRunResult,
    validate_result,
)
from md_benchmark.performance import (
    CudaPhaseProfiler,
    performance_profile_requested,
)


class CUDAGraphInputError(RuntimeError):
    """Raised when dynamic lower-interface inputs do not fit the capture."""


class CUDAGraphValidationError(RuntimeError):
    """Raised when replay stability or eager/replay parity validation fails."""


def _copy_fixed_tensor_(name: str, destination: Tensor, source: Tensor) -> None:
    """Copy one graph input without allowing shape/dtype/device substitution."""
    if destination.shape != source.shape:
        raise CUDAGraphInputError(
            f"CUDA Graph {name} shape changed from {tuple(destination.shape)} "
            f"to {tuple(source.shape)}; increase the fixed neighbor capacity"
        )
    if destination.dtype != source.dtype:
        raise CUDAGraphInputError(
            f"CUDA Graph {name} dtype changed from {destination.dtype} "
            f"to {source.dtype}"
        )
    if destination.device != source.device:
        raise CUDAGraphInputError(
            f"CUDA Graph {name} device changed from {destination.device} "
            f"to {source.device}"
        )
    destination.copy_(source)


@dataclass
class StaticLowerInputs:
    """Persistent fixed-address inputs consumed by the captured lower model."""

    extended_coord: Tensor
    extended_atype: Tensor
    nlist: Tensor
    mapping: Tensor

    @classmethod
    def from_dynamic(
        cls,
        extended_coord: Tensor,
        extended_atype: Tensor,
        nlist: Tensor,
        mapping: Tensor,
    ) -> StaticLowerInputs:
        return cls(
            extended_coord=extended_coord.detach().clone(),
            extended_atype=extended_atype.detach().clone(),
            nlist=nlist.detach().clone(),
            mapping=mapping.detach().clone(),
        )

    @torch.no_grad()
    def copy_from_(
        self,
        extended_coord: Tensor,
        extended_atype: Tensor,
        nlist: Tensor,
        mapping: Tensor,
    ) -> None:
        _copy_fixed_tensor_("extended_coord", self.extended_coord, extended_coord)
        _copy_fixed_tensor_("extended_atype", self.extended_atype, extended_atype)
        _copy_fixed_tensor_("nlist", self.nlist, nlist)
        _copy_fixed_tensor_("mapping", self.mapping, mapping)

    def addresses(self) -> dict[str, int]:
        return {
            "extended_coord": self.extended_coord.data_ptr(),
            "extended_atype": self.extended_atype.data_ptr(),
            "nlist": self.nlist.data_ptr(),
            "mapping": self.mapping.data_ptr(),
        }

    def shapes(self) -> dict[str, list[int]]:
        return {
            "extended_coord": list(self.extended_coord.shape),
            "extended_atype": list(self.extended_atype.shape),
            "nlist": list(self.nlist.shape),
            "mapping": list(self.mapping.shape),
        }


def _maximum_abs_error(candidate: Tensor, reference: Tensor) -> float:
    if candidate.numel() == 0:
        return 0.0
    return float((candidate - reference).abs().max().item())


def _iter_jit_nodes(block: Any) -> Iterator[Any]:
    """Yield TorchScript nodes recursively, including nodes in control flow."""
    for node in block.nodes():
        yield node
        for nested in node.blocks():
            yield from _iter_jit_nodes(nested)


def _rewrite_capture_unsafe_scalar_zeros_(graph: Any) -> int:
    """Replace ``torch.tensor(0, device=cuda)`` with a device-side scalar zero.

    The released DPA3 TorchScript archive predates the source-level
    ``torch.where`` fix and materializes a CPU scalar before copying it to the
    neighbor-list device.  CUDA Graph capture rejects that pageable host copy.
    ``torch.zeros((), dtype=..., device=...)`` is mathematically identical and
    is allocated directly from the CUDA graph pool.
    """
    replacements = 0
    for node in list(_iter_jit_nodes(graph)):
        if node.kind() != "aten::tensor":
            continue
        inputs = list(node.inputs())
        if len(inputs) != 4:
            continue
        scalar = inputs[0].toIValue()
        requires_grad = inputs[3].toIValue()
        if scalar != 0 or requires_grad not in {False, None}:
            continue

        empty_shape = graph.create("prim::ListConstruct", [], 1)
        empty_shape.output().setType(torch._C.ListType.ofInts())
        empty_shape.insertBefore(node)
        none_value = graph.insertConstant(None)
        none_value.node().moveBefore(node)
        zeros = graph.create(
            "aten::zeros",
            [
                empty_shape.output(),
                inputs[1],
                none_value,
                inputs[2],
                none_value,
            ],
            1,
        )
        zeros.output().setType(node.output().type())
        zeros.insertBefore(node)
        node.output().replaceAllUsesWith(zeros.output())
        node.destroy()
        replacements += 1

    if replacements:
        torch._C._jit_pass_dce(graph)
        graph.lint()
    return replacements


def _rewrite_capture_unsafe_index_put_zeros_(graph: Any) -> int:
    """Replace boolean zero ``index_put_`` with capture-safe ``masked_fill_``.

    Older released DPA3 archives implement ``x[x == -1] = 0`` through
    ``aten::index_put_``.  CUDA implements that boolean-index assignment via a
    dynamic-shape path, which is forbidden during stream capture.  The
    neighbor-list operation has exactly one boolean mask, a scalar zero value,
    and ``accumulate=False``; ``masked_fill_`` has identical semantics without
    the capture-unsafe dynamic indexing machinery.
    """
    replacements = 0
    for node in list(_iter_jit_nodes(graph)):
        if node.kind() != "aten::index_put_":
            continue
        inputs = list(node.inputs())
        if len(inputs) != 4 or inputs[3].toIValue() is not False:
            continue
        indices = inputs[1].node()
        if indices.kind() != "prim::ListConstruct":
            continue
        masks = list(indices.inputs())
        if len(masks) != 1:
            continue
        value_node = inputs[2].node()
        if value_node.kind() not in {"aten::tensor", "aten::zeros"}:
            continue

        zero = graph.insertConstant(0)
        zero.node().moveBefore(node)
        masked_fill = graph.create(
            "aten::masked_fill_",
            [inputs[0], masks[0], zero],
            1,
        )
        masked_fill.output().setType(node.output().type())
        masked_fill.insertBefore(node)
        node.output().replaceAllUsesWith(masked_fill.output())
        node.destroy()
        replacements += 1

    if replacements:
        torch._C._jit_pass_dce(graph)
        graph.lint()
    return replacements


def _patch_released_dpa3_jit_for_capture_(model: Any) -> int:
    """Patch only Repflows TorchScript methods and flush their execution plans."""
    replacements = 0
    for module_name, module in model.named_modules():
        module_type = type(module).__name__
        qualified_name = str(getattr(getattr(module, "_c", None), "qualified_name", ""))
        if "repflow" not in f"{module_name} {module_type} {qualified_name}".lower():
            continue
        script_module = getattr(module, "_c", None)
        if script_module is None:
            continue
        for method_name in script_module._method_names():
            method = script_module._get_method(method_name)
            method_replacements = _rewrite_capture_unsafe_scalar_zeros_(
                method.graph
            )
            method_replacements += _rewrite_capture_unsafe_index_put_zeros_(
                method.graph
            )
            if method_replacements:
                method._debug_flush_compilation_cache()
                replacements += method_replacements
    return replacements


class DPA3ModelCUDAGraphEvaluator:
    """Build neighbors eagerly and replay one fixed-address DPA3 model graph."""

    def __init__(
        self,
        eager_evaluator: DPA3EnergyForceEvaluator,
        *,
        capture_warmup: int = 3,
        energy_atol: float = 1.0e-6,
        force_atol: float = 1.0e-6,
        virial_atol: float = 1.0e-5,
        profiler: CudaPhaseProfiler | None = None,
    ) -> None:
        if eager_evaluator.device.type != "cuda":
            raise ValueError("DPA3 model-only CUDA Graph requires CUDA")
        if capture_warmup < 1:
            raise ValueError("cuda_graph_capture_warmup must be at least 1")
        if min(energy_atol, force_atol, virial_atol) < 0:
            raise ValueError("CUDA Graph validation tolerances must be non-negative")

        self.eager_evaluator = eager_evaluator
        self.device = eager_evaluator.device
        self.model_dtype = eager_evaluator.model_dtype
        self.neighbor_backend = eager_evaluator.neighbor_backend
        self.profiler = profiler or CudaPhaseProfiler(
            enabled=False,
            device=self.device,
            prefix="opt2",
        )
        self.capture_warmup = int(capture_warmup)
        self.energy_atol = float(energy_atol)
        self.force_atol = float(force_atol)
        self.virial_atol = float(virial_atol)

        self._backend = eager_evaluator._backend
        self._inner = self._backend.dp.model["Default"]
        self.jit_capture_scalar_zero_rewrites = (
            _patch_released_dpa3_jit_for_capture_(self._inner)
        )
        if self._backend._uses_edge_schema:
            raise NotImplementedError(
                "DPA3 Opt2 requires the extended-coordinate lower interface"
            )
        self.static_inputs: StaticLowerInputs | None = None
        self.static_force: Tensor | None = None
        self.static_energy: Tensor | None = None
        self.static_virial: Tensor | None = None
        self.graph: torch.cuda.CUDAGraph | None = None
        self.capture_stream: torch.cuda.Stream | None = None
        self.captured = False
        self.capture_count = 0
        self.capture_wall_time_s = 0.0
        self.validation_energy_abs_error = 0.0
        self.validation_force_max_abs_error = 0.0
        self.validation_virial_max_abs_error = 0.0
        self.replay_output_addresses_stable = False
        self.static_input_addresses_stable = False
        self.validation_passed = False
        self.validation_replays = 0
        self.production_replays = 0
        self._initial_input_addresses: dict[str, int] | None = None
        self._initial_output_addresses: dict[str, int] | None = None

    def _build_neighbor_inputs(
        self, positions: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        if positions.device != self.device:
            raise ValueError(
                f"positions must remain on {self.device}, got {positions.device}"
            )
        expected_shape = (self.eager_evaluator.atom_types.shape[1], 3)
        if positions.shape != expected_shape:
            raise ValueError(
                f"positions shape {tuple(positions.shape)} does not match "
                f"{expected_shape}"
            )
        with self.profiler.phase("model_input"):
            model_positions = positions.to(dtype=self.model_dtype).unsqueeze(0)
        # This call is intentionally outside capture on every force evaluation.
        extended_coord, extended_atype, nlist, mapping = (
            self._backend._nlist_builder.build(
                model_positions,
                self.eager_evaluator.atom_types,
                self.eager_evaluator.cell,
                self._backend.rcut,
                list(self._inner.get_sel()),
            )
        )
        if mapping is None:
            raise CUDAGraphInputError(
                "DPA3 fixed-slot neighbor construction did not return mapping"
            )
        return extended_coord, extended_atype, nlist, mapping

    def _static_model_forward(self) -> tuple[Tensor, Tensor, Tensor]:
        if self.static_inputs is None:
            raise RuntimeError("CUDA Graph static inputs have not been initialized")
        # Keep importing this PyTorch-backend helper lazy, matching Opt1's
        # policy of configuring the DeepMD inference environment first.
        from deepmd.pt.model.model.transform_output import (
            communicate_extended_output,
        )

        lower_kwargs: dict[str, Any] = {
            "fparam": self.eager_evaluator.fparam,
            "aparam": self.eager_evaluator.aparam,
        }
        if self.eager_evaluator.charge_spin is not None:
            lower_kwargs["charge_spin"] = self.eager_evaluator.charge_spin

        # Conservative forces and virials are generated through autograd.grad
        # inside forward_common_lower, so grad must remain enabled in capture.
        with torch.enable_grad():
            # Released DPA3 archives sanitize padding indices in-place.  Pass a
            # captured clone so that warmup/capture/replay never corrupt the
            # persistent fixed-address input copied by the eager neighbor path.
            model_nlist = self.static_inputs.nlist.clone()
            model_lower = self._inner.forward_common_lower(
                self.static_inputs.extended_coord,
                self.static_inputs.extended_atype,
                model_nlist,
                self.static_inputs.mapping,
                **lower_kwargs,
                do_atomic_virial=False,
            )
            prediction = communicate_extended_output(
                model_lower,
                self._backend.output_def,
                self.static_inputs.mapping,
                do_atomic_virial=False,
            )
            output = {
                backend: prediction[internal]
                for internal, backend in self._backend._OUTDEF_DP2BACKEND.items()
                if prediction.get(internal) is not None
            }
            try:
                force = output["force"].reshape(-1, 3).detach()
                energy = output["energy"].reshape(-1)[0].detach()
                virial = output["virial"].reshape(3, 3).detach()
            except KeyError as exc:
                raise RuntimeError(
                    f"DPA3 CUDA Graph model did not produce {exc.args[0]!r}; "
                    f"available outputs are {sorted(output)}"
                ) from exc
        return force, energy, virial

    def capture(self, positions: Tensor) -> None:
        """Warm and capture exactly one static-shape model/autograd graph."""
        if self.captured:
            raise RuntimeError("DPA3 model CUDA Graph has already been captured")
        dynamic_inputs = self._build_neighbor_inputs(positions)
        self.static_inputs = StaticLowerInputs.from_dynamic(*dynamic_inputs)
        self._initial_input_addresses = self.static_inputs.addresses()

        current_stream = torch.cuda.current_stream(self.device)
        side_stream = torch.cuda.Stream(device=self.device)
        self.capture_stream = side_stream
        side_stream.wait_stream(current_stream)
        reference: tuple[Tensor, Tensor, Tensor] | None = None
        with torch.cuda.stream(side_stream):
            for _ in range(self.capture_warmup):
                reference = self._static_model_forward()
            assert reference is not None
            reference = tuple(value.clone() for value in reference)
        current_stream.wait_stream(side_stream)
        torch.cuda.synchronize(self.device)

        capture_started = time.perf_counter()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=side_stream):
            static_force, static_energy, static_virial = (
                self._static_model_forward()
            )
        torch.cuda.synchronize(self.device)
        self.capture_wall_time_s = time.perf_counter() - capture_started
        self.graph = graph
        self.static_force = static_force
        self.static_energy = static_energy
        self.static_virial = static_virial
        self.capture_count = 1
        self.captured = True
        self._initial_output_addresses = self.output_addresses()

        graph.replay()
        first = tuple(
            value.clone()
            for value in (static_force, static_energy, static_virial)
        )
        graph.replay()
        second = tuple(
            value.clone()
            for value in (static_force, static_energy, static_virial)
        )
        torch.cuda.synchronize(self.device)
        self.validation_replays = 2
        self.replay_output_addresses_stable = (
            self.output_addresses() == self._initial_output_addresses
        )
        self.static_input_addresses_stable = (
            self.static_inputs.addresses() == self._initial_input_addresses
        )
        if not self.replay_output_addresses_stable:
            raise CUDAGraphValidationError(
                "DPA3 CUDA Graph output addresses changed across replay"
            )
        if not self.static_input_addresses_stable:
            raise CUDAGraphValidationError(
                "DPA3 CUDA Graph input addresses changed across replay"
            )

        replay_errors = tuple(
            _maximum_abs_error(new, old)
            for new, old in zip(second, first, strict=True)
        )
        reference_errors = tuple(
            _maximum_abs_error(new, old)
            for new, old in zip(second, reference, strict=True)
        )
        self.validation_force_max_abs_error = max(
            replay_errors[0], reference_errors[0]
        )
        self.validation_energy_abs_error = max(
            replay_errors[1], reference_errors[1]
        )
        self.validation_virial_max_abs_error = max(
            replay_errors[2], reference_errors[2]
        )
        failures = []
        for name, error, tolerance in (
            ("force", self.validation_force_max_abs_error, self.force_atol),
            ("energy", self.validation_energy_abs_error, self.energy_atol),
            ("virial", self.validation_virial_max_abs_error, self.virial_atol),
        ):
            if error > tolerance:
                failures.append(f"{name} error {error:.6g} > {tolerance:.6g}")
        if failures:
            raise CUDAGraphValidationError(
                "DPA3 CUDA Graph validation failed: " + "; ".join(failures)
            )
        self.validation_passed = True

    def reset_production_stats(self) -> None:
        self.production_replays = 0

    def output_addresses(self) -> dict[str, int]:
        if (
            self.static_force is None
            or self.static_energy is None
            or self.static_virial is None
        ):
            raise RuntimeError("DPA3 CUDA Graph outputs are not initialized")
        return {
            "force": self.static_force.data_ptr(),
            "energy": self.static_energy.data_ptr(),
            "virial": self.static_virial.data_ptr(),
        }

    def __call__(self, positions: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if not self.captured or self.graph is None or self.static_inputs is None:
            raise RuntimeError("DPA3 model CUDA Graph has not been captured")
        dynamic_inputs = self._build_neighbor_inputs(positions)
        with self.profiler.phase("cuda_graph_staticize"):
            self.static_inputs.copy_from_(*dynamic_inputs)
        if self.static_inputs.addresses() != self._initial_input_addresses:
            raise CUDAGraphValidationError(
                "DPA3 CUDA Graph static input addresses changed"
            )
        with self.profiler.phase("model_energy_force"):
            self.graph.replay()
        self.production_replays += 1
        assert self.static_force is not None
        assert self.static_energy is not None
        assert self.static_virial is not None
        return self.static_force, self.static_energy, self.static_virial

    def metadata(self) -> dict[str, Any]:
        if self.static_inputs is None:
            raise RuntimeError("DPA3 CUDA Graph has no static inputs")
        return {
            "capture_scope": "model-lower-forward-conservative-autograd-only",
            "capture_includes": [
                "forward_common_lower",
                "conservative_force_virial_autograd",
                "communicate_extended_output",
            ],
            "capture_excludes": [
                "neighbor_list",
                "fixed_input_copy",
                "md_integrator",
                "statistics_and_trajectory",
            ],
            "capture_count": self.capture_count,
            "capture_warmup": self.capture_warmup,
            "capture_wall_time_s": self.capture_wall_time_s,
            "validation_replays": self.validation_replays,
            "production_replays": self.production_replays,
            "static_input_shapes": self.static_inputs.shapes(),
            "static_input_addresses_stable": self.static_input_addresses_stable,
            "replay_output_addresses_stable": self.replay_output_addresses_stable,
            "validation_passed": self.validation_passed,
            "validation_energy_abs_error": self.validation_energy_abs_error,
            "validation_force_max_abs_error": (
                self.validation_force_max_abs_error
            ),
            "validation_virial_max_abs_error": (
                self.validation_virial_max_abs_error
            ),
            "capacity_overflow_policy": "device-assert-and-fail-no-fallback",
            "jit_capture_scalar_zero_rewrites": (
                self.jit_capture_scalar_zero_rewrites
            ),
        }


def run_md(request: MDRunRequest) -> MDRunResult:
    """Run DPA3 Opt2; never substitute Opt1 when capture is unsupported."""
    if request.model != "dpa3" or request.stage != "opt2":
        raise ValueError(f"DPA3 Opt2 route received {request.model}/{request.stage}")
    if request.backend not in {"model-only-cuda-graph", "gpu-resident"}:
        raise ValueError(
            "DPA3 Opt2 requires backend='model-only-cuda-graph'; "
            f"got {request.backend!r}"
        )
    if request.config.dtype != "float64":
        raise ValueError("DPA3 Opt2 requires --dtype float64 for the MD state")
    if request.config.ensemble.lower() != "nvt":
        raise ValueError("DPA3 Opt2 supports only NVT")
    if request.atoms.constraints:
        raise NotImplementedError("DPA3 Opt2 does not support ASE constraints")
    if not bool(np.asarray(request.atoms.pbc).all()):
        raise NotImplementedError(
            "DPA3 Opt2 currently requires fully periodic structures"
        )

    _configure_opt1()
    device = torch.device(request.config.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("DPA3 Opt2 requires config.device to name a CUDA device")

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
        prefix="opt2",
    )
    configured_capacity = request.options.get("neighbor_search_capacity")
    eager_evaluator = DPA3EnergyForceEvaluator(
        atoms,
        request.model_path,
        device=device,
        neighbor_capacity_factor=float(
            request.options.get("neighbor_capacity_factor", 1.25)
        ),
        neighbor_capacity_headroom=int(
            request.options.get("neighbor_capacity_headroom", 16)
        ),
        neighbor_search_capacity=(
            None if configured_capacity is None else int(configured_capacity)
        ),
        profiler=profiler,
    )
    evaluator = DPA3ModelCUDAGraphEvaluator(
        eager_evaluator,
        capture_warmup=int(
            request.options.get(
                "cuda_graph_capture_warmup",
                request.options.get("capture_warmup", 3),
            )
        ),
        energy_atol=float(request.options.get("cuda_graph_energy_atol", 1.0e-6)),
        force_atol=float(request.options.get("cuda_graph_force_atol", 1.0e-6)),
        virial_atol=float(request.options.get("cuda_graph_virial_atol", 1.0e-5)),
        profiler=profiler,
    )
    evaluator.capture(state.positions)

    if request.config.warmup_steps:
        warmup_integrator = _build_integrator(request, masses)
        _evaluate_state(state, evaluator)
        for _ in range(request.config.warmup_steps):
            warmup_integrator.step(state, evaluator)
        torch.cuda.synchronize(device)
        state.restore_initial_(initial_state)

    evaluator.reset_production_stats()
    integrator = _build_integrator(request, masses)
    elapsed, observations, trajectory, trajectory_path = _run_measured_loop(
        request,
        state,
        evaluator,
        integrator,
        masses,
        profiler,
    )
    expected_replays = request.config.steps + 1
    if evaluator.production_replays != expected_replays:
        raise RuntimeError(
            "DPA3 Opt2 production replay count mismatch: "
            f"expected={expected_replays}, actual={evaluator.production_replays}"
        )
    final_atoms = _state_to_atoms(atoms, state)
    metadata = {
        "engine": "gpu_resident",
        "backend": "model-only-cuda-graph",
        "requested_backend": request.backend,
        "model_path": str(Path(request.model_path).resolve()),
        "integrator": request.config.integrator,
        "neighborlist_backend": evaluator.neighbor_backend,
        "neighborlist_fixed_shape": dict(
            eager_evaluator.neighbor_shape_metadata
        ),
        "neighbor_rebuilt_each_force_evaluation": True,
        "md_state_precision": "float64",
        "model_precision": str(evaluator.model_dtype).removeprefix("torch."),
        "warmup_steps": request.config.warmup_steps,
        "torch_compile": False,
        "cuda_graph": True,
        "cuda_graph_scope": "model-only",
        "kernel_fusion": False,
        "triton": False,
        "amp": False,
        "tf32": False,
        "hot_loop_numpy_roundtrip": False,
        "deepmd_inference_env": dict(_DEEPMD_OPT1_ENV),
        "cuda_graph_details": evaluator.metadata(),
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


__all__ = [
    "CUDAGraphInputError",
    "CUDAGraphValidationError",
    "DPA3ModelCUDAGraphEvaluator",
    "StaticLowerInputs",
    "run_md",
]
