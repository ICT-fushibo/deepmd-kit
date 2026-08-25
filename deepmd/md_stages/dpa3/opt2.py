"""DPA3 Opt2: model-only CUDA Graph on top of GPU-resident MD.

The nvalchemiops neighbor search remains eager and runs before every replay.
Its fixed-shape lower-interface tensors are copied into persistent buffers;
only ``forward_common_lower`` (including conservative force/virial autograd)
and extended-output communication are captured.  The MD integrator and all
neighbor-list work are deliberately outside the graph.
"""

from __future__ import annotations

import copy
import math
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


_MISSING = object()


def _require_legacy_value(
    options: dict[str, Any], key: str, expected: Any
) -> bool:
    """Remove one legacy option only when current code has equal semantics."""
    value = options.pop(key, _MISSING)
    if value is _MISSING:
        return False
    if value != expected:
        raise ValueError(
            "DPA3 Opt2 cannot rehydrate checkpoint option "
            f"{key}={value!r}; the current implementation only supports "
            f"the compatibility value {expected!r}"
        )
    return True


def _map_legacy_alias(
    options: dict[str, Any], legacy_key: str, current_key: str
) -> bool:
    """Translate one renamed option while rejecting conflicting definitions."""
    value = options.pop(legacy_key, _MISSING)
    if value is _MISSING:
        return False
    if current_key in options and options[current_key] != value:
        raise ValueError(
            "DPA3 Opt2 checkpoint defines conflicting options: "
            f"{legacy_key}={value!r} and "
            f"{current_key}={options[current_key]!r}"
        )
    options[current_key] = value
    return True


def _normalize_released_dpa3_model_params(
    model_params: dict[str, Any],
) -> tuple[dict[str, Any], tuple[str, ...]]:
    """Normalize the released DPA-3.1 schema to the current public schema.

    The released checkpoint was frozen from an experimental descriptor schema.
    Most extra switches are disabled and therefore have no parameters or forward
    effect.  Two active legacy names have exact current equivalents:
    ``use_env_envelope`` is now ``use_exp_switch`` and ``edge_use_dist`` is now
    ``edge_init_use_dist``.  Anything that would activate an unsupported branch
    is rejected instead of being silently ignored; compact/padded E/F/virial
    parity remains the final scientific guard.
    """
    normalized = copy.deepcopy(model_params)
    descriptor = normalized.get("descriptor")
    if not isinstance(descriptor, dict):
        raise ValueError("DPA3 Opt2 checkpoint has no descriptor configuration")
    repflow = descriptor.get("repflow")
    if not isinstance(repflow, dict):
        raise ValueError("DPA3 Opt2 checkpoint has no descriptor.repflow mapping")

    consumed: list[str] = []
    if _require_legacy_value(descriptor, "use_torch_embed", False):
        consumed.append("descriptor.use_torch_embed")
    if _map_legacy_alias(repflow, "use_env_envelope", "use_exp_switch"):
        consumed.append("descriptor.repflow.use_env_envelope->use_exp_switch")
    if _map_legacy_alias(repflow, "edge_use_dist", "edge_init_use_dist"):
        consumed.append("descriptor.repflow.edge_use_dist->edge_init_use_dist")

    no_op_values = {
        "smooth_angle_init": False,
        "angle_init_use_sin": False,
        "angle_multi_freq": None,
        "use_new_sw": False,
        "update_dihedral": False,
        "use_ffn_node_edge_message": False,
        "use_ffn_edge_edge_message": False,
        "use_ffn_edge_angle_message": False,
        "use_ffn_angle_angle_message": False,
        "edge_use_concat_rbf": False,
        "edge_use_rbf": False,
        "embed_use_bias": True,
        "edge_use_attn": False,
        "edge_rbf_dot_self": False,
        "edge_rbf_dot_message": False,
        "edge_use_esen_rbf": False,
        "edge_use_esen_atom_ebd": False,
        "edge_use_esen_env": False,
        "residual_pref": [],
        "tebd_use_act": True,
        "message_use_self_concat": False,
        "use_combined_output": False,
        "use_slim_message": False,
    }
    for key, expected in no_op_values.items():
        if _require_legacy_value(repflow, key, expected):
            consumed.append(f"descriptor.repflow.{key}")

    # These tuning values are inactive because their owning feature switches
    # were required to be false above.  They cannot affect the rebuilt graph.
    inactive_tuning = (
        "d_dim",
        "d_sel",
        "d_rcut",
        "d_rcut_smth",
        "ffn_hidden_dim",
        "edge_attn_hidden",
        "edge_attn_head",
        "edge_attn_use_ln",
    )
    for key in inactive_tuning:
        if key in repflow:
            repflow.pop(key)
            consumed.append(f"descriptor.repflow.{key}")

    return normalized, tuple(consumed)


def _resolve_fixed_neighbor_slots(
    requested: Any,
    checkpoint_slots: int,
    initial_max_neighbors: int,
    *,
    capacity_factor: float,
    capacity_headroom: int,
) -> int:
    """Choose an adaptive fixed capacity or validate an explicit capacity."""
    if capacity_factor < 1.0:
        raise ValueError("neighbor_capacity_factor must be at least 1.0")
    if capacity_headroom < 0:
        raise ValueError("neighbor_capacity_headroom must be non-negative")
    if initial_max_neighbors > checkpoint_slots:
        raise ValueError(
            "DPA3 checkpoint e_sel is smaller than the initial neighbor count: "
            f"{checkpoint_slots} < {initial_max_neighbors}"
        )
    slots = (
        min(
            checkpoint_slots,
            max(
                1,
                initial_max_neighbors + capacity_headroom,
                math.ceil(initial_max_neighbors * capacity_factor),
            ),
        )
        if requested is None
        else int(requested)
    )
    if slots <= 0:
        raise ValueError("DPA3 Opt2 fixed_neighbor_slots must be positive")
    if slots < initial_max_neighbors:
        raise ValueError(
            "DPA3 Opt2 fixed_neighbor_slots is smaller than the initial "
            f"neighbor count: {slots} < {initial_max_neighbors}"
        )
    if slots > checkpoint_slots:
        raise ValueError(
            "DPA3 Opt2 fixed_neighbor_slots exceeds checkpoint e_sel: "
            f"{slots} > {checkpoint_slots}"
        )
    return slots


def _rehydrate_static_dynamic_model(
    scripted_model: Any,
    model_params: dict[str, Any],
    device: torch.device,
) -> tuple[Any, tuple[str, ...]]:
    """Rebuild the frozen DPA3 as eager PT with fixed padded dynamic selection."""
    from deepmd.pt.model.model import get_model

    normalized_params, compatibility_options = (
        _normalize_released_dpa3_model_params(model_params)
    )
    with torch.device(device):
        model = get_model(normalized_params).to(device)
    incompatible = model.load_state_dict(scripted_model.state_dict(), strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "DPA3 Opt2 could not strictly rehydrate the released checkpoint: "
            f"missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    try:
        repflows = model.atomic_model.descriptor.repflows
    except AttributeError as exc:
        raise RuntimeError(
            "DPA3 Opt2 rehydrated model does not expose descriptor.repflows"
        ) from exc
    if not repflows.use_dynamic_sel:
        raise ValueError(
            "DPA3 Opt2 fixed padded path requires a dynamic-selection checkpoint"
        )
    repflows._use_static_dynamic_sel = True
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, compatibility_options


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


@dataclass(frozen=True)
class JITCaptureRewriteStats:
    """Auditable counts for the released-checkpoint TorchScript rewrite."""

    methods_scanned: int
    scalar_zeros: int
    index_put_zeros: int
    tensor_conditionals: int
    caches_flushed: int

    @property
    def total(self) -> int:
        """Return the total number of graph nodes replaced."""
        return self.scalar_zeros + self.index_put_zeros + self.tensor_conditionals

    def as_dict(self) -> dict[str, int]:
        """Return JSON-serializable rewrite diagnostics."""
        return {
            "methods_scanned": self.methods_scanned,
            "scalar_zeros": self.scalar_zeros,
            "index_put_zeros": self.index_put_zeros,
            "tensor_conditionals": self.tensor_conditionals,
            "total": self.total,
            "caches_flushed": self.caches_flushed,
        }


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
        if value_node.kind() == "aten::tensor":
            value_inputs = list(value_node.inputs())
            scalar = value_inputs[0].toIValue() if value_inputs else None
            if not isinstance(scalar, (bool, int, float, complex)) or scalar != 0:
                continue
        elif value_node.kind() == "aten::zeros":
            value_inputs = list(value_node.inputs())
            if not value_inputs:
                continue
            shape = value_inputs[0]
            shape_node = shape.node()
            is_scalar_shape = (
                shape_node.kind() == "prim::ListConstruct"
                and not list(shape_node.inputs())
            )
            if not is_scalar_shape:
                shape_value = shape.toIValue()
                is_scalar_shape = isinstance(shape_value, (list, tuple)) and not shape_value
            if not is_scalar_shape:
                continue
        else:
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


def _is_capture_unsafe_tensor_conditional(node: Any) -> bool:
    """Return whether an If converts a CUDA ``any`` tensor to a host bool."""
    if node.kind() != "prim::If":
        return False
    inputs = list(node.inputs())
    if len(inputs) != 1:
        return False
    bool_node = inputs[0].node()
    if bool_node.kind() != "aten::Bool":
        return False
    bool_inputs = list(bool_node.inputs())
    return (
        len(bool_inputs) == 1
        and bool_inputs[0].node().kind() == "aten::any"
    )


def _count_capture_unsafe_tensor_conditionals(graph: Any) -> int:
    """Count CUDA-syncing tensor conditionals, including nested blocks."""
    return sum(
        _is_capture_unsafe_tensor_conditional(node)
        for node in _iter_jit_nodes(graph)
    )


def _rewrite_capture_unsafe_tensor_conditionals_(graph: Any) -> int:
    """Inline custom-activation branches guarded by ``bool(torch.any(...))``.

    Released DPA3 custom SiLU modules skip their tanh tail when no element is
    above the transition threshold.  The then branch already uses
    ``torch.where`` and is valid for both cases, while evaluating the tensor
    condition as a Python bool performs a capture-forbidden device sync.
    """
    replacements = 0
    # Filter before mutating the graph: destroying an If also invalidates all
    # of its block nodes.  Innermost-first order keeps nested candidates valid.
    candidates = [
        node
        for node in _iter_jit_nodes(graph)
        if _is_capture_unsafe_tensor_conditional(node)
    ]
    for node in reversed(candidates):
        inputs = list(node.inputs())
        blocks = list(node.blocks())
        if len(inputs) != 1 or len(blocks) != 2:
            continue
        bool_node = inputs[0].node()
        if len(list(node.outputs())) != 1:
            continue
        then_block, else_block = blocks
        then_return = list(then_block.returnNode().inputs())
        else_return = list(else_block.returnNode().inputs())
        if len(then_return) != 1 or len(else_return) != 1:
            continue
        if then_return[0].node().kind() != "aten::where":
            continue
        where_inputs = list(then_return[0].node().inputs())
        if else_return[0] not in where_inputs[1:]:
            continue

        value_map: dict[Any, Any] = {}

        def remap(value: Any) -> Any:
            return value_map.get(value, value)

        for branch_node in list(then_block.nodes()):
            clone = graph.createClone(branch_node, remap)
            clone.insertBefore(node)
            for original, replacement in zip(
                branch_node.outputs(), clone.outputs(), strict=True
            ):
                value_map[original] = replacement
        node.output().replaceAllUsesWith(remap(then_return[0]))
        node.destroy()
        replacements += 1

    if replacements:
        torch._C._jit_pass_dce(graph)
        graph.lint()
    return replacements


def _patch_released_dpa3_jit_for_capture_(model: Any) -> JITCaptureRewriteStats:
    """Patch capture-unsafe operations in every released-model JIT method.

    ``RecursiveScriptModule.named_modules()`` exposes loaded archive children as
    generic wrappers on some PyTorch releases.  In particular, the released
    DPA3 ``CustomSiLU`` child is not guaranteed to retain either its Python
    class name or a ``custom_silu`` component in its qualified name.  Selecting
    methods by those labels therefore left its ``bool(torch.any(mask))`` in the
    executable graph even though the graph rewrite itself was correct.

    All three rewrites below are selected by exact TorchScript operator and
    input patterns and preserve semantics, so applying them to every scripted
    method is both safer and more robust than relying on archive names.  Flush
    *all* method execution plans after the mutation: a parent plan may already
    contain an optimized ``prim::CallMethod`` path to the changed child.
    """
    scalar_zeros = 0
    index_put_zeros = 0
    tensor_conditionals = 0
    methods = []
    for _module_name, module in model.named_modules():
        script_module = getattr(module, "_c", None)
        if script_module is None:
            continue
        for method_name in script_module._method_names():
            method = script_module._get_method(method_name)
            methods.append(method)
            scalar_zeros += _rewrite_capture_unsafe_scalar_zeros_(method.graph)
            index_put_zeros += _rewrite_capture_unsafe_index_put_zeros_(
                method.graph
            )
            tensor_conditionals += _rewrite_capture_unsafe_tensor_conditionals_(
                method.graph
            )

    residual_tensor_conditionals = sum(
        _count_capture_unsafe_tensor_conditionals(method.graph)
        for method in methods
    )
    if residual_tensor_conditionals:
        raise RuntimeError(
            "DPA3 Opt2 TorchScript rewrite left "
            f"{residual_tensor_conditionals} capture-unsafe "
            "bool(torch.any(...)) conditional(s); refusing CUDA Graph capture"
        )

    total = scalar_zeros + index_put_zeros + tensor_conditionals
    caches_flushed = 0
    if total:
        for method in methods:
            method._debug_flush_compilation_cache()
            caches_flushed += 1
    return JITCaptureRewriteStats(
        methods_scanned=len(methods),
        scalar_zeros=scalar_zeros,
        index_put_zeros=index_put_zeros,
        tensor_conditionals=tensor_conditionals,
        caches_flushed=caches_flushed,
    )


class DPA3ModelCUDAGraphEvaluator:
    """Build neighbors eagerly and replay one fixed-address DPA3 model graph."""

    def __init__(
        self,
        eager_evaluator: DPA3EnergyForceEvaluator,
        *,
        fixed_neighbor_slots: int,
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
        self._archive_inner = self._backend.dp.model["Default"]
        checkpoint_slots = int(sum(self._archive_inner.get_sel()))
        initial_max = eager_evaluator.neighbor_shape_metadata.get(
            "initial_max_neighbors"
        )
        if initial_max is None:
            raise RuntimeError(
                "DPA3 Opt2 neighbor preflight did not report initial_max_neighbors"
            )
        self.fixed_neighbor_slots = _resolve_fixed_neighbor_slots(
            fixed_neighbor_slots,
            checkpoint_slots,
            int(initial_max),
            capacity_factor=1.0,
            capacity_headroom=0,
        )
        self.checkpoint_neighbor_slots = checkpoint_slots
        self.initial_max_neighbors = int(initial_max)
        self._inner, self.checkpoint_compatibility_options = (
            _rehydrate_static_dynamic_model(
                self._archive_inner,
                self._backend.get_model_def_script(),
                self.device,
            )
        )
        self.jit_capture_rewrite_stats = JITCaptureRewriteStats(
            methods_scanned=0,
            scalar_zeros=0,
            index_put_zeros=0,
            tensor_conditionals=0,
            caches_flushed=0,
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
        self.compact_parity_energy_abs_error = 0.0
        self.compact_parity_force_max_abs_error = 0.0
        self.compact_parity_virial_max_abs_error = 0.0
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
                [self.fixed_neighbor_slots],
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
        # Scientific reference remains the original released compact model.
        compact_reference = tuple(
            value.clone() for value in self.eager_evaluator(positions)
        )
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
        assert reference is not None
        candidate_reference_errors = tuple(
            _maximum_abs_error(new, old)
            for new, old in zip(reference, compact_reference, strict=True)
        )
        self.compact_parity_force_max_abs_error = candidate_reference_errors[0]
        self.compact_parity_energy_abs_error = candidate_reference_errors[1]
        self.compact_parity_virial_max_abs_error = candidate_reference_errors[2]
        for name, error, tolerance in (
            ("force", candidate_reference_errors[0], self.force_atol),
            ("energy", candidate_reference_errors[1], self.energy_atol),
            ("virial", candidate_reference_errors[2], self.virial_atol),
        ):
            if error > tolerance:
                raise CUDAGraphValidationError(
                    "DPA3 fixed-slot padded-dynamic parity failed before capture: "
                    f"{name} error {error:.6g} > {tolerance:.6g}"
                )

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
            "compact_parity_energy_abs_error": (
                self.compact_parity_energy_abs_error
            ),
            "compact_parity_force_max_abs_error": (
                self.compact_parity_force_max_abs_error
            ),
            "compact_parity_virial_max_abs_error": (
                self.compact_parity_virial_max_abs_error
            ),
            "capacity_overflow_policy": "device-assert-and-fail-no-fallback",
            "dynamic_selection_layout": "fixed-slot-padded",
            "fixed_neighbor_slots": self.fixed_neighbor_slots,
            "checkpoint_neighbor_slots": self.checkpoint_neighbor_slots,
            "initial_max_neighbors": self.initial_max_neighbors,
            "checkpoint_compatibility_options": list(
                self.checkpoint_compatibility_options
            ),
            "jit_capture_rewrites": self.jit_capture_rewrite_stats.as_dict(),
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
    capacity_factor = float(request.options.get("neighbor_capacity_factor", 1.25))
    capacity_headroom = int(request.options.get("neighbor_capacity_headroom", 16))
    requested_slots = request.options.get("fixed_neighbor_slots")
    configured_capacity = request.options.get("neighbor_search_capacity")
    if requested_slots is not None and configured_capacity is not None and (
        int(requested_slots) != int(configured_capacity)
    ):
        raise ValueError(
            "DPA3 Opt2 fixed_neighbor_slots and neighbor_search_capacity "
            f"disagree: {requested_slots} != {configured_capacity}"
        )
    if requested_slots is None:
        requested_slots = configured_capacity
    eager_evaluator = DPA3EnergyForceEvaluator(
        atoms,
        request.model_path,
        device=device,
        neighbor_capacity_factor=capacity_factor,
        neighbor_capacity_headroom=capacity_headroom,
        neighbor_search_capacity=None,
        profiler=profiler,
    )
    initial_max_neighbors = int(
        eager_evaluator.neighbor_shape_metadata["initial_max_neighbors"]
    )
    checkpoint_slots = int(
        sum(eager_evaluator._backend.dp.model["Default"].get_sel())
    )
    fixed_neighbor_slots = _resolve_fixed_neighbor_slots(
        requested_slots,
        checkpoint_slots,
        initial_max_neighbors,
        capacity_factor=capacity_factor,
        capacity_headroom=capacity_headroom,
    )
    # The initial preflight used the checkpoint selection to discover the true
    # maximum safely. Narrow the fixed raw-search capacity for all subsequent
    # calls; nv_search_matrix_fixed retains its device-side overflow assert.
    eager_evaluator._backend._nlist_builder.fixed_search_capacity = (
        fixed_neighbor_slots
    )
    eager_evaluator.neighbor_shape_metadata = (
        eager_evaluator._backend._nlist_builder.fixed_shape_metadata()
    )
    eager_evaluator.neighbor_shape_metadata.update(
        {
            "capacity_policy": (
                "adaptive-initial-preflight"
                if requested_slots is None
                else "explicit"
            ),
            "checkpoint_neighbor_slots": checkpoint_slots,
        }
    )
    evaluator = DPA3ModelCUDAGraphEvaluator(
        eager_evaluator,
        fixed_neighbor_slots=fixed_neighbor_slots,
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
