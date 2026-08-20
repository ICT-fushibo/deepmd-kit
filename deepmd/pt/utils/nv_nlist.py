# SPDX-License-Identifier: LGPL-3.0-or-later
"""Toolkit-Ops (``nvalchemiops``) neighbor-list strategy.

A :class:`~deepmd.dpmodel.utils.neighbor_list.NeighborList` implementation that
builds the extended representation ``(extended_coord, extended_atype, nlist,
mapping)`` using the device-resident neighbor-list kernels in ``nvalchemiops``.

Toolkit-Ops returns a dense ``[total_atoms, max_neighbors]`` neighbor matrix over
the flattened batch. The matrix is converted to the DeePMD extended-atom contract
by materializing each unique ghost ``(frame, src_local, shift)`` once; the
candidate list is then distance-sorted and truncated to ``sum(sel)`` so the
returned neighbor count is fixed. The search is non-differentiable and runs on
detached coordinates, while ``extended_coord`` is rebuilt from the input
coordinates so force and virial gradients propagate unchanged.
"""

from __future__ import (
    annotations,
)

import contextlib
import logging
import math
import os
import sys
from typing import (
    TYPE_CHECKING,
    Any,
)

import torch

from deepmd.dpmodel.utils.neighbor_list import (
    EdgeNeighborList,
    NeighborList,
)
from deepmd.pt_expt.utils.edge_schema import (
    edge_schema_from_neighbor_matrix,
)

NV_CELL_LIST_THRESHOLD = 1024
NV_NONPERIODIC_CELL_LIST_THRESHOLD = 4096
# CPU has far less parallelism than CUDA, so the O(N^2) ``batch_naive`` method
# is overtaken by the O(N) ``batch_cell_list`` at a much smaller atom count;
# switch over early regardless of periodicity.
NV_CPU_CELL_LIST_THRESHOLD = 128

log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import (
        Iterator,
    )

    from deepmd.dpmodel.utils.exclude_mask import (
        PairExcludeMask,
    )


@contextlib.contextmanager
def _suppress_native_stderr() -> Iterator[None]:
    """Redirect the process ``stderr`` file descriptor to ``os.devnull``.

    ``nvalchemiops`` initializes NVIDIA Warp on first import, which probes for a
    CUDA driver and prints a native ``Warp CUDA error 100`` line straight to the
    ``stderr`` fd on CPU-only hosts. That line bypasses Python logging, so the
    only way to mute it is at the descriptor level around the triggering import.
    """
    try:
        stderr_fd = sys.stderr.fileno()
    except (AttributeError, OSError, ValueError):
        # stderr is not a real file descriptor (e.g. captured in tests); the
        # native chatter cannot be redirected, so import without suppression.
        yield
        return
    saved_fd = os.dup(stderr_fd)
    with open(os.devnull, "w") as devnull:
        os.dup2(devnull.fileno(), stderr_fd)
        try:
            yield
        finally:
            os.dup2(saved_fd, stderr_fd)
            os.close(saved_fd)


def is_nv_available() -> bool:
    """Whether the ``nvalchemiops`` Toolkit-Ops neighbor list is importable."""
    # Warp's one-time CUDA probe prints to the native stderr on CPU-only hosts;
    # mute it there without hiding diagnostics on machines that have a GPU.
    import_ctx = (
        _suppress_native_stderr()
        if not torch.cuda.is_available()
        else contextlib.nullcontext()
    )
    try:
        with import_ctx:
            import nvalchemiops.torch.neighbors  # noqa: F401
    except (ImportError, OSError, RuntimeError) as err:
        log.debug("nvalchemiops Toolkit-Ops neighbor list is unavailable: %s", err)
        return False
    return True


def choose_nv_nlist_method(
    nloc: int, *, periodic: bool = True, device: torch.device | None = None
) -> str:
    """Choose the Toolkit-Ops neighbor method for a homogeneous batch.

    Parameters
    ----------
    nloc
        Number of local atoms per frame.
    periodic
        Whether the batch is periodic.
    device
        Target device. CPU uses a lower cell-list threshold than CUDA because
        the ``batch_naive`` method does not parallelize well there.

    Returns
    -------
    str
        Toolkit-Ops method name.
    """
    if device is not None and device.type == "cpu":
        threshold = NV_CPU_CELL_LIST_THRESHOLD
    elif periodic:
        threshold = NV_CELL_LIST_THRESHOLD
    else:
        threshold = NV_NONPERIODIC_CELL_LIST_THRESHOLD
    if nloc >= threshold:
        return "batch_cell_list"
    return "batch_naive"


@contextlib.contextmanager
def _input_device_context(device: torch.device) -> Iterator[None]:
    """Run third-party kernels with both default and current devices pinned."""
    if device.type == "cuda":
        with torch.device(device), torch.cuda.device(device):
            yield
    else:
        with torch.device(device):
            yield


class NvNeighborList(NeighborList):
    """Neighbor-list strategy using the ``nvalchemiops`` kernels.

    Implements the :class:`~deepmd.dpmodel.utils.neighbor_list.NeighborList`
    interface on torch tensors; the search runs on the device of the input
    coordinates. Periodic inputs materialize shifted ghost atoms; non-periodic
    inputs keep only local atoms.
    """

    def __init__(self, *, compile_truncation: bool = True) -> None:
        """Create the CUDA neighbor-list builder.

        Parameters
        ----------
        compile_truncation
            Use the historical Inductor-compiled distance-sort/truncation
            helper.  The default preserves the existing inference behaviour.
            GPU-resident Opt1 MD explicitly sets this to ``False`` because
            compilation belongs to a later optimization stage; the eager
            helper performs the same operation entirely on CUDA tensors.
        """
        self.compile_truncation = bool(compile_truncation)
        # Ordinary inference keeps the historical dynamic/compact path.
        # GPU-resident MD explicitly opts into fixed shapes during evaluator
        # initialization via ``prepare_fixed_shape``.
        self.fixed_search_capacity: int | None = None
        self.fixed_shift_bounds: tuple[int, int, int] | None = None
        self.fixed_shift_grid: torch.Tensor | None = None
        self.fixed_shift_cart: torch.Tensor | None = None
        self.fixed_capacity_initial_max: int | None = None

    def prepare_fixed_shape(
        self,
        coord: torch.Tensor,
        box: torch.Tensor | None,
        rcut: float,
        sel: list[int],
        *,
        capacity_factor: float = 1.25,
        capacity_headroom: int = 16,
        search_capacity: int | None = None,
    ) -> dict[str, Any]:
        """Freeze search and periodic-image shapes before an eager MD loop.

        The one preflight search deliberately synchronizes to discover a safe
        initial capacity and must run outside the measured loop.  Subsequent
        builds use a fixed-capacity search with a device-side overflow assert
        and deterministic ``(shift, source atom)`` ghost slots.  Thus their hot
        path has no data-dependent ``nonzero``/``unique`` allocation and no
        device-to-host scalar read.

        The fixed shift universe assumes the cell is constant (for example,
        NVT).  An underestimated search capacity raises instead of silently
        dropping neighbors; callers can retry with an explicit larger value.
        """
        if capacity_factor < 1.0:
            raise ValueError("capacity_factor must be at least 1.0")
        if capacity_headroom < 0:
            raise ValueError("capacity_headroom must be non-negative")

        target_neighbors = int(sum(sel))
        coord = coord.reshape(coord.shape[0], -1, 3)
        from deepmd.pt_expt.utils.nv_graph_builder import (
            nv_search_matrix,
        )

        _, cell, _, num_neighbors, _ = nv_search_matrix(
            coord, box, float(rcut), start_capacity=target_neighbors
        )
        initial_max = (
            int(num_neighbors.max().item()) if num_neighbors.numel() > 0 else 0
        )
        if search_capacity is None:
            search_capacity = max(
                target_neighbors,
                initial_max + int(capacity_headroom),
                math.ceil(initial_max * float(capacity_factor)),
            )
        elif search_capacity < initial_max:
            raise ValueError(
                "fixed search_capacity is smaller than the initial neighbor "
                f"count: {search_capacity} < {initial_max}"
            )
        if search_capacity <= 0:
            raise ValueError("fixed search_capacity must be positive")

        if cell is None:
            shift_bounds = None
            shift_grid = None
            shift_cart = None
        else:
            from deepmd.pt.utils.region import (
                to_face_distance,
            )

            face_distance = to_face_distance(cell)
            if torch.any(face_distance <= 0):
                raise ValueError("periodic cell must have positive face distances")
            bounds = torch.ceil(float(rcut) / face_distance).to(torch.int64)
            bounds = torch.amax(bounds, dim=0).detach().cpu().tolist()
            shift_bounds = tuple(int(value) for value in bounds)
            bx, by, bz = shift_bounds
            shift_values = [
                (ix, iy, iz)
                for ix in range(-bx, bx + 1)
                for iy in range(-by, by + 1)
                for iz in range(-bz, bz + 1)
                if (ix, iy, iz) != (0, 0, 0)
            ]
            shift_grid = torch.tensor(
                shift_values, dtype=coord.dtype, device=coord.device
            ).reshape(-1, 3)
            shift_cart = torch.einsum("sd,fdk->fsk", shift_grid, cell)

        self.fixed_search_capacity = int(search_capacity)
        self.fixed_shift_bounds = shift_bounds
        self.fixed_shift_grid = shift_grid
        self.fixed_shift_cart = shift_cart
        self.fixed_capacity_initial_max = initial_max
        return self.fixed_shape_metadata()

    def fixed_shape_metadata(self) -> dict[str, Any]:
        """Describe the opt-in fixed-shape neighbor-list policy."""
        return {
            "fixed_search_capacity": self.fixed_search_capacity,
            "fixed_shift_bounds": self.fixed_shift_bounds,
            "fixed_ghost_slots_per_atom": (
                None
                if self.fixed_shift_grid is None
                else int(self.fixed_shift_grid.shape[0])
            ),
            "initial_max_neighbors": self.fixed_capacity_initial_max,
            "overflow_policy": "device-assert",
            "ghost_layout": (
                "fixed-shift-source-slots"
                if self.fixed_search_capacity is not None
                else "dynamic-compact"
            ),
        }

    def build(
        self,
        coord: Any,
        atype: Any,
        box: Any,
        rcut: float,
        sel: list[int],
        return_mode: str = "extended",
        pair_excl: PairExcludeMask | None = None,
    ) -> tuple[Any, Any, Any, Any] | EdgeNeighborList:
        """Build the extended system and neighbor list.

        See :meth:`deepmd.dpmodel.utils.neighbor_list.NeighborList.build`. The
        returned ``nlist`` is distance-sorted and truncated to ``sum(sel)``.

        Parameters
        ----------
        pair_excl : PairExcludeMask or None, optional
            When provided, excluded type pairs are erased from the returned
            neighbor list (entries set to ``-1``) by
            :func:`~deepmd.dpmodel.utils.nlist.apply_pair_exclusion_nlist`.
            ``NvNeighborList`` is CUDA-only; the ``pair_excl`` parameter is
            accepted for API parity with the other strategies but cannot be
            validated on a CPU-only machine.
            ``return_mode='edges'`` does not support ``pair_excl``; a
            :class:`NotImplementedError` is raised in that combination.
        """
        if return_mode == "edges" and pair_excl is not None:
            raise NotImplementedError(
                "pair_excl is not supported with return_mode='edges'; "
                "use apply_pair_exclusion (graph variant) on the returned EdgeNeighborList."
            )
        device = coord.device
        nf, nloc = atype.shape[:2]
        target_neighbors = int(sum(sel))
        coord = coord.reshape(nf, nloc, 3)
        periodic = box is not None

        # Delegate the raw search to the shared helper in nv_graph_builder.
        # Function-level import avoids a module-level pt -> pt_expt cycle while
        # keeping the search logic in exactly one place (graph-builder primary,
        # legacy strategy is the deprecation-bound caller).
        from deepmd.pt_expt.utils.nv_graph_builder import (
            nv_search_matrix_fixed,
            nv_search_matrix,
        )

        if self.fixed_search_capacity is None:
            coord, cell, neighbor_matrix, num_neighbors, shifts = nv_search_matrix(
                coord, box, rcut, start_capacity=target_neighbors
            )
        else:
            coord, cell, neighbor_matrix, num_neighbors, shifts = (
                nv_search_matrix_fixed(
                    coord,
                    box,
                    rcut,
                    capacity=self.fixed_search_capacity,
                )
            )

        with _input_device_context(device):
            if return_mode == "edges":
                return edge_schema_from_neighbor_matrix(
                    coord=coord,
                    atype=atype,
                    cell=cell,
                    neighbor_matrix=neighbor_matrix,
                    num_neighbors=num_neighbors,
                    shifts=shifts,
                    rcut=float(rcut),
                )
            if return_mode != "extended":
                raise ValueError(
                    f"Unsupported neighbor-list return_mode: {return_mode!r}"
                )

            if self.fixed_search_capacity is None:
                extended_coord, extended_atype, mapping, nlist = (
                    _matrix_to_extended_inputs(
                        coord=coord,
                        atype=atype,
                        cell=cell,
                        nloc=nloc,
                        neighbor_matrix=neighbor_matrix,
                        num_neighbors=num_neighbors,
                        shifts=shifts,
                    )
                )
            else:
                extended_coord, extended_atype, mapping, nlist = (
                    _matrix_to_extended_inputs_fixed_slots(
                        coord=coord,
                        atype=atype,
                        cell=cell,
                        nloc=nloc,
                        neighbor_matrix=neighbor_matrix,
                        num_neighbors=num_neighbors,
                        shifts=shifts,
                        shift_bounds=self.fixed_shift_bounds,
                        shift_cart=self.fixed_shift_cart,
                    )
                )
            truncate = (
                _truncate_to_sel_compiled
                if self.compile_truncation
                else _truncate_to_sel
            )
            nlist = truncate(extended_coord, nlist, target_neighbors, float(rcut))
            if pair_excl is not None:
                from deepmd.dpmodel.utils.nlist import (
                    apply_pair_exclusion_nlist,
                )

                nlist = apply_pair_exclusion_nlist(nlist, extended_atype, pair_excl)
            return extended_coord, extended_atype, nlist, mapping


def _grow_search_capacity(capacity: int) -> int:
    """Increase Toolkit-Ops capacity by 1.25x, rounded up."""
    return (capacity * 5 + 3) // 4


@torch.no_grad()
def _truncate_to_sel(
    extended_coord: torch.Tensor,
    nlist: torch.Tensor,
    nsel: int,
    rcut: float,
) -> torch.Tensor:
    """Distance-sort the candidate neighbor list and keep the nearest ``nsel``
    within ``rcut``, padding with ``-1`` when fewer neighbors exist.

    The Toolkit-Ops search capacity may exceed ``sum(sel)`` on dense systems; this
    fixes the returned neighbor count at ``nsel``.

    The output is the integer ``nlist``; ``extended_coord`` is only read to rank
    candidates and is returned unchanged by the caller. The routine is therefore
    non-differentiable and runs under ``no_grad`` so it never participates in the
    autograd graph (forward, backward, or the second-order pass used to train
    forces), which also avoids retaining the distance temporaries for backward.
    """
    nf, nloc, nnei = nlist.shape
    if nnei < nsel:
        pad = torch.full(
            (nf, nloc, nsel - nnei), -1, dtype=nlist.dtype, device=nlist.device
        )
        return torch.cat([nlist, pad], dim=-1)
    if nnei == nsel:
        return nlist
    real_neighbor = nlist >= 0
    safe_nlist = torch.where(real_neighbor, nlist, torch.zeros_like(nlist))
    coord0 = extended_coord[:, :nloc, :]
    index = safe_nlist.view(nf, nloc * nnei, 1).expand(-1, -1, 3)
    coord1 = torch.gather(extended_coord, 1, index).view(nf, nloc, nnei, 3)
    rr = torch.linalg.norm(coord1 - coord0[:, :, None, :], dim=-1)
    rr = torch.where(real_neighbor, rr, float("inf"))
    rr, order = torch.sort(rr, dim=-1)
    sorted_nlist = torch.gather(safe_nlist, 2, order)
    sorted_nlist = torch.where(rr > rcut, -1, sorted_nlist)
    # ``.contiguous()`` is required: the bare ``[..., :nsel]`` slice keeps the
    # wider candidate stride, but the compiled lower interface freezes the nlist
    # sel axis and asserts a contiguous layout (``assert_size_stride``).
    return sorted_nlist[..., :nsel].contiguous()


# Lower the gather/distance-sort/mask pipeline of `_truncate_to_sel` into a single
# Inductor graph. ``dynamic=True`` keeps the per-system ``(nf, nloc, nnei)`` shapes
# on one compiled graph instead of recompiling per system size, and fusing the
# pipeline avoids materializing the full ``(nf, nloc, nnei, 3)`` distance
# temporaries.  Even construction of the OptimizedFunction is lazy so callers
# selecting ``compile_truncation=False`` never invoke ``torch.compile``.
_truncate_to_sel_inductor: Any | None = None


def _truncate_to_sel_compiled(
    extended_coord: torch.Tensor,
    nlist: torch.Tensor,
    nsel: int,
    rcut: float,
) -> torch.Tensor:
    global _truncate_to_sel_inductor
    if _truncate_to_sel_inductor is None:
        _truncate_to_sel_inductor = torch.compile(_truncate_to_sel, dynamic=True)
    return _truncate_to_sel_inductor(extended_coord, nlist, nsel, rcut)


def _matrix_to_extended_inputs_fixed_slots(
    *,
    coord: torch.Tensor,
    atype: torch.Tensor,
    cell: torch.Tensor | None,
    nloc: int,
    neighbor_matrix: torch.Tensor,
    num_neighbors: torch.Tensor,
    shifts: torch.Tensor,
    shift_bounds: tuple[int, int, int] | None,
    shift_cart: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert a fixed-width search matrix without data-dependent shapes.

    Periodic images are addressed by a deterministic dense universe.  For each
    non-zero integer cell shift in ``[-bound, bound]`` and every local source
    atom there is exactly one ghost slot.  Repeated edges therefore share the
    same differentiable ghost coordinate just as in the compact converter, but
    no ``nonzero``, ``unique``, or host scalar extraction is required.

    The unreferenced slots are harmless because DPA3's local-mapping descriptor
    reads extended coordinates only through ``nlist`` and maps ghost gradients
    back through ``mapping``.  Callers must opt in only for models with that
    contract.
    """
    nf = coord.shape[0]
    total_atoms, max_neighbors = neighbor_matrix.shape
    if total_atoms != nf * nloc:
        raise ValueError(
            "neighbor matrix atom dimension does not match coord: "
            f"{total_atoms} != {nf} * {nloc}"
        )
    device = coord.device
    neighbor_matrix = neighbor_matrix.reshape(nf, nloc, max_neighbors)
    shifts = shifts.reshape(nf, nloc, max_neighbors, 3).to(torch.long)
    slot = torch.arange(max_neighbors, dtype=torch.long, device=device)
    valid = slot.view(1, 1, -1) < num_neighbors.reshape(nf, nloc, 1)
    src_local = torch.remainder(neighbor_matrix.clamp_min(0).to(torch.long), nloc)

    local_mapping = torch.arange(nloc, dtype=torch.long, device=device)
    local_mapping = local_mapping.unsqueeze(0).expand(nf, -1)
    if cell is None:
        zero_shift = torch.all(shifts == 0, dim=-1)
        torch._assert_async(
            torch.all(torch.logical_or(~valid, zero_shift)),
            "non-periodic fixed-shape neighbor search returned a cell shift",
        )
        nlist = torch.where(valid, src_local, -1)
        return coord, atype, local_mapping, nlist
    if shift_bounds is None:
        raise ValueError("periodic fixed-shape conversion requires shift_bounds")
    if shift_cart is None:
        raise ValueError("periodic fixed-shape conversion requires shift_cart")

    bx, by, bz = shift_bounds
    within_bounds = (
        (shifts[..., 0] >= -bx)
        & (shifts[..., 0] <= bx)
        & (shifts[..., 1] >= -by)
        & (shifts[..., 1] <= by)
        & (shifts[..., 2] >= -bz)
        & (shifts[..., 2] <= bz)
    )
    torch._assert_async(
        torch.all(torch.logical_or(~valid, within_bounds)),
        "periodic image shift exceeds the prepared fixed-shape bounds",
    )

    ny = 2 * by + 1
    nz = 2 * bz + 1
    shift_code = (
        (shifts[..., 0] + bx) * ny * nz
        + (shifts[..., 1] + by) * nz
        + shifts[..., 2]
        + bz
    )
    zero_code = bx * ny * nz + by * nz + bz
    zero_shift = shift_code == zero_code
    # Remove the central-cell code from the ghost universe.  Values for direct
    # and invalid slots are not used, but clamp them to make all gathers safe.
    ghost_code = shift_code - (shift_code > zero_code).to(torch.long)
    nshift_ghost = (2 * bx + 1) * ny * nz - 1
    ghost_code = ghost_code.clamp(0, max(nshift_ghost - 1, 0))
    ghost_index = nloc + ghost_code * nloc + src_local
    nlist_values = torch.where(zero_shift, src_local, ghost_index)
    nlist = torch.where(valid, nlist_values, -1)

    # Build every possible image once.  The ordering mirrors ``shift_code`` so
    # the arithmetic above directly indexes the flattened ghost block.
    ghost_coord = coord[:, None, :, :] + shift_cart[:, :, None, :]
    ghost_coord = ghost_coord.reshape(nf, nshift_ghost * nloc, 3)
    ghost_atype = atype[:, None, :].expand(-1, nshift_ghost, -1)
    ghost_atype = ghost_atype.reshape(nf, nshift_ghost * nloc)
    ghost_mapping = local_mapping[:, None, :].expand(-1, nshift_ghost, -1)
    ghost_mapping = ghost_mapping.reshape(nf, nshift_ghost * nloc)
    return (
        torch.cat((coord, ghost_coord), dim=1),
        torch.cat((atype, ghost_atype), dim=1),
        torch.cat((local_mapping, ghost_mapping), dim=1),
        nlist,
    )


def _matrix_to_extended_inputs(
    *,
    coord: torch.Tensor,
    atype: torch.Tensor,
    cell: torch.Tensor | None,
    nloc: int,
    neighbor_matrix: torch.Tensor,
    num_neighbors: torch.Tensor,
    shifts: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert Toolkit-Ops matrix output to compact extended inputs.

    Toolkit-Ops returns neighbors as a dense matrix over flattened atoms:
    ``neighbor_matrix[dst_global, slot] = src_global`` and
    ``shifts[dst_global, slot] = (sx, sy, sz)``. Here ``dst_global`` and
    ``src_global`` are indices in the concatenated ``nf * nloc`` input.

    DeePMD lower paths use a different contract: ``nlist`` stores indices into
    ``extended_coord``. Local atoms occupy ``[0, nloc)`` in each frame, while
    shifted PBC images must be appended as ghost atoms. This conversion builds
    the minimal ghost set by materializing each unique
    ``(frame, src_local, shift)`` once, then redirects all shifted nlist entries
    to the corresponding compact ghost index.
    """
    nf = coord.shape[0]
    total_atoms, max_neighbors = neighbor_matrix.shape
    device = coord.device
    dtype = coord.dtype
    local_mapping = torch.arange(nloc, dtype=torch.long, device=device)
    local_mapping = local_mapping.unsqueeze(0).expand(nf, -1)
    nlist = torch.full((nf, nloc, max_neighbors), -1, dtype=torch.long, device=device)

    # === Step 1. Flatten valid Toolkit-Ops matrix slots ===
    # `edge_idx` indexes the flattened matrix layout `(total_atoms, max_neighbors)`.
    # This avoids constructing a full repeated destination tensor.
    slot = torch.arange(max_neighbors, dtype=torch.long, device=device).expand(
        total_atoms, max_neighbors
    )
    valid = (slot < num_neighbors.unsqueeze(1)).reshape(-1)
    edge_idx = torch.nonzero(valid, as_tuple=False).flatten()
    if edge_idx.numel() == 0:
        return coord, atype, local_mapping, nlist

    # Decode flattened edge slots:
    #   dst         : flattened center atom, in [0, nf * nloc)
    #   src         : flattened neighbor atom returned by Toolkit-Ops
    #   frame_idx   : batch frame/system containing both dst and src
    #   center_idx  : local center atom index inside the frame
    #   src_local   : local neighbor atom index before applying the PBC shift
    dst = edge_idx // max_neighbors
    src = neighbor_matrix.reshape(-1).index_select(0, edge_idx).to(dtype=torch.long)
    shift = shifts.reshape(-1, 3).index_select(0, edge_idx).to(dtype=torch.long)
    src_local = src % nloc
    frame_idx = dst // nloc
    center_idx = dst % nloc
    slot_idx = edge_idx % max_neighbors
    zero_shift = torch.all(shift == 0, dim=1)

    # === Step 2. Direct neighbors keep their local extended indices ===
    # Zero-shift neighbors already live in the leading local block of
    # `extended_coord`, so their DeePMD nlist value is simply `src_local`.
    direct_edge_idx = torch.nonzero(zero_shift, as_tuple=False).flatten()
    nlist[
        frame_idx.index_select(0, direct_edge_idx),
        center_idx.index_select(0, direct_edge_idx),
        slot_idx.index_select(0, direct_edge_idx),
    ] = src_local.index_select(0, direct_edge_idx)

    shifted_edge_idx = torch.nonzero(~zero_shift, as_tuple=False).flatten()
    if shifted_edge_idx.numel() == 0:
        return coord, atype, local_mapping, nlist
    if cell is None:
        raise RuntimeError("Non-periodic Toolkit-Ops neighbor list returned shifts.")

    # === Step 3. Materialize each unique shifted atom once per frame ===
    # A shifted source may appear in many center atoms' neighbor slots.  Dedup by
    # `(frame, src_local, shift)` so all such slots share one compact ghost atom.
    ghost_keys = torch.cat(
        [
            frame_idx.index_select(0, shifted_edge_idx).unsqueeze(1),
            src_local.index_select(0, shifted_edge_idx).unsqueeze(1),
            shift.index_select(0, shifted_edge_idx),
        ],
        dim=1,
    )
    unique_keys, inverse = torch.unique(ghost_keys, dim=0, return_inverse=True)
    ghost_frame = unique_keys[:, 0].to(dtype=torch.long)
    ghost_src = unique_keys[:, 1].to(dtype=torch.long)
    ghost_shift = unique_keys[:, 2:].to(dtype=dtype)

    # Assign per-frame compact ghost indices.  `ghost_rank` is the offset within
    # a frame's ghost block, so the final extended index is `nloc + ghost_rank`.
    # The `.item()` sync is only used to size the padded dense output.
    ghost_count = torch.bincount(ghost_frame, minlength=nf)
    max_extra = int(ghost_count.max().item())
    order = torch.argsort(ghost_frame)
    sorted_frame = ghost_frame.index_select(0, order)
    frame_start = torch.cumsum(ghost_count, dim=0) - ghost_count
    sorted_rank = torch.arange(
        unique_keys.shape[0], dtype=torch.long, device=device
    ) - frame_start.index_select(0, sorted_frame)
    ghost_rank = torch.empty_like(sorted_rank)
    ghost_rank[order] = sorted_rank
    ghost_index = nloc + ghost_rank

    extended_coord = torch.zeros((nf, nloc + max_extra, 3), dtype=dtype, device=device)
    extended_atype = torch.full(
        (nf, nloc + max_extra), -1, dtype=atype.dtype, device=device
    )
    mapping = torch.zeros((nf, nloc + max_extra), dtype=torch.long, device=device)
    extended_coord[:, :nloc] = coord
    extended_atype[:, :nloc] = atype
    mapping[:, :nloc] = local_mapping

    # Convert integer cell shifts to Cartesian ghost coordinates and record the
    # extended-to-local mapping used later to scatter forces/virials back.
    shift_cart = torch.bmm(
        ghost_shift.unsqueeze(1), cell.index_select(0, ghost_frame)
    ).squeeze(1)
    extended_coord[ghost_frame, ghost_index] = (
        coord[ghost_frame, ghost_src] + shift_cart
    )
    extended_atype[ghost_frame, ghost_index] = atype[ghost_frame, ghost_src]
    mapping[ghost_frame, ghost_index] = ghost_src

    # Redirect shifted neighbor slots to their compact ghost indices.  `inverse`
    # maps each shifted edge's key back to its row in `unique_keys`.
    shifted_nlist_values = ghost_index.index_select(0, inverse)
    shifted_frames = frame_idx.index_select(0, shifted_edge_idx)
    shifted_centers = center_idx.index_select(0, shifted_edge_idx)
    shifted_slots = slot_idx.index_select(0, shifted_edge_idx)
    nlist[shifted_frames, shifted_centers, shifted_slots] = shifted_nlist_values
    return extended_coord, extended_atype, mapping, nlist
