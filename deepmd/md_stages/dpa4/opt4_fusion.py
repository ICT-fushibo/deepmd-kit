"""DPA4 Opt4 FastEq rotate/radial-mix boundary.

Only the IEEE-fp32 rotate-to-local plus degree mixer is replaced.  The SO2
linears, gates, attention and destination reduction retain the released path.

Algorithmic adaptation of FastEq commit 40ba40e72bee769d74a869bb4a4ba820ee1c55c0
(MIT); the integration repository carries the complete third-party notice.
"""
from __future__ import annotations

import torch
from torch import nn

from md_benchmark.opt4_fx import (
    CheckedRegion,
    assert_float32_vjp_reassociation_close,
)
from md_benchmark.opt4_registry import FusionSetupError, record


class _RotateMixReference(nn.Module):
    def forward(self, x, src, wigner, kc, cb, lmax, n_focus, rank):
        from deepmd.kernels.triton.sezm.so2_value_path import _rotate_mix_reference

        return _rotate_mix_reference(
            x, src, wigner, kc, cb.detach(), lmax, n_focus, rank
        )


class _RotateMixCandidate(nn.Module):
    def forward(self, x, src, wigner, kc, cb, lmax, n_focus, rank):
        from deepmd.kernels.triton.sezm.so2_value_path import _rotate_mix_op

        return _rotate_mix_op(
            x, src, wigner, kc, cb.detach(), lmax, n_focus, rank
        )


class _FastEqSO2RotateMix(nn.Module):
    """Use DeepMD's explicit-VJP Triton primitive, then the native SO2 stack."""

    def __init__(self, convolution, detail: dict) -> None:
        super().__init__()
        object.__setattr__(self, "_convolution", convolution)
        object.__setattr__(self, "_detail", detail)
        self.region = CheckedRegion(
            _RotateMixReference(),
            detail,
            _RotateMixCandidate(),
            vjp_validator=self.validate_vjp,
        )

    def validate_vjp(self, actual, expected, args, index, output_probes):
        metrics = assert_float32_vjp_reassociation_close(actual, expected)
        rows = self._detail.setdefault("vjp_reassociation_validation", [])
        entry = {"input_index": int(index), **metrics}
        if entry not in rows:
            rows.append(entry)
        return True

    def forward(self, x, edge_cache, radial_feat):
        conv = self._convolution
        if conv.radial_hidden_proj is not None:
            rad_feat = conv.radial_hidden_proj(radial_feat)
        else:
            rad_feat = radial_feat
        mixer = conv.radial_degree_mixer
        if mixer is None:
            kc = rad_feat
            cb = rad_feat.new_zeros(1)
            rank = 0
        else:
            kc = torch.matmul(rad_feat.reshape(rad_feat.shape[0], -1), mixer.weight)
            cb = mixer.channel_basis.reshape(-1).detach()
            rank = mixer.rank

        x_local = self.region(
            x.contiguous(),
            edge_cache.src,
            edge_cache.D_full,
            kc.contiguous(),
            cb.contiguous(),
            conv.lmax,
            conv.n_focus,
            rank,
        ).view(
            conv.n_focus,
            edge_cache.src.shape[0],
            3 * conv.lmax + 1,
            conv.so2_focus_dim,
        )
        focus_gate_src = x_local[:, :, 0, :]
        for so2_linear, inter_norm, non_linear in zip(
            conv.so2_linears,
            conv.so2_inter_norms,
            conv.non_linearities,
            strict=True,
        ):
            residual = x_local
            x_local = non_linear(so2_linear(inter_norm(x_local)))
            x_local = residual + x_local
        if conv.focus_compete and conv.n_focus > 1:
            alpha = conv._focus_alpha(focus_gate_src.transpose(0, 1))
            x_local = x_local * alpha.transpose(0, 1).to(
                dtype=x_local.dtype
            ).unsqueeze(-1).unsqueeze(-1)
        x_local = x_local.permute(1, 0, 2, 3)
        self._detail["captured_output_bytes"] = (
            x_local.numel() * x_local.element_size()
            + rad_feat.numel() * rad_feat.element_size()
        )
        return x_local, rad_feat


def refresh(model, options) -> None:
    # This boundary has no CAP-dependent buffers.  A ROB1 recapture reuses the
    # installed operator and live edge_cache tensors.
    for module in model.modules():
        adapter = getattr(module, "_opt4_fasteq_rotate_mix", None)
        if isinstance(adapter, _FastEqSO2RotateMix):
            adapter.region.signatures.clear()


def install(model, passes, report, options):
    if "fasteq_so2_rotate_mix" not in passes:
        return
    try:
        from deepmd.kernels.triton.sezm.so2_value_path import (
            SO2_VALUE_PATH_TRITON_AVAILABLE,
            _is_supported,
        )
    except Exception as exc:
        raise FusionSetupError("DeepMD SO2 Triton primitive is unavailable") from exc
    if not SO2_VALUE_PATH_TRITON_AVAILABLE:
        raise FusionSetupError("DeepMD SO2 Triton primitive is unavailable")

    modules = []
    for path, module in list(model.named_modules()):
        if type(module).__name__ != "SO2Convolution":
            continue
        if not _is_supported(module):
            raise FusionSetupError(
                f"DPA4 SO2 module {path!r} is outside the validated FastEq layout"
            )
        if module._triton_value_path is not None or module._cute_value_path is not None:
            raise FusionSetupError(
                "fasteq_so2_rotate_mix requires the released value path; do not set "
                "DP_TRITON_INFER or DP_CUTE_INFER"
            )
        detail = {
            "module": path,
            "lmax": int(module.lmax),
            "mmax": int(module.mmax),
            "focus_dim": int(module.so2_focus_dim),
            "n_focus": int(module.n_focus),
            "radial_mixer_rank": int(
                getattr(module.radial_degree_mixer, "rank", 0)
            ),
            "validated_shapes": 0,
            "benchmark_requested": report.get("benchmark_boundaries", False),
        }
        module._opt4_fasteq_rotate_mix = _FastEqSO2RotateMix(module, detail)
        modules.append(detail)
    record(
        report,
        "fasteq_so2_rotate_mix",
        len(modules),
        "deepmd-triton-so2-rotate-mix-explicit-vjp",
        modules=modules,
        fused_boundaries=[
            "dynamic-source-gather",
            "wigner-rotate-to-local",
            "degree-radial-broadcast-multiply",
            "focus-major-store",
        ],
        so2_gemm="released-path",
        attention="released-path",
        destination_reduce="released-path",
        global_triton_infer=False,
        backward="explicit-node-wigner-radial-vjp",
        replay_runtime_compile=False,
    )
