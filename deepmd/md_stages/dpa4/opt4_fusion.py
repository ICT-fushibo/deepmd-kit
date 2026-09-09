"""Selective SeZM/SO2 boundaries; never enable the global Triton policy."""
import torch
from torch import nn
from md_benchmark.opt4_fx import CheckedRegion, checked_boundary
from md_benchmark.opt4_ops import focus_pack
from md_benchmark.opt4_registry import FusionSetupError, record


class Rotation(nn.Module):
    def __init__(self, module, back=False, fused=False):
        super().__init__()
        self.register_buffer("indices", module.coeff_index_m, persistent=False)
        self.dim, self.lmax = module.ebed_dim_full, module.lmax
        # Wigner matrices live in the block-diagonal SO(3) subspace. Off-degree
        # entries are structural zeros, not independent geometry variables.
        degree = torch.arange(self.dim, device=module.coeff_index_m.device).sqrt().floor().long()
        self.register_buffer("degree_mask", degree[:, None] == degree[None, :], persistent=False)
        self.back, self.fused = back, fused

    def forward(self, x, index_or_wigner, wigner=None):
        if self.fused:
            from .opt4_rotation import rotation
            if self.back:
                return rotation(x, self.indices, index_or_wigner, self.indices, self.degree_mask, self.lmax, True)
            return rotation(x, index_or_wigner, wigner, self.indices, self.degree_mask, self.lmax, False)
        if self.back:
            matrix = (index_or_wigner[:, :self.dim, :self.dim] * self.degree_mask).index_select(2, self.indices)
            return torch.bmm(matrix, x.transpose(1, 2).reshape(x.shape[0], x.shape[2], -1))
        matrix = (wigner[:, :self.dim, :self.dim] * self.degree_mask).index_select(1, self.indices)
        return torch.bmm(matrix, x.index_select(0, index_or_wigner))


def install(model, passes, report):
    details = {p: [] for p in passes}
    for path, module in list(model.named_modules()):
        if type(module).__name__ != "SO2Convolution":
            continue
        if module.edge_cartesian or not module.needs_local_frame:
            continue
        if module.training or module.use_triton_infer or module.use_cute_infer:
            raise FusionSetupError("Opt4 requires an eager eval SeZM checkpoint with global acceleration disabled")
        if "so2_rotation" in passes and module.mmax == 1 and module.compute_dtype == torch.float32:
            from .opt4_validation import RotationVJPComparison
            if not hasattr(torch.library, "triton_op"):
                raise FusionSetupError("this installed PyTorch lacks torch.library.triton_op")
            detail = {"module": path, "forward": {"benchmark_requested":report.get("benchmark_boundaries",False)}, "back": {"benchmark_requested":report.get("benchmark_boundaries",False)}}
            detail["forward"]["validation_reduction"] = "shared-native-index-put-accumulate; original float tolerances"
            module._opt4_rotation_to = CheckedRegion(Rotation(module), detail["forward"], Rotation(module, fused=True),
                                                    validation_context=RotationVJPComparison)
            module._opt4_rotation_back = CheckedRegion(Rotation(module, back=True), detail["back"], Rotation(module, back=True, fused=True))
            details["so2_rotation"].append(detail)
        if "so2_epilogue" in passes and module.radial_degree_mixer is None and module.node_wise_grid_product is None:
            module._opt4_epilogue = True
            pack_detail={"module":path,"benchmark_requested":report.get("benchmark_boundaries",False)}
            module._opt4_focus_op=checked_boundary(
                lambda x,r,f:(x*r).reshape(x.shape[0],x.shape[1],f,x.shape[2]//f).permute(2,0,1,3).contiguous(),focus_pack,pack_detail)
            parameter = next(module.parameters())
            module.register_buffer("_opt4_minus_one", parameter.new_tensor(-1.), persistent=False)
            gates = []
            for i, gate in enumerate(module.non_linearities):
                if type(gate).__name__ == "GatedActivation":
                    info = {"module": f"{path}.non_linearities.{i}","benchmark_requested":report.get("benchmark_boundaries",False)}
                    module.non_linearities[i] = CheckedRegion(gate, info)
                    gates.append(info)
            details["so2_epilogue"].append({"module": path, "boundaries": ["radial-focus-pack", "bias-correction", "scaled-residual"], "gates": gates,"pack":pack_detail})
    for p, modules in details.items():
        record(report, p, len(modules), "triton-ieee-autograd", modules=modules,
               precision="checkpoint dtype unchanged; rotation requires FP32 and mmax=1", gemm="SO2Linear unchanged",
               fusion_scope="forward-only" if p == "so2_rotation" else "forward-and-backward",
               backward_policy="native-bmm-sorted-index-put-vjp" if p == "so2_rotation" else "compiled")
