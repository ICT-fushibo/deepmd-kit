"""Contracts for DPA4 GPU-resident Opt1 MD."""

from __future__ import annotations

import inspect
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from ase import Atoms

from deepmd.md_stages.dpa4 import opt1
from md_benchmark.md_route import MDConfig, MDRunRequest


def _atoms() -> Atoms:
    return Atoms(
        "Cu2",
        positions=[[0.0, 0.0, 0.0], [1.8, 1.8, 1.8]],
        cell=np.eye(3) * 7.2,
        pbc=True,
    )


def test_route_rejects_non_gpu_resident_backend_before_cuda() -> None:
    request = MDRunRequest(
        model="dpa4",
        stage="opt1",
        model_path="model.pt",
        atoms=_atoms(),
        config=MDConfig(steps=1, observation_steps=(1,), dtype="float64"),
        backend="native",
    )
    with pytest.raises(ValueError, match="backend='gpu-resident'"):
        opt1.run_md(request)


def test_route_rejects_non_fp64_md_state_before_cuda() -> None:
    request = MDRunRequest(
        model="dpa4",
        stage="opt1",
        model_path="model.pt",
        atoms=_atoms(),
        config=MDConfig(steps=1, observation_steps=(1,), dtype="float32"),
        backend="gpu-resident",
    )
    with pytest.raises(ValueError, match="physical MD state to float64"):
        opt1.run_md(request)


@pytest.mark.parametrize("path", ["model.pt2", "model.pth", "model.ckpt"])
def test_opt1_rejects_precompiled_or_non_raw_artifacts(path: str) -> None:
    with pytest.raises(ValueError, match="raw '.pt'"):
        opt1._require_raw_pt(path)


def test_route_dispatches_dpa4_opt1(monkeypatch) -> None:
    import deepmd.md_route as route

    request = MDRunRequest(
        model="dpa4",
        stage="opt1",
        model_path="model.pt",
        atoms=_atoms(),
        config=MDConfig(steps=1, observation_steps=(1,), dtype="float64"),
        backend="gpu-resident",
    )
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
        "module_prefix": "deepmd.md_stages.dpa4",
    }


def test_hot_force_path_has_no_host_or_numpy_conversion() -> None:
    source = inspect.getsource(opt1.DPA4EnergyForceEvaluator.__call__)
    assert ".cpu(" not in source
    assert ".numpy(" not in source
    assert ".item(" not in source


def test_force_path_skips_atomic_virial_but_keeps_total_virial(monkeypatch) -> None:
    class FakeModel:
        kwargs = None

        def __call__(self, positions, atom_types, **kwargs):
            self.kwargs = kwargs
            assert positions.shape == (1, 2, 3)
            assert atom_types.shape == (1, 2)
            return {
                "force": torch.ones((1, 2, 3), dtype=torch.float64),
                "energy": torch.tensor([[-3.25]], dtype=torch.float64),
                "virial": torch.arange(9, dtype=torch.float64).reshape(1, 9),
            }

    evaluator = object.__new__(opt1.DPA4EnergyForceEvaluator)
    evaluator.device = torch.device("cpu")
    evaluator.model_dtype = torch.float64
    evaluator.atom_types = torch.zeros((1, 2), dtype=torch.long)
    evaluator.cell = torch.eye(3, dtype=torch.float64).reshape(1, 3, 3)
    evaluator.fparam = None
    evaluator.aparam = None
    evaluator.charge_spin = None
    evaluator._model = FakeModel()
    monkeypatch.setattr(torch.cuda, "device", lambda _device: nullcontext())

    force, energy, virial = evaluator(
        torch.zeros((2, 3), dtype=torch.float64)
    )

    assert evaluator._model.kwargs["do_atomic_virial"] is False
    assert force.shape == (2, 3)
    assert energy.item() == pytest.approx(-3.25)
    torch.testing.assert_close(
        virial, torch.arange(9, dtype=torch.float64).reshape(3, 3)
    )


@pytest.mark.parametrize(
    ("model_dtype", "expected"),
    [(torch.float32, "float32"), (torch.float64, "float64")],
)
def test_metadata_reports_actual_interface_and_checkpoint_precision(
    model_dtype, expected
) -> None:
    evaluator = SimpleNamespace(
        model_dtype=model_dtype,
        descriptor_precision="float32",
        fitting_precision="float32",
        do_atomic_virial=False,
    )

    metadata = opt1._evaluator_metadata(evaluator)

    assert metadata == {
        "model_precision": expected,
        "model_interface_precision": expected,
        "checkpoint_descriptor_precision": "float32",
        "checkpoint_fitting_precision": "float32",
        "do_atomic_virial": False,
    }


def test_opt1_matches_high_precision_baseline_interface() -> None:
    import deepmd.md_route as route

    assert opt1._DEEPMD_OPT1_ENV["DP_INTERFACE_PREC"] == "high"
    assert route._DEEPMD_BASELINE_ENV["DP_INTERFACE_PREC"] == "high"


def test_integrators_are_the_ase_aligned_gpu_implementations() -> None:
    from deepmd.md_stages.dpa3 import opt1 as shared

    assert opt1.GPUVelocityVerletBerendsen is shared.GPUVelocityVerletBerendsen
    assert opt1.GPUNoseHooverChain is shared.GPUNoseHooverChain


def test_matbench_snapshot_contains_step_energy_force_and_stress() -> None:
    atoms = _atoms()
    positions = torch.tensor(atoms.positions, dtype=torch.float64)
    state = opt1.GPUMDState(
        positions=positions,
        momenta=torch.zeros_like(positions),
        forces=torch.ones_like(positions),
        potential_energy=torch.tensor(-3.25, dtype=torch.float32),
        virial=torch.diag(torch.tensor([1.0, 2.0, 3.0])),
    )
    frame = opt1._state_to_atoms(atoms, state, step=0)

    assert frame.info["md_step"] == 0
    assert frame.get_potential_energy() == pytest.approx(-3.25)
    np.testing.assert_array_equal(frame.get_forces(), np.ones((2, 3)))
    volume = atoms.get_volume()
    np.testing.assert_allclose(
        frame.get_stress(), [-1 / volume, -2 / volume, -3 / volume, 0, 0, 0]
    )
