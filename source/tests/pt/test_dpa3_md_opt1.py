"""Lightweight contracts for DPA3 GPU-resident Opt1 MD."""

from __future__ import annotations

import inspect

import numpy as np
import pytest
import torch
from ase import Atoms, units
from ase.calculators.calculator import Calculator, all_changes
from ase.md.nose_hoover_chain import NoseHooverChainNVT
from ase.md.nvtberendsen import NVTBerendsen

from deepmd.md_stages.dpa3 import opt1
from md_benchmark.md_route import MDConfig, MDRunRequest


class _HarmonicASECalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def calculate(
        self,
        atoms=None,
        properties=("energy", "forces"),
        system_changes=all_changes,
    ) -> None:
        super().calculate(atoms, properties, system_changes)
        positions = np.asarray(self.atoms.positions)
        self.results = {
            "energy": 0.5 * float(np.square(positions).sum()),
            "forces": -positions.copy(),
        }


class _HarmonicTensorEvaluator:
    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.model_dtype = torch.float64
        self.neighbor_backend = "test"
        self.seen: list[torch.Tensor] = []

    def __call__(self, positions):
        assert isinstance(positions, torch.Tensor)
        assert positions.device == self.device
        self.seen.append(positions)
        return (
            -positions,
            0.5 * positions.square().sum(),
            torch.zeros((3, 3), dtype=positions.dtype, device=positions.device),
        )


def _atoms_and_state() -> tuple[Atoms, opt1.GPUMDState, torch.Tensor]:
    atoms = Atoms(
        "H2",
        positions=[[0.2, 0.1, 0.3], [0.7, 0.4, 0.2]],
        cell=np.eye(3) * 8.0,
        pbc=True,
    )
    momenta = np.array([[0.15, -0.12, 0.03], [-0.05, 0.02, 0.08]])
    atoms.set_momenta(momenta)
    masses = torch.tensor(atoms.get_masses(), dtype=torch.float64)
    state = opt1.GPUMDState(
        torch.tensor(atoms.positions, dtype=torch.float64),
        torch.tensor(momenta, dtype=torch.float64),
    )
    return atoms, state, masses


def test_berendsen_equations_match_ase() -> None:
    atoms, state, masses = _atoms_and_state()
    atoms.calc = _HarmonicASECalculator()
    ase_md = NVTBerendsen(
        atoms,
        timestep=0.5 * units.fs,
        temperature_K=300.0,
        taut=100.0 * units.fs,
    )
    ase_md.run(4)

    evaluator = _HarmonicTensorEvaluator(torch.device("cpu"))
    integrator = opt1.GPUVelocityVerletBerendsen(
        masses,
        timestep_fs=0.5,
        temperature_k=300.0,
        thermostat_time_fs=100.0,
    )
    for _ in range(4):
        integrator.step(state, evaluator)

    np.testing.assert_allclose(state.positions.numpy(), atoms.positions, atol=2e-14)
    np.testing.assert_allclose(state.momenta.numpy(), atoms.get_momenta(), atol=2e-14)
    assert all(value.device.type == "cpu" for value in evaluator.seen)


def test_nose_hoover_chain_equations_match_ase() -> None:
    atoms, state, masses = _atoms_and_state()
    atoms.calc = _HarmonicASECalculator()
    ase_md = NoseHooverChainNVT(
        atoms,
        timestep=0.25 * units.fs,
        temperature_K=300.0,
        tdamp=100.0 * units.fs,
    )
    ase_md.run(5)

    evaluator = _HarmonicTensorEvaluator(torch.device("cpu"))
    integrator = opt1.GPUNoseHooverChain(
        masses,
        timestep_fs=0.25,
        temperature_k=300.0,
        thermostat_time_fs=100.0,
    )
    for _ in range(5):
        integrator.step(state, evaluator)

    np.testing.assert_allclose(state.positions.numpy(), atoms.positions, atol=2e-14)
    np.testing.assert_allclose(state.momenta.numpy(), atoms.get_momenta(), atol=2e-14)


def test_snapshot_has_matbench_step_energy_force_stress() -> None:
    atoms, state, _ = _atoms_and_state()
    evaluator = _HarmonicTensorEvaluator(torch.device("cpu"))
    opt1._evaluate_state(state, evaluator)
    state.virial = torch.diag(torch.tensor([1.0, 2.0, 3.0]))
    frame = opt1._state_to_atoms(atoms, state, step=0)

    assert frame.info["md_step"] == 0
    assert np.isfinite(frame.get_potential_energy())
    assert frame.get_forces().shape == (2, 3)
    np.testing.assert_allclose(
        frame.get_stress(), [-1 / 512, -2 / 512, -3 / 512, 0, 0, 0]
    )


def test_route_rejects_non_gpu_resident_backend_before_cuda() -> None:
    atoms, _, _ = _atoms_and_state()
    request = MDRunRequest(
        model="dpa3",
        stage="opt1",
        model_path="model.pth",
        atoms=atoms,
        config=MDConfig(steps=1, observation_steps=(1,)),
        backend="native",
    )
    with pytest.raises(ValueError, match="backend='gpu-resident'"):
        opt1.run_md(request)


def test_route_dispatches_dpa3_opt1(monkeypatch) -> None:
    import deepmd.md_route as route

    atoms, _, _ = _atoms_and_state()
    request = MDRunRequest(
        model="dpa3",
        stage="opt1",
        model_path="model.pth",
        atoms=atoms,
        config=MDConfig(steps=1, observation_steps=(1,)),
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
        "module_prefix": "deepmd.md_stages.dpa3",
    }


def test_hot_step_paths_have_no_host_or_numpy_conversion() -> None:
    """Guard the state/evaluator hot path against accidental ASE round-trips."""
    hot_functions = (
        opt1.DPA3EnergyForceEvaluator.__call__,
        opt1.GPUVelocityVerletBerendsen.step,
        opt1.GPUNoseHooverChain.step,
    )
    for function in hot_functions:
        source = inspect.getsource(function)
        assert ".cpu(" not in source
        assert ".numpy(" not in source
        assert ".item(" not in source


def test_opt1_neighbor_builder_can_disable_compile() -> None:
    from deepmd.pt.utils.nv_nlist import NvNeighborList

    assert NvNeighborList().compile_truncation is True
    assert NvNeighborList(compile_truncation=False).compile_truncation is False
