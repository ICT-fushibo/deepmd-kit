import unittest
from types import SimpleNamespace

import torch

from deepmd.md_stages.dpa4.opt3 import DPA4WholeStepGraph
from deepmd.md_stages.dpa3.opt1 import GPUNoseHooverChain, GPUVelocityVerletBerendsen
from md_benchmark.stress_capture import deepmd_stress
from md_benchmark.stress_test_support import assert_replay_stable


class DPA4StressTests(unittest.TestCase):
    def test_zero_step_captures_stress_without_advancing_state(self):
        self._check_zero_step(GPUVelocityVerletBerendsen)

    def test_zero_step_preserves_nhc_state(self):
        self._check_zero_step(GPUNoseHooverChain)

    def _check_zero_step(self, integrator_type):
        runner = object.__new__(DPA4WholeStepGraph)
        runner.capture_stress = True
        runner.stress_volume = 60.0
        positions = torch.tensor(
            [[0.2, 0.5, 0.1], [0.7, 0.3, 0.8]], dtype=torch.float64, device="cuda:0"
        )
        runner.state = SimpleNamespace(
            positions=positions.clone(),
            momenta=torch.zeros_like(positions),
            forces=torch.zeros_like(positions),
            potential_energy=positions.new_zeros(()),
            virial=positions.new_zeros(3, 3),
            stress=positions.new_zeros(3, 3),
        )
        runner._integrator = integrator_type(
            torch.ones(2, device="cuda:0", dtype=torch.float64),
            timestep_fs=0.25,
            temperature_k=300,
            thermostat_time_fs=25,
        )
        runner.advance = positions.new_zeros(())
        runner.step_counter = torch.zeros((), dtype=torch.long, device="cuda:0")
        runner._last_edge_count = runner.step_counter.clone()
        runner._max_edge_count = runner.step_counter.clone()
        edge_count = torch.tensor(2, dtype=torch.long, device="cuda:0")
        runner._evaluate_positions = lambda p: (
            -p,
            0.5 * p.square().sum(),
            -(p.T @ p),
            edge_count,
        )

        def body():
            runner._step_body()
            return (
                runner.state.forces,
                runner.state.stress,
                runner.state.potential_energy,
            )

        assert_replay_stable(body)
        torch.testing.assert_close(runner.state.positions, positions)
        torch.testing.assert_close(runner.state.momenta, torch.zeros_like(positions))
        torch.testing.assert_close(
            runner.state.stress, deepmd_stress(-(positions.T @ positions), 60.0)
        )
        self.assertEqual(int(runner.step_counter), 0)
        self.assertIs(runner.state_tensors()["stress"], runner.state.stress)
        if isinstance(runner._integrator, GPUNoseHooverChain):
            self.assertEqual(float(runner._integrator.eta.abs().max()), 0)
            self.assertEqual(float(runner._integrator.p_eta.abs().max()), 0)


if __name__ == "__main__":
    unittest.main()
