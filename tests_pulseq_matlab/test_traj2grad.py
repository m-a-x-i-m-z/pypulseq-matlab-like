import numpy as np
import pypulseq_matlab_like as pp

from util import assert_equal


class TestTraj2grad:
    def test_linear_trajectory(self):
        system, count = pp.Opts(), 20
        trajectory = np.linspace(0, 1000, count).reshape(1, -1)
        gradient, _ = pp.traj_to_grad(trajectory, system=system)
        expected = (trajectory[0, 1] - trajectory[0, 0]) / system.grad_raster_time
        assert_equal(gradient[0], expected * np.ones(gradient.shape[1]), rel_tol=0.01)

    def test_output_dimensions(self):
        trajectory = np.linspace(0, 500, 30).reshape(1, -1)
        gradient, slew = pp.traj_to_grad(trajectory, system=pp.Opts())
        assert gradient.shape[1] == trajectory.shape[1] - 1
        assert slew.shape[1] == trajectory.shape[1] - 1

    def test_multi_channel(self):
        trajectory = np.vstack((np.linspace(0, 500, 20), np.linspace(0, 300, 20)))
        gradient, slew = pp.traj_to_grad(trajectory, system=pp.Opts())
        assert gradient.shape[0] == 2
        assert slew.shape[0] == 2

    def test_zero_trajectory(self):
        gradient, _ = pp.traj_to_grad(np.zeros((1, 10)), system=pp.Opts())
        assert_equal(gradient, np.zeros_like(gradient), abs_tol=1e-10)

    def test_custom_raster(self):
        system, raster = pp.Opts(), 5e-6
        trajectory = np.linspace(0, 100, 20).reshape(1, -1)
        first, _ = pp.traj_to_grad(trajectory, system=system, raster_time=raster)
        second, _ = pp.traj_to_grad(trajectory, system=system)
        assert_equal(first[0, 0], second[0, 0] * system.grad_raster_time / raster, rel_tol=0.01)
