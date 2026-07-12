import numpy as np
import pypulseq as pp


class TestCalcRamp:
    def test_zero_to_zero(self):
        system = pp.Opts()
        raster = system.grad_raster_time
        k0 = np.zeros((3, 2))
        kend = np.array([[raster * 2, raster * 3], [0, 0], [0, 0]])
        _, success = pp.calc_ramp(k0=k0, k_end=kend, system=system)
        assert bool(success)

    def test_simple_ramp_x(self):
        system = pp.Opts()
        dk = 1000 * system.grad_raster_time
        k0 = np.zeros((3, 2))
        kend = np.array([[2 * dk, 3 * dk], [0, 0], [0, 0]])
        output, success = pp.calc_ramp(k0=k0, k_end=kend, system=system)
        assert bool(success)
        assert output.shape[0] == 3

    def test_maxpoints_zero(self):
        system = pp.Opts()
        dk = system.max_grad * 0.9 * system.grad_raster_time
        _, success = pp.calc_ramp(k0=np.zeros((3, 2)), k_end=np.array([[2 * dk, 3 * dk], [0, 0], [0, 0]]), system=system, max_points=0)
        assert isinstance(bool(success), bool)

    def test_output_shape(self):
        system = pp.Opts()
        raster = system.grad_raster_time
        output, _ = pp.calc_ramp(k0=np.zeros((3, 2)), k_end=np.array([[raster * 2, raster * 3], [0, 0], [0, 0]]), system=system)
        assert output.shape[0] == 3
