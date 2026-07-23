import pytest

import pypulseq_matlab_like as pp

from util import assert_equal


class TestOpts:
    def test_default_fields(self):
        system = pp.Opts()
        fields = ('max_grad', 'max_slew', 'max_b1', 'rise_time', 'rf_dead_time', 'rf_ringdown_time', 'adc_dead_time', 'adc_raster_time', 'rf_raster_time', 'grad_raster_time', 'block_duration_raster', 'gamma', 'B0')
        assert all(hasattr(system, field) for field in fields)

    def test_default_values(self):
        system = pp.Opts()
        assert_equal(system.max_grad, pp.convert(40, 'mT/m', to_unit='Hz/m'), rel_tol=0.01)
        assert_equal(system.grad_raster_time, 10e-6, abs_tol=1e-10)
        assert_equal(system.rf_raster_time, 1e-6, abs_tol=1e-10)

    def test_custom_maxGrad(self):
        system = pp.Opts(max_grad=30, grad_unit='mT/m')
        assert_equal(system.max_grad, pp.convert(30, 'mT/m', to_unit='Hz/m'), rel_tol=0.01)

    def test_custom_maxSlew(self):
        system = pp.Opts(max_slew=100, slew_unit='T/m/s')
        assert_equal(system.max_slew, pp.convert(100, 'T/m/s', to_unit='Hz/m/s'), rel_tol=0.01)

    def test_riseTime(self):
        system = pp.Opts(rise_time=250e-6, max_grad=40, grad_unit='mT/m')
        assert_equal(system.rise_time, 250e-6)
        assert_equal(system.max_slew, system.max_grad / 250e-6, rel_tol=0.01)

    def test_gamma_B0(self):
        system = pp.Opts()
        assert_equal(system.gamma, 42576000, abs_tol=1)
        assert_equal(system.B0, 1.5, abs_tol=0.01)

    def test_custom_rfDeadTime(self):
        assert_equal(pp.Opts(rf_dead_time=100e-6).rf_dead_time, 100e-6)

    def test_consistency(self):
        first, second = pp.Opts(), pp.Opts()
        assert first.max_grad == second.max_grad
        assert first.max_slew == second.max_slew
        assert first.grad_raster_time == second.grad_raster_time
