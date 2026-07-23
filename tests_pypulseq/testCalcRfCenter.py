import numpy as np
import pypulseq_matlab_like as pp

from . import assert_equal


class TestCalcRfCenter:
    def test_block_pulse_center(self):
        duration = 1e-3
        rf = pp.make_block_pulse(np.pi / 2, duration=duration)
        time_center, index_center, fractional_index = pp.calc_rf_center(rf, return_fractional_index=True)
        assert_equal(time_center, duration / 2, abs_tol=1e-6)
        assert index_center >= 0
        assert abs(fractional_index) <= 0.5

    def test_sinc_pulse_center(self):
        duration = 4e-3
        rf = pp.make_sinc_pulse(np.pi / 2, duration=duration, time_bw_product=4)
        time_center, _, _ = pp.calc_rf_center(rf, return_fractional_index=True)
        assert_equal(time_center, duration / 2, abs_tol=1e-6)

    def test_explicit_center(self):
        rf = pp.make_block_pulse(np.pi / 2, duration=1e-3)
        rf.center = 0.3e-3
        time_center, _, _ = pp.calc_rf_center(rf, return_fractional_index=True)
        assert_equal(time_center, 0.3e-3, abs_tol=1e-9)

    def test_three_outputs(self):
        rf = pp.make_block_pulse(np.pi / 4, duration=0.5e-3)
        time_center, index_center, fractional_index = pp.calc_rf_center(rf, return_fractional_index=True)
        assert np.isscalar(time_center)
        assert np.isscalar(index_center)
        assert np.isscalar(fractional_index)

    def test_gauss_pulse_center(self):
        duration = 3e-3
        rf = pp.make_gauss_pulse(np.pi / 2, duration=duration)
        time_center, _, _ = pp.calc_rf_center(rf, return_fractional_index=True)
        assert_equal(time_center, duration / 2, abs_tol=1e-6)
