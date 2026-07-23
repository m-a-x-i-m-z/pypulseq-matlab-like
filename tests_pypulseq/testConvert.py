import numpy as np
import pypulseq_matlab_like as pp

from . import assert_equal


class TestConvert:
    def test_gradient_mTm_to_Hzm(self):
        assert_equal(pp.convert(1, 'mT/m', to_unit='Hz/m'), 1e-3 * 42.576e6, abs_tol=1)

    def test_slew_Tms_to_Hzms(self):
        assert_equal(pp.convert(1, 'T/m/s', to_unit='Hz/m/s'), 42.576e6, abs_tol=1)

    def test_b1_T_to_Hz(self):
        assert_equal(pp.convert(1, 'T', to_unit='Hz'), 42.576e6, abs_tol=1)

    def test_b1_uT_to_Hz(self):
        assert_equal(pp.convert(1, 'uT', to_unit='Hz'), 1e-6 * 42.576e6, abs_tol=0.1)

    def test_round_trip_gradient(self):
        assert_equal(pp.convert(pp.convert(12345, 'Hz/m', to_unit='mT/m'), 'mT/m', to_unit='Hz/m'), 12345, abs_tol=1e-6)

    def test_round_trip_slew(self):
        assert_equal(pp.convert(pp.convert(170e6, 'Hz/m/s', to_unit='T/m/s'), 'T/m/s', to_unit='Hz/m/s'), 170e6, abs_tol=1e-3)

    def test_slew_mTmms_to_Hzms(self):
        assert_equal(pp.convert(1, 'mT/m/ms', to_unit='Hz/m/s'), 42.576e6, abs_tol=1)

    def test_grad_radmsmm_to_Hzm(self):
        assert_equal(pp.convert(2 * np.pi * 1e-6, 'rad/ms/mm', to_unit='Hz/m'), 1, abs_tol=1e-6)

    def test_slew_radmsmmms_to_Hzms(self):
        assert_equal(pp.convert(2 * np.pi * 1e-9, 'rad/ms/mm/ms', to_unit='Hz/m/s'), 1, abs_tol=1e-6)

    def test_default_toUnit_gradient(self):
        assert_equal(pp.convert(1, 'mT/m'), 1e-3 * 42.576e6, abs_tol=1)

    def test_default_toUnit_slew(self):
        assert_equal(pp.convert(1, 'T/m/s'), 42.576e6, abs_tol=1)

    def test_custom_gamma(self):
        assert_equal(pp.convert(1, 'T', gamma=10000, to_unit='Hz'), 10000, abs_tol=1e-6)

    def test_identity_conversion(self):
        assert_equal(pp.convert(42576, 'Hz/m', to_unit='Hz/m'), 42576, abs_tol=1e-10)

    def test_b1_mT_to_Hz(self):
        assert_equal(pp.convert(1, 'mT', to_unit='Hz'), 1e-3 * 42.576e6, abs_tol=1)
