import pytest

import numpy as np
import pypulseq_matlab_like as pp

from util import assert_equal


class TestSimRf:
    def test_90deg_excitation(self):
        rf = pp.make_block_pulse(np.pi / 2, duration=0.5e-3)
        mz_z, mz_xy, frequency, *_ = pp.sim_rf(rf)
        zero = np.argmin(abs(frequency))
        assert_equal(mz_z[zero], 0, abs_tol=0.15)
        assert_equal(abs(mz_xy[zero]), 1, abs_tol=0.15)

    def test_180deg_inversion(self):
        rf = pp.make_block_pulse(np.pi, duration=0.7e-3)
        mz_z, _, frequency, *_ = pp.sim_rf(rf)
        assert_equal(mz_z[np.argmin(abs(frequency))], -1, abs_tol=0.15)

    def test_output_dimensions(self):
        rf = pp.make_block_pulse(np.pi / 2, duration=0.5e-3)
        mz_z, mz_xy, frequency, ref_eff, mx_xy, my_xy = pp.sim_rf(rf)
        count = len(frequency)
        assert len(mz_z) == count
        assert len(mz_xy) == count
        assert len(ref_eff) == count
        assert len(mx_xy) == count
        assert len(my_xy) == count

    def test_sinc_pulse_profile(self):
        rf = pp.make_sinc_pulse(np.pi / 2, duration=4e-3, time_bw_product=4)
        mz_z, _, frequency, *_ = pp.sim_rf(rf, 0)
        assert mz_z[np.argmin(abs(frequency))] < 0.3
