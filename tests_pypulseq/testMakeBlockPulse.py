import numpy as np
import pytest
import pypulseq as pp

from . import assert_equal


class TestMakeBlockPulse:
    def test_invalid_use_error(self):
        with pytest.raises(Exception):
            pp.make_block_pulse(np.pi, duration=1e-3, use='foo')

    def test_bandwidth_and_duration_error(self):
        with pytest.raises(Exception):
            pp.make_block_pulse(np.pi)

    def test_generation_methods(self):
        pulse = pp.make_block_pulse(np.pi, duration=1e-3)
        assert_equal(pulse.shape_dur, 1e-3)
        pulse = pp.make_block_pulse(np.pi, bandwidth=0.3e3)
        assert_equal(pulse.shape_dur, 1 / (4 * 0.3e3), abs_tol=1e-6)
        pulse = pp.make_block_pulse(np.pi, bandwidth=1e3, time_bw_product=5)
        assert_equal(pulse.shape_dur, 5 / 1e3, abs_tol=1e-6)

    def test_amp_calculation(self):
        assert_equal(abs(pp.make_block_pulse(np.pi, duration=1e-3).signal[-1]), 500, abs_tol=1e-3)
        assert_equal(abs(pp.make_block_pulse(0.5 * np.pi, duration=1e-3).signal[-1]), 250, abs_tol=1e-3)
        assert_equal(abs(pp.make_block_pulse(0.5 * np.pi, duration=2e-3).signal[-1]), 125, abs_tol=1e-3)
