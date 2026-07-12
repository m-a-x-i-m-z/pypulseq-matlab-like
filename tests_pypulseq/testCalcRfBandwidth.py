import numpy as np
import pypulseq as pp

from . import assert_equal


def _calc_rf_bandwidth_matlab_outputs(rf, cutoff=0.5):
    return pp.calc_rf_bandwidth(rf, cutoff=cutoff, return_all=True)


class TestCalcRfBandwidth:
    def test_block_pulse_bw(self):
        duration = 1e-3
        rf = pp.make_block_pulse(np.pi / 2, duration=duration)
        bandwidth, center_frequency, *_ = _calc_rf_bandwidth_matlab_outputs(rf)
        assert_equal(bandwidth, 1 / duration, rel_tol=0.3)
        assert_equal(center_frequency, 0, abs_tol=50)

    def test_sinc_pulse_bw(self):
        duration = 4e-3
        rf = pp.make_sinc_pulse(np.pi / 2, duration=duration, time_bw_product=4)
        bandwidth, *_ = _calc_rf_bandwidth_matlab_outputs(rf)
        assert_equal(bandwidth, 4 / duration, rel_tol=0.15)

    def test_custom_cutoff(self):
        rf = pp.make_block_pulse(np.pi / 2, duration=1e-3)
        bandwidth_half, *_ = _calc_rf_bandwidth_matlab_outputs(rf, 0.5)
        bandwidth_quarter, *_ = _calc_rf_bandwidth_matlab_outputs(rf, 0.25)
        assert bandwidth_quarter > bandwidth_half

    def test_six_outputs(self):
        rf = pp.make_block_pulse(np.pi / 6, duration=0.5e-3)
        result = pp.calc_rf_bandwidth(rf, return_all=True)
        assert isinstance(result, tuple) and len(result) == 6
        bandwidth, center_frequency, spectrum, frequency, resampled_rf, times = result
        assert np.isscalar(bandwidth)
        assert np.isscalar(center_frequency)
        assert spectrum.ndim == 1
        assert frequency.ndim == 1
        assert resampled_rf.ndim == 1
        assert times.ndim == 1
        assert len(spectrum) == len(frequency)
        assert len(resampled_rf) == len(times)
