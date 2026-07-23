import pytest
import pypulseq_matlab_like as pp

from util import assert_equal


class TestMakeAdc:
    def test_makeAdc_without_timing_should_fail(self):
        with pytest.raises(Exception):
            pp.make_adc(128)

    def test_makeAdc_given_numSamples_and_Dwell_is_valid(self):
        adc = pp.make_adc(128, dwell=10e-6)
        assert_equal(adc.num_samples, 128)
        assert_equal(adc.dwell, 10e-6)
        assert set(vars(adc)) == {'type', 'num_samples', 'dwell', 'delay', 'freq_offset', 'phase_offset', 'freq_ppm', 'phase_ppm', 'dead_time', 'phase_modulation'}

    def test_makeAdc_given_numSamples_and_Duration_is_valid(self):
        adc = pp.make_adc(128, duration=1280e-6)
        assert_equal(adc.num_samples, 128)
        assert_equal(adc.dwell, 10e-6)
        assert set(vars(adc)) == {'type', 'num_samples', 'dwell', 'delay', 'freq_offset', 'phase_offset', 'freq_ppm', 'phase_ppm', 'dead_time', 'phase_modulation'}
