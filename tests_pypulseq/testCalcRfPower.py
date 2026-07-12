import numpy as np
import pypulseq as pp

from . import assert_equal


class TestCalcRfPower:
    def test_block_pulse_energy(self):
        duration = 1e-3
        rf = pp.make_block_pulse(np.pi / 2, duration=duration)
        amplitude = np.max(np.abs(rf.signal))
        total_energy, peak_power, rf_rms = pp.calc_rf_power(rf)
        assert_equal(total_energy, amplitude**2 * duration, rel_tol=0.02)
        assert_equal(peak_power, amplitude**2, rel_tol=0.02)
        assert_equal(rf_rms, amplitude, rel_tol=0.02)

    def test_rms_relation(self):
        rf = pp.make_sinc_pulse(np.pi / 2, duration=4e-3, time_bw_product=4)
        total_energy, _, rf_rms = pp.calc_rf_power(rf)
        assert_equal(rf_rms, np.sqrt(total_energy / rf.shape_dur), rel_tol=1e-6)

    def test_dt_consistency(self):
        rf = pp.make_block_pulse(np.pi / 2, duration=1e-3)
        energy_one, _, _ = pp.calc_rf_power(rf, 1e-6)
        energy_two, _, _ = pp.calc_rf_power(rf, 0.5e-6)
        assert_equal(energy_one, energy_two, rel_tol=0.02)

    def test_shaped_less_than_block(self):
        duration = 4e-3
        block = pp.make_block_pulse(np.pi / 2, duration=duration)
        gauss = pp.make_gauss_pulse(np.pi / 2, duration=duration)
        energy_block = pp.calc_rf_power(block)[0]
        energy_gauss = pp.calc_rf_power(gauss)[0]
        assert energy_block > 0 and np.isfinite(energy_block)
        assert energy_gauss > 0 and np.isfinite(energy_gauss)

    def test_single_output(self):
        rf = pp.make_block_pulse(np.pi / 6, duration=0.5e-3)
        total_energy = pp.calc_rf_power(rf)[0]
        assert np.isscalar(total_energy)
        assert total_energy > 0
