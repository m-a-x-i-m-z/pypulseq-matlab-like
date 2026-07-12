import numpy as np
import pypulseq as pp

from . import assert_equal


class TestPts2waveform:
    def test_two_point_ramp(self):
        raster = 10e-6
        times, amplitudes = np.array([0, 1e-3]), np.array([0, 1000])
        waveform = pp.points_to_waveform(amplitudes, raster, times)
        assert_equal(len(waveform), round((max(times) - min(times)) / raster) - 1, abs_tol=1)
        assert_equal(waveform[0], 1000 * 0.5 * raster / 1e-3, abs_tol=1e-6)

    def test_constant_amplitude(self):
        waveform = pp.points_to_waveform(np.array([42000, 42000]), 10e-6, np.array([0, 5e-4]))
        assert_equal(waveform, 42000 * np.ones_like(waveform), abs_tol=1e-6)

    def test_multi_point(self):
        waveform = pp.points_to_waveform(np.array([0, 10000, 10000, 0]), 10e-6, np.array([0, 1e-4, 2e-4, 3e-4]))
        assert len(waveform) > 0
        assert np.all(np.isfinite(waveform))

    def test_output_vector(self):
        waveform = pp.points_to_waveform(np.array([0, 5000]), 10e-6, np.array([0, 2e-4]))
        assert np.issubdtype(waveform.dtype, np.number)
        assert waveform.ndim == 1
