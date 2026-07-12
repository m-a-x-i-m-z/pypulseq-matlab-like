import numpy as np
import pytest
from pypulseq.compress_shape import compress_shape
from pypulseq.decompress_shape import decompress_shape

from . import assert_equal


class TestCompressShape:
    def test_constant_waveform(self):
        waveform = np.ones(100)
        shape = compress_shape(waveform)
        assert shape.num_samples == 100
        assert len(shape.data) < len(waveform)

    def test_linear_ramp(self):
        waveform = np.linspace(0, 1, 100)
        shape = compress_shape(waveform)
        assert shape.num_samples == 100
        assert len(shape.data) < len(waveform)

    def test_random_waveform(self):
        waveform = np.random.default_rng(42).standard_normal(100)
        shape = compress_shape(waveform)
        assert shape.num_samples == 100
        assert len(shape.data) == shape.num_samples

    def test_short_waveform_no_compression(self):
        waveform = np.array([1, 2, 3, 4])
        shape = compress_shape(waveform)
        assert shape.num_samples == 4
        assert_equal(shape.data, waveform, abs_tol=1e-10)

    def test_short_waveform_force_compression(self):
        waveform = np.ones(4)
        shape = compress_shape(waveform, True)
        assert shape.num_samples == 4
        assert len(shape.data) <= len(waveform)

    def test_inf_sample_error(self):
        with pytest.raises(Exception):
            compress_shape(np.array([1, 2, np.inf, 4]))

    def test_nan_sample_error(self):
        with pytest.raises(Exception):
            compress_shape(np.array([1, np.nan, 3, 4, 5]))

    def test_num_samples_field(self):
        for count in (1, 2, 5, 50, 200):
            shape = compress_shape(np.sin(np.linspace(0, 2 * np.pi, count)))
            assert shape.num_samples == count

    def test_round_trip(self):
        waveform = np.sin(np.linspace(0, 4 * np.pi, 200))
        assert_equal(decompress_shape(compress_shape(waveform)), waveform, abs_tol=1e-6)
