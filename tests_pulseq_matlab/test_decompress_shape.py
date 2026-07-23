from types import SimpleNamespace

import numpy as np
from pypulseq_matlab_like.compress_shape import compress_shape
from pypulseq_matlab_like.decompress_shape import decompress_shape

from util import assert_equal


class TestDecompressShape:
    def test_round_trip_constant(self):
        waveform = np.ones(100)
        assert_equal(decompress_shape(compress_shape(waveform)), waveform, abs_tol=1e-6)

    def test_round_trip_linear(self):
        waveform = np.linspace(0, 1, 100)
        assert_equal(decompress_shape(compress_shape(waveform)), waveform, abs_tol=1e-6)

    def test_round_trip_sinusoid(self):
        waveform = np.sin(np.linspace(0, 4 * np.pi, 300))
        assert_equal(decompress_shape(compress_shape(waveform)), waveform, abs_tol=1e-6)

    def test_round_trip_random(self):
        waveform = np.random.default_rng(123).standard_normal(100)
        assert_equal(decompress_shape(compress_shape(waveform)), waveform, abs_tol=1e-6)

    def test_uncompressed_passthrough(self):
        shape = SimpleNamespace(num_samples=5, data=np.array([1, 2, 3, 4, 5]))
        assert_equal(decompress_shape(shape), np.array([1, 2, 3, 4, 5]))

    def test_force_decompression(self):
        waveform = np.linspace(0, 1, 50)
        shape = compress_shape(waveform, True)
        assert_equal(decompress_shape(shape, True), waveform, abs_tol=1e-6)

    def test_output_length(self):
        for count in (10, 50, 200):
            shape = compress_shape(np.sin(np.linspace(0, 2 * np.pi, count)))
            assert len(decompress_shape(shape)) == shape.num_samples
