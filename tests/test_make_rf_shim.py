from pathlib import Path
import tempfile

import numpy as np
import pypulseq_matlab_like as pp
import pytest

from util import assert_equal


class TestMakeRfShim:
    def test_basic_shim(self):
        vector = [1, 0.8, 0.9, 1.1]
        shim = pp.make_rf_shim(vector)
        assert shim.type == 'rf_shim'
        assert_equal(shim.shim_vector, np.asarray(vector).reshape(-1, 1), abs_tol=1e-10)

    def test_row_to_column(self):
        assert pp.make_rf_shim([1, 0.5, 0.7, 0.3]).shim_vector.shape[1] == 1

    def test_column_input(self):
        vector = np.array([[1], [0.5], [0.7]])
        assert_equal(pp.make_rf_shim(vector).shim_vector, vector, abs_tol=1e-10)

    def test_complex_shim(self):
        vector = np.array([1 + 0.5j, 0.8 - 0.3j, 0.9 + 0.1j])
        assert_equal(pp.make_rf_shim(vector).shim_vector, vector.reshape(-1, 1), abs_tol=1e-10)

    def test_single_element(self):
        assert_equal(pp.make_rf_shim(1.5).shim_vector, [[1.5]], abs_tol=1e-10)

    def test_addblock_getblock_roundtrip(self):
        vector = np.array([1, np.exp(1j * np.pi / 2), 0.8 * np.exp(-1j * 2.5), 0.5])
        sequence = pp.Sequence()
        sequence.add_block(pp.make_block_pulse(np.pi / 2, duration=1e-3, use='excitation'), pp.make_rf_shim(vector))
        assert_equal(sequence.get_block(1).rf_shim.shim_vector.reshape(-1), vector, abs_tol=1e-12)

    def test_writeread_getblock_roundtrip(self):
        vector = np.array([1, np.exp(1j * np.pi / 2), 0.8 * np.exp(-1j * 2.5), 0.5])
        sequence = pp.Sequence()
        sequence.add_block(pp.make_block_pulse(np.pi / 2, duration=1e-3, use='excitation'), pp.make_rf_shim(vector))
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / 'test_rfshim_roundtrip.seq'
            sequence.write(str(path))
            loaded = pp.Sequence()
            loaded.read(str(path))
            assert_equal(loaded.get_block(1).rf_shim.shim_vector.reshape(-1), vector, abs_tol=1e-5)
