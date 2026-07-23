import numpy as np
import pytest
import pypulseq_matlab_like as pp

from . import assert_equal


class TestMakeArbitraryRf:
    def test_rectangular_signal(self):
        system = pp.Opts()
        rf = pp.make_arbitrary_rf(np.ones(100), np.pi / 6, system=system)
        assert rf.type == 'rf'
        assert len(rf.signal) == 100
        assert_equal(abs(sum(rf.signal * system.rf_raster_time)) * 2 * np.pi, np.pi / 6, rel_tol=0.01)

    def test_complex_signal(self):
        signal = np.exp(1j * np.linspace(0, np.pi, 500))
        rf = pp.make_arbitrary_rf(signal, np.pi / 4, system=pp.Opts())
        assert rf.type == 'rf'
        assert len(rf.signal) == 500

    def test_explicit_center(self):
        rf = pp.make_arbitrary_rf(np.ones(100), np.pi / 6, system=pp.Opts(), center=30e-6)
        assert_equal(rf.center, 30e-6, abs_tol=1e-9)

    def test_with_slice_selection(self):
        rf, gz = pp.make_arbitrary_rf(np.ones(100), np.pi / 6, system=pp.Opts(), bandwidth=1000, slice_thickness=5e-3, return_gz=True)
        assert rf.type == 'rf'
        assert gz.type == 'trap'
        assert gz.channel == 'z'

    def test_bw_and_tbw_error(self):
        with pytest.raises(Exception):
            pp.make_arbitrary_rf(np.ones(100), np.pi / 6, system=pp.Opts(), bandwidth=1000, time_bw_product=4, slice_thickness=5e-3)

    def test_shape_dur(self):
        system = pp.Opts()
        rf = pp.make_arbitrary_rf(np.ones(200), np.pi / 6, system=system)
        assert_equal(rf.shape_dur, 200 * system.rf_raster_time, abs_tol=1e-10)
