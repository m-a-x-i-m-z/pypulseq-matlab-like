import importlib.util

import numpy as np
import pytest
import pypulseq_matlab_like as pp


pytestmark = pytest.mark.skipif(importlib.util.find_spec('sigpy') is None, reason='sigpy not available')


class TestMakeSLRpulse:
    def test_basic_slr(self):
        rf = pp.make_slr_pulse(np.pi / 2, duration=3e-3, time_bw_product=4)
        assert rf.type == 'rf'
        assert len(rf.signal) > 0

    def test_slr_with_slice(self):
        rf, gz, _ = pp.make_slr_pulse(np.pi / 2, duration=3e-3, time_bw_product=4, slice_thickness=5e-3, return_gz=True)
        assert rf.type == 'rf'
        assert gz.type == 'trap'

    def test_slr_excitation(self):
        rf = pp.make_slr_pulse(np.pi / 2, duration=3e-3, time_bw_product=4, use='excitation')
        assert rf.use == 'excitation'
