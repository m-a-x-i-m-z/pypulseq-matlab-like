import importlib.util

import pytest
import pypulseq_matlab_like as pp


pytestmark = pytest.mark.skipif(importlib.util.find_spec('sigpy') is None, reason='sigpy not available')


class TestMakeAdiabaticPulse:
    def test_hypsec(self):
        rf = pp.make_adiabatic_pulse('hypsec')
        assert rf.type == 'rf'
        assert len(rf.signal) > 0

    def test_wurst(self):
        rf = pp.make_adiabatic_pulse('wurst', duration=40e-3)
        assert rf.type == 'rf'
        assert len(rf.signal) > 0

    def test_hypsec_duration(self):
        rf = pp.make_adiabatic_pulse('hypsec', duration=15e-3)
        assert rf.shape_dur == pytest.approx(15e-3, rel=0.01)

    def test_with_slice_thickness(self):
        rf, gz, _ = pp.make_adiabatic_pulse('hypsec', slice_thickness=5e-3, return_gz=True)
        assert rf.type == 'rf'
        assert gz.type == 'trap'
        assert gz.channel == 'z'
