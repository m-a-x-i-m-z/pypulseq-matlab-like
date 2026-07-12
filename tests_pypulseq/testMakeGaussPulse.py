import pytest
import pypulseq as pp
from pypulseq.supported_labels_rf_use import get_supported_rf_uses


class TestMakeGaussPulse:
    def test_make_gauss_pulse(self):
        with pytest.raises(Exception):
            pp.make_gauss_pulse(1, use='invalid')
        for use in get_supported_rf_uses():
            assert pp.make_gauss_pulse(1, use=use) is not None
