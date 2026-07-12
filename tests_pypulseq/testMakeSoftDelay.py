import pytest
import pypulseq as pp


class TestMakeSoftDelay:
    def test_valid_creation(self):
        delay = pp.make_soft_delay('TE', numID=1)
        assert delay.type == 'soft_delay'
        assert delay.numID == 1
        assert delay.hint == 'TE'
        assert delay.offset == 0
        assert delay.factor == 1

    def test_custom_offset_factor(self):
        delay = pp.make_soft_delay('TR', numID=2, offset=-0.005, factor=2)
        assert delay.numID == 2
        assert delay.hint == 'TR'
        assert delay.offset == -0.005
        assert delay.factor == 2

    def test_whitespace_hint_error(self):
        with pytest.raises(Exception):
            pp.make_soft_delay('my delay', numID=1)

    def test_different_numids(self):
        first = pp.make_soft_delay('TE', numID=1)
        second = pp.make_soft_delay('TR', numID=99)
        assert first.numID == 1
        assert second.numID == 99

    def test_negative_factor(self):
        assert pp.make_soft_delay('delay', numID=1, offset=0, factor=-1).factor == -1
