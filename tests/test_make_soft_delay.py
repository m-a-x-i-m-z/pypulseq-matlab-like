import numpy as np
import pytest
import pypulseq_matlab_like as pp


def assert_matlab_soft_delay(delay, *, num_id, hint, offset, factor):
    """Check all soft-delay event fields."""
    assert delay.type == 'soft_delay'
    assert delay.numID == num_id
    assert delay.hint == hint
    assert delay.offset == offset
    assert delay.factor == factor


class TestMakeSoftDelay:
    def test_valid_creation(self):
        delay = pp.make_soft_delay('TE', numID=1)
        assert_matlab_soft_delay(delay, num_id=1, hint='TE', offset=0, factor=1)

    def test_custom_offset_factor(self):
        delay = pp.make_soft_delay('TR', numID=2, offset=-0.005, factor=2)
        assert_matlab_soft_delay(delay, num_id=2, hint='TR', offset=-0.005, factor=2)

    def test_whitespace_hint_error(self):
        with pytest.raises(ValueError, match="Parameter 'hint' may not contain white space characters"):
            pp.make_soft_delay('my delay', numID=1)

    def test_different_numids(self):
        first = pp.make_soft_delay('TE', numID=1)
        second = pp.make_soft_delay('TR', numID=99)
        assert_matlab_soft_delay(first, num_id=1, hint='TE', offset=0, factor=1)
        assert_matlab_soft_delay(second, num_id=99, hint='TR', offset=0, factor=1)

    def test_negative_factor(self):
        delay = pp.make_soft_delay('delay', numID=1, offset=0, factor=-1)
        assert_matlab_soft_delay(delay, num_id=1, hint='delay', offset=0, factor=-1)

    def test_zero_factor_error(self):
        with pytest.raises(ValueError, match="Parameter 'factor' must be nonzero"):
            pp.make_soft_delay('delay', numID=1, factor=0)

    def test_matlab_event_library_layout(self):
        """Match registerSoftDelayEvent data: [num, offset, factor, hintID]."""
        seq = pp.Sequence()
        delay = pp.make_soft_delay('TR', numID=2, offset=-0.005, factor=2)

        # MATLAB passes the block duration separately to addBlock().
        seq.add_block(0.01, delay)

        np.testing.assert_array_equal(seq.soft_delay_library.data[1], [2, -0.005, 2, 1])
        assert seq.soft_delay_hints2 == ['TR']

        restored = seq.get_block(1).soft_delay
        assert_matlab_soft_delay(restored, num_id=2, hint='TR', offset=-0.005, factor=2)
        assert restored.default_duration == 0.01
