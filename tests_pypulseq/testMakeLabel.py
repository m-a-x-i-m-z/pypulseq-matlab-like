import pytest
import pypulseq_matlab_like as pp


class TestMakeLabel:
    def test_set_label(self):
        label = pp.make_label('SLC', 'SET', 5)
        assert (label.type, label.label, label.value) == ('labelset', 'SLC', 5)

    def test_inc_label(self):
        label = pp.make_label('LIN', 'INC', 1)
        assert (label.type, label.label, label.value) == ('labelinc', 'LIN', 1)

    def test_negative_increment(self):
        assert pp.make_label('PAR', 'INC', -1).value == -1

    def test_invalid_label_error(self):
        with pytest.raises(Exception):
            pp.make_label('INVALID_LABEL', 'SET', 0)

    def test_invalid_type_error(self):
        with pytest.raises(Exception):
            pp.make_label('SLC', 'RESET', 0)

    def test_logical_flag(self):
        assert pp.make_label('REV', 'SET', True).value is True

    def test_missing_args_error(self):
        with pytest.raises(Exception):
            pp.make_label('SLC', 'SET')

    def test_standard_counters(self):
        for counter in ('SLC', 'SEG', 'REP', 'AVG', 'SET', 'ECO', 'PHS', 'LIN', 'PAR', 'ACQ'):
            assert pp.make_label(counter, 'SET', 0).label == counter

    def test_standard_flags(self):
        for flag in ('NAV', 'REV', 'SMS', 'REF', 'IMA', 'NOISE'):
            assert pp.make_label(flag, 'SET', True).label == flag
