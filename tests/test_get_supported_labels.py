import pytest

from pypulseq_matlab_like.supported_labels_rf_use import get_supported_labels


class TestGetSupportedLabels:
    def test_default_labels(self):
        labels = get_supported_labels()
        expected = ('SLC', 'SEG', 'REP', 'AVG', 'SET', 'ECO', 'PHS', 'LIN', 'PAR', 'ACQ', 'TRID')
        assert isinstance(labels, tuple)
        assert all(label in labels for label in expected)

    def test_default_labels_count(self):
        assert len(get_supported_labels()) >= 23

    def test_consistency(self):
        assert get_supported_labels() == get_supported_labels()

    def test_flags_present(self):
        labels = get_supported_labels()
        assert all(label in labels for label in ('NAV', 'REV', 'SMS', 'REF', 'IMA', 'NOISE'))

    def test_control_labels_present(self):
        labels = get_supported_labels()
        assert all(label in labels for label in ('PMC', 'NOROT', 'NOPOS', 'NOSCL', 'ONCE'))
