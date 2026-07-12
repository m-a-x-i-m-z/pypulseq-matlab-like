import pytest

from pypulseq import make_label
from pypulseq import supported_labels_rf_use as labels_module
from pypulseq.supported_labels_rf_use import add_supported_label, get_supported_labels


@pytest.fixture(autouse=True)
def _restore_supported_labels():
    original = list(labels_module._SUPPORTED_LABELS)
    yield
    labels_module._SUPPORTED_LABELS[:] = original


class TestAddCustomLabel:
    def test_add_valid_label(self):
        add_supported_label('MYCUSTOM')
        assert 'MYCUSTOM' in get_supported_labels()

    def test_non_char_error(self):
        with pytest.raises(Exception):
            add_supported_label(123)

    def test_multiple_labels(self):
        add_supported_label('CUSTOM1')
        add_supported_label('CUSTOM2')
        labels = get_supported_labels()
        assert 'CUSTOM1' in labels
        assert 'CUSTOM2' in labels

    def test_custom_label_with_makeLabel(self):
        add_supported_label('MYTEST')
        label = make_label('MYTEST', 'SET', 42)
        assert label.label == 'MYTEST'
        assert label.value == 42
