from pypulseq.supported_labels_rf_use import get_supported_rf_uses


class TestGetSupportedRfUse:
    def test_returns_cell(self):
        assert isinstance(get_supported_rf_uses(), tuple)

    def test_expected_entries(self):
        uses = get_supported_rf_uses()
        assert all(use in uses for use in ('excitation', 'refocusing', 'inversion', 'saturation', 'preparation'))

    def test_two_outputs(self):
        uses, short = get_supported_rf_uses(return_short_names=True)
        assert isinstance(uses, tuple)
        assert isinstance(short, str)
        assert len(short) == len(uses)

    def test_short_names(self):
        uses, short = get_supported_rf_uses(return_short_names=True)
        assert all(short[index] == use[0] for index, use in enumerate(uses))

    def test_undefined_and_other(self):
        uses = get_supported_rf_uses()
        assert 'undefined' in uses
        assert 'other' in uses
