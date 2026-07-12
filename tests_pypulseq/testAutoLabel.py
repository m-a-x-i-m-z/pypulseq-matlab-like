import numpy as np
import pypulseq as pp


def _build_test_seq(lines):
    sequence = pp.Sequence()
    rf, gz, _ = pp.make_sinc_pulse(np.pi / 8, duration=2e-3, slice_thickness=5e-3, use='excitation', return_gz=True)
    sequence.add_block(rf, gz)
    for index in range(lines):
        sequence.add_block(pp.make_trapezoid('x', area=1000))
        sequence.add_block(pp.make_trapezoid('y', area=-500 + index * 100))
        sequence.add_block(pp.make_trapezoid('x', area=-500))
        sequence.add_block(pp.make_trapezoid('x', area=1000, duration=10e-3), pp.make_adc(64, duration=10e-3))
    return sequence


def _build_test_seq_3d(lines, partitions):
    sequence = pp.Sequence()
    rf, gz, _ = pp.make_sinc_pulse(np.pi / 8, duration=1e-3, slice_thickness=100e-3, use='excitation', return_gz=True)
    for partition in range(partitions):
        for line in range(lines):
            sequence.add_block(rf, gz)
            sequence.add_block(pp.make_trapezoid('x', area=400))
            sequence.add_block(pp.make_trapezoid('y', area=-300 + line * 200), pp.make_trapezoid('z', area=-200 + partition * 200))
            sequence.add_block(pp.make_trapezoid('x', area=800, duration=8e-3), pp.make_adc(64, duration=8e-3))
            sequence.add_block(pp.make_trapezoid('x', area=-1200))
    return sequence


class TestAutoLabel:
    def test_detect_labels(self):
        sequence = _build_test_seq(6)
        labels, auxiliary = sequence.auto_label(skip_apply=True)
        adcs = sum(getattr(sequence.get_block(index), 'adc', None) is not None for index in sequence.block_events)
        assert labels
        for values in labels.values():
            assert len(values) == adcs
            assert min(values) >= 0
            assert np.all(abs(values - np.round(values)) < 1e-10)
        assert auxiliary

    def test_apply_labels(self):
        sequence = _build_test_seq(8)
        labels, _ = sequence.auto_label(skip_apply=True)
        sequence.auto_label()
        applied = sequence.evaluate_labels(evolution='adc')
        for key, values in labels.items():
            assert key in applied
            np.testing.assert_equal(applied[key], values)

    def test_reflect_reorder_3d(self):
        sequence = _build_test_seq_3d(4, 3)
        labels, _ = sequence.auto_label(skip_apply=True)
        assert 'LIN' in labels and 'PAR' in labels
        reordered, _ = sequence.auto_label(skip_apply=True, reorder=[1, 3, 2])
        np.testing.assert_equal(reordered['LIN'], labels['PAR'])
        np.testing.assert_equal(reordered['PAR'], labels['LIN'])
        reflected, _ = sequence.auto_label(skip_apply=True, reflect=[2, 3])
        np.testing.assert_equal(reflected['LIN'], max(labels['LIN']) - labels['LIN'])
        np.testing.assert_equal(reflected['PAR'], max(labels['PAR']) - labels['PAR'])
