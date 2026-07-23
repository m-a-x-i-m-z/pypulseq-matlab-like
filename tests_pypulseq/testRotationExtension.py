import math
from pathlib import Path

import matplotlib
import numpy as np
import pytest

import pypulseq_matlab_like as pp


matplotlib.use('Agg')



def _rotation(angle_degrees):
    angle = math.radians(angle_degrees)
    return np.array(((math.cos(angle), -math.sin(angle), 0), (math.sin(angle), math.cos(angle), 0), (0, 0, 1)))


def seq_make_radial():
    seq = pp.Sequence()
    rf = pp.make_block_pulse(math.pi / 2, duration=1e-3, use='excitation')
    gread = pp.make_trapezoid('x', area=1000)
    for angle in (0, 30, 45, 60, 90):
        seq.add_block(rf)
        seq.add_block(gread, pp.make_rotation(_rotation(angle)))
    seq.add_block(rf)
    seq.add_block(gread, pp.make_rotation(_rotation(0)))
    return seq


def seq_make_radial_norotext():
    seq = pp.Sequence()
    rf = pp.make_block_pulse(math.pi / 2, duration=1e-3, use='excitation')
    gread = pp.make_trapezoid('x', area=1000)
    for angle in (0, 30, 45, 60, 90):
        seq.add_block(rf)
        seq.add_block(*pp.rotate(gread, axis='z', angle=math.radians(angle)))
    seq.add_block(rf)
    seq.add_block(*pp.rotate(gread, axis='z', angle=0))
    return seq


def _assert_waveforms_close(first, second, atol, rtol):
    for one, two in zip(first.waveforms_and_times()[0], second.waveforms_and_times()[0]):
        assert one.shape == two.shape
        np.testing.assert_allclose(one[0], two[0], atol=atol, rtol=rtol)
        np.testing.assert_allclose(one[1], two[1], atol=1e2, rtol=rtol)


def _assert_kspace_close(first, second, atol):
    _, _, k1, t1, *_ = first.calculate_kspacePP()
    _, _, k2, t2, *_ = second.calculate_kspacePP()
    _, i1, i2 = np.intersect1d(t1, t2, return_indices=True)
    np.testing.assert_allclose(np.nan_to_num(k2[:, i2]), np.nan_to_num(k1[:, i1]), atol=atol)


class TestRotationExtension:
    def test_vs_rotate(self):
        first, second = seq_make_radial(), seq_make_radial_norotext()
        _assert_waveforms_close(first, second, 1e-5, 1e-5)
        _assert_kspace_close(first, second, 1e-1)

    def test_sequence_save_expected(self, tmp_path):
        actual_path = tmp_path / 'seq_make_radial.seq'
        seq_make_radial().write(str(actual_path))
        expected_path = Path(__file__).resolve().parent / 'expected_output' / 'seq_make_radial.seq'
        assert actual_path.read_text(encoding='utf-8') == expected_path.read_text(encoding='utf-8')

    def test_plot(self):
        pass

    def test_writeread(self, tmp_path):
        original = seq_make_radial(); path = tmp_path / 'seq_make_radial.seq'; original.write(str(path))
        reread = pp.Sequence(); reread.read(str(path))
        assert len(original.block_events) == len(reread.block_events)
        _assert_waveforms_close(original, reread, 1e-5, 1e-5)
        _assert_kspace_close(original, reread, 1e-1)
        assert original.evaluate_labels(evolution='blocks').keys() == reread.evaluate_labels(evolution='blocks').keys()

    def test_recreate(self):
        original = seq_make_radial(); recreated = pp.Sequence(original.system)
        for block_id in original.block_events:
            recreated.add_block(original.get_block(block_id))
        assert len(original.block_events) == len(recreated.block_events)
        _assert_waveforms_close(original, recreated, 1e-9, 1e-9)
        _assert_kspace_close(original, recreated, 1e-6)
        assert original.evaluate_labels(evolution='blocks').keys() == recreated.evaluate_labels(evolution='blocks').keys()
