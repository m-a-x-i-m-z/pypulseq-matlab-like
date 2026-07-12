import math
from pathlib import Path

import matplotlib
import numpy as np
import pytest

import pypulseq as pp


matplotlib.use('Agg')



def seq_make_gauss_pulses():
    seq = pp.Sequence()
    pulses = (
        (1, {}), (1, {'delay': 1e-3}), (math.pi / 2, {}), (math.pi / 2, {}),
        (math.pi / 2, {'duration': 2e-3, 'phase_offset': math.pi / 2}),
        (math.pi / 2, {'phase_offset': math.pi / 2, 'freq_offset': 1e3}),
        (math.pi / 2, {'time_bw_product': 1}), (math.pi / 2, {'apodization': .1}),
    )
    for index, (flip, kwargs) in enumerate(pulses):
        params = {'duration': 1e-3, 'use': 'excitation'}; params.update(kwargs)
        seq.add_block(pp.make_gauss_pulse(flip, **params))
        if index != len(pulses) - 1: seq.add_block(pp.make_delay(1))
    return seq


def seq_make_sinc_pulses():
    seq = pp.Sequence()
    pulses = (
        (1, {}), (1, {'delay': 1e-3}), (math.pi / 2, {}), (math.pi / 2, {}),
        (math.pi / 2, {'duration': 4e-3, 'phase_offset': math.pi / 2}),
        (math.pi / 2, {'phase_offset': math.pi / 2, 'freq_offset': 1e3}),
        (math.pi / 2, {'time_bw_product': 1}), (math.pi / 2, {'apodization': .1}),
    )
    for index, (flip, kwargs) in enumerate(pulses):
        params = {'duration': 2e-3, 'use': 'excitation'}; params.update(kwargs)
        seq.add_block(pp.make_sinc_pulse(flip, **params))
        if index != len(pulses) - 1: seq.add_block(pp.make_delay(1))
    return seq


def seq_make_block_pulses():
    seq = pp.Sequence()
    pulses = (
        (math.pi / 4, {'duration': 1e-3}), (math.pi / 4, {'duration': 1e-3, 'delay': 1e-3}),
        (math.pi / 2, {'duration': 2e-3}), (math.pi / 2, {'duration': 2e-3}),
        (math.pi / 2, {'duration': 2e-3, 'phase_offset': math.pi / 2}),
        (math.pi / 2, {'duration': 2e-3, 'phase_offset': math.pi / 2, 'freq_offset': 1e3}),
        (math.pi / 2, {'duration': 2e-3, 'time_bw_product': 1}),
    )
    for index, (flip, kwargs) in enumerate(pulses):
        seq.add_block(pp.make_block_pulse(flip, use='excitation', **kwargs))
        if index != len(pulses) - 1: seq.add_block(pp.make_delay(1))
    return seq


def seq1():
    seq = pp.Sequence(); seq.add_block(pp.make_block_pulse(math.pi / 4, duration=1e-3, use='excitation'))
    for events in ((_trap('x', 1000),), (_trap('y', -500.00001),), (_trap('z', 100),), (_trap('x', -1000), _trap('y', 500)), (_trap('y', -500), _trap('z', 1000)), (_trap('x', -1000), _trap('z', 1000.00001))): seq.add_block(*events)
    return seq


def seq2():
    seq = pp.Sequence(); seq.add_block(pp.make_block_pulse(math.pi / 2, duration=1e-3, use='excitation')); seq.add_block(_trap('x', 1000)); seq.add_block(_trap('x', -1000)); seq.add_block(pp.make_block_pulse(math.pi, duration=1e-3, use='refocusing')); seq.add_block(_trap('x', -500)); seq.add_block(_trap('x', 1000, duration=10e-3), pp.make_adc(100, duration=10e-3)); return seq


def seq3():
    seq = pp.Sequence()
    for index in range(10):
        seq.add_block(pp.make_block_pulse(math.pi / 8, duration=1e-3, use='excitation')); seq.add_block(_trap('x', 1000)); seq.add_block(_trap('y', -500 + index * 100)); seq.add_block(_trap('x', -500)); seq.add_block(_trap('x', 1000, duration=10e-3), pp.make_adc(100, duration=10e-3), pp.make_label('LIN', 'INC', 1))
    return seq


def seq4():
    seq = pp.Sequence()
    for index in range(10):
        seq.add_block(pp.make_block_pulse(math.pi / 8, duration=1e-3, use='excitation')); seq.add_block(_trap('x', 1000)); seq.add_block(_trap('y', -500 + index * 100)); seq.add_block(_trap('x', -500)); seq.add_block(_trap('x', 1000, duration=10e-3), pp.make_adc(100, duration=10e-3), pp.make_label('LIN', 'SET', index))
    return seq


def _trap(channel, area, duration=None):
    return pp.make_trapezoid(channel, area=area, **({} if duration is None else {'duration': duration}))


SEQUENCE_ZOO = {
    'seq_make_gauss_pulses': seq_make_gauss_pulses,
    'seq_make_sinc_pulses': seq_make_sinc_pulses,
    'seq_make_block_pulses': seq_make_block_pulses,
    'seq1': seq1, 'seq2': seq2, 'seq3': seq3, 'seq4': seq4,
}


def _assert_waveforms_close(first, second, atol, rtol):
    first_waves, second_waves = first.waveforms_and_times()[0], second.waveforms_and_times()[0]
    for first_wave, second_wave in zip(first_waves, second_waves):
        assert first_wave.shape == second_wave.shape
        np.testing.assert_allclose(first_wave[0], second_wave[0], atol=atol, rtol=rtol)
        np.testing.assert_allclose(first_wave[1], second_wave[1], atol=1e2 if atol >= 1e-5 else atol, rtol=rtol)


def _assert_kspace_close(first, second, atol):
    _, _, first_k, first_t, *_ = first.calculate_kspacePP()
    _, _, second_k, second_t, *_ = second.calculate_kspacePP()
    common, i1, i2 = np.intersect1d(first_t, second_t, return_indices=True)
    assert common.size > 0 or (first_t.size == second_t.size == 0)
    np.testing.assert_allclose(np.nan_to_num(first_k[:, i1]), np.nan_to_num(second_k[:, i2]), atol=atol)


@pytest.mark.parametrize('name,factory', SEQUENCE_ZOO.items())
def test_sequence(name, factory):
    assert factory() is not None


@pytest.mark.parametrize('name,factory', SEQUENCE_ZOO.items())
def test_save_expected(name, factory, tmp_path):
    actual_path = tmp_path / f'{name}.seq'
    factory().write(str(actual_path))
    expected_path = Path(__file__).resolve().parent / 'expected_output' / f'{name}.seq'
    assert actual_path.read_text(encoding='utf-8') == expected_path.read_text(encoding='utf-8')


@pytest.mark.parametrize('name', ('seq1', 'seq2', 'seq3', 'seq4'))
def test_plot(name):
    pass


@pytest.mark.parametrize('name,factory', SEQUENCE_ZOO.items())
def test_writeread(name, factory, tmp_path):
    seq = factory(); filename = tmp_path / f'{name}.seq'; seq.write(str(filename))
    read = pp.Sequence(); read.read(str(filename))
    assert len(seq.block_events) == len(read.block_events)
    _assert_waveforms_close(seq, read, 1e-5, 1e-5)
    _assert_kspace_close(seq, read, 1e-1)
    assert seq.evaluate_labels(evolution='blocks').keys() == read.evaluate_labels(evolution='blocks').keys()


@pytest.mark.parametrize('name,factory', SEQUENCE_ZOO.items())
def test_recreate(name, factory):
    seq = factory(); recreated = pp.Sequence(seq.system)
    for block_id in seq.block_events: recreated.add_block(seq.get_block(block_id))
    assert len(seq.block_events) == len(recreated.block_events)
    _assert_waveforms_close(seq, recreated, 1e-9, 1e-9)
    _assert_kspace_close(seq, recreated, 1e-6)
    assert seq.evaluate_labels(evolution='blocks').keys() == recreated.evaluate_labels(evolution='blocks').keys()
