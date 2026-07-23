import numpy as np
import pypulseq_matlab_like as pp

from . import assert_equal


def _system():
    return pp.Opts(max_grad=30, grad_unit='mT/m', max_slew=150, slew_unit='T/m/s')


def _verify(gradient, area, first, last, system):
    assert_equal(gradient.area, area, abs_tol=1)
    assert_equal(gradient.waveform[0], first, abs_tol=1e-3)
    assert_equal(gradient.waveform[-1], last, abs_tol=1e-3)
    slew = np.diff(gradient.waveform) / np.diff(gradient.tt)
    assert np.max(abs(slew)) <= system.max_slew * 1.01
    assert np.max(abs(gradient.waveform)) <= system.max_grad * 1.01


def _run_case(channel, start, end, area, *, compare_duration=True):
    system = _system()
    trapezoid, _, _ = pp.make_extended_trapezoid_area(area, channel, start, end, system=system)
    hexagon, _, _ = pp.make_hexagon_gradient_area(channel, start, end, area, system)
    _verify(trapezoid, area, start, end, system)
    _verify(hexagon, area, start, end, system)
    if compare_duration:
        assert hexagon.shape_dur <= trapezoid.shape_dur + 1e-9


def test_case_01_zero_zero_pos():
    _run_case('x', 0, 0, 5000)


def test_case_02_zero_zero_neg():
    _run_case('y', 0, 0, -5000)


def test_case_03_zero_zero_small():
    _run_case('z', 0, 0, 100)


def test_case_04_zero_zero_large():
    _run_case('x', 0, 0, 50000)


def test_case_05_posstart_zeroend():
    system = _system()
    _run_case('x', system.max_grad * 0.5, 0, 3000)


def test_case_06_zerostart_posend():
    system = _system()
    _run_case('y', 0, system.max_grad * 0.3, 4000)


def test_case_07_pos_pos_pos():
    system = _system()
    _run_case('z', system.max_grad * 0.2, system.max_grad * 0.4, 8000)


def test_case_08_neg_neg_neg():
    system = _system()
    _run_case('x', -system.max_grad * 0.3, -system.max_grad * 0.1, -6000)


def test_case_09_pos_neg():
    system = _system()
    _run_case('y', system.max_grad * 0.2, -system.max_grad * 0.2, 2000)


def test_case_10_neg_pos():
    system = _system()
    _run_case('z', -system.max_grad * 0.3, system.max_grad * 0.1, -3000)


def test_case_11_equal_edges():
    system = _system()
    _run_case('x', system.max_grad * 0.25, system.max_grad * 0.25, 7000)


def test_case_12_nearmax_small_area():
    system = _system()
    _run_case('y', system.max_grad * 0.8, system.max_grad * 0.8, 500, compare_duration=False)


def test_case_13_zero_area_equal_edges():
    system = _system()
    _run_case('x', system.max_grad * 0.3, system.max_grad * 0.3, 0)


def test_case_14_zero_area_antisym_edges():
    system = _system()
    _run_case('y', system.max_grad * 0.3, -system.max_grad * 0.3, 0)


def test_case_15_zero_area_onesided():
    system = _system()
    _run_case('z', system.max_grad * 0.3, 0, 0)


def test_case_16_zero_area_unequal_edges():
    system = _system()
    _run_case('x', -system.max_grad * 0.3, -system.max_grad * 0.075, 0)


def test_case_17_zero_area_nearmax_edges():
    system = _system()
    _run_case('y', system.max_grad * 0.8, system.max_grad * 0.8, 0)