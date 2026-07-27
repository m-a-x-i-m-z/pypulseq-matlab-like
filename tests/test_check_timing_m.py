import pytest

import pypulseq_matlab_like as pp


def _contains(report, block, text):
    return any(f'block:{block}' in entry.lower() and text.lower() in entry.lower() for entry in report)


def test_check_timing():
    """Direct Python counterpart of MATLAB test_check_timing (Python blocks are zero-based)."""
    system = pp.Opts(
        max_grad=28,
        grad_unit='mT/m',
        max_slew=200,
        slew_unit='T/m/s',
        rf_ringdown_time=20e-6,
        rf_dead_time=100e-6,
        adc_dead_time=10e-6,
    )
    broken = pp.Opts(
        max_grad=28,
        grad_unit='mT/m',
        max_slew=200,
        slew_unit='T/m/s',
        rf_ringdown_time=0,
        rf_dead_time=0,
        adc_dead_time=0,
    )
    seq = pp.Sequence(system)

    seq.add_block(pp.make_sinc_pulse(1, duration=1e-3, delay=system.rf_dead_time, use='excitation', system=system))
    seq.add_block(pp.make_sinc_pulse(1, duration=1e-3, use='excitation', system=broken))
    seq.add_block(pp.make_adc(100, duration=1e-3, delay=system.adc_dead_time, system=system))
    seq.add_block(pp.make_adc(123, duration=1e-3, delay=system.adc_dead_time, system=system))
    seq.add_block(pp.make_adc(100, duration=1e-3, system=broken))
    seq.add_block(pp.make_trapezoid('x', area=1, duration=1, system=system))
    seq.add_block(pp.make_trapezoid('x', area=1, duration=1.00001e-3, system=system))
    seq.add_block(pp.make_trapezoid('x', flat_area=1, rise_time=1e-6, flat_time=1e-3, fall_time=3e-6, system=system))
    seq.add_block(pp.make_trapezoid('x', area=1, duration=1e-3, delay=-1e-5, system=system))

    ok, report = seq.check_timing()
    assert not ok
    for block in (1, 3, 6):
        assert not any(f'block:{block}' in entry.lower() for entry in report)

    assert _contains(report, 2, 'rf dead time')
    assert _contains(report, 2, 'rf_ringdown_time')
    assert _contains(report, 4, 'adc_raster_time')
    assert _contains(report, 5, 'adc_dead_time')
    assert _contains(report, 5, 'post-adc')
    assert _contains(report, 7, 'block duration')
    assert _contains(report, 8, 'rise_time') and _contains(report, 8, 'fall_time')
    assert _contains(report, 9, 'delay:-')
