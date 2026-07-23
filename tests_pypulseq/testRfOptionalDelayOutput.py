import importlib.util
import math

import numpy as np
import pytest

import pypulseq_matlab_like as pp


def _verify_delay_duration(rf, delay_obj, ringdown_time):
    assert delay_obj.type == 'delay'
    assert delay_obj.delay == pytest.approx(rf.delay + rf.shape_dur + ringdown_time, abs=1e-12)


def test_optional_delay_output_duration():
    system = pp.Opts(rf_dead_time=80e-6, rf_ringdown_time=30e-6)
    signal = np.ones(128)

    rf_arb, _, _, delay_arb = pp.make_arbitrary_rf(
        signal,
        math.pi / 6,
        system=system,
        bandwidth=1500,
        slice_thickness=5e-3,
        delay=40e-6,
        return_gz=True,
        return_delay=True,
    )
    _verify_delay_duration(rf_arb, delay_arb, system.rf_ringdown_time)

    rf_block, delay_block = pp.make_block_pulse(
        math.pi / 2, system=system, duration=1.2e-3, delay=40e-6, return_delay=True
    )
    _verify_delay_duration(rf_block, delay_block, system.rf_ringdown_time)

    rf_gauss, _, _, delay_gauss = pp.make_gauss_pulse(
        math.pi / 2,
        system=system,
        duration=1.4e-3,
        time_bw_product=3,
        slice_thickness=5e-3,
        delay=40e-6,
        return_gz=True,
        return_delay=True,
    )
    _verify_delay_duration(rf_gauss, delay_gauss, system.rf_ringdown_time)

    rf_sinc, _, _, delay_sinc = pp.make_sinc_pulse(
        math.pi / 2,
        system=system,
        duration=1.6e-3,
        time_bw_product=4,
        slice_thickness=5e-3,
        delay=40e-6,
        return_gz=True,
        return_delay=True,
    )
    _verify_delay_duration(rf_sinc, delay_sinc, system.rf_ringdown_time)

    if importlib.util.find_spec('sigpy') is None:
        return

    rf_adiabatic, _, _, delay_adiabatic = pp.make_adiabatic_pulse(
        'hypsec',
        system=system,
        duration=8e-3,
        slice_thickness=5e-3,
        delay=40e-6,
        return_gz=True,
        return_delay=True,
    )
    _verify_delay_duration(rf_adiabatic, delay_adiabatic, system.rf_ringdown_time)

    rf_slr, _, _, delay_slr = pp.make_slr_pulse(
        math.pi / 2,
        system=system,
        duration=2e-3,
        time_bw_product=4,
        slice_thickness=5e-3,
        delay=40e-6,
        return_gz=True,
        return_delay=True,
    )
    _verify_delay_duration(rf_slr, delay_slr, system.rf_ringdown_time)
