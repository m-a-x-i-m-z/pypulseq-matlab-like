import numpy as np
import pytest
import pypulseq_matlab_like as pp

from . import assert_equal


class TestMakeArbitraryGrad:
    def test_triangle_waveform(self):
        waveform = np.array([0, 10000, 20000, 10000, 0])
        gradient = pp.make_arbitrary_grad('x', waveform, system=pp.Opts(), first=0, last=0)
        assert gradient.type == 'grad'
        assert gradient.channel == 'x'
        assert len(gradient.waveform) == len(waveform)
        assert gradient.area != 0

    def test_explicit_first_last(self):
        gradient = pp.make_arbitrary_grad('y', np.array([5000, 10000, 5000]), system=pp.Opts(), first=0, last=0)
        assert_equal(gradient.first, 0, abs_tol=1e-10)
        assert_equal(gradient.last, 0, abs_tol=1e-10)

    def test_no_first_last_warnings(self):
        gradient = pp.make_arbitrary_grad('z', np.array([0, 5000, 10000, 5000, 0]), system=pp.Opts())
        assert gradient.type == 'grad'

    def test_maxGrad_violation(self):
        system = pp.Opts()
        with pytest.raises(Exception):
            pp.make_arbitrary_grad('x', np.ones(10) * system.max_grad * 1.01, system=system, first=system.max_grad * 1.01, last=system.max_grad * 1.01)

    def test_maxSlew_violation(self):
        system = pp.Opts()
        step = system.max_slew * system.grad_raster_time * 1.05
        with pytest.raises(Exception):
            pp.make_arbitrary_grad('x', np.array([0, step, 0]), system=system, first=0, last=0)

    def test_oversampling_odd(self):
        waveform = np.array([0, 5000, 10000, 5000, 0])
        gradient = pp.make_arbitrary_grad('x', waveform, system=pp.Opts(), first=0, last=0, oversampling=True)
        assert gradient.type == 'grad'
        assert len(gradient.tt) == len(waveform)

    def test_oversampling_even_error(self):
        with pytest.raises(Exception):
            pp.make_arbitrary_grad('x', np.array([0, 5000, 10000, 15000, 20000, 0]), system=pp.Opts(), first=0, last=0, oversampling=True)

    def test_area_calculation(self):
        system = pp.Opts()
        waveform = 10000 * np.ones(20)
        gradient = pp.make_arbitrary_grad('x', waveform, system=system, first=10000, last=10000)
        assert_equal(gradient.area, 10000 * 20 * system.grad_raster_time, abs_tol=1)

    def test_valid_channels(self):
        for channel in ('x', 'y', 'z'):
            assert pp.make_arbitrary_grad(channel, np.array([0, 5000, 0]), system=pp.Opts(), first=0, last=0).channel == channel
