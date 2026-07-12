import numpy as np
import pytest
import pypulseq as pp

from . import assert_equal


class TestSplitGradientAt:
    def test_split_at_flat_top(self):
        system = pp.Opts()
        gradient = pp.make_trapezoid('x', area=5000, duration=4e-3, system=system)
        first, second = pp.split_gradient_at(gradient, pp.calc_duration(gradient) / 2, system)
        assert first.type == 'grad'
        assert second.type == 'grad'
        assert first.channel == 'x'
        assert second.channel == 'x'

    def test_split_at_rise(self):
        system = pp.Opts()
        gradient = pp.make_trapezoid('x', area=5000, duration=4e-3, system=system)
        first, second = pp.split_gradient_at(gradient, gradient.rise_time, system)
        assert first.type == 'grad'
        assert second.type == 'grad'

    def test_cut_after_end_error(self):
        system = pp.Opts()
        gradient = pp.make_trapezoid('x', area=1000, duration=1e-3, system=system)
        with pytest.raises(Exception):
            pp.split_gradient_at(gradient, pp.calc_duration(gradient) + 1e-3, system)

    def test_single_output(self):
        system = pp.Opts()
        gradient = pp.make_trapezoid('x', area=5000, duration=4e-3, system=system)
        parts = pp.split_gradient_at(gradient, pp.calc_duration(gradient) / 2, system)
        assert len(parts) == 2
        assert_equal(pp.calc_duration(parts[0]), parts[1].delay, abs_tol=1e-9)
        assert_equal(pp.calc_duration(parts[0]), pp.calc_duration(parts[1]) - parts[1].delay, abs_tol=1e-9)

    def test_split_arbitrary(self):
        system = pp.Opts()
        waveform = np.concatenate((np.linspace(0, 20000, 20), np.linspace(20000, 0, 20)))
        gradient = pp.make_arbitrary_grad('x', waveform, first=0, last=0, system=system)
        first, second = pp.split_gradient_at(gradient, pp.calc_duration(gradient) / 2, system)
        assert first.channel == second.channel == 'x'
        assert_equal(np.concatenate((first.waveform, second.waveform)), waveform)
        assert_equal(first.last, second.first, abs_tol=1e-6)
        assert_equal(first.shape_dur + second.shape_dur, len(waveform) * system.grad_raster_time, abs_tol=1e-9)

    def test_split_arbitrary_with_oversampling(self):
        system = pp.Opts()
        waveform = np.linspace(0, 20000, 21)
        waveform_one = np.concatenate((waveform[1:], waveform[-2:0:-1]))
        gradient = pp.make_arbitrary_grad('x', waveform_one, first=0, last=0, system=system, oversampling=True)
        first, second = pp.split_gradient_at(gradient, pp.calc_duration(gradient) / 2, system)
        assert first.channel == 'x'
        assert_equal(first.waveform, waveform[1:-1])
        assert_equal(second.waveform, waveform[-2:0:-1])
        assert_equal(first.last, second.first, abs_tol=1e-6)
        assert_equal(first.last, waveform[-1], abs_tol=1e-6)
        assert_equal(first.shape_dur + second.shape_dur, (len(waveform_one) + 1) * system.grad_raster_time * 0.5, abs_tol=1e-9)

    def test_split_extended_trap(self):
        system = pp.Opts()
        gradient = pp.make_extended_trapezoid('x', times=[0, 2e-4, 8e-4, 1e-3], amplitudes=[0, 50000, 50000, 0])
        first, second = pp.split_gradient_at(gradient, 5e-4, system)
        assert first.type == second.type == 'grad'
        assert_equal(first.shape_dur + second.shape_dur, gradient.shape_dur, abs_tol=1e-9)
