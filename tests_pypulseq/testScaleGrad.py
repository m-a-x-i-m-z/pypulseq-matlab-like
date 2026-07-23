import numpy as np
import pytest
import pypulseq_matlab_like as pp

from . import assert_equal


class TestScaleGrad:
    def test_scale_trap_by_2(self):
        gradient = pp.make_trapezoid('x', area=1000, duration=1e-3)
        scaled = pp.scale_grad(gradient, 2)
        assert_equal(scaled.amplitude, 2 * gradient.amplitude, abs_tol=1e-6)
        assert_equal(scaled.area, 2 * gradient.area, abs_tol=1e-3)
        assert_equal(scaled.flat_area, 2 * gradient.flat_area, abs_tol=1e-3)

    def test_scale_trap_negate(self):
        gradient = pp.make_trapezoid('x', area=1000, duration=1e-3)
        scaled = pp.scale_grad(gradient, -1)
        assert_equal(scaled.amplitude, -gradient.amplitude, abs_tol=1e-6)
        assert_equal(scaled.area, -gradient.area, abs_tol=1e-3)

    def test_scale_by_zero(self):
        gradient = pp.make_trapezoid('x', area=1000, duration=1e-3)
        scaled = pp.scale_grad(gradient, 0)
        assert_equal(scaled.amplitude, 0, abs_tol=1e-10)
        assert_equal(scaled.area, 0, abs_tol=1e-10)

    def test_scale_arbitrary_grad(self):
        gradient = pp.make_arbitrary_grad('x', np.array([0, 10000, 20000, 10000, 0]), first=0, last=0)
        scaled = pp.scale_grad(gradient, 0.5)
        assert_equal(scaled.waveform, 0.5 * gradient.waveform, abs_tol=1e-6)
        assert_equal(scaled.first, 0.5 * gradient.first, abs_tol=1e-6)
        assert_equal(scaled.last, 0.5 * gradient.last, abs_tol=1e-6)

    def test_id_field_removed(self):
        gradient = pp.make_trapezoid('x', area=1000, duration=1e-3)
        gradient.id = 42
        with pytest.raises(Exception):
            pp.scale_grad(gradient, 1.5)

    def test_maxGrad_violation(self):
        gradient = pp.make_trapezoid('x', area=1000, duration=1e-3)
        system = pp.Opts()
        factor = 2 * system.max_grad / abs(gradient.amplitude)
        with pytest.raises(Exception):
            pp.scale_grad(gradient, factor, system)

    def test_maxSlew_violation(self):
        gradient = pp.make_trapezoid('x', amplitude=1e6, rise_time=1e-4, flat_time=1e-4, fall_time=1e-4)
        with pytest.raises(Exception):
            pp.scale_grad(gradient, 100, pp.Opts())
