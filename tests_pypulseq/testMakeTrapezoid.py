import pytest
import pypulseq_matlab_like as pp

from . import assert_equal


class TestMakeTrapezoid:
    def test_make_trapezoid(self):
        with pytest.raises(Exception):
            pp.make_trapezoid('x')
        with pytest.raises(Exception):
            pp.make_trapezoid('x', flat_time=10, area=10)
        with pytest.raises(Exception):
            pp.make_trapezoid('x', area=1e6, duration=1e-6)
        with pytest.raises(Exception):
            pp.make_trapezoid('x', amplitude=1)
        with pytest.raises(Exception):
            pp.make_trapezoid('x', amplitude=1e10, duration=1)
        with pytest.raises(Exception):
            pp.make_trapezoid('x', area=1, duration=0.1, rise_time=0.1)

        trap = pp.make_trapezoid('x', amplitude=1, duration=1)
        assert_equal(trap.amplitude, 1, abs_tol=1e-10)
        assert_equal(trap.rise_time, 1e-5, abs_tol=1e-10)
        assert_equal(trap.flat_time, 1 - 2e-5, abs_tol=1e-10)
        assert_equal(trap.fall_time, 1e-5, abs_tol=1e-10)

        trap = pp.make_trapezoid('x', amplitude=1, flat_time=1)
        assert_equal(trap.amplitude, 1, abs_tol=1e-10)
        assert_equal(trap.rise_time, 1e-5, abs_tol=1e-10)
        assert_equal(trap.flat_time, 1, abs_tol=1e-10)
        assert_equal(trap.fall_time, 1e-5, abs_tol=1e-10)

        trap = pp.make_trapezoid('x', flat_area=1, flat_time=1)
        assert_equal(trap.amplitude, 1, abs_tol=1e-10)
        assert_equal(trap.rise_time, 1e-5, abs_tol=1e-10)
        assert_equal(trap.flat_time, 1, abs_tol=1e-10)
        assert_equal(trap.fall_time, 1e-5, abs_tol=1e-10)

        trap = pp.make_trapezoid('x', area=1)
        assert_equal(trap.amplitude, 5e4, abs_tol=1e-10)
        assert_equal(trap.rise_time, 2e-5, abs_tol=1e-10)
        assert_equal(trap.flat_time, 0, abs_tol=1e-10)
        assert_equal(trap.fall_time, 2e-5, abs_tol=1e-10)

        trap = pp.make_trapezoid('x', area=1, duration=1, rise_time=0.01)
        assert_equal(trap.amplitude, 1 / 0.99, abs_tol=1e-10)
        assert_equal(trap.rise_time, 0.01, abs_tol=1e-10)
        assert_equal(trap.flat_time, 0.98, abs_tol=1e-10)
        assert_equal(trap.fall_time, 0.01, abs_tol=1e-10)
