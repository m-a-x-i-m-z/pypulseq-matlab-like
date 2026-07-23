import pytest
import pypulseq_matlab_like as pp

from . import assert_equal


class TestAddGradients:
    def test_sum_identical_traps(self):
        gradient = pp.make_trapezoid('x', area=1000, duration=1e-3)
        total = pp.add_gradients([gradient, gradient])
        assert_equal(total.amplitude, 2 * gradient.amplitude, abs_tol=1e-6)
        assert_equal(total.area, 2 * gradient.area, abs_tol=1e-3)
        assert_equal(total.flat_area, 2 * gradient.flat_area, abs_tol=1e-3)

    def test_opposing_traps(self):
        first = pp.make_trapezoid('x', area=1000, duration=1e-3)
        second = pp.make_trapezoid('x', area=-1000, duration=1e-3)
        assert_equal(pp.add_gradients([first, second]).area, 0, abs_tol=1)

    def test_non_cell_error(self):
        with pytest.raises(Exception):
            pp.add_gradients(pp.make_trapezoid('x', area=1000, duration=1e-3))

    def test_single_gradient_error(self):
        with pytest.raises(Exception):
            pp.add_gradients([pp.make_trapezoid('x', area=1000, duration=1e-3)])

    def test_different_channels_error(self):
        with pytest.raises(Exception):
            pp.add_gradients([pp.make_trapezoid('x', area=1000, duration=1e-3), pp.make_trapezoid('y', area=1000, duration=1e-3)])

    def test_trap_plus_extended(self):
        system = pp.Opts()
        trap = pp.make_trapezoid('x', amplitude=1e5, duration=1e-3)
        extended = pp.make_extended_trapezoid('x', times=[0, 2e-4, 8e-4, 1e-3], amplitudes=[0, 50000, 50000, 0])
        total = pp.add_gradients([trap, extended], system=system)
        assert total.type == 'grad'
        assert_equal(pp.calc_duration(total), max(pp.calc_duration(trap), pp.calc_duration(extended)), abs_tol=system.grad_raster_time)

    def test_three_traps(self):
        system = pp.Opts()
        gradient = [pp.make_trapezoid('y', area=area, system=system, max_grad=system.max_grad / 3, max_slew=system.max_slew / 3) for area in (1000, 700, 500)]
        total = pp.add_gradients(gradient)
        assert_equal(total.area, sum(event.area for event in gradient), abs_tol=1)
