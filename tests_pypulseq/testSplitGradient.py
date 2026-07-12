import numpy as np
import pytest
import pypulseq as pp

from . import assert_equal


class TestSplitGradient:
    def test_split_standard_trap(self):
        parts = pp.split_gradient(pp.make_trapezoid('x', area=5000, duration=4e-3))
        assert len(parts) == 3
        assert all(part.channel == 'x' for part in parts)

    def test_split_durations(self):
        system = pp.Opts()
        gradient = pp.make_trapezoid('x', area=5000, duration=4e-3, system=system)
        parts = pp.split_gradient(gradient, system)
        reconstructed = pp.add_gradients(list(parts), system=system)
        assert_equal(pp.calc_duration(reconstructed), pp.calc_duration(gradient), abs_tol=1e-6)
        assert_equal(reconstructed.area, gradient.area, rel_tol=0.001)

    def test_split_triangular(self):
        parts = pp.split_gradient(pp.make_trapezoid('x', amplitude=100000, rise_time=5e-4, flat_time=0, fall_time=5e-4, system=pp.Opts()), pp.Opts())
        assert len(parts) == 2

    def test_arbitrary_grad_error(self):
        gradient = pp.make_arbitrary_grad('x', np.array([0, 10000, 20000, 10000, 0]), first=0, last=0)
        with pytest.raises(Exception):
            pp.split_gradient(gradient)

    def test_channel_preserved(self):
        assert all(part.channel == 'y' for part in pp.split_gradient(pp.make_trapezoid('y', area=3000, duration=3e-3)))
