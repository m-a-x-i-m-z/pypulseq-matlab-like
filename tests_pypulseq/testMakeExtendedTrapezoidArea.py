import numpy as np
import pypulseq_matlab_like as pp

from . import assert_equal


class TestMakeExtendedTrapezoidArea:
    def test_small_area(self):
        system = pp.Opts()
        gradient, _, _ = pp.make_extended_trapezoid_area(100, 'x', 0, 0, system=system)
        assert gradient.type == 'grad'
        assert gradient.channel == 'x'
        assert_equal(gradient.area, 100, abs_tol=1)

    def test_large_area(self):
        gradient, times, _ = pp.make_extended_trapezoid_area(50000, 'x', 0, 0, system=pp.Opts())
        assert_equal(gradient.area, 50000, abs_tol=1)
        assert len(times) >= 3

    def test_nonzero_gs_ge(self):
        gradient, _, _ = pp.make_extended_trapezoid_area(2000, 'x', 10000, 5000, system=pp.Opts())
        assert_equal(gradient.area, 2000, abs_tol=1)

    def test_negative_area(self):
        gradient, _, _ = pp.make_extended_trapezoid_area(-5000, 'x', 0, 0, system=pp.Opts())
        assert_equal(gradient.area, -5000, abs_tol=1)

    def test_amplitude_within_limits(self):
        system = pp.Opts()
        _, _, amplitudes = pp.make_extended_trapezoid_area(10000, 'x', 0, 0, system=system)
        assert np.all(np.abs(amplitudes) <= system.max_grad)

    def test_channel_y(self):
        gradient, _, _ = pp.make_extended_trapezoid_area(500, 'y', 0, 0, system=pp.Opts())
        assert gradient.channel == 'y'
