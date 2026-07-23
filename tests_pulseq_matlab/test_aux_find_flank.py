import numpy as np

from pypulseq_matlab_like.calc_rf_bandwidth import __find_flank as find_flank

from util import assert_equal


class TestAuxFindFlank:
    def test_gaussian_flank(self):
        x = np.linspace(-3, 3, 601)
        y = np.exp(-x**2 / (2 * 0.5**2))
        xf = find_flank(x, y, 0.5)
        assert xf > x[0]
        assert xf < 0

    def test_step_function(self):
        x = np.arange(1, 11)
        y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        xf = find_flank(x, y, 0.5)
        assert xf >= 5
        assert xf <= 6

    def test_cutoff_zero(self):
        x = np.arange(1, 6)
        y = np.array([0.1, 0.5, 1, 0.5, 0.1])
        assert_equal(find_flank(x, y, 0), x[0], abs_tol=0.01)

    def test_plateau(self):
        x = np.arange(1, 6)
        y = np.ones(5)
        assert_equal(find_flank(x, y, 0.5), x[0], abs_tol=0.01)
