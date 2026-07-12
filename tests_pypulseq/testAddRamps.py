import numpy as np
import pypulseq as pp

from . import assert_equal


class TestAddRamps:
    def test_single_channel(self):
        output = pp.add_ramps(np.linspace(0, 100, 20).reshape(1, -1), system=pp.Opts())[0]
        assert_equal(output[0, 0], 0, abs_tol=1e-6)
        assert_equal(output[0, -1], 0, abs_tol=1e-6)
        assert output.shape[1] > 20

    def test_multi_channel(self):
        output_x, output_y = pp.add_ramps([np.linspace(0, 100, 20), np.linspace(0, 50, 20)], system=pp.Opts())
        assert_equal(output_x[0], 0, abs_tol=1e-3)
        assert_equal(output_x[-1], 0, abs_tol=1e-3)
        assert_equal(output_y[0], 0, abs_tol=1e-3)
        assert_equal(output_y[-1], 0, abs_tol=1e-3)
        assert len(output_x) == len(output_y)

    def test_rf_padding(self):
        rf_shape = np.ones(200)
        trajectory, padded_rf = pp.add_ramps(np.linspace(0, 100, 20).reshape(1, -1), rf=rf_shape, system=pp.Opts())
        assert len(padded_rf) > len(rf_shape)
        assert_equal(padded_rf[0], 0, abs_tol=1e-10)
        assert_equal(padded_rf[-1], 0, abs_tol=1e-10)

    def test_zero_start_end(self):
        trajectory = np.concatenate((np.zeros(3), np.linspace(0, 100, 20), np.zeros(3)))
        output = pp.add_ramps(trajectory.reshape(1, -1), system=pp.Opts())[0]
        assert_equal(output[0, 0], 0, abs_tol=1e-6)
        assert_equal(output[0, -1], 0, abs_tol=1e-6)
