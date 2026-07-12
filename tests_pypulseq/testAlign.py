import pytest
import pypulseq as pp

from . import assert_equal


class TestAlign:
    def test_left_align(self):
        rf = pp.make_block_pulse(3.141592653589793 / 2, duration=0.5e-3)
        gx = pp.make_trapezoid('x', area=1000, duration=1e-3)
        rf_out, gx_out = pp.align('left', rf, gx)
        assert_equal(rf_out.delay, 0, abs_tol=1e-10)
        assert_equal(gx_out.delay, 0, abs_tol=1e-10)

    def test_right_align(self):
        rf = pp.make_block_pulse(3.141592653589793 / 2, duration=0.5e-3)
        gx = pp.make_trapezoid('x', area=1000, duration=1e-3)
        rf_out, gx_out = pp.align('right', rf, gx)
        duration = pp.calc_duration(rf, gx)
        assert_equal(pp.calc_duration(rf_out, gx_out), duration, abs_tol=1e-9)
        assert_equal(pp.calc_duration(rf_out), duration, abs_tol=1e-9)
        assert_equal(pp.calc_duration(gx_out), duration, abs_tol=1e-9)

    def test_center_align(self):
        gx = pp.make_trapezoid('x', area=1000, duration=2e-3)
        rf = pp.make_block_pulse(3.141592653589793 / 2, duration=gx.flat_time)
        rf_out, gx_out = pp.align('center', rf, gx)
        assert_equal(rf_out.delay, gx_out.rise_time, abs_tol=1e-6)
        assert_equal(gx_out.delay, 0, abs_tol=1e-6)

    def test_required_duration(self):
        rf = pp.make_block_pulse(3.141592653589793 / 2, duration=0.5e-3)
        gx = pp.make_trapezoid('x', area=1000, duration=1e-3)
        rf_out, gx_out = pp.align('right', rf, gx, 3e-3)
        assert_equal(pp.calc_duration(rf_out), 3e-3, abs_tol=1e-6)
        assert_equal(pp.calc_duration(gx_out), 3e-3, abs_tol=1e-6)

    def test_required_duration_too_short(self):
        with pytest.raises(Exception):
            pp.align('left', pp.make_trapezoid('x', area=1000, duration=2e-3), 0.1e-3)

    def test_first_param_must_be_string(self):
        gx = pp.make_trapezoid('x', area=1000, duration=1e-3)
        with pytest.raises(Exception):
            pp.align(gx, gx)

    def test_mixed_alignment(self):
        rf = pp.make_block_pulse(3.141592653589793 / 2, duration=0.5e-3)
        gx = pp.make_trapezoid('x', area=1000, duration=2e-3)
        rf_out, gx_out = pp.align('left', rf, 'right', gx)
        assert_equal(rf_out.delay, 0, abs_tol=1e-10)
        assert_equal(pp.calc_duration(gx_out), max(pp.calc_duration(rf_out), pp.calc_duration(gx_out)), abs_tol=1e-6)

    def test_single_output(self):
        output = pp.align('left', pp.make_block_pulse(3.141592653589793 / 2, duration=0.5e-3), pp.make_trapezoid('x', area=1000, duration=1e-3))
        assert isinstance(output, list)
