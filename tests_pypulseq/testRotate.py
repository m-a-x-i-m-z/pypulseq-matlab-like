import math

import numpy as np
import pytest

import pypulseq as pp
from pypulseq.rotate_3d import rotate_3d


def _grad(channel, area=1000, system=None):
    return pp.make_trapezoid(channel, area=area, duration=2e-3, system=system)


def _areas(events):
    result = dict.fromkeys(('x', 'y', 'z'), 0.0)
    for event in events:
        if hasattr(event, 'channel'):
            result[event.channel] = event.area
    return result


def _matrix(axis, angle):
    c, s = math.cos(angle), math.sin(angle)
    return {
        'x': np.array(((1, 0, 0), (0, c, -s), (0, s, c))),
        'y': np.array(((c, 0, s), (0, 1, 0), (-s, 0, c))),
        'z': np.array(((c, -s, 0), (s, c, 0), (0, 0, 1))),
    }[axis]


def _test_90_degree_rotation(axis, input_axis, output_axis):
    gradient = _grad(input_axis)
    areas = _areas(pp.rotate(gradient, axis=axis, angle=math.pi / 2))
    assert abs(areas[output_axis]) == pytest.approx(abs(gradient.area), abs=10)
    assert abs(areas[input_axis]) < 10


def _cross_validate(axis, channels, angle):
    system = pp.Opts()
    first = _grad(channels[0], 1000, system)
    second = _grad(channels[1], 700 if axis != 'z' else 500, system)
    with pytest.raises(Exception):
        pp.rotate(first, second, axis=axis, angle=angle)
    with pytest.raises(Exception):
        rotate_3d(_matrix(axis, angle), first, second)

    derated_system = pp.Opts(max_slew=system.max_slew / math.sqrt(2))
    first = _grad(channels[0], 1000, derated_system)
    second = _grad(channels[1], 700 if axis != 'z' else 500, derated_system)
    rotate_areas = _areas(pp.rotate(first, second, axis=axis, angle=angle))
    rotate3d_areas = _areas(rotate_3d(_matrix(axis, angle), first, second))
    for channel in ('x', 'y', 'z'):
        assert rotate_areas[channel] == pytest.approx(rotate3d_areas[channel], abs=5)


def test_zero_angle():
    gx = _grad('x')
    out = pp.rotate(gx, axis='z', angle=0)
    assert len(out) == 1
    assert out[0].channel == 'x'
    assert out[0].area == pytest.approx(gx.area, abs=1)


def test_90deg_z_rotation():
    _test_90_degree_rotation('z', 'x', 'y')


def test_90deg_x_rotation():
    _test_90_degree_rotation('x', 'y', 'z')


def test_90deg_y_rotation():
    _test_90_degree_rotation('y', 'z', 'x')


def test_parallel_gradient_unchanged():
    gz = _grad('z')
    out = pp.rotate(gz, axis='z', angle=math.pi / 3)
    assert len(out) == 1
    assert out[0].channel == 'z'
    assert out[0].area == pytest.approx(gz.area, abs=1)


def test_45deg_split():
    gx = _grad('x')
    areas = _areas(pp.rotate(gx, axis='z', angle=math.pi / 4))
    assert areas['x'] == pytest.approx(gx.area * math.cos(math.pi / 4), abs=10)
    assert areas['y'] == pytest.approx(gx.area * math.sin(math.pi / 4), abs=10)


def test_non_grad_passthrough():
    adc = pp.make_adc(128, duration=1e-3)
    assert any(event.type == 'adc' for event in pp.rotate(_grad('x'), adc, axis='z', angle=math.pi / 4))


def test_with_system():
    system = pp.Opts()
    gx = _grad('x', system=system)
    areas = _areas(pp.rotate(gx, axis='z', angle=math.pi / 4, system=system))
    assert areas['x'] == pytest.approx(gx.area * math.cos(math.pi / 4), abs=10)


def test_invalid_axis_error():
    with pytest.raises(Exception):
        pp.rotate(_grad('x'), axis='w', angle=math.pi / 4)


def test_non_scalar_angle_error():
    with pytest.raises(Exception):
        pp.rotate(_grad('x'), axis='z', angle=[math.pi / 4, math.pi / 2])


def test_two_gradients():
    system = pp.Opts()
    gx = _grad('x', 1000, system)
    gy = _grad('y', 2000, system)
    with pytest.raises(Exception):
        pp.rotate(gx, gy, axis='z', angle=math.pi / 6)

    derated_system = pp.Opts(max_slew=system.max_slew / math.sqrt(2))
    gx = _grad('x', 1000, derated_system)
    gy = _grad('y', 2000, derated_system)
    areas = _areas(pp.rotate(gx, gy, axis='z', angle=math.pi / 6))
    assert areas['x'] == pytest.approx(gx.area * math.cos(math.pi / 6) - gy.area * math.sin(math.pi / 6), abs=15)
    assert areas['y'] == pytest.approx(gx.area * math.sin(math.pi / 6) + gy.area * math.cos(math.pi / 6), abs=15)


def test_cross_validate_z():
    _cross_validate('z', ('x', 'y'), math.pi / 5)


def test_cross_validate_x():
    _cross_validate('x', ('y', 'z'), math.pi / 7)


def test_cross_validate_y():
    _cross_validate('y', ('x', 'z'), math.pi / 3)


def test_cross_validate_single_grad_all_axes():
    angles = {'x': math.pi / 6, 'y': math.pi / 4, 'z': math.pi / 3}
    for axis, angle in angles.items():
        for channel in ('x', 'y', 'z'):
            gradient = _grad(channel)
            rotate_areas = _areas(pp.rotate(gradient, axis=axis, angle=angle))
            rotate3d_areas = _areas(rotate_3d(_matrix(axis, angle), gradient))
            for component in ('x', 'y', 'z'):
                assert rotate_areas[component] == pytest.approx(rotate3d_areas[component], abs=10)