import math

import numpy as np
import pytest

import pypulseq_matlab_like as pp
from pypulseq_matlab_like.rotate_3d import rotate_3d


def _grad(channel, area=1000, system=None):
    return pp.make_trapezoid(channel, area=area, duration=2e-3, system=system)


def _areas(events):
    areas = dict.fromkeys(('x', 'y', 'z'), 0.0)
    for event in events:
        if hasattr(event, 'channel'):
            areas[event.channel] = event.area
    return areas


def _matrix(axis, angle):
    c, s = math.cos(angle), math.sin(angle)
    return {
        'x': np.array(((1, 0, 0), (0, c, -s), (0, s, c))),
        'y': np.array(((c, 0, s), (0, 1, 0), (-s, 0, c))),
        'z': np.array(((c, -s, 0), (s, c, 0), (0, 0, 1))),
    }[axis]


def _test_90_degree_rotation(axis, source, target):
    gradient = _grad(source)
    areas = _areas(rotate_3d(_matrix(axis, math.pi / 2), gradient))
    assert abs(areas[target]) == pytest.approx(abs(gradient.area), abs=10)


def _test_45_degree_rotation(axis, source, retained, created):
    gradient = _grad(source)
    areas = _areas(rotate_3d(_matrix(axis, math.pi / 4), gradient))
    assert areas[retained] == pytest.approx(gradient.area * math.cos(math.pi / 4), abs=10)
    assert areas[created] == pytest.approx(gradient.area * math.sin(math.pi / 4), abs=10)


def test_identity_rotation():
    out = rotate_3d(np.eye(3), _grad('x', 1000), _grad('y', 2000))
    assert len(out) == 2
    assert sorted(abs(event.area) for event in out) == pytest.approx([1000, 2000], abs=10)


def test_90deg_z_rotation():
    _test_90_degree_rotation('z', 'x', 'y')


def test_90deg_x_rotation():
    _test_90_degree_rotation('x', 'y', 'z')


def test_90deg_y_rotation():
    _test_90_degree_rotation('y', 'z', 'x')


def test_45deg_z_rotation():
    _test_45_degree_rotation('z', 'x', 'x', 'y')


def test_45deg_x_rotation():
    _test_45_degree_rotation('x', 'y', 'y', 'z')


def test_45deg_y_rotation():
    _test_45_degree_rotation('y', 'z', 'z', 'x')


def test_oblique_rotation_and_inverse():
    gx = _grad('x')
    axis = np.array((0.5, 0.3, 0.6))
    axis /= np.linalg.norm(axis)
    angle = math.radians(70)
    cross = np.array(((0, -axis[2], axis[1]), (axis[2], 0, -axis[0]), (-axis[1], axis[0], 0)))
    rotation = np.eye(3) * math.cos(angle) + (1 - math.cos(angle)) * np.outer(axis, axis) + cross * math.sin(angle)
    forward = rotate_3d(rotation, gx)
    areas = _areas(forward)
    assert all(abs(areas[channel]) > 1 for channel in ('x', 'y', 'z'))
    assert np.linalg.norm([areas[channel] for channel in ('x', 'y', 'z')]) == pytest.approx(abs(gx.area), abs=10)
    restored = _areas(rotate_3d(rotation.T, *forward))
    assert restored['x'] == pytest.approx(gx.area, abs=15)
    assert abs(restored['y']) == pytest.approx(0, abs=15)
    assert abs(restored['z']) == pytest.approx(0, abs=15)


def test_quaternion_input():
    angle = math.pi / 2
    out = rotate_3d([math.cos(angle / 2), 0, 0, math.sin(angle / 2)], _grad('x'))
    assert out


def test_non_grad_passthrough():
    out = rotate_3d(np.eye(3), _grad('x'), pp.make_adc(128, duration=1e-3))
    assert any(event.type == 'adc' for event in out)


def test_duplicate_axis_error():
    with pytest.raises(Exception):
        rotate_3d(np.eye(3), _grad('x', 1000), _grad('x', 500))


def test_invalid_rotation_error():
    with pytest.raises(Exception):
        rotate_3d([1, 2, 3], _grad('x'))


def test_with_system():
    assert rotate_3d(_matrix('z', math.pi / 4), _grad('x'), system=pp.Opts())