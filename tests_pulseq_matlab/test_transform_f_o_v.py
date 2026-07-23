import math

import numpy as np
import pytest

import pypulseq_matlab_like as pp


def _sys():
    return pp.Opts(max_grad=200, grad_unit='mT/m', max_slew=200, slew_unit='T/m/s')


def _rx(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array(((1, 0, 0), (0, c, -s), (0, s, c)))


def _ry(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array(((c, 0, s), (0, 1, 0), (-s, 0, c)))


def _rz(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array(((c, -s, 0), (s, c, 0), (0, 0, 1)))


def _event(events, kind, channel=None):
    for event in events:
        if getattr(event, 'type', None) == kind and (channel is None or getattr(event, 'channel', None) == channel):
            return event
    return None


def _areas(events):
    return np.array([getattr(_event(events, 'trap', axis) or _event(events, 'grad', axis), 'area', 0.0) for axis in 'xyz'])


def _trap(axis, area, system, duration=2e-3, **kwargs):
    return pp.make_trapezoid(axis, area=area, duration=duration, system=system, **kwargs)


def _constant_gradient(axis, flat_area, flat_time, system):
    return pp.make_trapezoid(axis, flat_area=flat_area, flat_time=flat_time, system=system)


def _rf(delay=0.0, system=None):
    return pp.make_sinc_pulse(math.pi / 6, duration=1e-3, slice_thickness=5e-3, delay=delay, use='excitation', system=system)


def _transform(**kwargs):
    return pp.transform_fov(**kwargs)


def test_constructor_no_args():
    with pytest.raises(ValueError):
        _transform()


def test_constructor_rotation():
    rotation = _rz(math.pi / 4)
    obj = _transform(rotation=rotation)
    np.testing.assert_allclose(obj.rotation, rotation, atol=1e-14)
    assert len(obj.translation) == len(obj.scale) == 0


def test_constructor_translation():
    translation = np.array((0.01, 0.02, 0.03))
    obj = _transform(translation=translation)
    np.testing.assert_allclose(obj.translation, translation, atol=1e-14)
    assert len(obj.rotation) == 0


def test_constructor_scale():
    scale = np.array((2, 0.5, 1))
    np.testing.assert_allclose(_transform(scale=scale).scale, scale, atol=1e-14)


def test_constructor_transform_with_rotation_error():
    with pytest.raises(ValueError):
        _transform(transform=np.eye(4), rotation=np.eye(3))


def test_constructor_transform_with_translation_error():
    with pytest.raises(ValueError):
        _transform(transform=np.eye(4), translation=(1, 1, 1))


def test_scale_trap_x2():
    system = _sys(); gx = _trap('x', 1000, system)
    gx2 = _event(_transform(scale=(0.5, 1, 1), system=system).apply_to_block(gx), 'trap', 'x')
    assert gx2.area == pytest.approx(.5 * gx.area, abs=1)
    assert gx2.amplitude == pytest.approx(.5 * gx.amplitude, abs=1e-3)


def test_scale_selective_channel():
    system = _sys(); g = [_trap('x', 1000, system), _trap('y', 2000, system), _trap('z', 3000, system)]
    result = _areas(_transform(scale=(1, 1, .5), system=system).apply_to_block(*g))
    np.testing.assert_allclose(result, (g[0].area, g[1].area, .5 * g[2].area), atol=1)


def test_scale_negate():
    system = _sys(); gx = _trap('x', 1000, system)
    assert _areas(_transform(scale=(-1, 1, 1), system=system).apply_to_block(gx))[0] == pytest.approx(-gx.area, abs=1)


def test_rotation_identity():
    system = _sys(); out = _transform(rotation=np.eye(3), system=system).apply_to_block(_trap('x', 1000, system), _trap('y', 2000, system))
    np.testing.assert_allclose(_areas(out), (1000, 2000, 0), atol=10)


def test_rotation_90z_gx_to_gy():
    system = _sys(); out = _transform(rotation=_rz(math.pi / 2), system=system).apply_to_block(_trap('x', 1000, system))
    np.testing.assert_allclose(_areas(out), (0, 1000, 0), atol=15)


def test_rotation_90x_gy_to_gz():
    system = _sys(); out = _transform(rotation=_rx(math.pi / 2), system=system).apply_to_block(_trap('y', 2000, system))
    np.testing.assert_allclose(_areas(out), (0, 0, 2000), atol=15)


def test_rotation_45z_splits_area():
    system = _sys(); out = _transform(rotation=_rz(math.pi / 4), system=system).apply_to_block(_trap('x', 1000, system))
    np.testing.assert_allclose(_areas(out)[:2], (1000 / math.sqrt(2), 1000 / math.sqrt(2)), rtol=.02)


def test_rotation_preserves_total_area():
    system = _sys(); area = np.array((1000, 2000, 1500)); g = [_trap(axis, a, system, max_slew=system.max_slew / math.sqrt(3)) for axis, a in zip('xyz', area)]
    result = _areas(_transform(rotation=_ry(math.radians(20)) @ _rz(math.radians(30)), system=system).apply_to_block(*g))
    assert np.linalg.norm(result) == pytest.approx(np.linalg.norm(area), rel=.02)


def test_rotation_analytic_area_vector():
    system = _sys(); area = np.array((1000, -500, 2000)); rotation = _rx(math.radians(53)) @ _rz(math.radians(37))
    result = _areas(_transform(rotation=rotation, system=system).apply_to_block(*[_trap(a, v, system, max_slew=system.max_slew / math.sqrt(3)) for a, v in zip('xyz', area)]))
    np.testing.assert_allclose(result, rotation @ area, rtol=.03)


def test_translation_rf_freq_offset():
    system = _sys(); gx = _constant_gradient('x', 5000, 1e-3, system); rf = _rf(gx.rise_time, system)
    out = _transform(translation=(.01, 0, 0), system=system).apply_to_block(rf, gx)
    assert _event(out, 'rf').freq_offset == pytest.approx(.01 * gx.amplitude, rel=.01)


def test_translation_adc_freq_offset():
    system = _sys(); gx = _constant_gradient('x', 5000, 2e-3, system); adc = pp.make_adc(64, duration=2e-3, delay=gx.rise_time, system=system)
    out = _transform(translation=(.02, 0, 0), system=system).apply_to_block(adc, gx)
    assert _event(out, 'adc').freq_offset == pytest.approx(.02 * gx.amplitude, rel=.01)


def test_translation_zero_no_change():
    system = _sys(); gx = _constant_gradient('x', 5000, 1e-3, system); rf = _rf(gx.rise_time, system); adc = pp.make_adc(32, duration=1e-3, delay=gx.rise_time, system=system)
    out = _transform(translation=(0, 0, 0), system=system).apply_to_block(rf, adc, gx)
    assert _event(out, 'rf').freq_offset == pytest.approx(rf.freq_offset, abs=1e-6)
    assert _event(out, 'adc').freq_offset == pytest.approx(adc.freq_offset, abs=1e-6)


def test_combined_rotation_translation():
    system = _sys(); gx = _constant_gradient('x', 5000, 1e-3, system); rf = _rf(gx.rise_time, system)
    out = _transform(rotation=_rz(math.pi / 2), translation=(.01, 0, 0), system=system).apply_to_block(rf, gx)
    assert _areas(out)[0] == pytest.approx(0, abs=20)
    assert abs(_areas(out)[1]) == pytest.approx(abs(gx.area), abs=20)
    assert abs(_event(out, 'rf').freq_offset) > 0


def test_combined_all_three():
    system = _sys(); area = np.array((1000, 2000, -1500)); scale = np.array((.7, .5, .6)); rotation = _rx(math.radians(20)) @ _rz(math.radians(30))
    out = _transform(scale=scale, rotation=rotation, translation=(.005, .01, -.003), system=system).apply_to_block(*[_trap(a, v, system, max_slew=system.max_slew / math.sqrt(3)) for a, v in zip('xyz', area)])
    np.testing.assert_allclose(_areas(out), rotation @ (scale * area), rtol=.03)


def test_double_rotation_sequential():
    system = _sys(); gx = _trap('x', 1000, system); first, second = _rz(math.pi / 6), _rx(math.pi / 4)
    combined = _areas(_transform(rotation=second @ first, system=system).apply_to_block(gx))
    sequential = _areas(_transform(rotation=second, system=system).apply_to_block(*_transform(rotation=first, system=system).apply_to_block(gx)))
    np.testing.assert_allclose(combined, sequential, rtol=.03)


def test_rotation_extension_produces_event():
    out = _transform(rotation=_rz(math.pi / 4), use_rotation_extension=True, system=_sys()).apply_to_block(_trap('x', 1000, _sys()))
    assert _event(out, 'rot3D') is not None


def test_rotation_extension_grads_not_rotated():
    system = _sys(); out = _transform(rotation=_rz(math.pi / 2), use_rotation_extension=True, system=system).apply_to_block(_trap('x', 1000, system))
    np.testing.assert_allclose(_areas(out), (1000, 0, 0), atol=15)


def test_transform_4x4_identity():
    system = _sys(); out = _transform(transform=np.eye(4), system=system).apply_to_block(_trap('x', 1000, system), _trap('y', 2000, system))
    np.testing.assert_allclose(_areas(out), (1000, 2000, 0), atol=10)


def test_applyToSeq_basic():
    system = _sys(); seq = pp.Sequence(system); gx = _trap('x', 1000, system)
    for _ in range(3): seq.add_block(gx)
    result = _transform(rotation=_rz(math.pi / 2), system=system).apply_to_seq(seq)
    assert len(result.block_durations) == 3
    for block_id in result.block_events:
        assert getattr(result.get_block(block_id).gx, 'area', 0) == pytest.approx(0, abs=15)
        assert abs(result.get_block(block_id).gy.area) == pytest.approx(abs(gx.area), abs=15)


def test_applyToSeq_blockRange():
    system = _sys(); seq = pp.Sequence(system); gx = _trap('x', 1000, system)
    for _ in range(3): seq.add_block(gx)
    assert len(_transform(rotation=_rz(math.pi / 2), system=system).apply_to_seq(seq, block_range=(2, 3)).block_durations) == 2


def _waveform_areas(seq):
    return np.array([np.trapezoid(w[1], w[0]) if w.size else 0.0 for w in seq.waveforms_and_times()[0]])


def test_kspace_rotation_analytic():
    system = _sys(); seq = pp.Sequence(system); gx = _trap('x', 5000, system)
    seq.add_block(gx, pp.make_adc(64, duration=2e-3 - gx.rise_time - gx.fall_time, delay=gx.rise_time, system=system))
    result = _waveform_areas(_transform(rotation=_rz(math.pi / 2), system=system).apply_to_seq(seq))
    assert result[0] == pytest.approx(0, abs=50)
    assert abs(result[1]) == pytest.approx(abs(_waveform_areas(seq)[0]), rel=.02)


def test_kspace_combined_transform_analytic():
    system = _sys(); area = np.array((3000, -1000, 2000)); scale = np.array((.5, .8, .7)); rotation = _ry(math.radians(25)) @ _rz(math.radians(40))
    seq = pp.Sequence(system); seq.add_block(*[_trap(a, v, system, duration=3e-3, max_slew=system.max_slew / math.sqrt(3)) for a, v in zip('xyz', area)])
    np.testing.assert_allclose(_waveform_areas(_transform(scale=scale, rotation=rotation, system=system).apply_to_seq(seq)), rotation @ (scale * area), rtol=.03)


def test_kspace_180_rotation_inverts():
    system = _sys(); seq = pp.Sequence(system); seq.add_block(_trap('x', 4000, system))
    assert _waveform_areas(_transform(rotation=_rz(math.pi), system=system).apply_to_seq(seq))[0] == pytest.approx(-_waveform_areas(seq)[0], rel=.02)


def test_three_axis_rotation_chain():
    system = _sys(); area = np.array((1500, -800, 2200)); rotation = _rz(math.radians(17)) @ _ry(math.radians(43)) @ _rx(math.radians(-29))
    out = _transform(rotation=rotation, system=system).apply_to_block(*[_trap(a, v, system, duration=3e-3, max_slew=system.max_slew / math.sqrt(3)) for a, v in zip('xyz', area)])
    np.testing.assert_allclose(_areas(out), rotation @ area, rtol=.03)


def test_translation_phase_accumulation():
    system = _sys(); seq = pp.Sequence(system); gx = _constant_gradient('x', 1000, 1e-3, system); adc = pp.make_adc(16, duration=1e-3, delay=gx.rise_time, system=system)
    for _ in range(3): seq.add_block(gx, adc)
    result = _transform(translation=(.01, 0, 0), system=system).apply_to_seq(seq)
    for block_id in result.block_events: assert abs(result.get_block(block_id).adc.freq_offset) > 0


def test_translation_gradient_only_block():
    system = _sys(); gx = _trap('x', 1000, system, duration=1e-3)
    assert _areas(_transform(translation=(.01, 0, 0), system=system).apply_to_block(gx))[0] == pytest.approx(1000, abs=5)


def test_translation_rf_only_block():
    system = _sys(); rf = pp.make_block_pulse(math.pi / 2, duration=1e-3, use='excitation', system=system)
    assert _event(_transform(translation=(.01, .02, .03), system=system).apply_to_block(rf), 'rf').freq_offset == pytest.approx(rf.freq_offset, abs=1e-6)
