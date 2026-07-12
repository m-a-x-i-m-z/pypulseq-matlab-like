from types import SimpleNamespace
from typing import Tuple
from warnings import warn

import numpy as np

from pypulseq.calc_rf_bandwidth import calc_rf_bandwidth
from pypulseq.opts import Opts


def sim_rf(
    rf: SimpleNamespace,
    rephase_factor: float = None,
    prephase_factor: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate an RF pulse with quaternion rotations.

    """
    bw_mul = 4.0
    df = 1.0
    dt = 10e-6

    if rephase_factor is None:
        if hasattr(rf, 'use') and rf.use == 'refocusing':
            rephase_factor = 0.0
        else:
            rephase_factor = -(rf.shape_dur - rf.center) / rf.shape_dur

    freq_ppm = getattr(rf, 'freq_ppm', 0.0)
    phase_ppm = getattr(rf, 'phase_ppm', 0.0)
    if abs(freq_ppm) > np.finfo(float).eps or abs(phase_ppm) > np.finfo(float).eps:
        warn('sim_rf() relies on Opts.default for B0 and gamma when ppm offsets are present.', stacklevel=2)
        sys = Opts.default
        full_freq_offset = rf.freq_offset + freq_ppm * 1e-6 * sys.gamma * sys.B0
        full_phase_offset = rf.phase_offset + phase_ppm * 1e-6 * sys.gamma * sys.B0
    else:
        full_freq_offset = rf.freq_offset
        full_phase_offset = rf.phase_offset

    f0 = full_freq_offset
    bw = calc_rf_bandwidth(rf, cutoff=0.5, return_axis=False, dw=df * 10.0, dt=dt)
    bw = abs(bw) + abs(f0)

    if bw > 4e3:
        dt = 5e-6
        if bw > 1e4:
            dt = 2e-6
            if bw > 2e4:
                dt = 1e-6

    t = (np.arange(1, int(np.round(rf.shape_dur / dt)) + 1) * dt) - 0.5 * dt
    f = 2 * np.pi * np.linspace(f0 - bw_mul * bw / 2.0, f0 + bw_mul * bw / 2.0, int(max(1, np.round(bw / df))))

    shape = 2 * np.pi * (
        np.interp(t, rf.t, np.real(rf.signal), left=0.0, right=0.0)
        + 1j * np.interp(t, rf.t, np.imag(rf.signal), left=0.0, right=0.0)
    )
    shape *= np.exp(1j * (full_phase_offset + 2 * np.pi * full_freq_offset * t))

    q = np.zeros((f.size, 4), dtype=float)
    q[:, 0] = 1.0

    w = -f * dt * t.size * prephase_factor
    q = _quat_multiply(q, np.column_stack((np.cos(w / 2.0), np.zeros((f.size, 2)), np.sin(w / 2.0))))

    for j in range(t.size):
        w = -dt * np.sqrt(np.abs(shape[j]) ** 2 + f**2)
        abs_w = np.abs(w)
        n = np.column_stack((np.real(shape[j]) * np.ones_like(f), np.imag(shape[j]) * np.ones_like(f), f))
        nz = abs_w > 0
        n[nz] *= dt / abs_w[nz, None]
        q = _quat_multiply(q, np.column_stack((np.cos(w / 2.0), np.sin(w / 2.0)[:, None] * n)))

    w = -f * dt * t.size * rephase_factor
    q = _quat_multiply(q, np.column_stack((np.cos(w / 2.0), np.zeros((f.size, 2)), np.sin(w / 2.0))))

    f_hz = f / (2 * np.pi)
    m = np.zeros((f.size, 4), dtype=float)

    m[:, 3] = 1.0
    m0rf = _quat_multiply(_quat_conj(q), _quat_multiply(m, q))
    mz_z = m0rf[:, 3]
    mz_xy = m0rf[:, 1] + 1j * m0rf[:, 2]

    m.fill(0.0)
    m[:, 1] = 1.0
    mx_xy = _quat_multiply(_quat_conj(q), _quat_multiply(m, q))
    mx_xy = mx_xy[:, 1] + 1j * mx_xy[:, 2]

    m.fill(0.0)
    m[:, 2] = 1.0
    my_xy = _quat_multiply(_quat_conj(q), _quat_multiply(m, q))
    my_xy = my_xy[:, 1] + 1j * my_xy[:, 2]
    ref_eff = (mx_xy + 1j * my_xy) / 2.0

    return mz_z, mz_xy, f_hz, ref_eff, mx_xy, my_xy


def _quat_multiply(q: np.ndarray, r: np.ndarray) -> np.ndarray:
    vec = (
        np.column_stack((q[:, 0] * r[:, 1], q[:, 0] * r[:, 2], q[:, 0] * r[:, 3]))
        + np.column_stack((r[:, 0] * q[:, 1], r[:, 0] * q[:, 2], r[:, 0] * q[:, 3]))
        + np.column_stack(
            (
                q[:, 2] * r[:, 3] - q[:, 3] * r[:, 2],
                q[:, 3] * r[:, 1] - q[:, 1] * r[:, 3],
                q[:, 1] * r[:, 2] - q[:, 2] * r[:, 1],
            )
        )
    )
    scalar = q[:, 0] * r[:, 0] - q[:, 1] * r[:, 1] - q[:, 2] * r[:, 2] - q[:, 3] * r[:, 3]
    return np.column_stack((scalar, vec))


def _quat_conj(q: np.ndarray) -> np.ndarray:
    out = q.copy()
    out[:, 1:4] *= -1.0
    return out
