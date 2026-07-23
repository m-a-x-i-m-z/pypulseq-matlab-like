import math
from typing import Tuple

import numpy as np
from scipy.interpolate import PPoly


def calc_moments_b_tensor(
    self,
    calcB: bool = True,
    calcm1: bool = False,
    calcm2: bool = False,
    calcm3: bool = False,
    Ndummy: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    _, _, _, _, t_excitation, t_refocusing, _, _, gw_pp, _ = self.calculate_kspacePP()

    R = len(t_excitation)
    if R == 0:
        return np.zeros((0, 3, 3)), np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 3))

    gx_pp_l, gy_pp_l, gz_pp_l = _ensure_three_pp(gw_pp, float(sum(self.block_durations.values())))

    t_seq = []
    t_echo = []
    for i in range(R):
        t_echo.append(2.0 * t_refocusing[i] - t_excitation[i])
        t_seq.extend([t_excitation[i], t_refocusing[i], t_echo[i]])
    t_seq = np.asarray(t_seq, dtype=float)

    t1 = gx_pp_l.x
    t2 = gy_pp_l.x
    t3 = gz_pp_l.x
    tn = np.unique(np.concatenate((t1, t2, t3, t_seq)))

    gx_pp_coefs = fillPpCoefs(gx_pp_l, tn)
    gx_pp_l = mkpp(tn, gx_pp_coefs)
    gy_pp_coefs = fillPpCoefs(gy_pp_l, tn)
    gy_pp_l = mkpp(tn, gy_pp_coefs)
    gz_pp_coefs = fillPpCoefs(gz_pp_l, tn)
    gz_pp_l = mkpp(tn, gz_pp_coefs)

    n = 1 + Ndummy
    gz_pp = [None] * R
    gy_pp = [None] * R
    gx_pp = [None] * R
    for i in range(len(gz_pp_l.x)):
        if n == R:
            break
        elif abs(gz_pp_l.x[i] - t_excitation[n]) == 0:
            gz_pp[n - Ndummy - 1] = fnbrk(gz_pp_l, [t_excitation[n - 1], t_excitation[n]])
            gy_pp[n - Ndummy - 1] = fnbrk(gy_pp_l, [t_excitation[n - 1], t_excitation[n]])
            gx_pp[n - Ndummy - 1] = fnbrk(gx_pp_l, [t_excitation[n - 1], t_excitation[n]])
            n = n + 1

    gx_pp[R - 1] = fnbrk(gx_pp_l, [t_excitation[-1], gx_pp_l.x[-1]])
    gy_pp[R - 1] = fnbrk(gy_pp_l, [t_excitation[-1], gy_pp_l.x[-1]])
    gz_pp[R - 1] = fnbrk(gz_pp_l, [t_excitation[-1], gz_pp_l.x[-1]])

    for j in range(R):
        if gx_pp[j] is None:
            continue
        gx_coefs = gx_pp[j].c.T.copy()
        gy_coefs = gy_pp[j].c.T.copy()
        gz_coefs = gz_pp[j].c.T.copy()
        for i in range(gx_coefs.shape[0]):
            if j + Ndummy < len(t_refocusing) and gx_pp[j].x[i] == t_refocusing[j + Ndummy]:
                gx_coefs[i:, :] = -gx_coefs[i:, :]
                gy_coefs[i:, :] = -gy_coefs[i:, :]
                gz_coefs[i:, :] = -gz_coefs[i:, :]
        gx_pp[j] = mkpp(gx_pp[j].x, gx_coefs)
        gy_pp[j] = mkpp(gy_pp[j].x, gy_coefs)
        gz_pp[j] = mkpp(gz_pp[j].x, gz_coefs)

    B = np.zeros((R, 3, 3))
    if calcB:
        qz_pp = [None] * R
        qy_pp = [None] * R
        qx_pp = [None] * R
        for i in range(R):
            if gx_pp[i] is None:
                continue
            qx_pp[i] = compat_fnint(gx_pp[i])
            qy_pp[i] = compat_fnint(gy_pp[i])
            qz_pp[i] = compat_fnint(gz_pp[i])
            qx_pp[i].c = qx_pp[i].c * 2 * np.pi
            qy_pp[i].c = qy_pp[i].c * 2 * np.pi
            qz_pp[i].c = qz_pp[i].c * 2 * np.pi

        for m in range(R):
            if qx_pp[m] is None:
                continue
            q_pp = [qx_pp[m], qy_pp[m], qz_pp[m]]
            for i in range(3):
                for j in range(3):
                    coefs = np.zeros((q_pp[i].c.shape[1], 5))
                    for k in range(q_pp[i].c.shape[1]):
                        coefs[k, :] = np.convolve(q_pp[i].c[:, k], q_pp[j].c[:, k])
                    bpp_div = mkpp(qx_pp[m].x, coefs)
                    bpp = compat_fnint(bpp_div)
                    te_idx = 2 + 3 * (m + Ndummy)
                    if te_idx < len(t_seq):
                        B[m, i, j] = ppval(bpp, t_seq[te_idx])

    m1 = np.zeros((R, 3))
    if calcm1:
        m1z_pp = [None] * R
        m1y_pp = [None] * R
        m1x_pp = [None] * R
        for i in range(R):
            if gx_pp[i] is None:
                continue
            t_pp_coefs = np.zeros((len(gx_pp[i].x) - 1, 2))
            t_pp_coefs[:, 0] = 1
            t_pp_coefs[:, 1] = gx_pp[i].x[:-1] - gx_pp[i].x[0]

            new_cx = np.zeros((gx_pp[i].c.shape[1], gx_pp[i].c.shape[0] + 1))
            new_cy = np.zeros((gy_pp[i].c.shape[1], gy_pp[i].c.shape[0] + 1))
            new_cz = np.zeros((gz_pp[i].c.shape[1], gz_pp[i].c.shape[0] + 1))

            gx_coefs = gx_pp[i].c.T
            gy_coefs = gy_pp[i].c.T
            gz_coefs = gz_pp[i].c.T
            for k in range(t_pp_coefs.shape[0]):
                new_cx[k, :] = np.convolve(gx_coefs[k, :], t_pp_coefs[k, :])
                new_cy[k, :] = np.convolve(gy_coefs[k, :], t_pp_coefs[k, :])
                new_cz[k, :] = np.convolve(gz_coefs[k, :], t_pp_coefs[k, :])

            tgx_pp = mkpp(gx_pp[i].x, new_cx)
            tgy_pp = mkpp(gy_pp[i].x, new_cy)
            tgz_pp = mkpp(gz_pp[i].x, new_cz)

            m1x_pp[i] = compat_fnint(tgx_pp)
            m1y_pp[i] = compat_fnint(tgy_pp)
            m1z_pp[i] = compat_fnint(tgz_pp)
            m1x_pp[i].c = m1x_pp[i].c * 2 * np.pi
            m1y_pp[i].c = m1y_pp[i].c * 2 * np.pi
            m1z_pp[i].c = m1z_pp[i].c * 2 * np.pi

        for m in range(R):
            te_idx = 2 + 3 * (m + Ndummy)
            if m1x_pp[m] is None or te_idx >= len(t_seq):
                continue
            m1[m, 0] = ppval(m1x_pp[m], t_seq[te_idx])
            m1[m, 1] = ppval(m1y_pp[m], t_seq[te_idx])
            m1[m, 2] = ppval(m1z_pp[m], t_seq[te_idx])

    m2 = np.zeros((R, 3))
    if calcm2:
        m2z_pp = [None] * R
        m2y_pp = [None] * R
        m2x_pp = [None] * R
        for i in range(R):
            if gx_pp[i] is None:
                continue
            t2_pp_coefs = np.zeros((len(gx_pp[i].x) - 1, 3))
            t2_pp_coefs[:, 0] = 1
            t2_pp_coefs[:, 1] = 2 * (gx_pp[i].x[:-1] - gx_pp[i].x[0])
            t2_pp_coefs[:, 2] = (gx_pp[i].x[:-1] - gx_pp[i].x[0]) ** 2

            new_cx = np.zeros((gx_pp[i].c.shape[1], gx_pp[i].c.shape[0] + 2))
            new_cy = np.zeros((gy_pp[i].c.shape[1], gy_pp[i].c.shape[0] + 2))
            new_cz = np.zeros((gz_pp[i].c.shape[1], gz_pp[i].c.shape[0] + 2))

            gx_coefs = gx_pp[i].c.T
            gy_coefs = gy_pp[i].c.T
            gz_coefs = gz_pp[i].c.T
            for k in range(t2_pp_coefs.shape[0]):
                new_cx[k, :] = np.convolve(gx_coefs[k, :], t2_pp_coefs[k, :])
                new_cy[k, :] = np.convolve(gy_coefs[k, :], t2_pp_coefs[k, :])
                new_cz[k, :] = np.convolve(gz_coefs[k, :], t2_pp_coefs[k, :])

            tgx_pp = mkpp(gx_pp[i].x, new_cx)
            tgy_pp = mkpp(gy_pp[i].x, new_cy)
            tgz_pp = mkpp(gz_pp[i].x, new_cz)

            m2x_pp[i] = compat_fnint(tgx_pp)
            m2y_pp[i] = compat_fnint(tgy_pp)
            m2z_pp[i] = compat_fnint(tgz_pp)
            m2x_pp[i].c = m2x_pp[i].c * 2 * np.pi
            m2y_pp[i].c = m2y_pp[i].c * 2 * np.pi
            m2z_pp[i].c = m2z_pp[i].c * 2 * np.pi

        for m in range(R):
            te_idx = 2 + 3 * (m + Ndummy)
            if m2x_pp[m] is None or te_idx >= len(t_seq):
                continue
            m2[m, 0] = ppval(m2x_pp[m], t_seq[te_idx]) - ppval(m2x_pp[m], m2x_pp[m].x[0])
            m2[m, 1] = ppval(m2y_pp[m], t_seq[te_idx]) - ppval(m2y_pp[m], m2y_pp[m].x[0])
            m2[m, 2] = ppval(m2z_pp[m], t_seq[te_idx]) - ppval(m2z_pp[m], m2z_pp[m].x[0])

    m3 = np.zeros((R, 3))
    if calcm3:
        m3z_pp = [None] * R
        m3y_pp = [None] * R
        m3x_pp = [None] * R
        for i in range(R):
            if gx_pp[i] is None:
                continue
            t3_pp_coefs = np.zeros((len(gx_pp[i].x) - 1, 4))
            t3_pp_coefs[:, 0] = 1
            t3_pp_coefs[:, 1] = 3 * (gx_pp[i].x[:-1] - gx_pp[i].x[0])
            t3_pp_coefs[:, 2] = 3 * (gx_pp[i].x[:-1] - gx_pp[i].x[0]) ** 2
            t3_pp_coefs[:, 3] = (gx_pp[i].x[:-1] - gx_pp[i].x[0]) ** 3

            new_cx = np.zeros((gx_pp[i].c.shape[1], gx_pp[i].c.shape[0] + 3))
            new_cy = np.zeros((gy_pp[i].c.shape[1], gy_pp[i].c.shape[0] + 3))
            new_cz = np.zeros((gz_pp[i].c.shape[1], gz_pp[i].c.shape[0] + 3))

            gx_coefs = gx_pp[i].c.T
            gy_coefs = gy_pp[i].c.T
            gz_coefs = gz_pp[i].c.T
            for k in range(t3_pp_coefs.shape[0]):
                new_cx[k, :] = np.convolve(gx_coefs[k, :], t3_pp_coefs[k, :])
                new_cy[k, :] = np.convolve(gy_coefs[k, :], t3_pp_coefs[k, :])
                new_cz[k, :] = np.convolve(gz_coefs[k, :], t3_pp_coefs[k, :])

            tgx_pp = mkpp(gx_pp[i].x, new_cx)
            tgy_pp = mkpp(gy_pp[i].x, new_cy)
            tgz_pp = mkpp(gz_pp[i].x, new_cz)

            m3x_pp[i] = compat_fnint(tgx_pp)
            m3y_pp[i] = compat_fnint(tgy_pp)
            m3z_pp[i] = compat_fnint(tgz_pp)
            m3x_pp[i].c = m3x_pp[i].c * 2 * np.pi
            m3y_pp[i].c = m3y_pp[i].c * 2 * np.pi
            m3z_pp[i].c = m3z_pp[i].c * 2 * np.pi

        for m in range(R):
            te_idx = 2 + 3 * (m + Ndummy)
            if m3x_pp[m] is None or te_idx >= len(t_seq):
                continue
            m3[m, 0] = ppval(m3x_pp[m], t_seq[te_idx]) - ppval(m3x_pp[m], m3x_pp[m].x[0])
            m3[m, 1] = ppval(m3y_pp[m], t_seq[te_idx]) - ppval(m3y_pp[m], m3y_pp[m].x[0])
            m3[m, 2] = ppval(m3z_pp[m], t_seq[te_idx]) - ppval(m3z_pp[m], m3z_pp[m].x[0])

    return B, m1, m2, m3


def fillPpCoefs(pp1: PPoly, xn: np.ndarray) -> np.ndarray:
    idx1 = slookup(xn[:-1], pp1.x[:-1])
    pp1_coefs = np.zeros((len(xn) - 1, pp1.c.shape[0]))
    src = pp1.c.T
    for i in range(pp1_coefs.shape[0]):
        if idx1[i] > 0:
            pp1_coefs[i, :] = src[idx1[i] - 1, :]
        elif i > 0:
            for k in range(pp1.c.shape[0]):
                for l in range(k + 1):
                    pp1_coefs[i, -(l + 1)] = (
                        pp1_coefs[i, -(l + 1)]
                        + pp1_coefs[i - 1, -(k + 1)] * math.comb(k, l) * (xn[i] - xn[i - 1]) ** (k - l)
                    )
    return pp1_coefs


def slookup(what: np.ndarray, where: np.ndarray) -> np.ndarray:
    idx = np.zeros_like(what, dtype=int)
    wb = 0
    for c in range(len(what)):
        hits = np.where(np.abs(where[wb:] - what[c]) < 1e-12)[0]
        if hits.size == 0:
            continue
        idx[c] = wb + hits[0] + 1
        wb = wb + hits[0] + 1
    return idx


def sintlookup(what: np.ndarray, where: np.ndarray) -> np.ndarray:
    idx = np.zeros_like(what, dtype=int)
    wb = 0
    for c in range(len(what)):
        hits = np.where(what[c] >= where[wb:])[0]
        if hits.size == 0:
            continue
        idx[c] = wb + hits[-1] + 1
        wb = idx[c] - 1
    return idx


def compat_fnint(pp_in: PPoly) -> PPoly:
    return pp_in.antiderivative()


def mkpp(breaks: np.ndarray, coefs_rows: np.ndarray) -> PPoly:
    return PPoly(np.asarray(coefs_rows, dtype=float).T, np.asarray(breaks, dtype=float), extrapolate=True)


def fnbrk(pp: PPoly, interval) -> PPoly:
    start = float(interval[0])
    stop = float(interval[1])
    start_idx = _find_break_index(pp.x, start)
    stop_idx = _find_break_index(pp.x, stop)
    if start_idx is None or stop_idx is None or stop_idx <= start_idx:
        raise ValueError('Invalid interval for fnbrk equivalent.')
    return PPoly(pp.c[:, start_idx:stop_idx].copy(), pp.x[start_idx : stop_idx + 1].copy(), extrapolate=True)


def ppval(pp: PPoly, x: float) -> float:
    return float(pp(x))


def _find_break_index(breaks: np.ndarray, value: float):
    hits = np.where(np.abs(breaks - value) < 1e-12)[0]
    if hits.size == 0:
        return None
    return int(hits[0])


def _ensure_three_pp(gw_pp, total_duration: float):
    out = []
    for pp in gw_pp[:3]:
        if pp is None:
            out.append(PPoly(np.zeros((2, 1)), np.array([0.0, total_duration]), extrapolate=True))
        else:
            out.append(pp)
    while len(out) < 3:
        out.append(PPoly(np.zeros((2, 1)), np.array([0.0, total_duration]), extrapolate=True))
    return out
