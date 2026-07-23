import math
from copy import copy
from types import SimpleNamespace
from typing import Tuple, Union
from warnings import warn

import numpy as np
from scipy import signal as sp_signal

try:
    import sigpy.mri.rf as rf
    import sigpy.plot as pl
except ModuleNotFoundError as err:
    raise ModuleNotFoundError(
        "SigPy is not installed. Install it using 'pip install sigpy' or 'pip install pypulseq[sigpy]'."
    ) from err

from pypulseq_matlab_like.calc_rf_center import calc_rf_center
from pypulseq_matlab_like.calc_duration import calc_duration
from pypulseq_matlab_like.make_delay import make_delay
from pypulseq_matlab_like.make_trapezoid import make_trapezoid
from pypulseq_matlab_like.opts import Opts
from pypulseq_matlab_like.sigpy_pulse_opts import SigpyPulseOpts
from pypulseq_matlab_like.supported_labels_rf_use import get_supported_rf_uses
from pypulseq_matlab_like.utils.tracing import trace, trace_enabled


def _dzrf_ls_matlab_like(
    n: int,
    tb: float,
    ptype: str,
    d1: float,
    d2: float,
    cancel_alpha_phs: bool,
) -> np.ndarray:
    """
    Version note:
    - Native SigPy path is kept below in commented form where we switch calls.
    - This helper is added to reproduce the MATLAB v7 LS-SLR behavior for
      `ftype='ls'` while keeping the public PyPulseq API unchanged.
    """

    def _calc_ripples(ptype_: str, d1in: float, d2in: float) -> tuple[float, float, float]:
        p = str(ptype_).strip().lower()
        if p == 'st':
            return 1.0, d1in, d2in
        if p == 'ex':
            return np.sqrt(0.5), np.sqrt(d1in / 2.0), d2in / np.sqrt(2.0)
        if p == 'se':
            return 1.0, d1in / 4.0, np.sqrt(d2in)
        if p == 'inv':
            return 1.0, d1in / 8.0, np.sqrt(d2in / 2.0)
        if p == 'sat':
            return np.sqrt(0.5), d1in / 2.0, np.sqrt(d2in)
        raise ValueError(f'Pulse type ("{ptype_}") is not recognized.')

    def _dinf_local(d1_: float, d2_: float) -> float:
        a1 = 5.309e-3
        a2 = 7.114e-2
        a3 = -4.761e-1
        a4 = -2.66e-3
        a5 = -5.941e-1
        a6 = -4.278e-1
        l10d1 = np.log10(d1_)
        l10d2 = np.log10(d2_)
        return (a1 * l10d1**2 + a2 * l10d1 + a3) * l10d2 + (a4 * l10d1**2 + a5 * l10d1 + a6)

    def _dzls(n_: int, tb_: float, d1_: float, d2_: float) -> np.ndarray:
        di = _dinf_local(d1_, d2_)
        w = di / tb_
        f = np.array([0.0, (1.0 - w) * (tb_ / 2.0), (1.0 + w) * (tb_ / 2.0), n_ / 2.0], dtype=float)
        f = f / (n_ / 2.0)
        f = np.clip(f, 0.0, np.nextafter(1.0, 0.0))
        f[0] = 0.0
        f[1] = min(f[1], f[2])
        f[2] = max(f[2], f[1] + np.finfo(float).eps)

        m = np.array([1.0, 1.0, 0.0, 0.0], dtype=float)
        wt = np.array([1.0, d1_ / d2_], dtype=float)
        h = sp_signal.firls(n_ + 1, f, m, weight=wt, fs=2.0)

        k = np.concatenate((np.arange(0, n_ // 2 + 1, dtype=float), np.arange(-(n_ // 2), 0, dtype=float)))
        c = np.exp(1j * 2.0 * np.pi / (2.0 * (n_ + 1)) * k)
        h = np.real(np.fft.ifft(np.fft.fft(h) * c))
        return np.asarray(h[:n_], dtype=np.complex128)

    def _mag2mp(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.complex128).reshape(-1)
        n_ = int(x.size)
        xlf = np.fft.fft(np.log(np.abs(x)))
        xlfp = xlf.copy()
        xlfp[0] = xlf[0]
        xlfp[1 : n_ // 2] = 2.0 * xlf[1 : n_ // 2]
        xlfp[n_ // 2] = xlf[n_ // 2]
        xlfp[n_ // 2 + 1 :] = 0.0
        return np.exp(np.fft.ifft(xlfp))

    def _b2a(b: np.ndarray) -> np.ndarray:
        b = np.asarray(b, dtype=np.complex128).reshape(-1)
        n_ = int(b.size)
        npad = n_ * 16
        bcp = np.zeros((npad,), dtype=np.complex128)
        bcp[:n_] = b
        bf = np.fft.fft(bcp)
        bfmax = np.max(np.abs(bf))
        if bfmax >= 1.0:
            bf = bf / (1e-7 + bfmax)
        afa = _mag2mp(np.sqrt(1.0 - np.abs(bf) ** 2))
        a = np.fft.fft(afa) / npad
        return np.flip(a[:n_])

    def _ab2rf(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        a = np.asarray(a, dtype=np.complex128).reshape(-1)
        b = np.asarray(b, dtype=np.complex128).reshape(-1)
        n_ = int(a.size)
        rf_out = np.zeros((n_,), dtype=np.complex128)
        for idx in range(n_ - 1, -1, -1):
            cj = np.sqrt(1.0 / (1.0 + np.abs(b[idx] / a[idx]) ** 2))
            sj = np.conj(cj * b[idx] / a[idx])
            theta = np.arctan2(np.abs(sj), cj)
            psi = np.angle(sj)
            rf_out[idx] = 2.0 * theta * np.exp(1j * psi)
            if idx > 0:
                at = cj * a + sj * b
                bt = -np.conj(sj) * a + cj * b
                a = at[1 : idx + 1]
                b = bt[:idx]
        return rf_out

    def _b2rf(b: np.ndarray, cancel_alpha_phs_: bool) -> np.ndarray:
        a = _b2a(b)
        if cancel_alpha_phs_:
            b_a_phase = np.fft.fft(b) * np.exp(-1j * np.angle(np.fft.fft(np.flip(a))))
            b = np.fft.ifft(b_a_phase)
        return _ab2rf(a, b)

    ptype_l = str(ptype).strip().lower()
    bsf, d1m, d2m = _calc_ripples(ptype_l, float(d1), float(d2))
    b = _dzls(int(n), float(tb), float(d1m), float(d2m))
    if ptype_l == 'st':
        pulse = b
    elif ptype_l == 'ex':
        pulse = _b2rf(bsf * b, bool(cancel_alpha_phs))
    else:
        pulse = _b2rf(bsf * b, False)
    return np.asarray(pulse, dtype=np.complex128)


def sigpy_n_seq(
    flip_angle: float,
    delay: float = 0.0,
    duration: float = 4e-3,
    dwell: float = 0.0,
    freq_offset: float = 0.0,
    center_pos: float = 0.5,
    max_grad: float = 0.0,
    max_slew: float = 0.0,
    phase_offset: float = 0.0,
    return_gz: bool = True,
    return_delay: bool = False,
    slice_thickness: float = 0.0,
    system: Union[Opts, None] = None,
    time_bw_product: float = 4.0,
    pulse_cfg: Union[SigpyPulseOpts, None] = None,
    use: str = 'undefined',
    plot: bool = True,
    freq_ppm: float = 0.0,
    phase_ppm: float = 0.0,
) -> Union[SimpleNamespace, Tuple[SimpleNamespace, SimpleNamespace, SimpleNamespace]]:
    """
    Creates a radio-frequency sinc pulse event using the sigpy rf pulse library and optionally accompanying slice select, slice select rephasing
    trapezoidal gradient events.

    Parameters
    ----------
    flip_angle : float
        Flip angle in radians.
    delay : float, optional, default=0
        Delay in seconds (s).
    duration : float, optional, default=4e-3
        Duration in seconds (s).
    freq_offset : float, optional, default=0
        Frequency offset in Hertz (Hz).
    center_pos : float, optional, default=0.5
        Position of peak.5 (midway).
    max_grad : float, optional, default=0
        Maximum gradient strength of accompanying slice select trapezoidal event.
    max_slew : float, optional, default=0
        Maximum slew rate of accompanying slice select trapezoidal event.
    phase_offset : float, optional, default=0
        Phase offset in Hertz (Hz).
    return_gz:bool, default=False
        Boolean flag to indicate if slice-selective gradient has to be returned.
    slice_thickness : float, optional, default=0
        Slice thickness of accompanying slice select trapezoidal event. The slice thickness determines the area of the
        slice select event.
    system : Opts, optional
        System limits. Default is a system limits object initialized to default values.
    time_bw_product : float, optional, default=4
        Time-bandwidth product.
    pulse_cfg: SigpyPulseOpts, optional, default=None
        Pulse configuration options. Possible keys are:
        - pulse_type: str, optional, default='slr'
            Pulse type. Must be one of 'slr' or 'sms'.
        - ptype: str, optional, default='st'
            Pulse design method. Must be one of 'st', 'ex', 'inv', 'sat', 'se', 'fi', 'fs', 'se'.
        - ftype: str, optional, default='ls'
            Filter type. Must be one of 'ls', 'pm', 'min', 'max', 'ap'.
        - d1: float, optional, default=0.01
            Passband ripple.
        - d2: float, optional, default=0.01
            Stopband ripple.
        - cancel_alpha_phs: bool, optional, default=False
            Cancel alpha phase.
        - n_bands: int, optional, default=3
            Number of bands. SMS only.
        - band_sep: float, optional, default=20
            Band separation. SMS only.
        - phs_0_pt: str, optional, default='None'
            Phase 0 point. SMS only.
    use : str, default='undefined'
        Use of radio-frequency Shinnar-LeRoux pulse event.
        Must be one of 'excitation', 'refocusing', 'inversion',
        'saturation', 'preparation', 'other', 'undefined'.
    plot: bool, optional, default=True
        Show sigpy plot outputs
    freq_ppm : float, default=0
        PPM frequency offset.
    phase_ppm : float, default=0
        PPM phase offset.

    Returns
    -------
    rf : SimpleNamespace
        Radio-frequency Shinnar-LeRoux pulse event.
    gz : SimpleNamespace, optional
        Accompanying slice select trapezoidal gradient event. Returned only if `slice_thickness` is provided.
    gzr : SimpleNamespace, optional
        Accompanying slice select rephasing trapezoidal gradient event. Returned only if `slice_thickness` is provided.

    Raises
    ------
    ValueError
        If invalid `use` parameter was passed.
        If `return_gz=True` and `slice_thickness` was not provided.
    """
    if system is None:
        system = Opts.default

    if pulse_cfg is None:
        pulse_cfg = SigpyPulseOpts()

    valid_pulse_uses = get_supported_rf_uses()
    if use != '' and use not in valid_pulse_uses:
        raise ValueError(f'Invalid use parameter. Must be one of {valid_pulse_uses}. Passed: {use}')

    if use == 'excitation':
        if flip_angle <= np.pi / 6:
            pulse_cfg.ptype = 'st'
        else:
            pulse_cfg.ptype = 'ex'
    elif use == 'refocusing':
        pulse_cfg.ptype = 'se'
    elif use == 'inversion':
        pulse_cfg.ptype = 'inv'
    elif use == 'saturation':
        pulse_cfg.ptype = 'sat'
    else:
        pulse_cfg.ptype = 'st'

    if pulse_cfg.pulse_type == 'slr':
        [signal, t, _] = make_slr(
            flip_angle=flip_angle,
            time_bw_product=time_bw_product,
            duration=duration,
            dwell=dwell,
            system=system,
            pulse_cfg=pulse_cfg,
            disp=plot,
        )
    if pulse_cfg.pulse_type == 'sms':
        [signal, t, _] = make_sms(
            flip_angle=flip_angle,
            time_bw_product=time_bw_product,
            duration=duration,
            system=system,
            pulse_cfg=pulse_cfg,
            disp=plot,
        )

    rfp = SimpleNamespace()
    rfp.type = 'rf'
    rfp.signal = signal
    rfp.t = t
    if dwell == 0:
        dwell = system.rf_raster_time
    rfp.dwell = dwell
    rfp.shape_dur = len(t) * dwell
    rfp.freq_offset = freq_offset
    rfp.phase_offset = phase_offset
    rfp.freq_ppm = freq_ppm
    rfp.phase_ppm = phase_ppm
    rfp.dead_time = system.rf_dead_time
    rfp.ringdown_time = system.rf_ringdown_time
    rfp.delay = delay
    rfp.use = use
    rfp.center = calc_rf_center(rfp)[0]

    if rfp.dead_time > rfp.delay:
        warn(
            f'Specified RF delay {rfp.delay * 1e6:.2f} us is less than the dead time {rfp.dead_time * 1e6:.0f} us. Delay was increased to the dead time.',
            stacklevel=2,
        )
        rfp.delay = rfp.dead_time

    if return_gz:
        if slice_thickness == 0:
            raise ValueError('Slice thickness must be provided')

        if max_grad > 0:
            system = copy(system)
            system.max_grad = max_grad

        if max_slew > 0:
            system = copy(system)
            system.max_slew = max_slew
        bandwidth = time_bw_product / duration
        amplitude = bandwidth / slice_thickness
        area = amplitude * duration
        gz = make_trapezoid(channel='z', system=system, flat_time=duration, flat_area=area)
        gzr = make_trapezoid(
            channel='z',
            system=system,
            area=-area * (1 - rfp.center / rfp.shape_dur) - 0.5 * (gz.area - area),
        )

        if rfp.delay > gz.rise_time:
            gz.delay = math.ceil((rfp.delay - gz.rise_time) / system.grad_raster_time) * system.grad_raster_time

        if rfp.delay < (gz.rise_time + gz.delay):
            rfp.delay = gz.rise_time + gz.delay

    # Following 2 lines of code are workarounds for numpy returning 3.14... for np.angle(-0.00...)
    negative_zero_indices = np.where(rfp.signal == -0.0)
    rfp.signal[negative_zero_indices] = 0

    if trace_enabled():
        rfp.trace = trace()

    if return_gz:
        if return_delay:
            return rfp, gz, gzr, make_delay(calc_duration(rfp))
        return rfp, gz, gzr
    if return_delay:
        return rfp, make_delay(calc_duration(rfp))
    return rfp


def make_slr(
    flip_angle: float,
    time_bw_product: float = 4.0,
    duration: float = 0.0,
    dwell: float = 0.0,
    system: Union[Opts, None] = None,
    pulse_cfg: Union[SigpyPulseOpts, None] = None,
    disp: bool = False,
):
    if system is None:
        system = Opts.default

    if pulse_cfg is None:
        pulse_cfg = SigpyPulseOpts()

    if dwell == 0:
        dwell = system.rf_raster_time
    n_samples = round(duration / dwell)
    t = (np.arange(1, n_samples + 1) - 0.5) * dwell

    # Insert sigpy
    ptype = pulse_cfg.ptype
    ftype = pulse_cfg.ftype
    d1 = pulse_cfg.d1
    d2 = pulse_cfg.d2
    cancel_alpha_phs = pulse_cfg.cancel_alpha_phs

    if str(ftype).strip().lower() == 'ls':
        pulse = _dzrf_ls_matlab_like(
            n=n_samples,
            tb=time_bw_product,
            ptype=ptype,
            d1=d1,
            d2=d2,
            cancel_alpha_phs=cancel_alpha_phs,
        )
        # Original native SigPy call kept for easy rollback/reference:
        # pulse = rf.slr.dzrf(
        #     n=n_samples,
        #     tb=time_bw_product,
        #     ptype=ptype,
        #     ftype=ftype,
        #     d1=d1,
        #     d2=d2,
        #     cancel_alpha_phs=cancel_alpha_phs,
        # )
    else:
        pulse = rf.slr.dzrf(
            n=n_samples,
            tb=time_bw_product,
            ptype=ptype,
            ftype=ftype,
            d1=d1,
            d2=d2,
            cancel_alpha_phs=cancel_alpha_phs,
        )
    flip = np.abs(np.sum(pulse)) * dwell * 2 * np.pi
    signal = pulse * flip_angle / flip

    if disp:
        pl.LinePlot(pulse)
        pl.LinePlot(signal)

        # Simulate it
        [a, b] = rf.sim.abrm(
            pulse,
            np.arange(-20 * time_bw_product, 20 * time_bw_product, 40 * time_bw_product / 2000),
            True,
        )
        mag_xy = 2 * np.multiply(np.conj(a), b)
        pl.LinePlot(mag_xy)

    return signal, t, pulse


def make_sms(
    flip_angle: float,
    time_bw_product: float = 4.0,
    duration: float = 0.0,
    system: Union[Opts, None] = None,
    pulse_cfg: Union[SigpyPulseOpts, None] = None,
    disp: bool = False,
):
    if system is None:
        system = Opts.default

    if pulse_cfg is None:
        pulse_cfg = SigpyPulseOpts()

    n_samples = round(duration / system.rf_raster_time)
    t = (np.arange(1, n_samples + 1) - 0.5) * system.rf_raster_time

    # Insert sigpy
    ptype = pulse_cfg.ptype
    ftype = pulse_cfg.ftype
    d1 = pulse_cfg.d1
    d2 = pulse_cfg.d2
    cancel_alpha_phs = pulse_cfg.cancel_alpha_phs
    n_bands = pulse_cfg.n_bands
    band_sep = pulse_cfg.band_sep
    phs_0_pt = pulse_cfg.phs_0_pt

    if str(ftype).strip().lower() == 'ls':
        pulse_in = _dzrf_ls_matlab_like(
            n=n_samples,
            tb=time_bw_product,
            ptype=ptype,
            d1=d1,
            d2=d2,
            cancel_alpha_phs=cancel_alpha_phs,
        )
        # Original native SigPy call kept for easy rollback/reference:
        # pulse_in = rf.slr.dzrf(
        #     n=n_samples,
        #     tb=time_bw_product,
        #     ptype=ptype,
        #     ftype=ftype,
        #     d1=d1,
        #     d2=d2,
        #     cancel_alpha_phs=cancel_alpha_phs,
        # )
    else:
        pulse_in = rf.slr.dzrf(
            n=n_samples,
            tb=time_bw_product,
            ptype=ptype,
            ftype=ftype,
            d1=d1,
            d2=d2,
            cancel_alpha_phs=cancel_alpha_phs,
        )
    pulse = rf.multiband.mb_rf(pulse_in, n_bands=n_bands, band_sep=band_sep, phs_0_pt=phs_0_pt)

    flip = np.abs(np.sum(pulse)) * system.rf_raster_time * 2 * np.pi
    signal = pulse * flip_angle / flip

    if disp:
        pl.LinePlot(pulse_in)
        pl.LinePlot(pulse)
        pl.LinePlot(signal)
        # Simulate it
        [a, b] = rf.sim.abrm(
            pulse,
            np.arange(-20 * time_bw_product, 20 * time_bw_product, 40 * time_bw_product / 2000),
            True,
        )
        mag_xy = 2 * np.multiply(np.conj(a), b)
        pl.LinePlot(mag_xy)

    return signal, t, pulse
