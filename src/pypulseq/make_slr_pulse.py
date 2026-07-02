import numpy as np
from types import SimpleNamespace
from typing import Tuple, Union

from pypulseq.make_sigpy_pulse import sigpy_n_seq
from pypulseq.opts import Opts
from pypulseq.sigpy_pulse_opts import SigpyPulseOpts
from pypulseq.supported_labels_rf_use import get_supported_rf_uses


def make_slr_pulse(
    flip_angle: float,
    system: Union[Opts, None] = None,
    duration: float = 1e-3,
    freq_offset: float = 0.0,
    phase_offset: float = 0.0,
    freq_ppm: float = 0.0,
    phase_ppm: float = 0.0,
    time_bw_product: float = 4.0,
    passband_ripple: float = 0.01,
    stopband_ripple: float = 0.01,
    filter_type: str = 'ms',
    max_grad: float = 0.0,
    max_slew: float = 0.0,
    slice_thickness: float = 0.0,
    delay: float = 0.0,
    dwell: float = 0.0,
    use: str = 'excitation',
    return_gz: bool = False,
    recenter_on_sample: bool = False,
) -> Union[SimpleNamespace, Tuple[SimpleNamespace, SimpleNamespace, SimpleNamespace]]:
    """
    MATLAB-parity wrapper for `makeSLRpulse.m`.

    This function exposes a dedicated SLR-pulse constructor in PyPulseq while
    reusing the existing SigPy-backed implementation in `sigpy_n_seq()`.
    """
    if system is None:
        system = Opts.default

    valid_pulse_uses = get_supported_rf_uses()
    if use != '' and use not in valid_pulse_uses:
        raise ValueError(f'Invalid use parameter. Must be one of {valid_pulse_uses}. Passed: {use}')

    # Match MATLAB ptype selection logic from makeSLRpulse.m
    if use == 'excitation':
        ptype = 'st' if flip_angle <= np.pi / 6 else 'ex'
    elif use == 'refocusing':
        ptype = 'se'
    elif use == 'inversion':
        ptype = 'inv'
    elif use == 'saturation':
        ptype = 'sat'
    else:
        ptype = 'st'

    pulse_cfg = SigpyPulseOpts(
        pulse_type='slr',
        ptype=ptype,
        ftype=filter_type,
        d1=passband_ripple,
        d2=stopband_ripple,
        cancel_alpha_phs=False,
    )

    result = sigpy_n_seq(
        flip_angle=flip_angle,
        delay=delay,
        duration=duration,
        dwell=dwell,
        freq_offset=freq_offset,
        max_grad=max_grad,
        max_slew=max_slew,
        phase_offset=phase_offset,
        return_gz=return_gz,
        slice_thickness=slice_thickness,
        system=system,
        time_bw_product=time_bw_product,
        pulse_cfg=pulse_cfg,
        use=use,
        plot=False,
        freq_ppm=freq_ppm,
        phase_ppm=phase_ppm,
    )
    if recenter_on_sample:
        rf = result[0] if isinstance(result, tuple) else result
        center_index = int(np.argmin(np.abs(np.asarray(rf.t) - rf.center)))
        rf.center = float(rf.t[center_index])
    return result
