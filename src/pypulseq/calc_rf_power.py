from types import SimpleNamespace
from typing import Tuple

import numpy as np

def calc_rf_power(rf: SimpleNamespace, dt: float = 1e-6) -> Tuple[float, float, float]:
    """
    Calculate the relative power of the RF pulse.

    Parameters
    ----------
    rf : SimpleNamespace
        RF pulse event.
    dt : float, default=1e-6
        Sampling rate for the calculation.

    Returns
    -------
    total_energy : float
        Total energy of the pulse (Hz^2 * s).
    peak_pwr : float
        Peak power (Hz^2).
    rf_rms : float
        RMS B1 amplitude (Hz).
    """
    if not all(hasattr(rf, attr) for attr in ['signal', 'shape_dur', 't']):
        raise ValueError("RF object must include 'signal', 'shape_dur', and 't' attributes.")

    nn = int(np.round(rf.shape_dur / dt))
    t = (np.arange(nn) + 0.5) * dt
    rfs = np.interp(t, rf.t, rf.signal, left=0, right=0)

    rfs_sq = rfs * np.conj(rfs)
    total_energy = float(np.sum(np.real(rfs_sq)) * dt)
    peak_pwr = float(np.max(np.real(rfs_sq)))
    rf_rms = float(np.sqrt(total_energy / rf.shape_dur))

    return total_energy, peak_pwr, rf_rms
