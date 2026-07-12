from types import SimpleNamespace
from typing import Tuple

import numpy as np


def calc_rf_center(
    rf: SimpleNamespace, return_fractional_index: bool = False
) -> Tuple[float, int] | Tuple[float, int, float]:
    """Calculate the RF center using zero-based indexing."""
    if hasattr(rf, 'center'):
        time_center = rf.center
        id_center = np.argmin(abs(rf.t - time_center)).item()
    else:
        rf_max = np.max(np.abs(rf.signal))
        i_peak = np.where(np.abs(rf.signal) >= rf_max * 0.99999)[0]
        time_center = (rf.t[i_peak[0]] + rf.t[i_peak[-1]]) / 2
        id_center = i_peak[int(np.ceil(len(i_peak) / 2)) - 1].item()

    if not return_fractional_index:
        return time_center, id_center

    fractional_time = time_center - rf.t[id_center]
    if id_center < len(rf.t) - 1 and fractional_time > 1e-9:
        fractional_index = fractional_time / (rf.t[id_center + 1] - rf.t[id_center])
    elif id_center > 0 and fractional_time < -1e-9:
        fractional_index = fractional_time / (rf.t[id_center] - rf.t[id_center - 1])
    else:
        fractional_index = 0.0
    return time_center, id_center, fractional_index