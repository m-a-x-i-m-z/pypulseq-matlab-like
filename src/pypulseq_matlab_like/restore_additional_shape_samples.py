from typing import Optional, Tuple
from warnings import warn

import numpy as np


def restore_additional_shape_samples(
    tt: np.ndarray,
    waveform: np.ndarray,
    first: float,
    last: float,
    grad_raster_time: float,
    i_block: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Restore edge samples for compressed gradient shapes.

    MATLAB counterpart: `mr.restoreAdditionalShapeSamples`.
    """
    tt = np.asarray(tt, dtype=float).reshape(-1)
    waveform = np.asarray(waveform, dtype=float).reshape(-1)

    max_abs = float(np.max(np.abs(waveform))) if waveform.size > 0 else 0.0
    odd_step1 = np.concatenate(([first], 2.0 * waveform))
    odd_sign = np.where((np.arange(odd_step1.size) % 2) == 0, 1.0, -1.0)
    waveform_odd_rest = np.cumsum(odd_step1 * odd_sign) * odd_sign
    waveform_odd_interp = np.concatenate(([first], 0.5 * (waveform[:-1] + waveform[1:]), [last]))

    threshold = 2e-5 * max_abs
    if abs(waveform_odd_rest[-1] - last) > threshold:
        block_info = f'[block {i_block}] ' if i_block is not None else ''
        deviation = abs(waveform_odd_rest[-1] - last)
        warn(
            block_info
            + 'Last restored point differs too much from the recorded last; '
            + f'skipping shape restoration. Deviation: {deviation} Hz/m.',
            stacklevel=2,
        )
        tt_chg = np.concatenate(([0.0], tt, [tt[-1] + grad_raster_time / 2.0]))
        waveform_chg = np.concatenate(([first], waveform, [last]))
        return tt_chg, waveform_chg

    waveform_odd_mask = np.abs(waveform_odd_rest - waveform_odd_interp) <= np.finfo(float).eps + threshold
    waveform_odd = waveform_odd_interp * waveform_odd_mask + waveform_odd_rest * (~waveform_odd_mask)

    comb = np.vstack((np.concatenate(([0.0], waveform)), waveform_odd))
    waveform_os = comb.T.reshape(-1)[1:]
    tt_os = np.arange(waveform_os.size, dtype=float) * grad_raster_time * 0.5

    mask_changes = np.abs(np.concatenate(([1.0], np.diff(waveform_os, n=2), [1.0]))) > 1e-8
    waveform_chg = waveform_os[mask_changes]
    tt_chg = tt_os[mask_changes]
    return tt_chg, waveform_chg
