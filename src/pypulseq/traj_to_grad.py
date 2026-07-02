from typing import Tuple, Union

import numpy as np

from pypulseq.opts import Opts


def traj_to_grad(
    k: np.ndarray,
    raster_time: Union[float, None] = None,
    first: Union[float, np.ndarray, None] = None,
    first_grad_step_half_raster: bool = True,
    conservative_slew_estimate: bool = False,
    system: Union[Opts, None] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert k-space trajectory `k` into gradient waveform in compliance with `raster_time` gradient raster time.

    Parameters
    ----------
    k : numpy.ndarray
        K-space trajectory to be converted into gradient waveform.
    raster_time : float, default=Opts().grad_raster_time
        Gradient raster time.
    first : float or numpy.ndarray, optional
        Initial gradient value. Default is 0.
    first_grad_step_half_raster : bool, default=True
        Whether the first gradient step is half of the raster time.
    conservative_slew_estimate : bool, default=False
        Whether to use conservative slew rate estimation (taking max absolute of neighbors).
    system : Opts, optional
        System limits.

    Returns
    -------
    g : numpy.ndarray
        Gradient waveform.
    sr : numpy.ndarray
        Slew rate.
    """
    if system is None:
        system = Opts.default

    if raster_time is None:
        raster_time = system.grad_raster_time

    if first is None:
        first = np.zeros(k.shape[0])
    if np.isscalar(first):
        first = np.full(k.shape[0], first)

    # Compute finite difference for gradients in Hz/m along the last axis.
    g = (k[..., 1:] - k[..., :-1]) / raster_time

    # Compute the slew rate
    first = np.asarray(first, dtype=g.dtype)
    first_shape = [1] * g.ndim
    first_shape[0] = k.shape[0]
    first = first.reshape(first_shape)
    first = np.broadcast_to(first, g.shape[:-1] + (1,))

    g_with_first = np.concatenate((first, g), axis=-1)
    sr0 = (g_with_first[..., 1:] - g_with_first[..., :-1]) / raster_time

    if first_grad_step_half_raster:
        sr0[..., 0] *= 2

    sr = np.zeros(sr0.shape)
    sr[..., 0] = sr0[..., 0]

    if conservative_slew_estimate:
        if first_grad_step_half_raster:
            sr[..., 1] = sr0[..., 1]
            if sr0.shape[-1] > 2:
                sr[..., 2:] = np.where(
                    np.abs(sr0[..., 1:-1]) >= np.abs(sr0[..., 2:]), sr0[..., 1:-1], sr0[..., 2:]
                )
        else:
            if sr0.shape[-1] > 1:
                sr[..., 1:] = np.where(
                    np.abs(sr0[..., :-1]) >= np.abs(sr0[..., 1:]), sr0[..., :-1], sr0[..., 1:]
                )
    else:
        if first_grad_step_half_raster:
            sr[..., 1] = sr0[..., 1]
            if sr0.shape[-1] > 2:
                sr[..., 2:] = 0.5 * (sr0[..., 1:-1] + sr0[..., 2:])
        else:
            if sr0.shape[-1] > 1:
                sr[..., 1:] = 0.5 * (sr0[..., :-1] + sr0[..., 1:])

    return g, sr
