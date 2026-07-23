from types import SimpleNamespace
from typing import Union

import numpy as np
import warnings

from pypulseq_matlab_like.opts import Opts
from pypulseq_matlab_like.utils.tracing import trace, trace_enabled


def make_arbitrary_grad(
    channel: str,
    waveform: np.ndarray,
    first: Union[float, None] = None,
    last: Union[float, None] = None,
    delay: float = 0.0,
    max_grad: Union[float, None] = None,
    max_slew: Union[float, None] = None,
    system: Union[Opts, None] = None,
    oversampling: bool = False,
) -> SimpleNamespace:
    """
    Creates a gradient event from an arbitrary waveform.

    Note that the sample points are assumed to be equally spaced by `system.grad_raster_time`
    and that the given waveform values are the values in the middle of each raster interval.

    The duration of the gradient is thus given by the number of samples times `system.grad_raster_time`.

    See also `pypulseq.Sequence.sequence.Sequence.add_block()`.

    Parameters
    ----------
    channel : str
        Orientation of gradient event of arbitrary shape. Must be one of `x`, `y` or `z`.
    waveform : numpy.ndarray
        Arbitrary waveform.
    first : float
        Gradient value at the start of the gradient event. (t=0)
        Will default to a linear extrapolated value if not provided.
    last : float
        Gradient value at the end of the gradient event. (t=duration)
        Will default to a linear extrapolated value if not provided.
    system : Opts, default=Opts()
        System limits.
        Will default to `pypulseq.opts.default` if not provided.
    max_grad : float
        Maximum gradient strength.
        Will default to `system.max_grad` if not provided.
    max_slew : float
        Maximum slew rate.
        Will default to `system.max_slew` if not provided.
    delay : float, default=0
        Delay in seconds (s).
    oversampling : bool, default=False
        Boolean flag to indicate if gradient is oversampled by a factor of 2.

    Returns
    -------
    grad : SimpleNamespace
        Gradient event with arbitrary waveform.

    Raises
    ------
    ValueError
        If invalid `channel` is passed. Must be one of x, y or z.
        If slew rate is violated.
        If gradient amplitude is violated.
    """
    if system is None:
        system = Opts.default

    if max_grad is None or max_grad <= 0:
        max_grad = system.max_grad

    if max_slew is None or max_slew <= 0:
        max_slew = system.max_slew

    channel = str(channel).lower()
    if channel not in ['x', 'y', 'z']:
        raise ValueError(f'Invalid channel. Must be one of x, y or z. Passed: {channel}')

    g = np.asarray(waveform).reshape(-1)

    def is_finite_scalar(x) -> bool:
        if x is None:
            return False
        xa = np.asarray(x)
        return xa.size == 1 and bool(np.isfinite(xa).reshape(-1)[0])

    if is_finite_scalar(first):
        first = float(np.asarray(first).reshape(-1)[0])
    else:
        warnings.warn(
            'it will be compulsory to provide the first point of the gradient shape in the future releases; '
            'finding the first by extrapolation for now...',
            stacklevel=2,
        )
        if oversampling:
            first = 2 * g[0] - g[1]
        else:
            first = 0.5 * (3 * g[0] - g[1])

    if is_finite_scalar(last):
        last = float(np.asarray(last).reshape(-1)[0])
    else:
        warnings.warn(
            'it will be compulsory to provide the last point of the gradient shape in the future releases; '
            'finding the last by extrapolation for now...',
            stacklevel=2,
        )
        if oversampling:
            last = 2 * g[-1] - g[-2]
        else:
            last = 0.5 * (3 * g[-1] - g[-2])

    # Slew rate calculation
    if oversampling:
        # [(first-g1), diff(g), (last-gN)] / dt * 2
        edge_scale = system.grad_raster_time * 0.5
        pre = first - g[0]
        post = last - g[-1]
    else:
        # [(first-g1)*2, diff(g), (gN-last)*2] / dt
        edge_scale = system.grad_raster_time
        pre = 2 * (first - g[0])
        post = 2 * (g[-1] - last)

    slew_rate = np.concatenate([[pre], np.diff(g), [post]]) / edge_scale

    slew_peak = float(np.max(np.abs(slew_rate)))
    grad_peak = float(np.max(np.abs(g)))
    if slew_peak > max_slew:
        raise ValueError(f'Slew rate violation ({slew_peak / max_slew * 100:.0f}%)')
    if grad_peak > max_grad:
        raise ValueError(f'Gradient amplitude violation ({grad_peak / max_grad * 100:.0f}%)')

    grad = SimpleNamespace()
    grad.type = 'grad'
    grad.channel = channel
    grad.waveform = g
    grad.delay = delay
    if oversampling:
        grad.area = (g[::2] * system.grad_raster_time).sum()
        if len(g) % 2 == 0:
            raise ValueError('when oversampling is active the gradient shape vector must contain an odd number of samples')
        grad.tt = np.arange(1, len(g) + 1) * 0.5 * system.grad_raster_time
        grad.shape_dur = (len(g) + 1) * 0.5 * system.grad_raster_time
    else:
        grad.area = (g * system.grad_raster_time).sum()
        grad.tt = (np.arange(len(g)) + 0.5) * system.grad_raster_time
        grad.shape_dur = len(g) * system.grad_raster_time
    grad.first = first
    grad.last = last

    if trace_enabled():
        grad.trace = trace()

    return grad
