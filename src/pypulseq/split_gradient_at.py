from copy import deepcopy
from types import SimpleNamespace
from typing import Tuple, Union
from warnings import warn

import numpy as np

from pypulseq.make_extended_trapezoid import make_extended_trapezoid
from pypulseq.opts import Opts
from pypulseq.utils.tracing import trace, trace_enabled


def split_gradient_at(
    grad: SimpleNamespace, time_point: float, system: Union[Opts, None] = None
) -> Union[SimpleNamespace, Tuple[SimpleNamespace, SimpleNamespace]]:
    """
    Splits a trapezoidal gradient into two extended trapezoids defined by the cut line. Returns the two gradient parts
    by cutting the original 'grad' at 'time_point'. For the input type 'trapezoid' the results are returned as extended
    trapezoids, for 'arb' as arbitrary gradient objects. The delays in the individual gradient events are adapted such
    that add_gradients(...) produces a gradient equivalent to 'grad'.

    See Also
    --------
    - `pypulseq.split_gradient()`
    - `pypulseq.make_extended_trapezoid()`
    - `pypulseq.make_trapezoid()`
    - `pypulseq.Sequence.sequence.Sequence.add_block()`
    - `pypulseq.opts.Opts`

    Parameters
    ----------
    grad : SimpleNamespace
        Gradient event to be split into two gradient events.
    time_point : float
        Time point at which `grad` will be split into two gradient waveforms.
    system : Opts, default=Opts()
        System limits.

    Returns
    -------
    grad1, grad2 : SimpleNamespace
        Gradient waveforms after splitting.

    Raises
    ------
    ValueError
        If non-gradient event is passed.
    """
    if system is None:
        system = Opts.default

    if hasattr(grad, 'id'):
        raise ValueError(
            'split_gradient_at() was passed a gradient with an id field. Split gradients before registration or remove the id field.'
        )

    # copy() to emulate pass-by-value; otherwise passed grad is modified
    grad = deepcopy(grad)

    grad_raster_time = system.grad_raster_time

    # MATLAB parity: snap split point to gradient raster only
    time_index = round(time_point / grad_raster_time)
    if abs(time_point - time_index * grad_raster_time) > 1e-6:
        warn(
            'splitting the gradient at a point that is not on a gradient raster edge, substantial rounding is applied',
            stacklevel=2,
        )
    time_point = time_index * grad_raster_time
    channel = grad.channel

    if grad.type == 'grad':
        # Check if we have an arbitrary gradient or an extended trapezoid
        if abs(grad.tt[0] - 0.5 * grad_raster_time) < 1e-10:
            is_arb = np.all(abs(grad.tt[1:] - grad.tt[:-1] - grad_raster_time) < 1e-10)
            is_arb_os = np.all(abs(grad.tt[1:] - grad.tt[:-1] - grad_raster_time * 0.5) < 1e-10)
            if is_arb or is_arb_os:
                # MATLAB convention for this branch uses 1-based index
                time_index_mat = time_index + 1
                if is_arb_os:
                    time_index_mat = (time_index_mat - 1) * 2

                # If time point is out of range we have nothing to do
                if time_index_mat == 1 or time_index_mat >= len(grad.tt):
                    return grad

                grad1 = deepcopy(grad)
                grad2 = deepcopy(grad)
                if is_arb_os:
                    grad1.last = grad.waveform[time_index_mat - 1]
                else:
                    grad1.last = 0.5 * (grad.waveform[time_index_mat - 2] + grad.waveform[time_index_mat - 1])
                grad2.first = grad1.last
                grad2.delay = grad.delay + time_point
                grad1.tt = grad.tt[: time_index_mat - 1]
                grad1.waveform = grad.waveform[: time_index_mat - 1]
                if is_arb_os:
                    grad2.tt = grad.tt[time_index_mat:] - time_point
                    grad2.waveform = grad.waveform[time_index_mat:]
                else:
                    grad2.tt = grad.tt[time_index_mat - 1 :] - time_point
                    grad2.waveform = grad.waveform[time_index_mat - 1 :]

                grad1.shape_dur = grad1.tt[-1] - grad1.tt[0] + grad_raster_time
                grad2.shape_dur = grad2.tt[-1] - grad2.tt[0] + grad_raster_time

                if trace_enabled():
                    t = trace()
                    grad1.trace = t
                    grad2.trace = t

                return grad1, grad2

        # Extended trapezoid
        times = grad.tt
        amplitudes = grad.waveform
    elif grad.type == 'trap':
        grad.delay = round(grad.delay / grad_raster_time) * grad_raster_time
        grad.rise_time = round(grad.rise_time / grad_raster_time) * grad_raster_time
        grad.flat_time = round(grad.flat_time / grad_raster_time) * grad_raster_time
        grad.fall_time = round(grad.fall_time / grad_raster_time) * grad_raster_time

        # Prepare the extended trapezoid structure
        if grad.flat_time == 0:
            times = [0, grad.rise_time, grad.rise_time + grad.fall_time]
            amplitudes = [0, grad.amplitude, 0]
        else:
            times = [
                0,
                grad.rise_time,
                grad.rise_time + grad.flat_time,
                grad.rise_time + grad.flat_time + grad.fall_time,
            ]
            amplitudes = [0, grad.amplitude, grad.amplitude, 0]
    else:
        raise ValueError('Splitting of unsupported event.')

    # If the split line is behind the gradient, there is no second gradient to create
    if time_point >= grad.delay + times[-1]:
        raise ValueError('Splitting of gradient at time point after the end of gradient.')

    # If the split line goes through the delay
    if time_point < grad.delay:
        times = np.concatenate(([0], grad.delay + times))
        amplitudes = np.concatenate(([0], amplitudes))
        grad.delay = 0
    else:
        time_point -= grad.delay

    amplitudes = np.array(amplitudes)
    times = np.array(times)

    # Sample at time point
    amp_tp = np.interp(x=time_point, xp=times, fp=amplitudes)
    t_eps = 1e-10
    times1 = np.concatenate((times[np.where(times < time_point - t_eps)], [time_point]))
    amplitudes1 = np.concatenate((amplitudes[np.where(times < time_point - t_eps)], [amp_tp]))
    times2 = np.concatenate(([time_point], times[times > time_point + t_eps])) - time_point
    amplitudes2 = np.concatenate(([amp_tp], amplitudes[times > time_point + t_eps]))

    # Recreate gradients
    grad1 = make_extended_trapezoid(
        channel=channel,
        system=system,
        times=times1,
        amplitudes=amplitudes1,
        skip_check=True,
    )
    grad1.delay = grad.delay
    grad2 = make_extended_trapezoid(
        channel=channel,
        system=system,
        times=times2,
        amplitudes=amplitudes2,
        skip_check=True,
    )
    grad2.delay = time_point + grad.delay

    if trace_enabled():
        t = trace()
        grad1.trace = t
        grad2.trace = t

    return grad1, grad2
