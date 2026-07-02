import math
from types import SimpleNamespace
from typing import Union

from pypulseq import eps
from pypulseq.opts import Opts
from pypulseq.utils.tracing import trace, trace_enabled


def calculate_shortest_params_for_area(area: float, max_slew: float, max_grad: float, grad_raster_time: float):
    """Calculate the shortest possible rise_time, flat_time, and fall_time for a given area."""

    # Calculate initial rise time constrained by max slew rate
    rise_time = math.ceil(math.sqrt(abs(area) / max_slew) / grad_raster_time) * grad_raster_time
    rise_time = max(rise_time, grad_raster_time)

    # Calculate initial amplitude
    amplitude = area / rise_time
    effective_time = rise_time

    # Adjust for max gradient constraint
    if abs(amplitude) > max_grad + eps:
        effective_time = math.ceil(abs(area) / max_grad / grad_raster_time) * grad_raster_time
        amplitude = area / effective_time
        rise_time = math.ceil(abs(amplitude) / max_slew / grad_raster_time) * grad_raster_time
        rise_time = max(rise_time, grad_raster_time)

    # Calculate flat and fall times
    flat_time = effective_time - rise_time
    fall_time = rise_time

    return amplitude, rise_time, flat_time, fall_time


def make_trapezoid(
    channel: str,
    amplitude: Union[float, None] = None,
    area: Union[float, None] = None,
    delay: float = 0.0,
    duration: Union[float, None] = None,
    fall_time: Union[float, None] = None,
    flat_area: Union[float, None] = None,
    flat_time: Union[float, None] = None,
    max_grad: Union[float, None] = None,
    max_slew: Union[float, None] = None,
    rise_time: Union[float, None] = None,
    system: Union[Opts, None] = None,
) -> SimpleNamespace:
    """
    Create a trapezoidal gradient event.

    The user must supply any of the following sets of parameters:
    Area based:
    - area
    - area and duration
    - area and duration and rise_time
    - flat_time, area and rise_time
    Amplitude based:
    - amplitude and duration
    - amplitude and flat_time
    Flat area based:
    - flat_area and flat_time
    Additional options may be supplied with the above.

    See Also
    --------
    - `pypulseq.Sequence.sequence.Sequence.add_block()`
    - `pypulseq.opts.Opts`

    Parameters
    ----------
    channel : str
        Orientation of trapezoidal gradient event. Must be one of `x`, `y` or `z`.
    amplitude : float, default=None
        Peak amplitude (Hz/m).
    area : float, default=None
        Area (1/m).
    delay : float, default=0
        Delay in seconds (s).
    duration : float, default=None
        Duration in seconds (s). Duration is defined as rise_time + flat_time + fall_time.
    fall_time : float, default=None
        Fall time in seconds (s).
    flat_area : float, default=None
        Flat area (1/m).
    flat_time : float, default=None
        Flat duration in seconds (s). Default is -1 to allow for triangular pulses.
    max_grad : float, default=None
        Maximum gradient strength (Hz/m).
    max_slew : float, default=None
        Maximum slew rate (Hz/m/s).
    rise_time : float, default=0
        Rise time in seconds (s).
    system : Opts, default=Opts()
        System limits.

    Returns
    -------
    grad : SimpleNamespace
        Trapezoidal gradient event created based on the supplied parameters.

    Raises
    ------
    ValueError
        If none of `area`, `flat_area` and `amplitude` are passed
        If requested area is too large for this gradient
        If `flat_time`, `duration` and `area` are not supplied.
        Amplitude violation
    """
    if system is None:
        system = Opts.default

    if channel not in ['x', 'y', 'z']:
        raise ValueError(f'Invalid channel. Must be one of `x`, `y` or `z`. Passed: {channel}')

    if max_grad is None or max_grad <= 0:
        max_grad = system.max_grad

    if max_slew is None or max_slew <= 0:
        max_slew = system.max_slew

    if fall_time is not None and rise_time is None:
        raise ValueError("Must always supply `rise_time` if `fall_time` is specified explicitly.")
    if area is None and flat_area is None and amplitude is None:
        raise ValueError("Must supply either 'area', 'flat_area' or 'amplitude'.")
    if sum(x is None for x in (area, flat_area, amplitude)) != 2:
        raise ValueError(
            "Must supply either 'area', 'flat_area' or 'amplitude', and only one of the three may be specified."
        )

    if flat_time is not None:
        if amplitude is not None:
            amplitude2 = amplitude
        else:
            if flat_area is None:
                raise ValueError(
                    "When 'flat_time' is provided either 'flat_area' or 'amplitude' must be provided as well."
                )
            amplitude2 = flat_area / flat_time

        if rise_time is None:
            rise_time = abs(amplitude2) / max_slew
            rise_time = math.ceil(rise_time / system.grad_raster_time) * system.grad_raster_time
            if rise_time == 0:
                rise_time = system.grad_raster_time
        if fall_time is None:
            fall_time = rise_time

    elif duration is not None and duration > 0:
        if amplitude is not None:
            amplitude2 = amplitude
        else:
            if area is None:
                raise ValueError("Must supply area when duration is provided without amplitude.")
            if rise_time is None:
                d_c = 1 / abs(2 * max_slew) + 1 / abs(2 * max_slew)
                possible = duration**2 > 4 * abs(area) * d_c
                if not possible:
                    _, t1, t2, t3 = calculate_shortest_params_for_area(area, max_slew, max_grad, system.grad_raster_time)
                    raise ValueError(
                        'Requested area is too large for this gradient. '
                        f'Minimum required duration is {(t1 + t2 + t3) * 1e6} us'
                    )
                amplitude2 = (duration - math.sqrt(duration**2 - 4 * abs(area) * d_c)) / (2 * d_c)
            else:
                if fall_time is None:
                    fall_time = rise_time
                if duration <= 0.5 * rise_time + 0.5 * fall_time:
                    raise ValueError('The `duration` is too short for the given `rise_time`.')
                amplitude2 = area / (duration - 0.5 * rise_time - 0.5 * fall_time)
                possible = duration >= (rise_time + fall_time) and abs(amplitude2) < max_grad
                assert possible, (
                    'Requested area is too large for this gradient duration. '
                    f'Probably amplitude is violated ({round(abs(amplitude2) / max_grad * 100)}%)'
                )

        if rise_time is None:
            rise_time = math.ceil(abs(amplitude2) / max_slew / system.grad_raster_time) * system.grad_raster_time
            if rise_time == 0:
                rise_time = system.grad_raster_time
        if fall_time is None:
            fall_time = rise_time

        flat_time = duration - rise_time - fall_time
        if amplitude is None:
            amplitude2 = area / (rise_time / 2 + fall_time / 2 + flat_time)

    else:
        if area is None:
            raise ValueError('Must supply area or duration.')
        amplitude2, rise_time, flat_time, fall_time = calculate_shortest_params_for_area(
            area, max_slew, max_grad, system.grad_raster_time
        )

    if abs(amplitude2) > max_grad:
        if area is None:
            raise ValueError(f'Amplitude violation ({round(abs(amplitude2) / max_grad * 100)}%)')
        _, t1, t2, t3 = calculate_shortest_params_for_area(area, max_slew, max_grad, system.grad_raster_time)
        raise ValueError(
            'Requested duration is too short for the area to be realized within system limits. '
            f'Minimum duration is {(t1 + t2 + t3) * 1e6} us'
        )

    grad = SimpleNamespace()
    grad.type = 'trap'
    grad.channel = channel
    grad.amplitude = amplitude2
    grad.rise_time = rise_time
    grad.flat_time = flat_time
    grad.fall_time = fall_time
    grad.area = amplitude2 * (flat_time + rise_time / 2 + fall_time / 2)
    grad.flat_area = amplitude2 * flat_time
    grad.delay = delay
    grad.first = 0
    grad.last = 0

    if trace_enabled():
        grad.trace = trace()

    return grad
