from types import SimpleNamespace
from typing import Callable, Tuple, Union

import numpy as np

from pypulseq.make_extended_trapezoid import make_extended_trapezoid
from pypulseq.opts import Opts
from pypulseq.utils.tracing import trace, trace_enabled


def _to_raster(time_val: float, raster_time: float) -> float:
    return np.ceil(time_val / raster_time) * raster_time


def _calc_ramp_time(g1: float, g2: float, max_slew: float, raster_time: float) -> float:
    return _to_raster(abs(g1 - g2) / max_slew, raster_time)


def _binary_search(fun: Callable[[int], Union[None, Tuple[int, int, int, float]]], low: int, high: int):
    while low < high - 1:
        mid = (low + high) // 2
        if fun(mid) is not None:
            high = mid
        else:
            low = mid
    return fun(high)


def _find_solution(
    duration: int,
    area: float,
    grad_start: float,
    grad_end: float,
    max_slew: float,
    max_grad: float,
    raster_time: float,
):
    sign_area = np.sign(area) if area != 0 else 1.0
    grad_amp = sign_area * max_grad

    ru_min = abs(grad_amp - grad_start) / max_slew / raster_time
    rd_min = abs(grad_amp - grad_end) / max_slew / raster_time
    flat_time = max(duration - ru_min - rd_min, 0.0)

    approx_area = (
        ru_min * (grad_amp + grad_start) + rd_min * (grad_amp + grad_end) + 2.0 * flat_time * grad_amp
    )
    if abs(2.0 * area / raster_time) > abs(approx_area):
        return None

    ru = (duration * max_slew * raster_time + sign_area * (grad_end - grad_start)) / (2.0 * max_slew * raster_time)
    if sign_area * grad_start + ru * max_slew * raster_time > max_grad + 1e-5:
        ru_steps = int(round(abs(grad_start - sign_area * max_grad) / max_slew / raster_time))
        rd_steps = int(round(abs(grad_end - sign_area * max_grad) / max_slew / raster_time))
        flat_steps = int(duration - ru_steps - rd_steps)
        if flat_steps > 0:
            grad_amp = -(
                ru_steps * raster_time * grad_start + rd_steps * raster_time * grad_end - 2.0 * area
            ) / ((ru_steps + 2 * flat_steps + rd_steps) * raster_time)
            amps = np.array([grad_start, grad_amp, grad_amp, grad_end], dtype=float)
            t = np.cumsum([0, ru_steps, flat_steps, rd_steps], dtype=float) * raster_time
            slew = np.diff(amps) / np.diff(t)
            if np.max(np.abs(slew)) < max_slew + 1e-5 and np.max(np.abs(amps)) < max_grad:
                return ru_steps, flat_steps, rd_steps, float(grad_amp)

    while abs(2.0 * area / raster_time) < abs(approx_area):
        grad_amp = grad_amp / 2.0
        if abs(grad_amp) < abs(max_grad) / 10.0:
            ru_min = 0.0
            rd_min = 0.0
            flat_time = max(duration - ru_min - rd_min, 0.0)
            break
        ru_min = abs(grad_amp - grad_start) / max_slew / raster_time
        rd_min = abs(grad_amp - grad_end) / max_slew / raster_time
        flat_time = max(duration - ru_min - rd_min, 0.0)
        approx_area = (
            ru_min * (grad_amp + grad_start) + rd_min * (grad_amp + grad_end) + 2.0 * flat_time * grad_amp
        )

    ru_min = int(np.floor(ru_min))
    rd_min = int(np.floor(rd_min))
    ru_limit = int(np.ceil(abs(sign_area * max_grad - grad_start) / max_slew / raster_time) + 1)
    rd_limit = int(np.ceil(abs(sign_area * max_grad - grad_end) / max_slew / raster_time) + 1)

    ru_grid, rd_grid = np.meshgrid(np.arange(ru_min, ru_limit + 1), np.arange(rd_min, rd_limit + 1))
    ru = ru_grid.ravel()
    rd = rd_grid.ravel()
    valid_mask = rd < (duration - ru)
    ru = ru[valid_mask]
    rd = rd[valid_mask]

    num = (2.0 * area - duration * (grad_end + grad_start) * raster_time)
    denom_min = (grad_start - grad_end + sign_area * duration * max_slew * raster_time)
    denom_max = (-grad_start + grad_end + sign_area * duration * max_slew * raster_time)
    if denom_min != 0 and denom_max != 0:
        ru_flat0_min = int(round(num / denom_min / raster_time))
        ru_flat0_max = int(duration - round(num / denom_max / raster_time))
        if ru_flat0_max >= ru_flat0_min:
            ext = np.arange(ru_flat0_min, ru_flat0_max + 1, dtype=int)
            ru = np.concatenate((ru, ext))
            rd = np.concatenate((rd, duration - ext))

    flat = duration - ru - rd
    valid = (flat >= 0) & (ru > 0) & (rd > 0)
    ru = ru[valid]
    rd = rd[valid]
    flat = flat[valid]
    if ru.size == 0:
        return None

    grad_amp = -(
        ru * raster_time * grad_start + rd * raster_time * grad_end - 2.0 * area
    ) / ((ru + 2 * flat + rd) * raster_time)

    slew1 = np.abs(grad_start - grad_amp) / (ru * raster_time)
    slew2 = np.abs(grad_end - grad_amp) / (rd * raster_time)
    valid = (np.abs(grad_amp) <= max_grad + 1e-5) & (slew1 <= max_slew + 1e-5) & (slew2 <= max_slew + 1e-5)
    if not np.any(valid):
        return None

    idx = np.where(valid)[0]
    best = idx[np.argmin(slew1[idx] + slew2[idx])]
    return int(ru[best]), int(flat[best]), int(rd[best]), float(grad_amp[best])


def make_extended_trapezoid_area(
    area: float,
    channel: str,
    grad_start: float,
    grad_end: float,
    convert_to_arbitrary: bool = False,
    system: Union[Opts, None] = None,
) -> Tuple[SimpleNamespace, np.ndarray, np.ndarray]:
    if system is None:
        system = Opts.default

    max_slew = system.max_slew * 0.99
    max_grad = system.max_grad * 0.99
    raster_time = system.grad_raster_time

    min_duration = max(int(round(_calc_ramp_time(grad_end, grad_start, max_slew, raster_time) / raster_time)), 2)
    max_duration = max(
        int(round(_calc_ramp_time(0.0, grad_start, max_slew, raster_time) / raster_time)),
        int(round(_calc_ramp_time(0.0, grad_end, max_slew, raster_time) / raster_time)),
        min_duration,
    )

    solution = None
    for duration in range(min_duration, max_duration + 1):
        solution = _find_solution(duration, area, grad_start, grad_end, max_slew, max_grad, raster_time)
        if solution is not None:
            break

    if solution is None:
        duration = max_duration
        while solution is None:
            duration *= 2
            solution = _find_solution(duration, area, grad_start, grad_end, max_slew, max_grad, raster_time)
        solution = _binary_search(
            lambda d: _find_solution(d, area, grad_start, grad_end, max_slew, max_grad, raster_time),
            duration // 2,
            duration,
        )

    ru_steps, flat_steps, rd_steps, grad_amp = solution
    time_ramp_up = ru_steps * raster_time
    flat_time = flat_steps * raster_time
    time_ramp_down = rd_steps * raster_time

    if flat_time > 0:
        times = np.cumsum([0.0, time_ramp_up, flat_time, time_ramp_down])
        amplitudes = np.array([grad_start, grad_amp, grad_amp, grad_end], dtype=float)
    else:
        times = np.cumsum([0.0, time_ramp_up, time_ramp_down])
        amplitudes = np.array([grad_start, grad_amp, grad_end], dtype=float)

    grad = make_extended_trapezoid(
        channel=channel,
        system=system,
        times=times,
        amplitudes=amplitudes,
        convert_to_arbitrary=convert_to_arbitrary,
    )
    if abs(grad.area - area) >= 1e-3:
        raise ValueError(f'Could not find a solution for area={area:.6f}.')

    if trace_enabled():
        grad.trace = trace()

    return grad, times, amplitudes
