from types import SimpleNamespace
from typing import Any, List, Tuple

import numpy as np

from pypulseq import Sequence, eps
from pypulseq.calc_duration import calc_duration
from pypulseq.utils.tracing import format_trace

error_messages = {
    'RASTER': '{value*multiplier:.2f} {unit} does not align to {raster} (Nearest valid value: {value_rounded*multiplier:.0f} {unit}, error: {error*multiplier:.2f} {unit})',
    'ADC_DEAD_TIME': 'ADC delay is smaller than ADC dead time ({value*multiplier:.2f} {unit} < {dead_time*multiplier:.0f} {unit})',
    'POST_ADC_DEAD_TIME': 'Post-ADC dead time exceeds block duration ({value*multiplier:.2f} {unit} + {dead_time*multiplier:.0f} {unit} > {duration*multiplier} {unit})',
    'BLOCK_DURATION_MISMATCH': 'Inconsistency between the stored block duration ({duration*multiplier:.2f} {unit}) and the content of the block ({value*multiplier:.2f} {unit})',
    'RF_DEAD_TIME': 'Delay of {value*multiplier:.2f} {unit} is smaller than the RF dead time {dead_time*multiplier:.0f} {unit}',
    'RF_RINGDOWN_TIME': 'Time between the end of the RF pulse at {value*multiplier:.2f} {unit} and the end of the block at {duration * multiplier:.2f} {unit} is shorter than rf_ringdown_time ({ringdown_time*multiplier:.0f} {unit})',
    'NEGATIVE_DELAY': 'Delay is negative {value*multiplier:.2f} {unit}',
    'SOFT_DELAY_FACTOR': 'Soft delay {hint}/{numID} has factor parameter as zero, which makes duration calculation undefined.',
    'SOFT_DELAY_DUR_INCONSISTENCY': 'Soft delay {hint}/{numID} default duration derived from this block ({value*1e6} us) is inconsistent with the previous default.',
    'SOFT_DELAY_HINT_INCONSISTENCY': 'Soft delay {hint}/{numID}: Soft delays with the same numeric ID are expected to share the same text hint but previous hint recorded is {prev_hint}.',
    'SOFT_DELAY_INVALID_NUMID': 'Soft delay {hint}/{numID} has an invalid numeric ID {numID}. Numeric IDs must be positive integers.',
    'ADC_SAMPLES_DIVISOR': 'ADC num_samples is not an integer multiple of adc_samples_divisor ({value} / {divisor}).',
}


def check_timing(seq: Sequence) -> Tuple[bool, List[SimpleNamespace]]:
    error_report: List[SimpleNamespace] = []

    def div_check(a: float, b: float, event: str, field: str, raster: str):
        """
        Checks whether `a` can be divided by `b` to an accuracy of 1e-9.
        """
        c = a / b
        c_rounded = round(c)
        is_ok = abs(c - c_rounded) < 1e-9

        if not is_ok:
            error_report.append(
                SimpleNamespace(
                    block=block_counter,
                    event=event,
                    field=field,
                    value=a,
                    value_rounded=c_rounded * b,
                    error=(a - c_rounded * b),
                    raster=raster,
                    error_type='RASTER',
                )
            )

    soft_delay_defaults = {}
    soft_delay_hints_by_num = {}

    # Loop over all blocks
    for block_counter in seq.block_events:
        block = seq.get_block(block_counter)

        # Check block duration
        duration = calc_duration(block)
        div_check(
            duration, seq.system.block_duration_raster, event='block', field='duration', raster='block_duration_raster'
        )

        if abs(duration - seq.block_durations[block_counter]) > eps:
            error_report.append(
                SimpleNamespace(
                    block=block_counter,
                    event='block',
                    field='duration',
                    error_type='BLOCK_DURATION_MISMATCH',
                    value=duration,
                    duration=seq.block_durations[block_counter],
                )
            )
            duration = seq.block_durations[block_counter]

        # Check block events
        for event, e in block.__dict__.items():
            if e is None or isinstance(e, (float, int)):  # Special handling for block_duration
                continue
            if isinstance(e, list):
                if len(e) > 1:
                    # For now this is only the case for arrays of extensions, but we cannot actually check extensions anyway...
                    continue
                if len(e) == 0:
                    continue
                e = e[0]
            elif not isinstance(e, (dict, SimpleNamespace)):
                raise ValueError('Wrong data type of variable arguments, list[SimpleNamespace] expected.')

            if hasattr(e, 'type') and e.type in ('rf', 'adc', 'output'):
                raster = seq.system.rf_raster_time
                raster_str = 'rf_raster_time'
            else:
                raster = seq.system.grad_raster_time
                raster_str = 'grad_raster_time'

            if hasattr(e, 'delay'):
                if e.delay < -eps:
                    error_report.append(
                        SimpleNamespace(
                            block=block_counter, event=event, field='delay', error_type='NEGATIVE_DELAY', value=e.delay
                        )
                    )

                div_check(e.delay, raster, event=event, field='delay', raster=raster_str)

            if hasattr(e, 'duration'):
                div_check(e.duration, raster, event=event, field='duration', raster=raster_str)

            if hasattr(e, 'dwell'):
                if e.dwell < seq.system.adc_raster_time - eps:
                    error_report.append(
                        SimpleNamespace(
                            block=block_counter,
                            event=event,
                            field='dwell',
                            value=e.dwell,
                            value_rounded=seq.system.adc_raster_time,
                            error=e.dwell - seq.system.adc_raster_time,
                            raster='adc_raster_time',
                            error_type='RASTER',
                        )
                    )
                div_check(e.dwell, seq.system.adc_raster_time, event=event, field='dwell', raster='adc_raster_time')
                if hasattr(e, 'num_samples'):
                    div = getattr(seq.system, 'adc_samples_divisor', 1)
                    if div and abs(e.num_samples / div - round(e.num_samples / div)) > eps:
                        error_report.append(
                            SimpleNamespace(
                                block=block_counter,
                                event=event,
                                field='num_samples',
                                error_type='ADC_SAMPLES_DIVISOR',
                                value=e.num_samples,
                                divisor=div,
                            )
                        )

            if hasattr(e, 'type') and e.type == 'trap':
                div_check(
                    e.rise_time, seq.system.grad_raster_time, event=event, field='rise_time', raster='grad_raster_time'
                )
                div_check(
                    e.flat_time, seq.system.grad_raster_time, event=event, field='flat_time', raster='grad_raster_time'
                )
                div_check(
                    e.fall_time, seq.system.grad_raster_time, event=event, field='fall_time', raster='grad_raster_time'
                )
            if hasattr(e, 'type') and e.type == 'rf':
                if hasattr(e, 'shape_dur'):
                    div_check(e.shape_dur, seq.system.rf_raster_time, event=event, field='shape_dur', raster='rf_raster_time')

                if hasattr(e, 't') and len(e.t) >= 4:
                    rt = np.asarray(e.t) / seq.system.rf_raster_time
                    drt = np.diff(rt)
                    if np.all(np.abs(drt[1:] - drt[0]) < 1e-9 / seq.system.rf_raster_time):
                        dwell = e.t[1] - e.t[0]
                        div_check(
                            dwell,
                            min(seq.system.adc_raster_time, seq.system.rf_raster_time),
                            event=event,
                            field='dwell',
                            raster='min(adc_raster_time,rf_raster_time)',
                        )
                    elif np.any(np.abs(rt - np.round(rt)) > 1e-6):
                        # MATLAB flags this as invalid RF timing for extended-shape RF definitions.
                        max_err = np.max(np.abs(rt - np.round(rt))) * seq.system.rf_raster_time
                        error_report.append(
                            SimpleNamespace(
                                block=block_counter,
                                event=event,
                                field='t',
                                value=max_err,
                                value_rounded=0.0,
                                error=max_err,
                                raster='rf_raster_time',
                                error_type='RASTER',
                            )
                        )

        # Check RF dead times
        if block.rf is not None:
            if block.rf.delay - block.rf.dead_time < -eps:
                error_report.append(
                    SimpleNamespace(
                        block=block_counter,
                        event='rf',
                        field='delay',
                        error_type='RF_DEAD_TIME',
                        value=block.rf.delay,
                        dead_time=block.rf.dead_time,
                    )
                )

            if block.rf.delay + block.rf.t[-1] + block.rf.ringdown_time - duration > eps:
                error_report.append(
                    SimpleNamespace(
                        block=block_counter,
                        event='rf',
                        field='duration',
                        error_type='RF_RINGDOWN_TIME',
                        value=block.rf.delay + block.rf.t[-1],
                        duration=duration,
                        ringdown_time=block.rf.ringdown_time,
                    )
                )

        # Check ADC dead times
        if block.adc is not None:
            if block.adc.delay - seq.system.adc_dead_time < -eps:
                error_report.append(
                    SimpleNamespace(
                        block=block_counter,
                        event='adc',
                        field='delay',
                        error_type='ADC_DEAD_TIME',
                        value=block.adc.delay,
                        dead_time=seq.system.adc_dead_time,
                    )
                )

            adc_end = block.adc.delay + block.adc.num_samples * block.adc.dwell + seq.system.adc_dead_time

            if adc_end > duration + eps:
                error_report.append(
                    SimpleNamespace(
                        block=block_counter,
                        event='adc',
                        field='duration',
                        error_type='POST_ADC_DEAD_TIME',
                        value=block.adc.delay + block.adc.num_samples * block.adc.dwell,
                        duration=duration,
                        dead_time=seq.system.adc_dead_time,
                    )
                )
        soft_delay = getattr(block, 'soft_delay', None)
        if soft_delay is not None:
            num_id_raw = getattr(soft_delay, 'numID', None)
            try:
                num_id = int(float(num_id_raw))
            except (TypeError, ValueError):
                num_id = None

            if num_id is None or num_id <= 0:
                error_report.append(
                    SimpleNamespace(
                        block=block_counter,
                        event='soft_delay',
                        field='delay',
                        error_type='SOFT_DELAY_INVALID_NUMID',
                        value=num_id_raw,
                        hint=soft_delay.hint,
                        numID=num_id_raw,
                    )
                )
                continue

            if soft_delay.factor == 0:
                error_report.append(
                    SimpleNamespace(
                        block=block_counter,
                        event='soft_delay',
                        field='delay',
                        error_type='SOFT_DELAY_FACTOR',
                        value=soft_delay.factor,
                        hint=soft_delay.hint,
                        numID=num_id,
                    )
                )
            # Calculate default delay based on the current block duration
            default_delay = (seq.block_durations[block_counter] - float(soft_delay.offset)) * float(soft_delay.factor)
            if num_id not in soft_delay_defaults:
                soft_delay_defaults[num_id] = default_delay
            elif (
                abs(default_delay - soft_delay_defaults[num_id])
                > 1e-7  # 0.1 μs threshold for duration consistency
            ):
                error_report.append(
                    SimpleNamespace(
                        block=block_counter,
                        event='soft_delay',
                        field='delay',
                        error_type='SOFT_DELAY_DUR_INCONSISTENCY',
                        value=default_delay,
                        hint=soft_delay.hint,
                        numID=num_id,
                    )
                )

            if num_id not in soft_delay_hints_by_num:
                soft_delay_hints_by_num[num_id] = soft_delay.hint
            elif soft_delay_hints_by_num[num_id] != soft_delay.hint:
                error_report.append(
                    SimpleNamespace(
                        block=block_counter,
                        event='soft_delay',
                        field='delay',
                        error_type='SOFT_DELAY_HINT_INCONSISTENCY',
                        value=soft_delay.hint,
                        hint=soft_delay.hint,
                        prev_hint=soft_delay_hints_by_num[num_id],
                        numID=num_id,
                    )
                )

    return len(error_report) == 0, error_report


def format_string(template: str, **kwargs: Any) -> str:
    """
    Evaluate a formatted string using the f-string syntax. Similar to
    `'{x}'.format(x=1)`, but allows arbitrary computations, e.g.
    `'{x*y}'.format(x=2,y=2)`.

    Parameters
    ----------
    template : str
        Format string.
    **kwargs : Any
        Variables to use in the formatted string.

    Returns
    -------
    str
        Formatted string.
    """
    return eval(f'f"""{template}"""', kwargs)


def indent_string(x: str, n: int = 2) -> str:
    """
    Adds indentations (`n` spaces) to every line in a string
    """
    return '\n'.join(' ' * n + y for y in x.splitlines())


def print_error_report(
    seq: Sequence,
    error_report: List[SimpleNamespace],
    full_report: bool = False,
    max_errors: int = 10,
    colored: bool = True,
) -> None:
    current_block = None

    if full_report:
        max_errors = len(error_report)

    for e in error_report[:max_errors]:
        if e.block != current_block:
            print(f'Block {e.block}:')
            current_block = e.block
            trace = seq.block_trace.get(current_block, None)

            if hasattr(trace, 'block'):
                print(
                    ('\x1b[38;5;8m' if colored else '')
                    + 'Block created here:\n'
                    + format_trace(trace.block)
                    + ('\x1b[0m' if colored else '')
                )

        unit = 'us'
        multiplier = 1e6
        if e.field == 'dwell':
            unit = 'ns'
            multiplier = 1e9

        error_message = format_string(error_messages[e.error_type], **e.__dict__, unit=unit, multiplier=multiplier)
        print(
            f'- {e.event}.{e.field}: '
            + ('\x1b[38;5;9m' if colored else '')
            + error_message
            + ('\x1b[0m' if colored else '')
        )

        if hasattr(trace, e.event) and e.event != 'block':
            print(
                ('\x1b[38;5;8m' if colored else '')
                + f'  `{e.event}` created here:\n'
                + format_trace(getattr(trace, e.event), indent=2)
                + ('\x1b[0m' if colored else '')
            )

    if len(error_report) > max_errors:
        blocks = [e.block for e in error_report[max_errors:]]

        print(f'--- {len(error_report) - max_errors} more errors in blocks {min(blocks)} to {max(blocks)} hidden ---')
