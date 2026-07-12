import math
import numbers
from types import SimpleNamespace
from typing import List, Tuple, Union
from warnings import warn

import numpy as np
from scipy.spatial.transform import Rotation

from pypulseq import eps
from pypulseq.block_to_events import block_to_events
from pypulseq.compress_shape import compress_shape
from pypulseq.decompress_shape import decompress_shape
from pypulseq.event_lib import EventLibrary
from pypulseq.supported_labels_rf_use import get_supported_labels
from pypulseq.utils.tracing import trace_enabled


def set_block(self, block_index: int, *args: Union[SimpleNamespace, float]) -> None:
    """
    Replace block at index with new block provided as block structure, add sequence block, or create a new block
    from events and store at position specified by index. The block or events are provided in uncompressed form and
    will be stored in the compressed, non-redundant internal libraries.

    See Also
    --------
    - `pypulseq.Sequence.sequence.Sequence.get_block()`
    - `pypulseq.Sequence.sequence.Sequence.add_block()`

    Parameters
    ----------
    block_index : int
        Index at which block is replaced.
    args : SimpleNamespace
        Block or events to be replaced/added or created at `block_index`.
        If a floating point number is provided, it is interpreted as the duration of the block.

    Raises
    ------
    ValueError
        If trigger event that is passed is of unsupported control event type.
        If delay is set for a gradient even that starts with a non-zero amplitude.
    RuntimeError
        If two consecutive gradients to not have the same amplitude at the connection point.
        If the first gradient in the block does not start with 0.
        If a gradient that doesn't end at zero is not aligned to the block boundary.
        If multiple soft_delay extensions are used in a block.
        If a soft delay extension is used in a block of zero duration.
        If a soft delay extension is used in a block containing conventional events.
    """
    events = block_to_events(*args)
    new_block = np.zeros(7, dtype=np.int32)
    duration = 0.0
    required_duration = None
    round_up_block_duration = False
    rot_quaternion = None

    check_g = [None, None, None]
    extensions = []
    sequence_id = id(self)

    def get_cached_registration(event: SimpleNamespace, cache_key=None):
        cache = getattr(event, '_pypulseq_sequence_event_cache', None)
        if cache is None:
            return None

        seq_cache = cache.get(sequence_id)
        if seq_cache is None:
            return None

        key = cache_key if cache_key is not None else ('__default__',)
        return seq_cache.get(key)

    def set_cached_registration(event: SimpleNamespace, cache_key=None, **values) -> None:
        cache = getattr(event, '_pypulseq_sequence_event_cache', None)
        if cache is None:
            cache = {}
            setattr(event, '_pypulseq_sequence_event_cache', cache)
        seq_cache = cache.setdefault(sequence_id, {})
        key = cache_key if cache_key is not None else ('__default__',)
        seq_cache[key] = values

    def cache_value(value):
        if isinstance(value, np.ndarray):
            value = np.ascontiguousarray(value)
            return (str(value.dtype), value.shape, value.tobytes())
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, float):
            return ('float', value.hex())
        if isinstance(value, (list, tuple)):
            return tuple(cache_value(item) for item in value)
        return value

    def event_cache_key(event: SimpleNamespace, *field_names):
        return tuple((field_name, cache_value(getattr(event, field_name, None))) for field_name in field_names)

    for event in events:
        if isinstance(event, str):
            if event == 'roundUpBlockDuration':
                round_up_block_duration = True
                continue
            warn(f'Unknown parameter passed to block {block_index}', stacklevel=2)
            continue

        if isinstance(event, numbers.Real):
            numeric_value = float(event)
            if required_duration is None:
                required_duration = numeric_value
            else:
                raise ValueError('More than one numeric parameter given to set_block()')
            duration = max(duration, numeric_value)
            continue

        if event.type == 'rf':
            if new_block[1] != 0:
                raise ValueError('Multiple RF events were specified in set_block')

            cache_key = event_cache_key(
                event,
                'signal',
                't',
                'center',
                'delay',
                'freq_ppm',
                'phase_ppm',
                'freq_offset',
                'phase_offset',
                'use',
            )
            cache = get_cached_registration(event, cache_key)
            if cache is not None and 'id' in cache:
                rf_id = cache['id']
            else:
                rf_id, shape_IDs = register_rf_event(self, event)
                set_cached_registration(event, cache_key, id=rf_id, shape_IDs=shape_IDs)

            new_block[1] = rf_id
            duration = max(duration, event.shape_dur + event.delay + event.ringdown_time)

            if trace_enabled() and hasattr(event, 'trace'):
                self.block_trace[block_index].rf = event.trace
        elif event.type == 'grad':
            channel_num = ['x', 'y', 'z'].index(event.channel)
            idx = 2 + channel_num

            if new_block[idx] != 0:
                raise ValueError(
                    f'Trying to add more than one gradient per axis on axis {event.channel} in block {block_index}'
                )

            grad_start = event.delay + math.floor(event.tt[0] / self.grad_raster_time + 1e-10) * self.grad_raster_time
            grad_duration = event.delay + math.ceil(event.tt[-1] / self.grad_raster_time - 1e-10) * self.grad_raster_time

            check_g[channel_num] = SimpleNamespace(idx=idx, start=(grad_start, event.first), stop=(grad_duration, event.last))

            cache_key = event_cache_key(
                event,
                'waveform',
                'tt',
                'first',
                'last',
                'delay',
            )
            cache = get_cached_registration(event, cache_key)
            if cache is not None and 'id' in cache:
                grad_id = cache['id']
            else:
                grad_id, shape_IDs = register_grad_event(self, event)
                set_cached_registration(event, cache_key, id=grad_id, shape_IDs=shape_IDs)

            new_block[idx] = grad_id
            duration = max(duration, grad_duration)

            if trace_enabled() and hasattr(event, 'trace'):
                setattr(self.block_trace[block_index], 'g' + event.channel, event.trace)
        elif event.type == 'trap':
            channel_num = ['x', 'y', 'z'].index(event.channel)
            idx = 2 + channel_num

            if new_block[idx] != 0:
                raise ValueError(
                    f'Trying to add more than one gradient per axis on axis {event.channel} in block {block_index}'
                )

            cache_key = event_cache_key(
                event,
                'amplitude',
                'rise_time',
                'flat_time',
                'fall_time',
                'delay',
            )
            cache = get_cached_registration(event, cache_key)
            if cache is not None and 'id' in cache:
                trap_id = cache['id']
            else:
                trap_id = register_grad_event(self, event)
                set_cached_registration(event, cache_key, id=trap_id)

            new_block[idx] = trap_id
            duration = max(duration, event.delay + event.rise_time + event.flat_time + event.fall_time)

            if trace_enabled() and hasattr(event, 'trace'):
                setattr(self.block_trace[block_index], 'g' + event.channel, event.trace)
        elif event.type == 'adc':
            if new_block[5] != 0:
                raise ValueError('Multiple ADC events were specified in set_block')

            cache_key = event_cache_key(
                event,
                'num_samples',
                'dwell',
                'delay',
                'dead_time',
                'freq_ppm',
                'phase_ppm',
                'freq_offset',
                'phase_offset',
                'phase_modulation',
            )
            cache = get_cached_registration(event, cache_key)
            if cache is not None and 'id' in cache:
                adc_id = cache['id']
            else:
                adc_id, shape_id = register_adc_event(self, event)
                set_cached_registration(event, cache_key, id=adc_id, shape_id=shape_id)

            new_block[5] = adc_id
            duration = max(duration, event.delay + event.num_samples * event.dwell + event.dead_time)

            if trace_enabled() and hasattr(event, 'trace'):
                self.block_trace[block_index].adc = event.trace
        elif event.type == 'delay':
            # MATLAB behavior in v1.5.x: delay contributes only to block duration,
            # it is not stored as a standalone delay event in new_block(1).
            duration = max(duration, event.delay)
        elif event.type in ['output', 'trigger']:
            if hasattr(event, 'id'):
                event_id = event.id
            else:
                event_id = register_control_event(self, event)

            ext = {'type': self.get_extension_type_ID('TRIGGERS'), 'ref': event_id}
            extensions.append(ext)
            duration = max(duration, event.delay + event.duration)
        elif event.type in ['labelset', 'labelinc']:
            if hasattr(event, 'id'):
                label_id = event.id
            else:
                label_id = register_label_event(self, event)

            ext = {
                'type': self.get_extension_type_ID(event.type.upper()),
                'ref': label_id,
            }
            extensions.append(ext)
        elif event.type == 'soft_delay':
            if hasattr(event, 'id'):
                event_id = event.id
            else:
                event_id = register_soft_delay_event(self, event)

            duration = max(duration, event.default_duration)
            ext = {'type': self.get_extension_type_ID('DELAYS'), 'ref': event_id}
            extensions.append(ext)
        elif event.type == 'rot3D':
            if rot_quaternion is not None:
                raise ValueError("Only one 'rotation' extension event can be added per block")
            rot_quaternion = np.asarray(event.rot_quaternion, dtype=float).flatten()
            if hasattr(event, 'id'):
                event_id = event.id
            else:
                cache = get_cached_registration(event)
                if cache is not None and 'id' in cache:
                    event_id = cache['id']
                else:
                    event_id = self.register_rotation_event(event)
                    set_cached_registration(event, id=event_id)

            ext = {'type': self.get_extension_type_ID('ROTATIONS'), 'ref': event_id}
            extensions.append(ext)
        elif event.type == 'rf_shim':
            if hasattr(event, 'id'):
                event_id = event.id
            else:
                event_id = self.register_rf_shim_event(event)

            ext = {'type': self.get_extension_type_ID('RF_SHIMS'), 'ref': event_id}
            extensions.append(ext)
        else:
            raise ValueError(f'Unknown event type {event.type} passed to set_block().')

    # =========
    # ADD EXTENSIONS
    # =========
    if len(extensions) > 0:
        """
        Add extensions now... but it's tricky actually we need to check whether the exactly the same list of extensions
        already exists, otherwise we have to create a new one... ooops, we have a potential problem with the key
        mapping then... The trick is that we rely on the sorting of the extension IDs and then we can always find the
        last one in the list by setting the reference to the next to 0 and then proceed with the other elements.
        """
        # Build chain by sorting by label reference ID (ascending) and iterating FORWARDS.
        # This matches MATLAB behavior where smaller label IDs (like LIN) are at the tail.
        extensions = sorted(extensions, key=lambda e: e['ref'])

        extension_id = 0
        for i in range(len(extensions)):
            data = (extensions[i]['type'], extensions[i]['ref'], extension_id)
            extension_id, _ = self.extensions_library.find_or_insert(new_data=data)

        # Sanity checks for the soft delays
        # Match MATLAB behavior: unconditionally register 'DELAYS' type ID when any extension is present
        # (MATLAB setBlock calls getExtensionTypeID('DELAYS') unconditionally at this point)
        n_soft_delays = sum([1 for e in extensions if e['type'] == self.get_extension_type_ID('DELAYS')])
        if n_soft_delays:
            if n_soft_delays > 1:
                raise RuntimeError('Only one soft delay extension is allowed per block.')
            if duration == 0 and required_duration is None:
                raise RuntimeError(
                    'Soft delay extension can only be used in conjunction with blocks of non-zero duration.'
                )  # otherwise the gradient checks get tedious
            if new_block[1:6].any():
                raise RuntimeError(
                    'Soft delay extension can only be used in empty blocks (blocks containing no conventional events such as RF, adc or gradients).'
                )
        # Now we add the ID
        new_block[6] = extension_id

    # =========
    # PERFORM GRADIENT CHECKS
    # =========
    if duration > 0:
        if round_up_block_duration:
            duration = math.ceil(duration / self.system.block_duration_raster) * self.system.block_duration_raster

        grad_check_data = self.grad_check_data
        slew_eps = self.system.max_slew * self.system.grad_raster_time
        next_block_index = None
        # Performance: in the common append path (add_block), block_index is not
        # in block_events yet, so there is no "next block" to validate against.
        # Avoid building/scanning a key list on every append.
        if block_index in self.block_events:
            blocks = list(self.block_events)
            idx = blocks.index(block_index)
            next_block_index = blocks[idx + 1] if idx < len(blocks) - 1 else None

        if block_index > 1 and grad_check_data['validForBlockNum'] != block_index - 1:
            grad_check_data['validForBlockNum'] = block_index - 1
            grad_check_data['lastGradVals'][:] = 0

            prev_nonempty_block = None
            for prev_idx, prev_dur in self.block_durations.items():
                if prev_idx < block_index and prev_dur > 0:
                    if prev_nonempty_block is None or prev_idx > prev_nonempty_block:
                        prev_nonempty_block = prev_idx

            if prev_nonempty_block is not None:
                for i in range(3):
                    prev_id = self.block_events[prev_nonempty_block][i + 2]
                    if prev_id != 0:
                        prev_lib = self.grad_library.get(prev_id)
                        if prev_lib['type'] == 'g':
                            grad_check_data['lastGradVals'][i] = prev_lib['data'][2]

        if rot_quaternion is not None:
            # scipy Rotation uses [x, y, z, w], while Pulseq stores [w, x, y, z].
            rot_obj = Rotation.from_quat(
                [rot_quaternion[1], rot_quaternion[2], rot_quaternion[3], rot_quaternion[0]]
            )
            grad_check_data['lastGradVals'] = rot_obj.apply(np.asarray(grad_check_data['lastGradVals'], dtype=float), inverse=True)

        for i in range(3):
            grad_to_check = check_g[i]
            if grad_to_check is None:
                if abs(grad_check_data['lastGradVals'][i]) > slew_eps:
                    raise RuntimeError(
                        f'Error in block {block_index} on gradient axis {i + 1}: previous block ended with non-zero amplitude but the current block has no compatible gradient.'
                    )
                grad_check_data['lastGradVals'][i] = 0
                continue

            if abs(grad_to_check.start[1]) > slew_eps:
                if grad_to_check.start[0] != 0:
                    raise RuntimeError('No delay allowed for gradients which start with a non-zero amplitude')
                if block_index > 1:
                    if abs(grad_check_data['lastGradVals'][i] - grad_to_check.start[1]) > slew_eps:
                        raise RuntimeError(
                            'Two consecutive gradients need to have the same amplitude at the connection point'
                        )
                else:
                    raise RuntimeError('First gradient in the the first block has to start at 0.')

            if abs(grad_to_check.stop[1]) > slew_eps and abs(grad_to_check.stop[0] - duration) > 1e-7:
                raise RuntimeError("A gradient that doesn't end at zero needs to be aligned to the block boundary.")

            if next_block_index is not None:
                next_id = self.block_events[next_block_index][grad_to_check.idx]
                if next_id != 0:
                    next_lib = self.grad_library.get(next_id)
                    next_type = next_lib['type']

                    if next_type == 't':
                        first = 0
                    elif next_type == 'g':
                        first = next_lib['data'][1]
                    else:
                        first = 0
                else:
                    first = 0

                if abs(first - grad_to_check.stop[1]) > slew_eps:
                    raise RuntimeError(
                        'Two consecutive gradients need to have the same amplitude at the connection point'
                    )

            grad_check_data['lastGradVals'][i] = grad_to_check.stop[1]

        if rot_quaternion is not None:
            grad_check_data['lastGradVals'] = rot_obj.apply(np.asarray(grad_check_data['lastGradVals'], dtype=float))

    self.grad_check_data['validForBlockNum'] = block_index
    # =========
    # END GRADIENT CHECKS
    # =========

    if required_duration is not None:
        if duration - required_duration > eps:
            raise ValueError(
                f'Required block duration is {required_duration:g} s but actual block duration is {duration:g} s'
            )
        duration = required_duration

    self.block_events[block_index] = new_block
    self.block_durations[block_index] = float(duration)


def get_raw_block_content_IDs(self, block_index: int) -> SimpleNamespace:
    """
    Returns PyPulseq block content IDs at `block_index` position in `self.block_events`.

    No block events are created, only the IDs of the objects are returned.

    Parameters
    ----------
    block_index : int
        Index of PyPulseq block to be retrieved from `self.block_events`.

    Returns
    -------
    block : SimpleNamespace
        PyPulseq block content IDs at 'block_index' position in `self.block_events`.
    """
    raw_block = SimpleNamespace(block_duration=0, rf=0, gx=0, gy=0, gz=0, adc=0, ext=np.zeros((2, 0), dtype=int))
    event_ind = self.block_events[block_index]

    # Extensions
    if event_ind[6] > 0:
        next_ext_id = event_ind[6]
        ext_items = []
        while next_ext_id != 0:
            ext_data = self.extensions_library.data[next_ext_id]
            ext_items.append(np.asarray(ext_data[:2], dtype=int))
            next_ext_id = ext_data[2]
        if ext_items:
            raw_block.ext = np.stack(ext_items, axis=-1)

    # RF
    if event_ind[1] > 0:
        raw_block.rf = event_ind[1]

    # Gradients
    grad_channels = ['gx', 'gy', 'gz']
    for i in range(len(grad_channels)):
        if event_ind[2 + i] > 0:
            setattr(raw_block, grad_channels[i], event_ind[2 + i])

    # ADC
    if event_ind[5] > 0:
        raw_block.adc = event_ind[5]

    raw_block.block_duration = self.block_durations[block_index]
    return raw_block


def get_block(self, block_index: int, add_ids: bool = False) -> SimpleNamespace:
    """
    Returns PyPulseq block at `block_index` position in `self.block_events`.

    The block is created from the sequence data with all events and shapes decompressed.

    Parameters
    ----------
    block_index : int
        Index of PyPulseq block to be retrieved from `self.block_events`.

    Returns
    -------
    block : SimpleNamespace
        PyPulseq block at 'block_index' position in `self.block_events`.

    Raises
    ------
    ValueError
        If a trigger event of an unsupported control type is encountered.
        If a label object of an unknown extension ID is encountered.
    """
    # Check if block exists in the block cache. If so, return that
    if self.use_block_cache and not add_ids and block_index in self.block_cache:
        return self.block_cache[block_index]

    raw_block = get_raw_block_content_IDs(self, block_index)
    block = SimpleNamespace(
        block_duration=0.0,
        rf=None,
        gx=None,
        gy=None,
        gz=None,
        adc=None,
        delay=None,
        label=None,
        soft_delay=None,
    )
    event_ind = self.block_events[block_index]

    if event_ind[0] > 0:  # Delay
        delay = SimpleNamespace()
        delay.type = 'delay'
        delay.delay = self.delay_library.data[event_ind[0]][0]
        if add_ids:
            delay.id = int(event_ind[0])
        block.delay = delay

    if raw_block.rf:  # RF
        if raw_block.rf in self.rf_library.type:
            block.rf = self.rf_from_lib_data(self.rf_library.data[raw_block.rf], self.rf_library.type[raw_block.rf])
        else:
            block.rf = self.rf_from_lib_data(self.rf_library.data[raw_block.rf], 'u')
        if add_ids:
            block.rf.id = int(raw_block.rf)
            block.rf.shape_ids = np.array(self.rf_library.data[raw_block.rf][1:4], dtype=int)

    # Gradients
    grad_channels = ['gx', 'gy', 'gz']
    for i in range(len(grad_channels)):
        grad_id = getattr(raw_block, grad_channels[i])
        if grad_id:
            grad, compressed = SimpleNamespace(), SimpleNamespace()
            grad_type = self.grad_library.type[grad_id]
            lib_data = self.grad_library.data[grad_id]
            grad.type = 'trap' if grad_type == 't' else 'grad'
            grad.channel = grad_channels[i][1]
            if grad.type == 'grad':
                amplitude = lib_data[0]
                shape_id = lib_data[3]  # change in v150: changed from lib_data[1] to lib_data[3]
                time_id = lib_data[4]  # change in v150: changed from lib_data[2] to lib_data[4]
                delay = lib_data[5]  # change in v150: changed from lib_data[3] to lib_data[5]
                shape_data = self.shape_library.data[shape_id]
                compressed.num_samples = shape_data[0]
                compressed.data = shape_data[1:]
                g = decompress_shape(compressed)
                grad.waveform = amplitude * g

                if time_id == 0:
                    grad.tt = (np.arange(1, len(g) + 1) - 0.5) * self.grad_raster_time
                    t_end = len(g) * self.grad_raster_time
                    grad.area = sum(grad.waveform) * self.grad_raster_time
                elif time_id == -1:
                    # Gradient with oversampling by a factor of 2
                    grad.tt = 0.5 * (np.arange(1, len(g) + 1)) * self.grad_raster_time
                    if len(grad.tt) != len(grad.waveform):
                        raise ValueError(
                            f'Mismatch between time shape length ({len(grad.tt)}) and gradient shape length ({len(grad.waveform)}).'
                        )
                    if len(grad.waveform) % 2 != 1:
                        raise ValueError('Oversampled gradient waveforms must have odd number of samples')
                    t_end = (len(g) + 1) * 0.5 * self.grad_raster_time
                    grad.area = sum(grad.waveform[::2]) * self.grad_raster_time  # remove oversampling
                else:
                    t_shape_data = self.shape_library.data[time_id]
                    compressed.num_samples = t_shape_data[0]
                    compressed.data = t_shape_data[1:]
                    grad.tt = decompress_shape(compressed) * self.grad_raster_time
                    if len(grad.tt) != len(grad.waveform):
                        raise ValueError(
                            f'Mismatch between time shape length ({len(grad.tt)}) and gradient shape length ({len(grad.waveform)}).'
                        )
                    t_end = grad.tt[-1]
                    grad.area = 0.5 * sum((grad.tt[1:] - grad.tt[:-1]) * (grad.waveform[1:] + grad.waveform[:-1]))

                grad.shape_id = shape_id
                grad.time_id = time_id
                grad.delay = delay
                grad.shape_dur = t_end
                grad.first = lib_data[1]  # change in v150 - we always have first/last now
                grad.last = lib_data[2]  # change in v150 - we always have first/last now
                if add_ids:
                    grad.id = int(grad_id)
                    grad.shape_ids = np.array([shape_id, time_id], dtype=int)
            else:
                grad.amplitude = lib_data[0]
                grad.rise_time = lib_data[1]
                grad.flat_time = lib_data[2]
                grad.fall_time = lib_data[3]
                grad.delay = lib_data[4]
                grad.area = grad.amplitude * (grad.flat_time + grad.rise_time / 2 + grad.fall_time / 2)
                grad.flat_area = grad.amplitude * grad.flat_time
                if add_ids:
                    grad.id = int(grad_id)

            setattr(block, grad_channels[i], grad)

    # ADC
    if raw_block.adc:
        lib_data = self.adc_library.data[raw_block.adc]
        shape_id_phase_modulation = lib_data[7]
        if shape_id_phase_modulation:
            shape_data = self.shape_library.data[shape_id_phase_modulation]
            compressed = SimpleNamespace()
            compressed.num_samples = shape_data[0]
            compressed.data = shape_data[1:]
            phase_shape = decompress_shape(compressed)
        else:
            phase_shape = np.array([], dtype=float)

        adc = SimpleNamespace()
        adc.num_samples = lib_data[0]
        adc.dwell = lib_data[1]
        adc.delay = lib_data[2]
        adc.freq_ppm = lib_data[3]
        adc.phase_ppm = lib_data[4]
        adc.freq_offset = lib_data[5]
        adc.phase_offset = lib_data[6]
        adc.phase_modulation = phase_shape
        adc.dead_time = self.system.adc_dead_time
        adc.num_samples = int(adc.num_samples)
        adc.type = 'adc'
        if add_ids:
            adc.id = int(raw_block.adc)

        block.adc = adc

    if raw_block.ext.size != 0:
        # We have extensions - triggers, labels, etc.
        trig_list = []
        label_list = []
        n_soft_delay = 0
        n_rotation = 0
        n_rf_shim = 0
        for ext_type_id, ext_ref_id in raw_block.ext.T:
            # Format: ext_type, ext_id, next_ext_id
            ext_type = self.get_extension_type_string(int(ext_type_id))

            # Triggers
            if ext_type == 'TRIGGERS':
                trigger_types = ['output', 'trigger']
                data = self.trigger_library.data[int(ext_ref_id)]
                trigger = SimpleNamespace()
                trigger.type = trigger_types[int(data[0]) - 1]
                if data[0] == 1:
                    trigger_channels = ['osc0', 'osc1', 'ext1']
                    trigger.channel = trigger_channels[int(data[1]) - 1]
                elif data[0] == 2:
                    trigger_channels = ['physio1', 'physio2']
                    trigger.channel = trigger_channels[int(data[1]) - 1]
                else:
                    raise ValueError('Unsupported trigger event type')

                trigger.delay = data[2]
                trigger.duration = data[3]
                if add_ids:
                    trigger.id = int(ext_ref_id)
                # Allow for multiple triggers per block
                trig_list.append(trigger)
            elif ext_type in ['LABELSET', 'LABELINC']:
                label = SimpleNamespace()
                label.type = ext_type.lower()
                supported_labels = get_supported_labels()
                if ext_type == 'LABELSET':
                    data = self.label_set_library.data[int(ext_ref_id)]
                else:
                    data = self.label_inc_library.data[int(ext_ref_id)]

                label.label = supported_labels[int(data[1] - 1)]
                label.value = data[0]
                if add_ids:
                    label.id = int(ext_ref_id)
                # Allow for multiple labels per block
                label_list.append(label)
            elif ext_type == 'DELAYS':
                n_soft_delay += 1
                if n_soft_delay > 1:
                    raise RuntimeError('Only one soft delay extension object per block is allowed')
                data = self.soft_delay_library.data[int(ext_ref_id)]
                hint_id = int(float(data[3]))
                hint = self.soft_delay_hints2[hint_id - 1] if hint_id > 0 else ''

                soft_delay = SimpleNamespace(
                    type='soft_delay',
                    numID=int(float(data[0])),
                    offset=data[1],
                    factor=data[2],
                    hint=hint,
                    default_duration=self.block_durations[block_index],
                )
                if add_ids:
                    soft_delay.id = int(ext_ref_id)
                block.soft_delay = soft_delay

            elif ext_type == 'ROTATIONS':
                n_rotation += 1
                if n_rotation > 1:
                    raise RuntimeError('Only one rotation extension object per block is allowed')
                data = self.rotation_library.data[int(ext_ref_id)]
                rotation = SimpleNamespace(
                    type='rot3D',
                    rot_quaternion=data
                )
                if add_ids:
                    rotation.id = int(ext_ref_id)
                block.rotation = rotation

            elif ext_type == 'RF_SHIMS':
                n_rf_shim += 1
                if n_rf_shim > 1:
                    raise RuntimeError('Only one RF shim extension object per block is allowed')
                data = self.rf_shim_library.data[int(ext_ref_id)]
                # data is [mag0, ph0, mag1, ph1 ...]
                # data is [magnitude, phase, magnitude, phase, ...]
                # Ensure data is numpy array for slicing
                if not isinstance(data, np.ndarray):
                    data = np.array(data)
                
                # Careful with shapes if data is not flat?
                # It should be flat.
                shim_vector = data[0::2] * np.exp(1j * data[1::2])
                
                rf_shim = SimpleNamespace(
                    type='rf_shim',
                    shim_vector=shim_vector
                )
                if add_ids:
                    rf_shim.id = int(ext_ref_id)
                block.rf_shim = rf_shim

            else:
                warn(f'unknown extension ID {int(ext_type_id)}', stacklevel=2)

        if trig_list:
            block.trig = list(reversed(trig_list))

        if label_list:
            block.label = list(reversed(label_list))

    block.block_duration = self.block_durations[block_index]

    # Enter block into the block cache
    if self.use_block_cache and not add_ids:
        self.block_cache[block_index] = block

    return block


def register_adc_event(self, event: EventLibrary) -> Tuple[int, int]:
    """

    Parameters
    ----------
    event : SimpleNamespace
        ADC event to be registered.

    Returns
    -------
    int, int
        ID of registered ADC event, shape ID
    """
    surely_new = False

    # Handle phase modulation
    if not hasattr(event, 'phase_modulation') or event.phase_modulation is None or len(event.phase_modulation) == 0:
        shape_id = 0
    else:
        if hasattr(event, 'shape_id'):
            shape_id = event.shape_id
        else:
            phase_shape = compress_shape(np.asarray(event.phase_modulation).flatten())
            shape_data = np.concatenate(([phase_shape.num_samples], phase_shape.data))
            shape_id, shape_found = self.shape_library.find_or_insert(shape_data)
            if not shape_found:
                surely_new = True

    # Construct the ADC event data
    data = (
        event.num_samples,
        event.dwell,
        max(event.delay, event.dead_time),
        event.freq_ppm,
        event.phase_ppm,
        event.freq_offset,
        event.phase_offset,
        shape_id,
    )

    # Insert or find/insert into libraryAdd commentMore actions
    if surely_new:
        adc_id = self.adc_library.insert(0, data)
    else:
        adc_id, found = self.adc_library.find_or_insert(data)

        # Clear block cache if overwritten
        if self.use_block_cache and found:
            self.block_cache.clear()

    # Optional mapping
    if hasattr(event, 'name'):
        self.adc_id_to_name_map[adc_id] = event.name

    return adc_id, shape_id


def register_control_event(self, event: SimpleNamespace) -> int:
    """

    Parameters
    ----------
    event : SimpleNamespace
        Control event to be registered.

    Returns
    -------
    int
        ID of registered control event.
    """
    event_type = ['output', 'trigger'].index(event.type)
    if event_type == 0:
        # Trigger codes supported by the Siemens interpreter as of May 2019
        event_channel = ['osc0', 'osc1', 'ext1'].index(event.channel)
    elif event_type == 1:
        # Trigger codes supported by the Siemens interpreter as of June 2019
        event_channel = ['physio1', 'physio2'].index(event.channel)
    else:
        raise ValueError('Unsupported control event type')

    data = (event_type + 1, event_channel + 1, event.delay, event.duration)
    control_id, found = self.trigger_library.find_or_insert(new_data=data)

    # Clear block cache because trigger was overwritten
    # TODO: Could find only the blocks that are affected by the changes
    if self.use_block_cache and found:
        self.block_cache.clear()

    return control_id


def register_grad_event(self, event: SimpleNamespace) -> Union[int, Tuple[int, List[int]]]:
    """
    Parameters
    ----------
    event : SimpleNamespace
        Gradient event to be registered.

    Returns
    -------
    int, [int, ...]
        For gradient events: ID of registered gradient event, list of shape IDs
    int
        For trapezoid gradient events: ID of registered gradient event
    """
    may_exist = True
    any_changed = False

    if event.type == 'grad':
        amplitude = max((abs(value) for value in event.waveform), default=0.0)
        if amplitude > 0:
            fnz = event.waveform[np.nonzero(event.waveform)[0][0]]
            amplitude *= np.sign(fnz) if fnz != 0 else 1

        # Shape ID initialization
        if hasattr(event, 'shape_IDs'):
            shape_IDs = event.shape_IDs
        else:
            shape_IDs = [0, 0]

            # Shape for waveform
            g = event.waveform / amplitude if amplitude != 0 else event.waveform
            c_shape = compress_shape(g)
            s_data = np.concatenate(([c_shape.num_samples], c_shape.data))
            shape_IDs[0], found = self.shape_library.find_or_insert(s_data)
            may_exist = may_exist and found
            any_changed = any_changed or found

            # Shape for timing
            c_time = compress_shape(event.tt / self.grad_raster_time)
            if len(c_time.data) == 4 and np.allclose(c_time.data, [0.5, 1, 1, c_time.num_samples - 3]):
                # Standard raster: leave shape_IDs[1] as 0
                pass
            elif len(c_time.data) == 3 and np.allclose(c_time.data, [0.5, 0.5, c_time.num_samples - 2]):
                # Half-raster: set to -1 as special flag
                shape_IDs[1] = -1
            else:
                t_data = np.concatenate(([c_time.num_samples], c_time.data))
                shape_IDs[1], found = self.shape_library.find_or_insert(t_data)
                may_exist = may_exist and found
                any_changed = any_changed or found

        # Updated data layout to match MATLAB v1.5.0 ordering
        data = (amplitude, event.first, event.last, *shape_IDs, event.delay)

    elif event.type == 'trap':
        data = (
            event.amplitude,
            event.rise_time,
            event.flat_time,
            event.fall_time,
            event.delay,
        )
    else:
        raise ValueError('Unknown gradient type passed to register_grad_event()')

    if may_exist:
        grad_id, found = self.grad_library.find_or_insert(new_data=data, data_type=event.type[0])
        any_changed = any_changed or found
    else:
        grad_id = self.grad_library.insert(0, data, event.type[0])

    # Clear block cache because grad event or shapes were overwritten
    # TODO: Could find only the blocks that are affected by the changes
    if self.use_block_cache and any_changed:
        self.block_cache.clear()

    if hasattr(event, 'name'):
        self.grad_id_to_name_map[grad_id] = event.name

    if event.type == 'grad':
        return grad_id, shape_IDs
    elif event.type == 'trap':
        return grad_id


def register_label_event(self, event: SimpleNamespace) -> int:
    """
    Parameters
    ----------
    event : SimpleNamespace
        ID of label event to be registered.

    Returns
    -------
    int
        ID of registered label event.
    """
    label_id = get_supported_labels().index(event.label) + 1
    data = (event.value, label_id)
    if event.type == 'labelset':
        lib = self.label_set_library
        ext_type = self.get_extension_type_ID('LABELSET')
    elif event.type == 'labelinc':
        lib = self.label_inc_library
        ext_type = self.get_extension_type_ID('LABELINC')
    else:
        raise ValueError('Unsupported label type passed to register_label_event()')

    label_id, found = lib.find_or_insert(new_data=data)

    # Clear block cache because label event was overwritten
    # TODO: Could find only the blocks that are affected by the changes
    if self.use_block_cache and found:
        self.block_cache.clear()

    return label_id


def register_soft_delay_event(self, event: SimpleNamespace) -> int:
    """
    Parameters
    ----------
    event : SimpleNamespace
        ID of soft delay event to be registered.

    Returns
    -------
    int
        ID of registered soft delay event.
    """
    num_id = getattr(event, 'numID', None)
    if num_id is None:
        if event.hint in self.soft_delay_hints:
            num_id = self.soft_delay_hints[event.hint]
        else:
            num_id = max([-1, *self.soft_delay_hints.values()]) + 1
            self.soft_delay_hints[event.hint] = num_id
    else:
        num_id = int(num_id)
        if event.hint in self.soft_delay_hints and self.soft_delay_hints[event.hint] != num_id:
            raise ValueError(
                f"Soft delay hint '{event.hint}' is already assigned to numID {self.soft_delay_hints[event.hint]}"
            )
        for known_hint, known_num_id in self.soft_delay_hints.items():
            if known_hint != event.hint and known_num_id == num_id:
                raise ValueError(f"numID {num_id} is already used by soft delay '{known_hint}'")
        self.soft_delay_hints[event.hint] = num_id

    event.numID = int(num_id)

    # Map hint string to hintID (1-based), matching MATLAB behavior
    if event.hint in self.soft_delay_hint_ids:
        hint_id = self.soft_delay_hint_ids[event.hint]
    else:
        hint_id = len(self.soft_delay_hints2) + 1
        self.soft_delay_hint_ids[event.hint] = hint_id
        self.soft_delay_hints2.append(event.hint)

    data = (event.numID, event.offset, event.factor, hint_id)
    soft_delay_id, found = self.soft_delay_library.find_or_insert(new_data=data)
    if self.use_block_cache and found:
        self.block_cache.clear()
    return soft_delay_id


def register_rf_event(self, event: SimpleNamespace) -> Tuple[int, List[int]]:
    """
    Parameters
    ----------
    event : SimpleNamespace
        RF event to be registered.

    Returns
    -------
    int, [int, ...]
        ID of registered RF event, list of shape IDs
    """
    mag = np.abs(event.signal)
    amplitude = np.max(mag)
    mag /= amplitude
    # Following line of code is a workaround for numpy's divide functions returning NaN when mathematical
    # edge cases are encountered (eg. divide by 0)
    mag[np.isnan(mag)] = 0
    phase = np.angle(event.signal)
    phase[phase < 0] += 2 * np.pi
    phase /= 2 * np.pi
    may_exist = True

    if hasattr(event, 'shape_IDs'):
        shape_IDs = event.shape_IDs
    else:
        shape_IDs = [0, 0, 0]

        mag_shape = compress_shape(mag)
        data = np.concatenate(([mag_shape.num_samples], mag_shape.data))
        shape_IDs[0], found = self.shape_library.find_or_insert(data)
        may_exist = may_exist & found

        phase_shape = compress_shape(phase)
        data = np.concatenate(([phase_shape.num_samples], phase_shape.data))
        shape_IDs[1], found = self.shape_library.find_or_insert(data)
        may_exist = may_exist & found

        t_regular = (np.floor(event.t / self.rf_raster_time) == np.arange(len(event.t))).all()

        if t_regular:
            shape_IDs[2] = 0
        else:
            time_shape = compress_shape(event.t / self.rf_raster_time)
            data = [time_shape.num_samples, *time_shape.data]
            shape_IDs[2], found = self.shape_library.find_or_insert(data)
            may_exist = may_exist & found

    use = 'u'  # Undefined
    if hasattr(event, 'use'):
        if event.use in [
            'excitation',
            'refocusing',
            'inversion',
            'saturation',
            'preparation',
            'other',
        ]:
            use = event.use[0]
        else:
            if event.use == 'u':
                event.use = 'undefined'
            warn(
                f"Unknown or undefined RF pulse intended use 'use'={event.use}. Keep in mind that the 'use' "
                'parameter is not optional since v1.5.0',
                stacklevel=2,
            )
            use = 'u'
    else:
        raise ValueError('Parameter "use" is not optional since v1.5.0')

    data = (
        amplitude,
        *shape_IDs,
        event.center,
        event.delay,
        event.freq_ppm,
        event.phase_ppm,
        event.freq_offset,
        event.phase_offset,
    )

    if may_exist:
        rf_id, found = self.rf_library.find_or_insert(new_data=data, data_type=use)

        # Clear block cache because RF event was overwritten
        # TODO: Could find only the blocks that are affected by the changes
        if self.use_block_cache and found:
            self.block_cache.clear()
    else:
        rf_id = self.rf_library.insert(key_id=0, new_data=data, data_type=use)

    if hasattr(event, 'name'):
        self.rf_id_to_name_map[rf_id] = event.name

    return rf_id, shape_IDs


def _retrieve_grad_event(self, grad_id: int, channel: str) -> SimpleNamespace:
    grad, compressed = SimpleNamespace(), SimpleNamespace()
    grad_type = self.grad_library.type[grad_id]
    lib_data = self.grad_library.data[grad_id]
    grad.type = 'trap' if grad_type == 't' else 'grad'
    grad.channel = channel
    
    if grad.type == 'grad':
        amplitude = lib_data[0]
        shape_id = lib_data[3]
        time_id = lib_data[4]
        delay = lib_data[5]
        shape_data = self.shape_library.data[shape_id]
        compressed.num_samples = shape_data[0]
        compressed.data = shape_data[1:]
        g = decompress_shape(compressed)
        grad.waveform = amplitude * g

        if time_id == 0:
            grad.tt = (np.arange(1, len(g) + 1) - 0.5) * self.grad_raster_time
        elif time_id == -1:
            grad.tt = 0.5 * (np.arange(1, len(g) + 1)) * self.grad_raster_time
        else:
            t_shape_data = self.shape_library.data[time_id]
            compressed.num_samples = t_shape_data[0]
            compressed.data = t_shape_data[1:]
            grad.tt = decompress_shape(compressed) * self.grad_raster_time
            
        grad.first = lib_data[1]
        grad.last = lib_data[2]
        grad.delay = delay
    else:
        grad.amplitude = lib_data[0]
        grad.rise_time = lib_data[1]
        grad.flat_time = lib_data[2]
        grad.fall_time = lib_data[3]
        grad.delay = lib_data[4]
        grad.area = grad.amplitude * (grad.flat_time + grad.rise_time / 2 + grad.fall_time / 2)
        grad.flat_area = grad.amplitude * grad.flat_time

    return grad
