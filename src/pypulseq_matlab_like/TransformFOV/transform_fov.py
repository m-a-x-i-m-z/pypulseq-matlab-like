from types import SimpleNamespace
from typing import Optional, Sequence as TypingSequence
from warnings import warn

import numpy as np
from scipy.spatial.transform import Rotation

from pypulseq_matlab_like.block_to_events import block_to_events
from pypulseq_matlab_like.make_rotation import make_rotation
from pypulseq_matlab_like.opts import Opts
from pypulseq_matlab_like.rotate_3d import rotate_3d
from pypulseq_matlab_like.scale_grad import scale_grad
from pypulseq_matlab_like.utils.event_helpers import copy_without_id


class transform_fov:
    """
    MATLAB-aligned transform_fov helper.

    This implementation follows the structure and logic of
    `matlab/+mr/@TransformFOV/TransformFOV.m`. The `scale` vector is a
    gradient scaling vector in non-rotated logical Pulseq coordinates; FOV
    size is divided by these values, and a zero scale on an axis zeros that
    axis' gradients.
    """

    def __init__(
        self,
        rotation=None,
        translation=None,
        scale=None,
        transform=None,
        use_rotation_extension: bool = False,
        prior_phase_cycle: float = 0.0,
        system=None,
    ):
        if transform is not None:
            if rotation is not None or translation is not None:
                raise ValueError("Neither 'translation' nor 'rotation' can be provided in combination with 'transform'.")
            transform = np.asarray(transform, dtype=float)
            if transform.shape != (4, 4):
                raise ValueError("'transform' must be 4x4.")
            rotation = transform[:3, :3]
            off = transform[3, :3]
            translation = rotation @ off
        elif rotation is None and translation is None and scale is None:
            raise ValueError('At least one transforming parameter needs to be provided')

        self.rotation = [] if rotation is None else np.asarray(rotation, dtype=float)
        self.translation = [] if translation is None else np.asarray(translation, dtype=float).reshape(3)
        self.scale = [] if scale is None else np.asarray(scale, dtype=float).reshape(3)
        self.use_rotation_extension = bool(use_rotation_extension)
        self.prior_phase_cycle = float(prior_phase_cycle)
        self.rotation_quaternion = []
        self.system = system if system is not None else Opts()
        self.labels = {'NOPOS': 0, 'NOROT': 0, 'NOSCL': 0}

        if len(self.rotation) != 0 and self.use_rotation_extension:
            q_xyzw = Rotation.from_matrix(self.rotation).as_quat()
            self.rotation_quaternion = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=float)

    def apply_to_block(self, *varargin):
        if not any(isinstance(item, (list, tuple)) for item in varargin):
            block_events = list(varargin)
        else:
            block_events = []
            for item in varargin:
                if isinstance(item, (list, tuple)):
                    block_events.extend(item)
                else:
                    block_events.append(item)

        if len(block_events) == 1 and hasattr(block_events[0], 'block_duration'):
            block_events = block_events[0]

        if hasattr(block_events, 'block_duration') or (
            isinstance(block_events, list)
            and len(block_events) > 0
            and hasattr(block_events[0], 'block_duration')
        ):
            block_events = list(block_to_events(block_events))

        rf = None
        adc = None
        grads = [None, None, None]
        other = []
        rot_ext_quaternion = []
        block_duration = None

        for event in block_events:
            current_events = event if isinstance(event, (list, tuple)) else [event]
            for e in current_events:
                if isinstance(e, (int, float, np.integer, np.floating)):
                    # MATLAB parity:
                    # block2events() carries blockDuration as a numeric event that
                    # is passed through to addBlock(B2). Preserve it here.
                    if block_duration is None:
                        block_duration = float(e)
                    continue
                if hasattr(e, 'type') and not isinstance(e, (list, tuple, dict)):
                    if e.type == 'rf':
                        rf = e
                    elif e.type == 'adc':
                        adc = e
                    elif e.type in ['trap', 'grad']:
                        if not hasattr(e, 'channel'):
                            raise ValueError('unspecified gradient channel for the gradient object')
                        if e.channel == 'x':
                            grads[0] = e
                        elif e.channel == 'y':
                            grads[1] = e
                        elif e.channel == 'z':
                            grads[2] = e
                        else:
                            raise ValueError(f'unsupported gradient channel {e.channel} for the gradient object')
                    elif e.type == 'labelset':
                        # MATLAB supports arrays of labelset structs.
                        label_events = e if isinstance(e, (list, tuple)) else [e]
                        for le in label_events:
                            if hasattr(le, 'label') and le.label in ['NOPOS', 'NOROT', 'NOSCL']:
                                self.labels[le.label] = le.value
                            elif hasattr(le, 'label') and le.label == 'SLC':
                                # MATLAB transform_fov path does not carry over this
                                # single-block slice counter reset during translated
                                # block replication.
                                continue
                            else:
                                # Preserve regular labelset events (e.g. LIN/NAV/REV/SEG/AVG)
                                # so transformed blocks keep the same extension content.
                                other.append(le)
                    elif e.type == 'rot3D':
                        rot_ext_quaternion = np.asarray(e.rot_quaternion, dtype=float)
                    else:
                        other.append(e)
                else:
                    other.append(e)

        grad_raster_time = self.system.grad_raster_time

        if len(self.scale) != 0:
            for i in range(3):
                if grads[i] is not None:
                    grads[i] = scale_grad(copy_without_id(grads[i]), float(self.scale[i]), self.system)

        if len(self.translation) != 0:
            if rf is None:
                if adc is None:
                    t_start = None
                    t_end = None
                else:
                    t_start, t_end = extract_time(adc)
            else:
                if adc is None:
                    t_start, t_end = extract_time(rf)
                else:
                    t_start_adc, t_end_adc = extract_time(adc)
                    t_start_rf, t_end_rf = extract_time(rf)
                    t_start = min(t_start_adc, t_start_rf)
                    t_end = max(t_end_adc, t_end_rf)

            if rf is not None and hasattr(rf, 'id'):
                delattr(rf, 'id')
            if adc is not None and hasattr(adc, 'id'):
                delattr(adc, 'id')

            if not self.use_rotation_extension and len(rot_ext_quaternion) != 0:
                grads = rotate_grad_list(rot_ext_quaternion, grads, self.system)
                rot_ext_quaternion = []

            grads_backup = None
            if self.labels['NOROT'] and len(self.rotation) != 0:
                grads_backup = grads
                grads = rotate_grad_list(self.rotation.T, grads, self.system)

            phase_cycle_this_block = 0.0

            if not self.labels['NOPOS']:
                if rf is not None:
                    rf.phase_offset = rf.phase_offset + 2 * np.pi * self.prior_phase_cycle
                if adc is not None:
                    adc.phase_offset = adc.phase_offset + 2 * np.pi * self.prior_phase_cycle

            for i in range(3):
                if abs(float(self.translation[i])) <= np.finfo(float).eps:
                    continue

                g = grads[i]
                if g is None:
                    continue

                if g.type == 'trap':
                    if abs(g.flat_time) > np.finfo(float).eps:
                        tt = g.delay + np.cumsum([0.0, g.rise_time, g.flat_time, g.fall_time])
                        waveform = g.amplitude * np.array([0.0, 1.0, 1.0, 0.0], dtype=float)
                    else:
                        if abs(g.rise_time) > np.finfo(float).eps and abs(g.fall_time) > np.finfo(float).eps:
                            tt = g.delay + np.cumsum([0.0, g.rise_time, g.fall_time])
                            waveform = g.amplitude * np.array([0.0, 1.0, 0.0], dtype=float)
                        else:
                            if abs(g.amplitude) > np.finfo(float).eps:
                                warn("'empty' gradient with non-zero magnitude detected", stacklevel=2)
                            continue
                else:
                    tt = g.delay + np.asarray(g.tt, dtype=float)
                    waveform = np.asarray(g.waveform, dtype=float)

                breaks, coefs, tt_extended, waveform_extended = generate_breaks_coefs(
                    g, tt, waveform, grad_raster_time, t_start, t_end
                )
                fi_breaks, fi_coefs = linear_pp_integral(breaks, coefs)

                if not self.labels['NOPOS']:
                    for event in [rf, adc]:
                        if event is None:
                            continue

                        t_s, t_e = extract_time(event)
                        is_const = is_grad_const(tt_extended, waveform_extended, t_s, t_e)

                        if is_const:
                            freq = float(self.translation[i]) * ppval_linear(breaks, coefs, t_s)
                            if hasattr(event, 't'):
                                rf.freq_offset = rf.freq_offset + freq
                                phase_cycle = local_frac(
                                    accurate_mod_pp(fi_breaks, fi_coefs, t_s, float(self.translation[i]))
                                    - freq * (t_s - rf.delay)
                                )
                                rf.phase_offset = rf.phase_offset + 2 * np.pi * phase_cycle
                            else:
                                adc.freq_offset = adc.freq_offset + freq
                                phase_cycle = local_frac(
                                    accurate_mod_pp(fi_breaks, fi_coefs, t_s, float(self.translation[i]))
                                    - freq * (t_s - adc.delay)
                                )
                                adc.phase_offset = adc.phase_offset + 2 * np.pi * phase_cycle
                        else:
                            if hasattr(event, 't'):
                                ppval_f_center = ppval_linear(breaks, coefs, rf.delay + rf.center)
                                freq = float(self.translation[i]) * ppval_f_center
                                rf.freq_offset = rf.freq_offset + freq
                                ppval_fi_center_shift = accurate_mod_pp(
                                    fi_breaks, fi_coefs, rf.delay + rf.center, float(self.translation[i])
                                )
                                phase_cycle = local_frac(ppval_fi_center_shift - freq * rf.center)
                                rf.phase_offset = rf.phase_offset + 2 * np.pi * phase_cycle
                                phase_cycle_vector_tmp = accurate_mod_pp(
                                    fi_breaks, fi_coefs, np.asarray(rf.t, dtype=float) + rf.delay, float(self.translation[i])
                                )
                                phase_cycle_vector = local_frac(
                                    phase_cycle_vector_tmp
                                    - ppval_f_center * (np.asarray(rf.t, dtype=float) - rf.center)
                                    * float(self.translation[i])
                                    - ppval_fi_center_shift
                                )
                                rf.signal = rf.signal * np.exp(1j * 2 * np.pi * phase_cycle_vector)
                            else:
                                adc_center = 0.5 * adc.dwell * adc.num_samples
                                ppval_f_center = ppval_linear(breaks, coefs, adc.delay + adc_center)
                                freq = float(self.translation[i]) * ppval_f_center
                                adc.freq_offset = adc.freq_offset + freq
                                ppval_fi_center_shift = accurate_mod_pp(
                                    fi_breaks, fi_coefs, adc.delay + adc_center, float(self.translation[i])
                                )
                                phase_cycle = local_frac(
                                    -0.5 + local_frac(0.5 + ppval_fi_center_shift - freq * adc_center)
                                )
                                adc.phase_offset = adc.phase_offset + 2 * np.pi * phase_cycle
                                adc_t = adc.dwell * (np.arange(adc.num_samples, dtype=float) + 0.5)
                                phase_cycle_vector_tmp = accurate_mod_pp(
                                    fi_breaks, fi_coefs, adc_t + adc.delay, float(self.translation[i])
                                )
                                phase_cycle_vector = local_frac(
                                    (-0.5 + local_frac(0.5 - ppval_f_center * (adc_t - adc_center)))
                                    * float(self.translation[i])
                                    + ppval_fi_center_shift
                                    + phase_cycle_vector_tmp
                                )
                                if (
                                    not hasattr(adc, 'phase_modulation')
                                    or adc.phase_modulation is None
                                    or len(adc.phase_modulation) == 0
                                ):
                                    adc.phase_modulation = 2 * np.pi * np.asarray(phase_cycle_vector).reshape(-1)
                                else:
                                    adc.phase_modulation = (
                                        np.asarray(adc.phase_modulation).reshape(-1)
                                        + 2 * np.pi * np.asarray(phase_cycle_vector).reshape(-1)
                                    )

                phase_cycle_this_block = local_frac(
                    phase_cycle_this_block
                    + accurate_mod_pp(fi_breaks, fi_coefs, tt_extended[-1], float(self.translation[i]))
                )

            if grads_backup is not None:
                grads = grads_backup

            self.prior_phase_cycle = local_frac(self.prior_phase_cycle + phase_cycle_this_block)

        if len(self.rotation) != 0 and not self.labels['NOROT']:
            if self.use_rotation_extension:
                if len(rot_ext_quaternion) == 0:
                    rot_ext_quaternion = np.asarray(self.rotation_quaternion, dtype=float)
                else:
                    q2 = np.asarray(rot_ext_quaternion, dtype=float).reshape(4)
                    q1 = np.asarray(self.rotation_quaternion, dtype=float).reshape(4)
                    r2 = Rotation.from_quat([q2[1], q2[2], q2[3], q2[0]])
                    r1 = Rotation.from_quat([q1[1], q1[2], q1[3], q1[0]])
                    q_xyzw = (r2 * r1).as_quat()
                    rot_ext_quaternion = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=float)
            else:
                grads = rotate_grad_list(self.rotation, grads, self.system)

        if len(rot_ext_quaternion) != 0:
            other.append(make_rotation(rot_ext_quaternion))

        out = [block_duration, rf, adc, *grads, *other]
        return [item for item in out if item is not None]

    def apply_to_seq(self, seq, same_seq: bool = False, block_range: Optional[TypingSequence[int]] = None):
        if block_range is None:
            block_range = [1, np.inf]

        if not np.isfinite(block_range[1]):
            block_end = len(seq.block_durations)
        else:
            block_end = int(block_range[1])
        block_start = int(block_range[0])

        if same_seq:
            seq2 = seq
        else:
            from pypulseq_matlab_like.Sequence.sequence import Sequence

            seq2 = Sequence(seq.system)
            seq2.copy_definitions(seq)

        self.labels = {'NOPOS': 0, 'NOROT': 0, 'NOSCL': 0}

        for iB in range(block_start, block_end + 1):
            block = seq.get_block(iB, add_ids=same_seq)
            block2 = self.apply_to_block(block)
            seq2.add_block(*block2)

        return seq2

def extract_time(event):
    if event.type == 'adc':
        t_s = event.delay + event.dwell * 0.5
        t_e = event.delay + event.dwell * (event.num_samples - 0.5)
    elif event.type == 'rf':
        t_s = event.delay + event.t[0]
        t_e = event.delay + event.t[-1]
    else:
        raise ValueError(f'Unsupported event type {event.type}')
    return float(t_s), float(t_e)


def is_grad_const(t, amp, t_start, t_end):
    t = np.asarray(t, dtype=float)
    amp = np.asarray(amp, dtype=float)

    index_s = np.where(t <= t_start)[0]
    index_e = np.where(t >= t_end)[0]
    index_s = int(index_s[-1]) if len(index_s) > 0 else 0
    index_e = int(index_e[0]) if len(index_e) > 0 else len(t) - 1
    return bool(np.all(np.abs(amp[index_s : index_e + 1] - amp[index_s]) <= 1e-10))


def generate_breaks_coefs(g, tt, waveform, grad_raster_time, t_start, t_end):
    tt = np.asarray(tt, dtype=float).reshape(-1)
    waveform = np.asarray(waveform, dtype=float).reshape(-1)

    if g.type == 'grad':
        if abs(float(g.tt[0]) - grad_raster_time / 2) < np.finfo(float).eps:
            tt = np.concatenate(([tt[0] - grad_raster_time / 2], tt, [tt[-1] + grad_raster_time / 2]))
            waveform = np.concatenate(([float(g.first)], waveform, [float(g.last)]))

    tt_extended = tt.copy()
    waveform_extended = waveform.copy()

    breaks = tt_extended.copy()
    coefs = np.zeros((len(tt) - 1, 2), dtype=float)
    coefs[:, 0] = np.diff(waveform) / np.diff(tt)
    coefs[~np.isfinite(coefs)] = 0.0
    coefs[:, 1] = waveform[:-1]

    if t_start is not None and t_start < tt[0]:
        breaks = np.concatenate(([t_start], breaks))
        coefs = np.vstack(([0.0, 0.0], coefs))
        tt_extended = np.concatenate(([t_start], tt_extended))
        waveform_extended = np.concatenate(([0.0], waveform_extended))
    if t_end is not None and t_end > tt[-1]:
        breaks = np.concatenate((breaks, [t_end]))
        coefs = np.vstack((coefs, [0.0, 0.0]))
        tt_extended = np.concatenate((tt_extended, [t_end]))
        waveform_extended = np.concatenate((waveform_extended, [0.0]))

    return breaks, coefs, tt_extended, waveform_extended


def linear_pp_integral(breaks, coefs):
    breaks = np.asarray(breaks, dtype=float).reshape(-1)
    coefs = np.asarray(coefs, dtype=float)
    fi_coefs = np.zeros_like(coefs, dtype=float)
    fi_coefs[:, 0] = 0.5 * coefs[:, 0]
    fi_coefs[:, 1] = coefs[:, 1]
    return breaks.copy(), fi_coefs


def local_frac(value):
    value = np.asarray(value, dtype=float)
    out = value - np.floor(value)
    if out.ndim == 0:
        return float(out)
    return out


def accurate_mod_pp(breaks, coefs, t, shift):
    breaks = np.asarray(breaks, dtype=float).reshape(-1)
    coefs = np.asarray(coefs, dtype=float)
    t_values = np.atleast_1d(np.asarray(t, dtype=float))

    mod = np.zeros(t_values.shape, dtype=float)
    areas = []
    i_breaks = 0

    for c, t0 in enumerate(t_values):
        index_candidates = np.where(breaks <= t0)[0]
        if len(index_candidates) == 0:
            index = 1
        else:
            index = int(index_candidates[-1]) + 1

        while index > (i_breaks + 1):
            i_breaks = i_breaks + 1
            ab = coefs[i_breaks - 1, 0] * shift * ((breaks[i_breaks] - breaks[i_breaks - 1]) ** 2 - 0.0)
            cd = coefs[i_breaks - 1, 1] * shift * ((breaks[i_breaks] - breaks[i_breaks - 1]) - 0.0)
            areas.append(local_frac(local_frac(ab) + local_frac(cd)))

        if t0 == breaks[i_breaks]:
            area = local_frac(np.sum(areas)) if len(areas) > 0 else 0.0
        else:
            row = min(i_breaks, coefs.shape[0] - 1)
            delta = t0 - breaks[row]
            ab_n = coefs[row, 0] * shift * (delta**2 - 0.0)
            cd_n = coefs[row, 1] * shift * (delta - 0.0)
            area = local_frac(
                (local_frac(np.sum(areas)) if len(areas) > 0 else 0.0) + local_frac(ab_n) + local_frac(cd_n)
            )

        mod[c] = area

    if np.asarray(t).ndim == 0:
        return float(mod[0])
    return mod


def ppval_linear(breaks, coefs, t):
    breaks = np.asarray(breaks, dtype=float).reshape(-1)
    coefs = np.asarray(coefs, dtype=float)
    t_values = np.atleast_1d(np.asarray(t, dtype=float))
    out = np.zeros(t_values.shape, dtype=float)

    for idx, t0 in enumerate(t_values):
        interval = np.searchsorted(breaks, t0, side='right') - 1
        interval = max(0, min(interval, coefs.shape[0] - 1))
        delta = t0 - breaks[interval]
        out[idx] = coefs[interval, 0] * delta + coefs[interval, 1]

    if np.asarray(t).ndim == 0:
        return float(out[0])
    return out


def rotate_grad_list(rotation, grads, system):
    present = [grad for grad in grads if grad is not None]
    if len(present) == 0:
        return list(grads)

    rotated = rotate_3d(rotation, *present, system=system)
    grads_out = [None, None, None]
    for grad in rotated:
        if hasattr(grad, 'type') and grad.type in ['grad', 'trap'] and hasattr(grad, 'channel'):
            grads_out['xyz'.index(grad.channel)] = grad
    return grads_out
