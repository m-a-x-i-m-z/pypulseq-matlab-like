from types import SimpleNamespace
import numbers

import numpy as np
from scipy.spatial.transform import Rotation

from pypulseq_matlab_like.add_gradients import add_gradients
from pypulseq_matlab_like.block_to_events import block_to_events
from pypulseq_matlab_like.scale_grad import scale_grad
from pypulseq_matlab_like.utils.event_helpers import copy_without_id

def _get_grad_abs_mag(grad: SimpleNamespace) -> float:
    if grad.type == 'trap':
        return abs(grad.amplitude)
    return np.max(np.abs(grad.waveform))


def _quat_to_rot_mat(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float).flatten()
    if q.size != 4:
        raise ValueError('Quaternion must have 4 components [w, x, y, z].')
    # scipy uses [x, y, z, w] ordering --- reorder to match our matlab pulseq [w, x, y, z] convention.
    return Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()


def _rotation_to_matrix(rotation) -> np.ndarray:
    arr = np.asarray(rotation, dtype=float)
    if arr.shape == (3, 3):
        return arr

    flat = arr.flatten()
    if flat.size == 4:
        return _quat_to_rot_mat(flat)

    if flat.size == 2:
        phi = float(flat[0])
        theta = float(flat[1])
        if not (-np.pi <= phi < 2 * np.pi):
            raise ValueError(f'rotation angle phi ({phi:.2f}) is invalid. should be within [-pi,2*pi] radians')
        if not (-np.pi <= theta <= np.pi):
            raise ValueError(f'rotation angle theta ({theta:.2f}) is invalid. should be within [-pi,pi] radians')
        r_y = Rotation.from_rotvec([0.0, theta, 0.0])
        r_z = Rotation.from_rotvec([0.0, 0.0, phi])
        # MATLAB uses q = qz * qy.
        return (r_z * r_y).as_matrix()

    if flat.size == 1:
        phi = float(flat[0])
        if not (-np.pi <= phi < 2 * np.pi):
            raise ValueError(f'rotation angle phi ({phi:.2f}) is invalid. should be within [-pi,2*pi] radians')
        return Rotation.from_rotvec([0.0, 0.0, phi]).as_matrix()

    raise ValueError("The parameter 'rotation' must either bi a 3x3 matrix or a quaternion")


def _is_valid_system_obj(system_obj) -> bool:
    if system_obj is None:
        return False
    if isinstance(system_obj, dict):
        return 'grad_raster_time' in system_obj
    return hasattr(system_obj, 'grad_raster_time')


def _normalize_system_obj(system_obj):
    if system_obj is None:
        return None

    if isinstance(system_obj, dict):
        return SimpleNamespace(**system_obj)

    # object-like input
    if hasattr(system_obj, '__dict__'):
        dst = SimpleNamespace(**vars(system_obj))
    else:
        dst = system_obj

    return dst


def _parse_system_from_args(args, system_kw):
    events = list(args)
    parsed_system = system_kw

    # MATLAB-compatible: optional positional 'system', sys can be at the beginning...
    if len(events) >= 2 and isinstance(events[0], str) and events[0].lower() == 'system':
        if not _is_valid_system_obj(events[1]):
            raise ValueError("Error parsing input parameters, keyword 'system' is not followed by a valid system struct")
        if parsed_system is not None:
            raise ValueError("System provided both positionally and as keyword argument.")
        parsed_system = events[1]
        events = events[2:]
    # ...or at the end.
    elif len(events) >= 2 and isinstance(events[-2], str) and events[-2].lower() == 'system':
        if not _is_valid_system_obj(events[-1]):
            raise ValueError("Error parsing input parameters, keyword 'system' is not followed by a valid system struct")
        if parsed_system is not None:
            raise ValueError("System provided both positionally and as keyword argument.")
        parsed_system = events[-1]
        events = events[:-2]

    return tuple(events), _normalize_system_obj(parsed_system)


def rotate_3d(rotation, *args, system=None):
    """
    Rotate gradient events by a 3x3 matrix, quaternion, or Euler-style angle input.

    Non-gradient events (e.g. RF, ADC, delays, numeric placeholders) are passed
    through unchanged.

    Parameters
    ----------
    rotation : array_like or list
        Rotation specification. Supported forms are:
        - 3x3 rotation matrix
        - Quaternion [w, x, y, z] (scalar first)
        - Single angle [phi] for z-axis rotation
        - Two angles [phi, theta] for Rz(phi) * Ry(theta)
    args : SimpleNamespace or block-like containers
        Input events to rotate. Gradient events ('grad', 'trap') are rotated.
        Other events are returned unchanged.
    system : Opts or dict-like, optional
        System limits used by ``add_gradients``. This argument is optional.
        It can be provided as keyword ``system=...`` or MATLAB-style positional
        pair ``'system', system_obj``.

    Returns
    -------
    rotated_grads : list
        List of events including unchanged bypass events and rotated gradients.

    Notes
    -----
    - At most one gradient per axis ('x', 'y', 'z') is accepted.
    - If ``system`` is omitted, gradient accumulation uses
      ``add_gradients([...])`` default behavior.

    Examples
    --------
    >>> import numpy as np
    >>> import pypulseq_matlab_like as pp
    >>> from pypulseq_matlab_like.rotate_3d import rotate_3d
    >>>
    >>> system = pp.Opts()
    >>> gx = pp.make_trapezoid(channel='x', system=system, amplitude=100, duration=2e-3)
    >>> gy = pp.make_trapezoid(channel='y', system=system, amplitude=50, duration=2e-3)
    >>> gz = pp.make_trapezoid(channel='z', system=system, amplitude=20, duration=2e-3)
    >>> rf = pp.make_block_pulse(flip_angle=np.pi / 2, duration=1e-3, system=system)
    >>> adc = pp.make_adc(num_samples=64, dwell=10e-6, delay=0, system=system)
    >>>
    >>> angle = np.pi / 4
    >>> r_z = np.array([
    ...     [np.cos(angle), -np.sin(angle), 0],
    ...     [np.sin(angle),  np.cos(angle), 0],
    ...     [0,              0,             1],
    ... ])
    >>>
    >>> q = np.array([np.cos(angle / 2), 0.0, 0.0, np.sin(angle / 2)], dtype=float)
    >>>
    >>> out1 = rotate_3d(r_z, gx, gy, gz, rf, adc, system=system)
    >>> out2 = rotate_3d(q, gx, gy, gz, rf, adc, system=system)
    >>> out3 = rotate_3d([angle], gx, gy, gz, system=system)
    >>> out4 = rotate_3d([angle, np.pi / 6], gx, gy, gz, system=system)
    >>> out5 = rotate_3d(r_z, gx, gy, 'system', system)
    """
    rot_mat = _rotation_to_matrix(rotation)

    events_in, system = _parse_system_from_args(args, system)

    if len(events_in) == 0:
        return []

    # MATLAB block2events-style conversion for blocks and nested containers.
    events = list(block_to_events(*events_in))

    # Filter gradients
    grads3_in = [None, None, None]  # x, y, z
    ibypass = []
    rotation_axes = ['x', 'y', 'z']

    for i, event in enumerate(events):
        if event is None:
            continue
        if isinstance(event, numbers.Number):
            ibypass.append(i)
            continue

        if hasattr(event, 'type') and event.type in ['grad', 'trap']:
            if not hasattr(event, 'channel'):
                raise ValueError('Gradient event has no channel field.')
            if event.channel in rotation_axes:
                axis_idx = rotation_axes.index(event.channel)
                if grads3_in[axis_idx] is not None:
                    raise ValueError(f"More than one gradient on the same axis {event.channel} provided")
                grads3_in[axis_idx] = copy_without_id(event, deep=True)
            else:
                raise ValueError(f"Invalid gradient channel '{event.channel}'. Expected one of x, y, z.")
        else:
            ibypass.append(i)

    max_mag = 0
    for g in grads3_in:
        if g is not None:
            max_mag = max(max_mag, _get_grad_abs_mag(g))
            
    fthresh = 1e-6
    thresh = fthresh * max_mag
    
    grads_out = []
    
    for j in range(3):  # For each output axis (x, y, z)
        grad_out_curr = None
        for i in range(3):  # For each input axis
            if grads3_in[i] is None or abs(rot_mat[j, i]) < fthresh:
                continue

            # Scale gradient
            g = scale_grad(grads3_in[i], rot_mat[j, i])
            g.channel = rotation_axes[j]

            if grad_out_curr is None:
                grad_out_curr = g
            else:
                if system is None:
                    grad_out_curr = add_gradients([grad_out_curr, g])
                else:
                    grad_out_curr = add_gradients([grad_out_curr, g], system=system)

        if grad_out_curr is not None and _get_grad_abs_mag(grad_out_curr) >= thresh:
            grads_out.append(grad_out_curr)

    # Reassemble output
    bypass_events = [events[i] for i in ibypass]
    return bypass_events + grads_out
