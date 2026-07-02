from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation


def make_rotation(*args) -> SimpleNamespace:
    """
    Create a rotation extension event (`type='rot3D'`).

    The returned object stores a unit quaternion in scalar-first format
    `[w, x, y, z]` inside `rot_quaternion`. The extension can then be attached
    to a sequence block, for example:

    `seq.add_block(gx, adc, make_rotation(phi))`

    Parameters
    ----------
    *args : tuple
        MATLAB-style call forms:
        - `make_rotation(phi)` for z-axis rotation
        - `make_rotation(phi, theta)` for `Rz(phi) * Ry(theta)`
        - `make_rotation(axis, angle)` for axis-angle rotation
        - `make_rotation(quaternion)` with `[w, x, y, z]`
        - `make_rotation(rot_mat)` with a 3x3 matrix
        - `make_rotation(rot_mats)` with an Nx3x3 matrix stack
    Returns
    -------
    rot : SimpleNamespace
        Rotation extension with:
        - `rot.type == 'rot3D'`
        - `rot.rot_quaternion` as normalized `[w, x, y, z]`

    Notes
    -----
    - Quaternion input is normalized before being stored.
    - Angle ranges follow MATLAB parity checks implemented in this function.
    - When using block extensions, only one rotation extension per block is
      allowed.

    Examples
    --------
    >>> import numpy as np
    >>> import pypulseq as pp
    >>> from pypulseq.make_rotation import make_rotation
    >>>
    >>> # 1) z-axis rotation by phi
    >>> phi = np.pi / 6
    >>> rot_z = make_rotation(phi)
    >>> rot_z.type
    'rot3D'
    >>>
    >>> # 2) combined Rz(phi) * Ry(theta)
    >>> theta = np.pi / 12
    >>> rot_zy = make_rotation(phi, theta)
    >>>
    >>> # 3) axis-angle input: rotate by phi about axis v
    >>> v = np.array([1.0, 1.0, 0.0])
    >>> rot_axis = make_rotation(v, np.pi / 4)
    >>>
    >>> # 4) quaternion input [w, x, y, z]
    >>> q = np.array([np.cos(phi / 2), 0.0, 0.0, np.sin(phi / 2)], dtype=float)
    >>> rot_q = make_rotation(q)
    >>>
    >>> # 5) matrix input
    >>> r_z = np.array([
    ...     [np.cos(phi), -np.sin(phi), 0.0],
    ...     [np.sin(phi),  np.cos(phi), 0.0],
    ...     [0.0,          0.0,         1.0],
    ... ])
    >>> rot_m = make_rotation(r_z)
    >>>
    >>> # 6) Attach extension to a block
    >>> system = pp.Opts()
    >>> seq = pp.Sequence(system)
    >>> gx = pp.make_trapezoid(channel='x', area=1, duration=1e-3, system=system)
    >>> seq.add_block(gx, rot_z)
    """
    if len(args) < 1:
        raise ValueError('make_rotation - invalid arguments: must supply rotation parameter(s).')

    first = args[0]
    arr = np.asarray(first, dtype=float)
    flat = arr.flatten()

    rot = SimpleNamespace()
    rot.type = 'rot3D'

    if arr.ndim == 3 and arr.shape[1:] == (3, 3):
        if len(args) != 1:
            raise ValueError('make_rotation(rot_mats) accepts exactly one argument.')
        q_xyzw = Rotation.from_matrix(np.ascontiguousarray(arr, dtype=np.float64)).as_quat()
        rot_events = []
        for q in q_xyzw:
            rot_i = SimpleNamespace()
            rot_i.type = 'rot3D'
            rot_i.rot_quaternion = np.array([q[3], q[0], q[1], q[2]], dtype=float)
            rot_events.append(rot_i)
        return rot_events

    if arr.shape == (3, 3):
        if len(args) != 1:
            raise ValueError('make_rotation(rot_mat) accepts exactly one argument.')
        q_xyzw = Rotation.from_matrix(np.ascontiguousarray(arr, dtype=np.float64)).as_quat()
        rot.rot_quaternion = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=float)
        return rot

    if flat.size == 1:
        if len(args) > 2:
            raise ValueError('make_rotation(phi) or make_rotation(phi, theta) expected.')
        phi = float(flat[0])
        theta = 0.0 if len(args) == 1 else float(args[1])
        if not (-np.pi <= phi < 2 * np.pi):
            raise ValueError(f'rotation angle phi ({phi:.2f}) is invalid. should be within [-pi,2*pi) radians')
        if not (-np.pi <= theta <= np.pi):
            raise ValueError(f'rotation angle theta ({theta:.2f}) is invalid. should be within [-pi,pi] radians')

        # MATLAB: q = qz * qy.
        r_z = Rotation.from_rotvec([0.0, 0.0, phi])
        r_y = Rotation.from_rotvec([0.0, theta, 0.0])
        q_xyzw = (r_z * r_y).as_quat()
        rot.rot_quaternion = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=float)
        return rot

    if flat.size == 3:
        if len(args) != 2:
            raise ValueError('make_rotation(axis, angle) expected for 3-element axis input.')
        v = flat.astype(float)
        norm_v = np.linalg.norm(v)
        if norm_v == 0:
            raise ValueError('rotation axis vector norm must be non-zero.')
        phi = float(args[1])
        if not (abs(phi) <= np.pi):
            raise ValueError(f'rotation angle phi ({phi:.2f}) is invalid. should be within [0,pi] radians')
        v = v / norm_v
        rot.rot_quaternion = np.array([np.cos(phi / 2.0), *(np.sin(phi / 2.0) * v)], dtype=float)
        return rot

    if flat.size == 4:
        if len(args) != 1:
            raise ValueError('make_rotation(quaternion) accepts exactly one argument.')
        q = flat.astype(float)
        n = np.linalg.norm(q)
        if n == 0:
            raise ValueError('Quaternion norm must be non-zero.')
        rot.rot_quaternion = q / n
        return rot

    raise ValueError('unexpected input to make_rotation')
