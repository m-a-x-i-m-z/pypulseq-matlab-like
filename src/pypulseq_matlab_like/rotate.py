from types import SimpleNamespace
from typing import List, Union

import numpy as np

from pypulseq_matlab_like.add_gradients import add_gradients
from pypulseq_matlab_like.block_to_events import block_to_events
from pypulseq_matlab_like.opts import Opts
from pypulseq_matlab_like.scale_grad import scale_grad
from pypulseq_matlab_like.utils.event_helpers import copy_without_id
from pypulseq_matlab_like.utils.tracing import trace, trace_enabled


def __get_grad_abs_mag(grad: SimpleNamespace) -> np.ndarray:
    if grad.type == 'trap':
        return abs(grad.amplitude)
    return np.max(np.abs(grad.waveform))

def rotate(*args: SimpleNamespace, angle: float, axis: str, system: Union[Opts, None] = None) -> List[SimpleNamespace]:
    """
    Rotates the corresponding gradient(s) about the given axis by the specified amount. Gradients parallel to the
    rotation axis and non-gradient(s) are not affected. Possible rotation axes are 'x', 'y' or 'z'.

    See also `pypulseq.Sequence.sequence.add_block()`.

    Parameters
    ----------
    axis : str
        Axis about which the gradient(s) will be rotated.
    angle : float
        Angle by which the gradient(s) will be rotated.
    args : SimpleNamespace
        Gradient(s).

    Returns
    -------
    rotated_grads : [SimpleNamespace]
        Rotated gradient(s).
    """
    if system is None:
        system = Opts.default

    if not np.isscalar(angle):
        raise ValueError('Rotation angle needs to be a scalar value.')

    axes = ['x', 'y', 'z']
    if axis not in axes:
        raise ValueError('Incorrect axes specification.')

    va: Union[List[SimpleNamespace], tuple]
    if len(args) == 1 and hasattr(args[0], 'rf'):
        va = block_to_events(*args)
    elif len(args) == 1 and isinstance(args[0], (list, tuple)):
        va = args[0]
    else:
        va = args

    # Cycle through the objects and rotate gradients non-parallel to the given rotation axis. Rotated gradients
    # assigned to the same axis are then added together.

    # First create indexes of the objects to be bypassed or rotated
    i_rotate1 = []
    i_rotate2 = []
    i_bypass = []

    axes.remove(axis)
    axes_to_rotate = axes
    if axis == 'y':
        # Preserve rotation handedness to match MATLAB behavior
        axes_to_rotate = axes_to_rotate[::-1]
    if len(axes_to_rotate) != 2:
        raise ValueError('Incorrect axes specification.')

    for i in range(len(va)):
        event = va[i]

        if (event.type != 'grad' and event.type != 'trap') or event.channel == axis:
            i_bypass.append(i)
        else:
            if event.channel == axes_to_rotate[0]:
                i_rotate1.append(i)
            else:
                if event.channel == axes_to_rotate[1]:
                    i_rotate2.append(i)
                else:
                    i_bypass.append(i)  # Should never happen

    # Now every gradient to be rotated generates two new gradients: one on the original axis and one on the other from
    # the axes_to_rotate list
    rotated1 = []
    rotated2 = []
    max_mag = 0  # Measure of relevant amplitude
    for i in range(len(i_rotate1)):
        g = copy_without_id(va[i_rotate1[i]])
        max_mag = max(max_mag, __get_grad_abs_mag(g))
        rotated1.append(scale_grad(grad=g, scale=np.cos(angle)))
        g = scale_grad(grad=g, scale=np.sin(angle))
        g.channel = axes_to_rotate[1]
        rotated2.append(g)

    for i in range(len(i_rotate2)):
        g = copy_without_id(va[i_rotate2[i]])
        max_mag = max(max_mag, __get_grad_abs_mag(g))
        rotated2.append(scale_grad(grad=g, scale=np.cos(angle)))
        g = scale_grad(grad=g, scale=-np.sin(angle))
        g.channel = axes_to_rotate[0]
        rotated1.append(g)

    # Eliminate zero-amplitude gradients
    threshold = 1e-6 * max_mag
    for i in range(len(rotated1) - 1, -1, -1):
        if __get_grad_abs_mag(rotated1[i]) < threshold:
            rotated1.pop(i)
    for i in range(len(rotated2) - 1, -1, -1):
        if __get_grad_abs_mag(rotated2[i]) < threshold:
            rotated2.pop(i)

    # Add gradients on the corresponding axis together
    g = []
    if len(rotated1) > 1:
        g.append(add_gradients(grads=rotated1, system=system))
    elif len(rotated1) == 1:
        g.append(rotated1[0])

    if len(rotated2) > 1:
        g.append(add_gradients(grads=rotated2, system=system))
    elif len(rotated2) == 1:
        g.append(rotated2[0])

    # Eliminate zero amplitude gradients
    for i in range(len(g) - 1, -1, -1):
        if __get_grad_abs_mag(g[i]) < threshold:
            g.pop(i)

    # Export
    bypass = [va[i] for i in i_bypass]
    rotated_grads = [*bypass, *g]

    if trace_enabled():
        for grad in rotated_grads:
            grad.trace = trace()

    return rotated_grads
