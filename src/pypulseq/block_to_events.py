from types import SimpleNamespace
from typing import Tuple


def block_to_events(*args: SimpleNamespace | float) -> Tuple[SimpleNamespace, ...]:
    """
    Converts `args` from a block to a list of events. If `args` is already a list of event(s), returns it unmodified.

    Parameters
    ----------
    args : SimpleNamespace
        Block to be flattened into a list of events.

    Returns
    -------
    events : list[SimpleNamespace]
        List of events comprising `args` if it was a block, otherwise `args` unmodified.
    """
    items = tuple(args)

    # MATLAB parity: strip away nested 1x1 cell wrappers.
    while (
        len(items) == 1
        and isinstance(items[0], (list, tuple))
        and len(items[0]) == 1
        and isinstance(items[0][0], (list, tuple))
    ):
        items = tuple(items[0])

    if len(items) == 0:
        return tuple()

    first = items[0]
    if hasattr(first, 'rf'):
        if len(items) != 1:
            raise ValueError('Only a single block structure can be added.')

        # MATLAB parity with +mr/block2events.m:
        # struct2cell(first), then remove empties.
        events = []
        for value in vars(first).values():
            if value is None:
                continue
            if isinstance(value, (list, tuple)):
                events.extend(value)
            else:
                events.append(value)
        return tuple(events)

    if isinstance(first, (list, tuple)):
        return tuple(first)

    return tuple(items)
