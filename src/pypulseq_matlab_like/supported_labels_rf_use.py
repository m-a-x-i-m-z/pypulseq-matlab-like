from typing import List, Tuple
from warnings import warn

_SUPPORTED_LABELS: List[str] = [
    'SLC',
    'SEG',
    'REP',
    'AVG',
    'SET',
    'ECO',
    'PHS',
    'LIN',
    'PAR',
    'ACQ',
    'TRID',
    'NAV',
    'REV',
    'SMS',
    'REF',
    'IMA',
    'OFF',
    'NOISE',
    'PMC',
    'NOROT',
    'NOPOS',
    'NOSCL',
    'ONCE',
]


def get_supported_labels() -> Tuple[str, ...]:
    """
    Returns
    -------
    tuple
        Supported labels.
    """
    return tuple(_SUPPORTED_LABELS)


def add_supported_label(new_label: str) -> Tuple[str, ...]:
    """
    Register a custom label, matching MATLAB's `addCustomLabel.m`.
    """
    if not isinstance(new_label, str):
        raise ValueError('New label should be a string.')

    if new_label in _SUPPORTED_LABELS:
        warn(f'addCustomLabel: label {new_label} is already known', stacklevel=2)

    _SUPPORTED_LABELS.append(new_label)
    return tuple(_SUPPORTED_LABELS)


def get_supported_rf_uses(return_short_names: bool = False):
    """Return supported RF uses and their abbreviations when requested."""
    supported_rf_use = ('excitation', 'refocusing', 'inversion', 'saturation', 'preparation', 'other', 'undefined')
    if return_short_names:
        return supported_rf_use, ''.join(use[0] for use in supported_rf_use)
    return supported_rf_use