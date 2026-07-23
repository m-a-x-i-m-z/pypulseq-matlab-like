from typing import Iterable, Union

import numpy as np


def convert(
    from_value: Union[float, Iterable],
    from_unit: str,
    gamma: float = 42.576e6,
    to_unit: str = str(),
) -> Union[float, Iterable]:
    """
    Converts gradient amplitude or slew rate from unit `from_unit` to unit `to_unit` with gyromagnetic ratio `gamma`.

    Parameters
    ----------
    from_value : float
        Gradient amplitude or slew rate to convert from.
    from_unit : str
        Unit of gradient amplitude or slew rate to convert from.
    to_unit : str, default=''
        Unit of gradient amplitude or slew rate to convert to.
    gamma : float, default=42.576e6
        Gyromagnetic ratio. Default is 42.576e6, for Hydrogen.

    Returns
    -------
    out : float
        Converted gradient amplitude or slew rate.

    Raises
    ------
    ValueError
        If an invalid `from_unit` is passed. Must be one of 'Hz/m', 'mT/m', or 'rad/ms/mm'.
        If an invalid `to_unit` is passed. Must be one of 'Hz/m/s', 'mT/m/ms', 'T/m/s', 'rad/ms/mm/ms'.
    """
    valid_b1_units = ['Hz', 'T', 'mT', 'uT']
    valid_grad_units = ['Hz/m', 'mT/m', 'rad/ms/mm']
    valid_slew_units = ['Hz/m/s', 'mT/m/ms', 'T/m/s', 'rad/ms/mm/ms']
    valid_units = valid_b1_units + valid_grad_units + valid_slew_units

    if from_unit not in valid_units:
        raise ValueError(
            "Invalid from_unit. Must be one of "
            "'Hz', 'T', 'mT', 'uT', 'Hz/m', 'mT/m', 'rad/ms/mm', "
            "'Hz/m/s', 'mT/m/ms', 'T/m/s', 'rad/ms/mm/ms'."
        )

    if to_unit != '' and to_unit not in valid_units:
        raise ValueError(
            "Invalid to_unit. Must be one of "
            "'Hz', 'T', 'mT', 'uT', 'Hz/m', 'mT/m', 'rad/ms/mm', "
            "'Hz/m/s', 'mT/m/ms', 'T/m/s', 'rad/ms/mm/ms'."
        )

    if to_unit == '':
        if from_unit in valid_b1_units:
            to_unit = valid_b1_units[0]
        elif from_unit in valid_grad_units:
            to_unit = valid_grad_units[0]
        elif from_unit in valid_slew_units:
            to_unit = valid_slew_units[0]

    def _unit_category(unit: str) -> str:
        if unit in valid_b1_units:
            return 'B1'
        if unit in valid_grad_units:
            return 'gradient'
        if unit in valid_slew_units:
            return 'slew rate'
        raise ValueError(f'Invalid unit {unit}')

    from_category = _unit_category(from_unit)
    to_category = _unit_category(to_unit)
    if from_category != to_category:
        raise ValueError(
            f"from_unit '{from_unit}' ({from_category}) and to_unit '{to_unit}' ({to_category}) "
            'are in different unit categories.'
        )

    from_value = np.asarray(from_value)

    # Convert to standard units
    if from_unit in ['Hz', 'Hz/m', 'Hz/m/s']:
        standard = from_value
    elif from_unit in ['mT', 'mT/m']:
        standard = from_value * 1e-3 * gamma
    elif from_unit == 'uT':
        standard = from_value * 1e-6 * gamma
    elif from_unit == 'rad/ms/mm':
        standard = from_value * 1e6 / (2 * np.pi)
    elif from_unit in ['T', 'mT/m/ms', 'T/m/s']:
        standard = from_value * gamma
    elif from_unit == 'rad/ms/mm/ms':
        standard = from_value * 1e9 / (2 * np.pi)

    # Convert from standard units
    if to_unit in ['Hz', 'Hz/m', 'Hz/m/s']:
        out = standard
    elif to_unit in ['mT', 'mT/m']:
        out = 1e3 * standard / gamma
    elif to_unit == 'uT':
        out = 1e6 * standard / gamma
    elif to_unit == 'rad/ms/mm':
        out = standard * 2 * np.pi * 1e-6
    elif to_unit in ['T', 'mT/m/ms', 'T/m/s']:
        out = standard / gamma
    elif to_unit == 'rad/ms/mm/ms':
        out = standard * 2 * np.pi * 1e-9

    if np.isscalar(from_value) or from_value.ndim == 0:
        return float(out)
    return out
