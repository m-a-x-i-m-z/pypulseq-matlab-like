import numpy as np
from types import SimpleNamespace
from typing import List


def make_rf_shim(shim_vector: List[complex], system=None) -> SimpleNamespace:
    """
    Create an RF shimming extension event (`type='rf_shim'`).

    The shim coefficients are stored as a complex column vector and can be attached
    to blocks that contain RF events, for example:

    `seq.add_block(rf, gz, make_rf_shim([1, np.exp(1j * phase)]))`

    Parameters
    ----------
    shim_vector : List[complex]
        Complex RF shim coefficients. Input is converted to a MATLAB-style
        column vector (`N x 1`) via reshape.
    system : Opts, optional
        Reserved for API compatibility. Unused.

    Returns
    -------
    rf_shim : SimpleNamespace
        RF shim extension with:
        - `rf_shim.type == 'rf_shim'`
        - `rf_shim.shim_vector`

    Notes
    -----
    - Vector layout follows MATLAB `makeRfShim(shimVec(:))`, i.e. always
      column-vector storage.
    - This helper does not enforce vector length; sequence-level validation is
      performed when writing/using the extension.
    - Use at most one RF shim extension per block.

    Examples
    --------
    >>> import numpy as np
    >>> import pypulseq as pp
    >>> from pypulseq.make_rf_shim import make_rf_shim
    >>>
    >>> # 1) Build standalone RF shim extension
    >>> shim = make_rf_shim([1.0 + 0j, 0.8 * np.exp(1j * np.pi / 3)])
    >>> shim.type
    'rf_shim'
    >>>
    >>> # 2) Attach RF shim to an RF block
    >>> system = pp.Opts()
    >>> seq = pp.Sequence(system)
    >>> rf, gz, _ = pp.make_sinc_pulse(
    ...     flip_angle=np.pi / 6,
    ...     duration=3e-3,
    ...     slice_thickness=5e-3,
    ...     apodization=0.5,
    ...     time_bw_product=4,
    ...     system=system,
    ...     use='excitation',
    ...     return_gz=True,
    ... )
    >>> seq.add_block(rf, gz, make_rf_shim([1.0, np.exp(1j * np.pi / 2)]))
    """
    rf_shim = SimpleNamespace()
    rf_shim.type = 'rf_shim'
    # MATLAB parity: shimVec(:) uses column-major linearization.
    shim_arr = np.reshape(np.array(shim_vector, dtype=complex), (-1, 1), order='F')
    rf_shim.shim_vector = shim_arr

    return rf_shim
