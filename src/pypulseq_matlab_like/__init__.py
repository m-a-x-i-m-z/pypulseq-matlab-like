import importlib.metadata
import math
import numpy as np

# =========
# VERSION
# =========
__version__ = "1.5.1"

# =========
# BANKER'S ROUNDING FIX
# =========
def round_half_up(n, decimals=0):
    """
    Avoid banker's rounding inconsistencies; from https://realpython.com/python-rounding/#rounding-half-up
    """
    multiplier = 10**decimals
    return math.floor(abs(n) * multiplier + 0.5) / multiplier


# =========
# EPSILON (Precision of floating point numbers)
# =========

# Instead of np.finfo(np.float64).eps, which was used before, we now try to estimate our precision based on the largest
# expected value for times, amplitudes etc (we choose 1E6) and consider another factor 10 for compounding of rounding errors.
# We then round the value to the closest power of 10.
eps = 10 ** np.floor(np.log10(np.spacing(1e6) * 10))  # this is 1e-9 for np.float64


# =========
# PACKAGE-LEVEL IMPORTS
# =========
from pypulseq_matlab_like.SAR.SAR_calc import calc_SAR
from pypulseq_matlab_like.Sequence.sequence import Sequence
from pypulseq_matlab_like.TransformFOV.transform_fov import transform_fov
from pypulseq_matlab_like.add_gradients import add_gradients
from pypulseq_matlab_like.add_ramps import add_ramps
from pypulseq_matlab_like.align import align
from pypulseq_matlab_like.block_to_events import block_to_events
from pypulseq_matlab_like.calc_duration import calc_duration
from pypulseq_matlab_like.calc_ramp import calc_ramp
from pypulseq_matlab_like.calc_rf_bandwidth import calc_rf_bandwidth
from pypulseq_matlab_like.calc_rf_center import calc_rf_center
from pypulseq_matlab_like.calc_rf_power import calc_rf_power
from pypulseq_matlab_like.convert import convert
from pypulseq_matlab_like.make_adc import make_adc, calc_adc_segments
from pypulseq_matlab_like.make_adiabatic_pulse import make_adiabatic_pulse
from pypulseq_matlab_like.make_arbitrary_grad import make_arbitrary_grad
from pypulseq_matlab_like.make_arbitrary_rf import make_arbitrary_rf
from pypulseq_matlab_like.make_block_pulse import make_block_pulse
from pypulseq_matlab_like.make_delay import make_delay
from pypulseq_matlab_like.make_soft_delay import make_soft_delay
from pypulseq_matlab_like.make_digital_output_pulse import make_digital_output_pulse
from pypulseq_matlab_like.make_extended_trapezoid import make_extended_trapezoid
from pypulseq_matlab_like.make_extended_trapezoid_area import make_extended_trapezoid_area
from pypulseq_matlab_like.make_gauss_pulse import make_gauss_pulse
from pypulseq_matlab_like.make_hexagon_gradient_area import make_hexagon_gradient_area
from pypulseq_matlab_like.make_label import make_label
from pypulseq_matlab_like.make_rf_shim import make_rf_shim
from pypulseq_matlab_like.make_rotation import make_rotation
from pypulseq_matlab_like.make_slr_pulse import make_slr_pulse
from pypulseq_matlab_like.make_sinc_pulse import make_sinc_pulse
from pypulseq_matlab_like.make_trapezoid import make_trapezoid
from pypulseq_matlab_like.sigpy_pulse_opts import SigpyPulseOpts
from pypulseq_matlab_like.make_trigger import make_trigger
from pypulseq_matlab_like.opts import Opts
from pypulseq_matlab_like.points_to_waveform import points_to_waveform
from pypulseq_matlab_like.rotate import rotate
from pypulseq_matlab_like.restore_additional_shape_samples import restore_additional_shape_samples
from pypulseq_matlab_like.scale_grad import scale_grad
from pypulseq_matlab_like.sim_rf import sim_rf
from pypulseq_matlab_like.split_gradient import split_gradient
from pypulseq_matlab_like.split_gradient_at import split_gradient_at
from pypulseq_matlab_like.supported_labels_rf_use import add_supported_label, get_supported_labels
from pypulseq_matlab_like.traj_to_grad import traj_to_grad
from pypulseq_matlab_like.utils.tracing import enable_trace, disable_trace
from pypulseq_matlab_like.verify_file_signature import verify_file_signature
