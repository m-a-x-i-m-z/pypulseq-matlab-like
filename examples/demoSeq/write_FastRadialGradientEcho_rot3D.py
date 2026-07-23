
import numpy as np
import sys
import os

from scipy.spatial.transform import Rotation as R

# Add pypulseq source to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from pypulseq_matlab_like.Sequence.sequence import Sequence
from pypulseq_matlab_like.opts import Opts
from pypulseq_matlab_like.make_trapezoid import make_trapezoid
from pypulseq_matlab_like.make_adc import make_adc
from pypulseq_matlab_like.make_sinc_pulse import make_sinc_pulse
from pypulseq_matlab_like.make_delay import make_delay
from pypulseq_matlab_like.add_gradients import add_gradients
from pypulseq_matlab_like.calc_duration import calc_duration
from pypulseq_matlab_like.rotate_3d import rotate_3d
from pypulseq_matlab_like.align import align

# System limits
system = Opts(max_grad=28, grad_unit='mT/m', max_slew=120, slew_unit='T/m/s',
              rf_ringdown_time=10e-6, rf_dead_time=100e-6, adc_dead_time=10e-6)

# Sequence object
seq = Sequence(system)  # Create a new sequence object
fov = 240e-3
Nx = 240
Ny = Nx
alpha = 5  # flip angle
slice_thickness = 6e-3  # slice
Nr = 256  # number of radial spokes
Ndummy = 10  # number of dummy scans
delta = np.pi / Nr  # angular increment
ro_dur = 1200e-6  # RO duration
ro_os = 2  # readout oversampling
ro_spoil = 0.5  # additional k-max excursion for RO spoiling
sl_spoil = 2  # spoil area compared to the slice thickness

rf_spoiling_inc = 84
rf_time_bw_product = 2

# Create alpha-degree slice selection pulse and gradient
rf, gz, gzReph = make_sinc_pulse(flip_angle=alpha * np.pi / 180, duration=400e-6,
                                 slice_thickness=slice_thickness, apodization=0.5, time_bw_product=rf_time_bw_product,
                                 system=system, use='excitation', return_gz=True)

gzReph.delay = calc_duration(gz)
gzComb = add_gradients(grads=[gz, gzReph], system=system)

# Define other gradients and ADC events
deltak = 1 / fov
gx = make_trapezoid(channel='x', amplitude=Nx * deltak / ro_dur,
                    flat_time=np.ceil(ro_dur / system.grad_raster_time) * system.grad_raster_time, system=system)
adc = make_adc(num_samples=Nx * ro_os, duration=ro_dur, delay=gx.rise_time, system=system)
gxPre = make_trapezoid(channel='x',
                       area=-gx.amplitude * (ro_dur / Nx / ro_os * (Nx * ro_os / 2 - 0.5) + 0.5 * gx.rise_time),
                       system=system)
#
gxPre, gzComb = align(right=[gxPre, gzComb])

addDelay = calc_duration(rf) - gxPre.delay
if addDelay > 0:
    gxPre.delay = gxPre.delay + np.ceil(addDelay / system.grad_raster_time) * system.grad_raster_time

# gradient spoiling
if sl_spoil > 0:
    sp_area_needed = sl_spoil / slice_thickness - gz.area / 2
    gzSpoil = make_trapezoid(channel='z', area=sp_area_needed, system=system, delay=gx.rise_time + gx.flat_time)
else:
    gzSpoil = None

if ro_spoil > 0:
    ro_add_time = np.ceil(((gx.area / Nx * (Nx / 2 + 1) * ro_spoil) / gx.amplitude) / system.grad_raster_time) * system.grad_raster_time
    gx.flat_time = gx.flat_time + ro_add_time

# start the sequence
rf_phase = 0
rf_inc = 0
TR = 0

def rotz(angle_rad):
    return R.from_euler('z', angle_rad).as_matrix()

for i in range(1 - Ndummy, Nr + 1):
    rf.phase_offset = rf_phase / 180 * np.pi
    adc.phase_offset = rf_phase / 180 * np.pi
    rf_inc = (rf_inc + rf_spoiling_inc) % 360.0
    rf_phase = (rf_phase + rf_inc) % 360.0

    phi = delta * (i - 1)

    # rotate_3d returns a list of gradients. add_block accepts *args.
    # My rotz implementation using scipy expects radians. So I should pass phi directly.

    rot_mat = rotz(phi)

    seq.add_block(*rotate_3d(rot_mat, rf, gzComb, gxPre, system=system))

    if i > 0:
        if gzSpoil is not None:
             seq.add_block(*rotate_3d(rot_mat, gx, adc, gzSpoil, system=system))
        else:
             seq.add_block(*rotate_3d(rot_mat, gx, adc, system=system))
    else:
        if gzSpoil is not None:
             seq.add_block(*rotate_3d(rot_mat, gx, gzSpoil, system=system))
        else:
             seq.add_block(*rotate_3d(rot_mat, gx, system=system))

    if TR <= 0:
        TR = seq.duration()[0]

# Timing validation
ok, error_report = seq.check_timing()

if ok:
    print('Timing check passed successfully')
else:
    print('Timing check failed! Error listing follows:')
    print(error_report)

# export definitions
seq.set_definition('FOV', [fov, fov, slice_thickness])
seq.set_definition('Name', 'gre_rad')

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '../demoSeq_pypulseq_results')
output_path = os.path.join(RESULTS_DIR, 'FastRadialGradientEcho_rot3D_py.seq')
print(f"Writing to: {os.path.abspath(output_path)}")
# Export sequence
os.makedirs(RESULTS_DIR, exist_ok=True)
seq.write(output_path)
