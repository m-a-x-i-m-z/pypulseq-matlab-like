
import numpy as np
import sys
import os

# Add pypulseq source to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from pypulseq_matlab_like.Sequence.sequence import Sequence
from pypulseq_matlab_like.opts import Opts
from pypulseq_matlab_like.make_trapezoid import make_trapezoid
from pypulseq_matlab_like.make_adc import make_adc
from pypulseq_matlab_like.make_sinc_pulse import make_sinc_pulse
from pypulseq_matlab_like.make_delay import make_delay
from pypulseq_matlab_like.make_extended_trapezoid import make_extended_trapezoid
from pypulseq_matlab_like.make_extended_trapezoid_area import make_extended_trapezoid_area
from pypulseq_matlab_like.add_gradients import add_gradients
from pypulseq_matlab_like.split_gradient_at import split_gradient_at
from pypulseq_matlab_like.align import align
from pypulseq_matlab_like.calc_duration import calc_duration
from pypulseq_matlab_like.rotate import rotate

# set system limits (slew rate 130 and max_grad 30 work on Prisma)
# System limits
system = Opts(max_grad=28, grad_unit='mT/m', max_slew=120, slew_unit='T/m/s',
              rf_ringdown_time=10e-6, rf_dead_time=100e-6, adc_dead_time=10e-6)

# Sequence object
seq = Sequence(system)  # Create a new sequence object
fov = 240e-3
Nx = 240
Ny = Nx  # Define FOV and resolution
alpha = 5  # flip angle
slice_thickness = 6e-3  # slice
Nr = 256  # number of radial spokes
Ndummy = 10  # number of dummy scans
delta = np.pi / Nr  # angular increment; try golden angle pi*(3-5^0.5) or 0.5 of it
ro_dur = 1200e-6  # RO duration
ro_os = 2  # readout oversampling
ro_spoil = 0.5  # additional k-max excursion for RO spoiling
sl_spoil = 1.5  # adjusted slightly if needed

# TE & TR are as short as possible derived from the above parameters and
# the system specs below

# more in-depth parameters
rf_spoiling_inc = 84  # RF spoiling increment
rf_time_bw_prod = 2  # time-bandwidth product for the RF pulse

# Create alpha-degree slice selection pulse and gradient
rf, gz, gzReph = make_sinc_pulse(flip_angle=alpha * np.pi / 180, duration=400e-6,
                                 slice_thickness=slice_thickness, apodization=0.5, time_bw_product=rf_time_bw_prod,
                                 system=system, use='excitation', return_gz=True)

# gradient spoiling in slice direction
if sl_spoil > 0:
    sp_area_needed = sl_spoil / slice_thickness * rf_time_bw_prod - gz.flat_area / 2
    gzSpoil, _, _ = make_extended_trapezoid_area(
        area=sp_area_needed,
        channel='z',
        grad_start=0,
        grad_end=gz.amplitude,
        system=system,
    )
    gz = make_extended_trapezoid(
        channel='z',
        times=np.array([0, gz.flat_time, gz.flat_time + gz.fall_time]),
        amplitudes=np.array([gz.amplitude, gz.amplitude, 0]),
        system=system,
    )
    gz.delay = gzSpoil.shape_dur
    gz = add_gradients(grads=[gz, gzSpoil], system=system)
    rf.delay = max(gzSpoil.shape_dur, system.rf_dead_time)
    gz.delay = rf.delay - gzSpoil.shape_dur

# join rephaser
gzReph.delay = calc_duration(gz)
gzComb = add_gradients(grads=[gz, gzReph], system=system)
gzSpoil, gz = split_gradient_at(grad=gzComb, time_point=rf.delay - system.rf_dead_time, system=system)
gz.delay = gz.delay - calc_duration(gzSpoil)
rf.delay = system.rf_dead_time

# Define other gradients and ADC events
deltak = 1 / fov
gx = make_trapezoid(channel='x', amplitude=Nx * deltak / ro_dur,
                    flat_time=np.ceil(ro_dur / system.grad_raster_time) * system.grad_raster_time, system=system)
adc = make_adc(num_samples=Nx * ro_os, duration=ro_dur, delay=gx.rise_time, system=system)
gxPre = make_trapezoid(channel='x',
                       area=-gx.amplitude * (ro_dur / Nx / ro_os * (Nx * ro_os / 2 - 0.5) + 0.5 * gx.rise_time),
                       system=system)  # 0.5 is necessary to acount for the Siemens sampling in the center of the dwell periods
#
gxPre, gz = align(right=[gxPre, gz])
addDelay = calc_duration(rf) - gxPre.delay
if addDelay > 0:
    gxPre.delay = gxPre.delay + np.ceil(addDelay / system.grad_raster_time) * system.grad_raster_time

# gradient spoiling in slice direction
if ro_spoil > 0:
    # ro_spoil_area=(gx.area-gx.flatArea)/2;
    ro_add_time = np.ceil(((gx.area / Nx * (Nx / 2 + 1) * ro_spoil) / gx.amplitude) / system.grad_raster_time) * system.grad_raster_time
    gx.flat_time = gx.flat_time + ro_add_time  # careful, areas stored in the object are now wrong

# join slice spoiler with the slice selection
# if (rf.delay>mr.calcDuration()) no, this does not work to be really optimal we need a new function with start, stop and area
# could be done with mr.makeExtendedTrapezoidArea()

# Calculate timing

# start the sequence
rf_phase = 0
rf_inc = 0
TR = 0
seq.add_block(gzSpoil)
gzSpoil, _, _ = align(right=gzSpoil, left=gx, center=make_delay(np.ceil((adc.delay + adc.dwell * adc.num_samples) / system.grad_raster_time) * system.grad_raster_time + gzSpoil.shape_dur))

for i in range(1 - Ndummy, Nr + 1):
    rf.phase_offset = rf_phase / 180 * np.pi
    adc.phase_offset = rf_phase / 180 * np.pi
    rf_inc = (rf_inc + rf_spoiling_inc) % 360.0
    rf_phase = (rf_phase + rf_inc) % 360.0

    phi = delta * (i - 1)

    # seq.addBlock(mr.rotate('z',phi,rf,gz,gxPre));
    seq.add_block(*rotate(rf, gz, gxPre, angle=phi, axis='z', system=system))

    if i > 0:
        seq.add_block(*rotate(gx, adc, gzSpoil, angle=phi, axis='z', system=system))
    else:
        seq.add_block(*rotate(gx, gzSpoil, angle=phi, axis='z', system=system))

    if TR <= 0:
        TR = seq.duration()[0]

# dummy slice select to ramp down to 0 Z spoil gradient
seq.add_block(gz)

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
output_path = os.path.join(RESULTS_DIR, 'FastRadialGradientEcho_py.seq')
print(f"Writing to: {os.path.abspath(output_path)}")
# Export sequence
os.makedirs(RESULTS_DIR, exist_ok=True)
seq.write(output_path)
