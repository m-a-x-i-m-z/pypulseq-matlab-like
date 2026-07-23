
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
from pypulseq_matlab_like.calc_duration import calc_duration

# step
# 0 ... Basic sequence
# 1 ... Add spoiler in read, phase and slice (vary spoiler - line 49)
# 2 ... Refocus in phase
# 3 ... Vary RF phase quasi-randomly
# 4 ... Make receiver phase follow transmitter phase
# 5 ... Add dummy scans
step = 0

# Define FOV and resolution
fov = 256e-3
sliceThickness = 5e-3
Nx = 128
Ny = Nx

# Define sequence parameters
TE = 8e-3
TR = 16e-3
alpha = 30

# System limits
system = Opts(max_grad=20, grad_unit='mT/m', max_slew=120, slew_unit='T/m/s',
              rf_ringdown_time=20e-6, rf_dead_time=100e-6)
sys_default = Opts()

# Create a new sequence object
# Sequence object
seq = Sequence(system)

# Create slice selective alpha-pulse and corresponding gradients
rf, gz, gzReph = make_sinc_pulse(flip_angle=alpha * np.pi / 180, duration=4e-3,
                                 slice_thickness=sliceThickness, apodization=0.5, time_bw_product=4,
                                 system=system, use='excitation', return_gz=True)

# Define other gradients and ADC events
deltak = 1 / fov  # Pulseq toolbox defaults to k-space units of m^-1
gx = make_trapezoid(channel='x', flat_area=Nx * deltak, flat_time=6.4e-3, system=sys_default)
adc = make_adc(num_samples=Nx, duration=gx.flat_time, delay=gx.rise_time, system=system)
gxPre = make_trapezoid(channel='x', area=-gx.area / 2, duration=2e-3, system=sys_default)
phaseAreas = ((np.arange(Ny)) - Ny / 2) * deltak

# Calculate timing
delayTE = np.round((TE - calc_duration(gxPre) - calc_duration(gz) / 2
                    - calc_duration(gx) / 2) / seq.grad_raster_time) * seq.grad_raster_time
delayTR = np.round((TR - calc_duration(gxPre) - calc_duration(gz)
                    - calc_duration(gx) - delayTE) / seq.grad_raster_time) * seq.grad_raster_time

if step > 0:
    spoilArea = 4 * gx.area  # 4 "looks" good
    # Add spoilers in read, refocus in phase and spoiler in slice
    gxPost = make_trapezoid(channel='x', area=spoilArea, system=system)  # we pass 'system' here to calculate shortest time gradient
    gyPost = make_trapezoid(channel='y', area=spoilArea, system=system)
    gzPost = make_trapezoid(channel='z', area=spoilArea, system=system)

if step > 1:
    gyPost = make_trapezoid(channel='y', area=-np.max(phaseAreas), duration=2e-3, system=system)

if step > 0:
    delayTR = delayTR - calc_duration(gxPost, gyPost, gzPost)

if step > 4:
    start = -30  # dummy scans
else:
    start = 0

# Loop over phase encodes and define sequence blocks
for i in range(start, Ny):
    if step > 2:
        # Vary RF phase quasi-randomly
        idx = i + 1
        rand_phase = (84 * (idx**2 + idx + 2) % 360) * np.pi / 180
        rf, gz, _ = make_sinc_pulse(flip_angle=alpha * np.pi / 180, duration=4e-3,
                                 slice_thickness=5e-3,
                                 apodization=0.5,
                                 time_bw_product=4,
                                 system=system,
                                 phase_offset=rand_phase,
                                 use='excitation', return_gz=True) # Re-create pulse with phase offset

    seq.add_block(rf, gz)

    if i >= 0:
        gyPre = make_trapezoid(channel='y', area=phaseAreas[i], duration=2e-3, system=sys_default)
    else:
        gyPre = make_trapezoid(channel='y', area=0, duration=2e-3, system=sys_default)

    seq.add_block(gxPre, gyPre, gzReph)
    seq.add_block(make_delay(delayTE))

    if step > 3:
        # Make receiver phase follow transmitter phase
        adc.phase_offset = rand_phase

    if i >= 0:
        seq.add_block(gx, adc)
    else:
        seq.add_block(gx)

    if step > 1:
        gyPost = make_trapezoid(channel='y', area=-gyPre.area, duration=2e-3)

    if step > 0:
        # Add spoilers in read and slice and may be in phase
        seq.add_block(gxPost, gyPost, gzPost)

    seq.add_block(make_delay(delayTR))

# Timing validation
ok, error_report = seq.check_timing()

if ok:
    print('Timing check passed successfully')
else:
    print('Timing check failed! Error listing follows:')
    print(error_report)

seq.set_definition('FOV', [fov, fov, sliceThickness])
seq.set_definition('Name', 'DEMO_gre' + str(step))

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '../demoSeq_pypulseq_results')
output_path = os.path.join(RESULTS_DIR, 'GRE_live_demo_py.seq')
print(f"Writing to: {os.path.abspath(output_path)}")
# Export sequence
os.makedirs(RESULTS_DIR, exist_ok=True)
seq.write(output_path)
