
import numpy as np
import sys
import os

# Add pypulseq source to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from pypulseq_matlab_like.Sequence.sequence import Sequence
from pypulseq_matlab_like.opts import Opts
from pypulseq_matlab_like.make_trapezoid import make_trapezoid
from pypulseq_matlab_like.make_adc import make_adc
from pypulseq_matlab_like.make_block_pulse import make_block_pulse
from pypulseq_matlab_like.make_delay import make_delay
from pypulseq_matlab_like.calc_duration import calc_duration
from pypulseq_matlab_like.calc_rf_center import calc_rf_center

fov = [200e-3, 200e-3, 160e-3]  # Define FOV
Nx = 64
Ny = Nx
Nz = Nx  # Define FOV and resolution
Tread = 3.2e-3
Tpre = 3e-3
riseTime = 400e-6
Ndummy = 50

# define system properties
# System limits
system = Opts(max_grad=20, grad_unit='mT/m', rise_time=riseTime,
              rf_ringdown_time=30e-6, rf_dead_time=100e-6)
# Sequence object
seq = Sequence(system)  # Create a new sequence object

# Create non-selective pulse
rf_duration = 0.2e-3
rf = make_block_pulse(flip_angle=8 * np.pi / 180, system=system, duration=rf_duration, use='excitation')
rf_delay = make_delay(calc_duration(rf))

# Define other gradients and ADC events
deltak = 1 / np.array(fov)
gx = make_trapezoid(channel='x', system=system, flat_area=Nx * deltak[0], flat_time=Tread)
adc = make_adc(num_samples=Nx, duration=gx.flat_time, delay=gx.rise_time)
gxPre = make_trapezoid(channel='x', system=system, area=-gx.area / 2, duration=Tpre)
gxSpoil = make_trapezoid(channel='x', system=system, area=gx.area, duration=Tpre)
areaY = ((np.arange(Ny)) - Ny / 2) * deltak[1]
areaZ = -((np.arange(Nz)) - Nz / 2) * deltak[2]

# Calculate timing
TE = 10e-3
TR = 40e-3

# delayTE = ceil((TE - mr.calcDuration(rf) + mr.calcRfCenter(rf) + rf.delay - mr.calcDuration(gxPre) - mr.calcDuration(gx)/2)/seq.gradRasterTime)*seq.gradRasterTime;
# calc_rf_center returns a tuple (center, variance? or similar), we need the first element.
# rf.delay is included in timing calculation.
# Note: calc_duration(rf) includes delay and ringdown.
# We want Time from Center of RF to Center of Readout.
# Center of RF (relative to block start) = rf.delay + calc_rf_center(rf)[0]
# Block start to Center of RF = rf.delay + calc_rf_center(rf)[0]
# Block duration = calc_duration(rf)
# So Time from Block End to RF Center (backwards) = calc_duration(rf) - (rf.delay + calc_rf_center(rf)[0])
# Wait, TE is defined as center-to-center.
# Time from RF center to end of RF block = calc_duration(rf) - (rf.delay + calc_rf_center(rf)[0])
# Then gxPre block.
# Then delayTE block.
# gx delay is 0 here (created without delay).
# calc_duration(gx) / 2 is roughly the center if symmetric.
# So TE = (Time from RF Center to End of RF Block) + calc_duration(gxPre) + delayTE + calc_duration(gx)/2
# TE = [calc_duration(rf) - (rf.delay + calc_rf_center(rf)[0])] + calc_duration(gxPre) + delayTE + calc_duration(gx)/2
# Solve for delayTE:
# delayTE = TE - [calc_duration(rf) - (rf.delay + calc_rf_center(rf)[0])] - calc_duration(gxPre) - calc_duration(gx)/2
# delayTE = TE - calc_duration(rf) + rf.delay + calc_rf_center(rf)[0] - calc_duration(gxPre) - calc_duration(gx)/2
# (Assuming calcRfCenter returns center relative to pulse start, not including delay).
# PyPulseq calc_rf_center returns center relative to shape.

delayTE = np.ceil((TE - calc_duration(rf) + calc_rf_center(rf)[0] + rf.delay - calc_duration(gxPre)
                   - calc_duration(gx) / 2) / seq.grad_raster_time) * seq.grad_raster_time
delayTR = np.ceil((TR - calc_duration(rf) - calc_duration(gxPre)
                   - calc_duration(gx) - calc_duration(gxSpoil) - delayTE) / seq.grad_raster_time) * seq.grad_raster_time

dTE = make_delay(delayTE)
dTR = make_delay(delayTR)

system_fallback = Opts()


# Make trapezoids for inner loop to save computation
gyPre_list = []
gyReph_list = []
for iY in range(Ny):
    gyPre_list.append(make_trapezoid(channel='y', area=areaY[iY], duration=Tpre, system=system_fallback))
    gyReph_list.append(make_trapezoid(channel='y', area=-areaY[iY], duration=Tpre, system=system_fallback))

seq.register_grad_event(gxPre)
seq.register_grad_event(gx)
seq.register_grad_event(gxSpoil)
_, rf.shape_IDs = seq.register_rf_event(rf)

for iY in range(Ny):
    seq.register_grad_event(gyPre_list[iY])
    seq.register_grad_event(gyReph_list[iY])

iRF = 0
# Drive magnetization to the steady state
for iY in range(Ndummy):
    # RF
    iRF += 1
    rf.phase_offset = np.deg2rad((84 * (iRF**2 + iRF)) % 360)
    seq.add_block(rf, rf_delay)
    # Gradients
    seq.add_block(gxPre, gyPre_list[int(np.floor(Ny / 2)) - 1])
    seq.add_block(dTE)
    seq.add_block(gx)
    seq.add_block(gyReph_list[int(np.floor(Ny / 2)) - 1], gxSpoil)
    seq.add_block(dTR)

# Loop over phase encodes and define sequence blocks
for iZ in range(Nz):
    gzPre = make_trapezoid(channel='z', area=areaZ[iZ], duration=Tpre, system=system_fallback)
    gzReph = make_trapezoid(channel='z', area=-areaZ[iZ], duration=Tpre, system=system_fallback)
    seq.register_grad_event(gzPre)
    seq.register_grad_event(gzReph)

    for iY in range(Ny):
        # RF spoiling
        iRF += 1
        rf.phase_offset = np.deg2rad((84 * (iRF**2 + iRF)) % 360)
        adc.phase_offset = rf.phase_offset

        # Excitation
        seq.add_block(rf, rf_delay)

        # Encoding
        seq.add_block(gxPre, gyPre_list[iY], gzPre)
        seq.add_block(dTE)
        seq.add_block(gx, adc)
        seq.add_block(gyReph_list[iY], gzReph, gxSpoil)
        seq.add_block(dTR)

print('Sequence ready')

# Timing validation
ok, error_report = seq.check_timing()

if ok:
    print('Timing check passed successfully')
else:
    print('Timing check failed! Error listing follows:')
    print(error_report)

seq.auto_label(mirror_fourier=True)
seq.set_definition('FOV', fov)
seq.set_definition('Name', 'gre3d')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '../demoSeq_pypulseq_results')
output_path = os.path.join(RESULTS_DIR, 'GradientEcho3D_py.seq')
print(f"Writing to: {os.path.abspath(output_path)}")
# Export sequence
os.makedirs(RESULTS_DIR, exist_ok=True)
seq.write(output_path, create_signature=False)
