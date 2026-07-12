
import numpy as np
import sys
import os

# Add pypulseq source to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from pypulseq.Sequence.sequence import Sequence
from pypulseq.opts import Opts
from pypulseq.make_trapezoid import make_trapezoid
from pypulseq.make_adc import make_adc
from pypulseq.make_sinc_pulse import make_sinc_pulse
from pypulseq.make_delay import make_delay
from pypulseq.make_extended_trapezoid_area import make_extended_trapezoid_area
from pypulseq.make_label import make_label
from pypulseq.calc_duration import calc_duration
from pypulseq.calc_rf_center import calc_rf_center
from pypulseq.split_gradient_at import split_gradient_at
from pypulseq.align import align
from pypulseq.add_gradients import add_gradients
from pypulseq.scale_grad import scale_grad

# System limits
system = Opts(max_grad=30, grad_unit='mT/m', max_slew=140, slew_unit='T/m/s',
              rf_ringdown_time=20e-6, rf_dead_time=100e-6, adc_dead_time=20e-6)

# Sequence object
seq = Sequence(system)
fov = 220e-3
Nx = 256
Ny = 256
adc_dur = 2560 # us
alpha = 40
thick = 4
rf_dur = 600 # us
rf_apo = 0.5
rf_bwt = 1.5

# Create 'alpha' degree slice selection pulse and gradient
rf, gz, gzReph = make_sinc_pulse(flip_angle=alpha * np.pi / 180, duration=rf_dur * 1e-6,
                                 slice_thickness=thick * 1e-3, apodization=rf_apo, time_bw_product=rf_bwt, system=system,
                                 use='excitation', return_gz=True)

# Define other gradients and ADC events
deltak = 1 / fov
gx = make_trapezoid(channel='x', flat_area=Nx * deltak, flat_time=adc_dur * 1e-6, system=system)
adc = make_adc(num_samples=Nx, duration=gx.flat_time, delay=gx.rise_time, system=system)
gxPre = make_trapezoid(channel='x', area=-gx.area / 2, system=system)
phaseAreas = (np.arange(Ny) - Ny / 2) * deltak

# Reshuffle gradients
# Split gz
gz_parts = split_gradient_at(grad=gz, time_point=calc_duration(rf), system=system)
gz_parts[0].delay = calc_duration(gzReph)
gz_1 = add_gradients(grads=(gzReph, gz_parts[0]), system=system)
gz_1, rf = align(left=gz_1, right=rf)

gz_parts[1].delay = 0
gzReph.delay = calc_duration(gz_parts[1])
gz_2 = add_gradients(grads=(gz_parts[1], gzReph), system=system)

# Split gx
gx_parts = split_gradient_at(grad=gx, time_point=np.ceil(calc_duration(adc) / system.grad_raster_time) * system.grad_raster_time, system=system)
gx_parts[0].delay = calc_duration(gxPre)
gx_1 = add_gradients(grads=(gxPre, gx_parts[0]), system=system)
adc.delay = adc.delay + calc_duration(gxPre)
gx_parts[1].delay = 0
gxPre.delay = calc_duration(gx_parts[1])
gx_2 = add_gradients(grads=(gx_parts[1], gxPre), system=system)

# Calculate timing
gxPre.delay = 0
pe_dur = calc_duration(gx_2)

# Adjust delays
gz_1.delay = max(calc_duration(gx_2) - rf.delay + rf.ringdown_time, 0)
rf.delay = rf.delay + gz_1.delay

# Finish timing
TR = calc_duration(gz_1) + calc_duration(gx_1)
TE = TR / 2

# Alpha/2 prep
# PyPulseq `rf` is arbitrary if modified? Sinc pulse object has signal array.
# Copying object and scaling signal is risky if properties don't update.
# Better to create new pulse or valid copy.
# `rf05.signal = 0.5 * rf.signal`.
import copy
rf05 = copy.deepcopy(rf)
rf05.signal = 0.5 * rf05.signal

seq.add_block(rf05, gz_1, make_label(label='ONCE', type='SET', value=1))

prepDelay = make_delay(np.round((TR / 2 - calc_duration(gz_1)) / system.grad_raster_time) * system.grad_raster_time)
gx_1_1, _, _ = make_extended_trapezoid_area(channel='x', grad_start=0, grad_end=gx_2.first, area=-gx_2.area, system=system)

gyPre_2 = make_trapezoid(channel='y', area=phaseAreas[-1], duration=pe_dur, system=system)

prepDelay, gz_2, gyPre_2, gx_1_1 = align(left=[prepDelay, gz_2, gyPre_2], right=gx_1_1)
seq.add_block(prepDelay, gz_2, gyPre_2, gx_1_1)
seq.add_block(make_label(label='ONCE', type='SET', value=0))

for i in range(1, Ny + 1):
    rf.phase_offset = np.pi * (i % 2)
    adc.phase_offset = np.pi * (i % 2)

    gyPre_1 = scale_grad(gyPre_2, -1)
    gyPre_2 = make_trapezoid(channel='y', area=phaseAreas[i - 1], duration=pe_dur, system=system)

    seq.add_block(rf, gz_1, gyPre_1, gx_2)
    seq.add_block(gx_1, gyPre_2, gz_2, adc)

# Finish x-grad
seq.add_block(gx_2, make_label(label='ONCE', type='SET', value=2))

# Check TR
assert TR == (seq.block_durations[4] + seq.block_durations[5])

# Timing validation
ok, error_report = seq.check_timing()

if ok:
    print('Timing check passed successfully')
else:
    print('Timing check failed! Error listing follows:')
    print(error_report)

seq.auto_label(mirror_fourier=True)
seq.set_definition('FOV', [fov, fov, thick * 1e-3])
seq.set_definition('Name', 'trufi')

# Export sequence
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '../demoSeq_pypulseq_results')
os.makedirs(RESULTS_DIR, exist_ok=True)
seq.write(os.path.join(RESULTS_DIR, 'Trufi_py.seq'))