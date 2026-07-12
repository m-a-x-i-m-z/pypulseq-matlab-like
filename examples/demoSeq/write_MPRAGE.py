
import numpy as np
import sys
import os

# Add pypulseq source to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from pypulseq.Sequence.sequence import Sequence
from pypulseq.opts import Opts
from pypulseq.make_trapezoid import make_trapezoid
from pypulseq.make_adc import make_adc
from pypulseq.make_block_pulse import make_block_pulse
from pypulseq.make_adiabatic_pulse import make_adiabatic_pulse
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
system = Opts(max_grad=24, grad_unit='mT/m', max_slew=100, slew_unit='T/m/s',
              rf_ringdown_time=20e-6, rf_dead_time=100e-6, adc_dead_time=10e-6)

# Sequence object
seq = Sequence(system)           # Create a new sequence object
alpha = 7                        # flip angle
ro_dur = 5017.6e-6 # BW=200Hz/pix
ro_os = 1                        # readout oversampling
ro_spoil = 3                     # additional k-max excursion for RO spoiling
TI = 1.1
TRout = 2.5
# TE & TR in the inner loop are as short as possible derived from the above parameters and the system specs
rfSpoilingInc = 84              # RF spoiling increment
rfLen = 100e-6

# Encoding axes
class Ax:
    pass
ax = Ax()
fov = np.array([192, 240, 256]) * 1e-3
N = np.array([192, 240, 256])
ax.d1 = 'z' # fastest dimension (readout)
ax.d2 = 'x' # second-fast dimension (inner pe loop)
ax.d3 = 'y' # slowest dimension (outer loop) # derived from setdiff('xyz', [ax.d1, ax.d2])
ax.n1 = 2 # z is index 2 in [x,y,z] (0-based) ? No, Fov/N arrays are [192, 240, 256].
# 'z' is usually slice, 'x' read, 'y' phase.
# Here d1='z' (readout). So N(3)=256 is readout resolution.
# d2='x'. N(1)=192.
# d3='y'. N(2)=240.
n1 = 2
n2 = 0
n3 = 1

# Create alpha-degree hard pulse and gradient
rf = make_block_pulse(flip_angle=alpha * np.pi / 180, system=system, duration=rfLen, use='excitation')

# Adiabatic pulse
# Python implementation:
rf180 = make_adiabatic_pulse(pulse_type='hypsec', system=system, duration=10.24e-3, delay=0, dwell=1e-5, use='inversion')

# Define other gradients and ADC events
deltak = 1.0 / fov
gro = make_trapezoid(channel=ax.d1, system=system, amplitude=N[n1] * deltak[n1] / ro_dur, flat_time=np.ceil((ro_dur + system.adc_dead_time) / system.grad_raster_time) * system.grad_raster_time)
adc = make_adc(num_samples=N[n1] * ro_os, duration=ro_dur, delay=gro.rise_time, system=system)
groPre = make_trapezoid(channel=ax.d1, system=system, area=-gro.amplitude * (adc.dwell * (adc.num_samples / 2 + 0.5) + 0.5 * gro.rise_time))
gpe1 = make_trapezoid(channel=ax.d2, system=system, area=-deltak[n2] * (N[n2] / 2))
gpe2 = make_trapezoid(channel=ax.d3, system=system, area=-deltak[n3] * (N[n3] / 2))
gslSp = make_trapezoid(channel=ax.d3, system=system, area=np.max(deltak * N) * 4, duration=10e-3)

# Split gradient
gro1, groSp = split_gradient_at(grad=gro, time_point=gro.rise_time + gro.flat_time, system=system)

# Gradient spoiling
if ro_spoil > 0:
    # Python returns tuple (grad, times, amplitudes)
    groSp, _, _ = make_extended_trapezoid_area(channel=gro.channel, grad_start=gro.amplitude, grad_end=0, area=deltak[n1] / 2 * N[n1] * ro_spoil, system=system)

# Calculate timing
rf.delay = calc_duration(groSp, gpe1, gpe2)
groPre = align(right=[groPre, gpe1, gpe2])[0]

gro1.delay = calc_duration(groPre)
adc.delay = gro1.delay + gro.rise_time
gro1 = add_gradients(grads=(gro1, groPre), system=system)
TRinner = calc_duration(rf) + calc_duration(gro1)

pe1Steps = ((np.arange(N[n2]) - N[n2] / 2) / N[n2] * 2)
pe2Steps = ((np.arange(N[n3]) - N[n3] / 2) / N[n3] * 2)

# TI calc
# find(pe1Steps==0)
center_idx = np.where(pe1Steps == 0)[0][0]
TIdelay = np.floor((TI - center_idx * TRinner - (calc_duration(rf180) - calc_rf_center(rf180)[0] - rf180.delay) - rf.delay - calc_rf_center(rf)[0]) / system.block_duration_raster + 0.5) * system.block_duration_raster
TRoutDelay = TRout - TRinner * N[n2] - TIdelay - calc_duration(rf180)

# Labels
lblIncLin = make_label(label='LIN', type='INC', value=1)
lblIncPar = make_label(label='PAR', type='INC', value=1)
lblResetPar = make_label(label='PAR', type='SET', value=0)

# pre-register objects that do not change while looping
seq.register_grad_event(gslSp)
seq.register_grad_event(groSp)
seq.register_grad_event(gro1)
seq.register_rf_event(rf)
seq.register_rf_event(rf180)
seq.register_label_event(lblIncPar)

# Start sequence
for j in range(N[n3]):
    seq.add_block(rf180)
    seq.add_block(make_delay(TIdelay), gslSp)
    rf_phase = 0
    rf_inc = 0

    gpe2je = scale_grad(gpe2, pe2Steps[j])
    gpe2jr = scale_grad(gpe2, -pe2Steps[j])

    for i in range(N[n2]):
        rf.phase_offset = rf_phase / 180 * np.pi
        adc.phase_offset = rf_phase / 180 * np.pi
        rf_inc = (rf_inc + rfSpoilingInc) % 360.0
        rf_phase = (rf_phase + rf_inc) % 360.0

        if i == 0:
            seq.add_block(rf)
        else:
            seq.add_block(rf, groSp, scale_grad(gpe1, -pe1Steps[i - 1]), gpe2jr, lblIncPar)

        seq.add_block(adc, gro1, scale_grad(gpe1, pe1Steps[i]), gpe2je)

    seq.add_block(groSp, make_delay(TRoutDelay), lblResetPar, lblIncLin)

# Timing validation
ok, error_report = seq.check_timing()

if ok:
    print('Timing check passed successfully')
else:
    print('Timing check failed! Error listing follows:')
    print(error_report)

seq.set_definition('FOV', fov)
seq.set_definition('Name', 'mprage')
seq.set_definition('OrientationMapping', 'SAG')
seq.set_definition('ReceiverGainHigh', 1)

# Export sequence
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '../demoSeq_pypulseq_results')
os.makedirs(RESULTS_DIR, exist_ok=True)
seq.write(os.path.join(RESULTS_DIR, 'MPRAGE_py.seq'))
