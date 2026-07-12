
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
ro_dur = 5120e-6
ro_os = 2                        # readout oversampling
ro_spoil = 3                     # additional k-max excursion for RO spoiling
TI = 1.1
TRout = 2.5
rfSpoilingInc = 84
rfLen = 100e-6

# Encoding axes
class Ax:
    pass
ax = Ax()
fov = np.array([192, 240, 256]) * 1e-3
N = np.array([192, 240, 256])
phaseResolution = fov[2] / N[2] / (fov[1] / N[1])
ax.d1 = 'z' # fastest dimension (readout)
ax.d2 = 'x' # second-fast dimension (inner pe loop)
ax.d3 = 'y' # slowest dimension (outer loop)
n1 = 2 # z
n2 = 0 # x
n3 = 1 # y

# Create alpha-degree hard pulse and gradient
rf = make_block_pulse(flip_angle=alpha * np.pi / 180, system=system, duration=rfLen, use='excitation')
rf180 = make_adiabatic_pulse(pulse_type='hypsec', system=system, duration=10.24e-3, dwell=1e-5, delay=0, use='inversion')

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
if ro_spoil > 0:
    groSp, _, _ = make_extended_trapezoid_area(channel=gro.channel, grad_start=gro.amplitude, grad_end=0, area=deltak[n1] / 2 * N[n1] * ro_spoil, system=system)

# Calculate timing
rf.delay = calc_duration(groSp, gpe1, gpe2)

# We use max duration to align
max_dur = calc_duration(groPre, gpe1, gpe2)
groPre.delay = max_dur - calc_duration(groPre)
# But they need to be played *with* groPre.
# In `add_gradients` below, `gro1` and `groPre` are added.
# In loop, `scaleGrad(gpe1)` is used.
# `gpe1` duration is typically rise+flat+fall.
# If `groPre` is longer than `gpe1`, we are fine.
# In python, we can just ensure they fit in the block.
# But `gro1.delay=mr.calcDuration(groPre)` suggests `groPre` determines delay of `gro1` relative to start of block.
# If `gpe1` is longer than `groPre`, `groPre` might be delayed.
gro1.delay = max_dur
# If `groPre` was extended/shifted by align, `gro1` follows it.
# So `gro1` starts after `groPre` finishes.
adc.delay = gro1.delay + gro.rise_time
gro1 = add_gradients(grads=(gro1, groPre), system=system)
TRinner = calc_duration(rf) + calc_duration(gro1)

pe1Steps = ((np.arange(N[n2]) - N[n2] / 2) / N[n2] * 2)
pe2Steps = ((np.arange(N[n3]) - N[n3] / 2) / N[n3] * 2)

center_idx = np.where(pe1Steps == 0)[0][0]
TIdelay = np.floor((TI - center_idx * TRinner - (calc_duration(rf180) - calc_rf_center(rf180)[0] - rf180.delay) - rf.delay - calc_rf_center(rf)[0]) / system.block_duration_raster + 0.5) * system.block_duration_raster
TRoutDelay = TRout - TRinner * N[n2] - TIdelay - calc_duration(rf180)

# Labels
lblIncPar = make_label(label='PAR', type='INC', value=1)
lblResetPar = make_label(label='PAR', type='SET', value=0)
lblSetRefScan = make_label(label='REF', type='SET', value=1)
lblSetRefAndImaScan = make_label(label='IMA', type='SET', value=1)
lblResetRefScan = make_label(label='REF', type='SET', value=0)
lblResetRefAndImaScan = make_label(label='IMA', type='SET', value=0)

seq.register_rf_event(rf)
seq.register_rf_event(rf180)
seq.register_label_event(lblSetRefScan)
seq.register_label_event(lblSetRefAndImaScan)
seq.register_label_event(lblResetRefScan)
seq.register_label_event(lblResetRefAndImaScan)

# GRAPPA setup
nY = N[n3]
accelFactorPE = 2
ACSnum = 32
centerLineIdx = int(np.floor(nY / 2))
# Python `int(nY/2)` = 120.
# PEsamp_u.
PEsamp_u = []
for i in range(nY):
    # `i_mat - center_mat`
    # `(i_py+1) - (center_py+1)` = `i_py - center_py`.
    # So logic: `(i - centerLineIdx) % accelFactorPE == 0` is same.
    if (i - centerLineIdx) % accelFactorPE == 0:
        PEsamp_u.append(i)

minPATRefLineIdx = centerLineIdx - int(ACSnum / 2)
maxPATRefLineIdx = centerLineIdx + int((ACSnum - 1) / 2)
PEsamp_ACS = list(range(minPATRefLineIdx, maxPATRefLineIdx + 1))
PEsamp = sorted(list(set(PEsamp_u) | set(PEsamp_ACS)))
nPEsamp = len(PEsamp)
# PEsamp_INC = diff([PEsamp, PEsamp(end)])
# `diff` of size N+1 gives N elements.
PEsamp_INC = [PEsamp[i+1] - PEsamp[i] for i in range(len(PEsamp)-1)]
PEsamp_INC.append(0)
# `LIN` counter roughly tracks encoded line index.
# We initialize LIN at PEsamp[0]. Then increment by `PEsamp_INC`.
# `PEsamp_INC`. `diff` of `[p0, p1, ..., pn, pn]` -> `[p1-p0, p2-p1, ..., pn-pn=0]`.
# So inside loop, we increment LIN to match NEXT index.
# Correct.

# Reverse polarity of gradients to match Siemens product (Gz readout reversed)
groSp = scale_grad(groSp, -1)
gro1 = scale_grad(gro1, -1)
gslSp = scale_grad(gslSp, -1)
gpe1.amplitude = -gpe1.amplitude

# Register static gradients after final polarity/sign adjustments
seq.register_grad_event(groSp)
seq.register_grad_event(gro1)
seq.register_grad_event(gslSp)


# Add noise scans
seq.add_block(adc, make_label(label='LIN', type='SET', value=0), make_label(label='NOISE', type='SET', value=1), lblResetRefAndImaScan, lblResetRefScan)
seq.add_block(make_label(label='NOISE', type='SET', value=0))

# Set LIN for first acquired PE line
seq.add_block(make_label(label='LIN', type='SET', value=PEsamp[0])) # 0-based

for count in range(nPEsamp):
    current_pe_idx = PEsamp[count]

    # Set PAT labels
    if current_pe_idx in PEsamp_ACS:
        if current_pe_idx in PEsamp_u:
            seq.add_block(lblSetRefAndImaScan, lblSetRefScan)
        else:
            seq.add_block(lblResetRefAndImaScan, lblSetRefScan)
    else:
        seq.add_block(lblResetRefAndImaScan, lblResetRefScan)

    seq.add_block(rf180)
    seq.add_block(make_delay(TIdelay), gslSp)
    rf_phase = 0
    rf_inc = 0

    gpe2je = scale_grad(gpe2, pe2Steps[current_pe_idx])
    seq.register_grad_event(gpe2je)
    gpe2jr = scale_grad(gpe2, -pe2Steps[current_pe_idx])
    seq.register_grad_event(gpe2jr)

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

    seq.add_block(groSp, scale_grad(gpe1, -pe1Steps[N[n2] - 1]), gpe2jr,
                  make_delay(TRoutDelay), make_label(label='LIN', type='INC', value=PEsamp_INC[count]), lblResetPar)

# Timing validation
ok, error_report = seq.check_timing()

if ok:
    print('Timing check passed successfully')
else:
    print('Timing check failed! Error listing follows:')
    print(error_report)

seq.set_definition('FOV', fov)
seq.set_definition('Name', 'mp_gt')
seq.set_definition('ReadoutOversamplingFactor', ro_os)
seq.set_definition('OrientationMapping', 'SAG')
seq.set_definition('kSpaceCenterLine', centerLineIdx) # 0-based
seq.set_definition('PhaseResolution', phaseResolution)

# Export sequence
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '../demoSeq_pypulseq_results')
os.makedirs(RESULTS_DIR, exist_ok=True)
seq.write(os.path.join(RESULTS_DIR, 'MPRAGE_grappa_py.seq'))
