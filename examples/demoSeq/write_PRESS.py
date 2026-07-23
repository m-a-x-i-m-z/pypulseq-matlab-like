
import numpy as np
import sys
import os

# Add pypulseq source to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from pypulseq_matlab_like.Sequence.sequence import Sequence
from pypulseq_matlab_like.opts import Opts
from pypulseq_matlab_like.make_trapezoid import make_trapezoid
from pypulseq_matlab_like.make_adc import make_adc
from pypulseq_matlab_like.make_slr_pulse import make_slr_pulse
from pypulseq_matlab_like.make_gauss_pulse import make_gauss_pulse
from pypulseq_matlab_like.make_delay import make_delay
from pypulseq_matlab_like.make_extended_trapezoid import make_extended_trapezoid
from pypulseq_matlab_like.make_extended_trapezoid_area import make_extended_trapezoid_area
from pypulseq_matlab_like.calc_duration import calc_duration
from pypulseq_matlab_like.calc_rf_center import calc_rf_center
from pypulseq_matlab_like.add_gradients import add_gradients

# Check for check_timing
from pypulseq_matlab_like.check_timing import check_timing

# System limits
system = Opts(max_grad=15, grad_unit='mT/m', max_slew=100, slew_unit='T/m/s',
              rf_ringdown_time=20e-6, rf_dead_time=100e-6, adc_dead_time=20e-6, B0=2.89)

# Sequence object
seq = Sequence(system)              # Create a new sequence object
voxel = np.array([20, 30, 40]) * 1e-3 # voxel size
Nx = 4096
Nrep = 1
Ndummy = 0
adcDur = 256e-3
rfDurEx = 3000e-6
rfDurRef = 6000e-6
TR = 3000e-3
TE = 120e-3
spA = 0.6e3 # spoiler area in 1/m (=Hz/m*s)
spB = 2.0e3 # spoiler area in 1/m (=Hz/m*s)

# Create slice-selective excitation and refocusing pulses
rf_ex, g_ex, g_exReph = make_slr_pulse(
    flip_angle=np.pi / 2,
    system=system,
    duration=rfDurEx,
    slice_thickness=voxel[0],
    time_bw_product=6,
    passband_ripple=1.0,
    stopband_ripple=1e-2,
    filter_type='ms',
    use='excitation',
    return_gz=True,
)

rf_ref1, g_ref1, _ = make_slr_pulse(
    flip_angle=np.pi,
    system=system,
    duration=rfDurRef,
    phase_offset=np.pi / 2,
    slice_thickness=voxel[1],
    time_bw_product=6,
    passband_ripple=1.0,
    stopband_ripple=1e-2,
    filter_type='ms',
    use='refocusing',
    return_gz=True,
)

rf_ref2, g_ref2, _ = make_slr_pulse(
    flip_angle=np.pi,
    system=system,
    duration=rfDurRef,
    phase_offset=np.pi / 2,
    slice_thickness=voxel[2],
    time_bw_product=6,
    passband_ripple=1.0,
    stopband_ripple=1e-2,
    filter_type='ms',
    use='refocusing',
    return_gz=True,
)

# fix channels for the gradients
g_ex.channel = 'x'
g_ref1.channel = 'y'
# g_ref2 is default 'z'? make_sinc_pulse returns z gradients default.

# join spoilers with the slice selection pulses of the refocusing gradients
# step 1: create pre-gradient to merge into the plato
g_ref1_pre, _, _ = make_extended_trapezoid_area(channel=g_ref1.channel, grad_start=0, grad_end=g_ref1.amplitude, area=spA, system=system)
# step 2: create post-gradient to start at the plato
g_ref1_post, _, _ = make_extended_trapezoid_area(channel=g_ref1.channel, grad_start=g_ref1.amplitude, grad_end=0, area=spA, system=system)
# step 3: create a composite gradient
# Stitching: pre.tt, (post.tt + pre.dur + flat).
# pre ends at pre.dur.
times_c1 = np.concatenate((g_ref1_pre.tt, g_ref1_post.tt + g_ref1_pre.shape_dur + g_ref1.flat_time))
amps_c1 = np.concatenate((g_ref1_pre.waveform, g_ref1_post.waveform))
g_refC1 = make_extended_trapezoid(channel=g_ref1_pre.channel, system=system, times=times_c1, amplitudes=amps_c1)

# same procedure for the second refocusing pulse slice selection
g_ref2_pre, _, _ = make_extended_trapezoid_area(channel=g_ref2.channel, grad_start=0, grad_end=g_ref2.amplitude, area=spB, system=system)
g_ref2_post, _, _ = make_extended_trapezoid_area(channel=g_ref2.channel, grad_start=g_ref2.amplitude, grad_end=0, area=spB, system=system)
times_c2 = np.concatenate((g_ref2_pre.tt, g_ref2_post.tt + g_ref2_pre.shape_dur + g_ref2.flat_time))
amps_c2 = np.concatenate((g_ref2_pre.waveform, g_ref2_post.waveform))
g_refC2 = make_extended_trapezoid(channel=g_ref2_pre.channel, system=system, times=times_c2, amplitudes=amps_c2)

# update RF pulses delays to center them on the central flat parts of the combined gradients
rf_ref1.delay = g_ref1_pre.shape_dur
rf_ref2.delay = g_ref2_pre.shape_dur

# now calculate other spoiler gradients
g_spAz1 = make_trapezoid(channel='z', area=spA, system=system)
g_spAz2 = make_trapezoid(channel='z', area=spA, system=system, delay=calc_duration(g_spAz1) + g_ref1.flat_time)
g_spAx1 = make_trapezoid(channel='x', area=spA + g_exReph.area, system=system)
g_spAx2 = make_trapezoid(channel='x', area=spA, system=system, delay=calc_duration(g_spAz1) + g_ref1.flat_time)
g_spBy1 = make_trapezoid(channel='y', area=spB, system=system)
g_spBy2 = make_trapezoid(channel='y', area=spB, system=system, delay=calc_duration(g_spBy1) + g_ref2.flat_time)
g_spBx1 = make_trapezoid(channel='x', area=spB, system=system)
g_spBx2 = make_trapezoid(channel='x', area=spB, system=system, delay=calc_duration(g_spBy1) + g_ref2.flat_time)

# combine spoilers to composite gradients
g_spAz = add_gradients(grads=(g_spAz1, g_spAz2), system=system)
g_spAx = add_gradients(grads=(g_spAx1, g_spAx2), system=system)
g_spBy = add_gradients(grads=(g_spBy1, g_spBy2), system=system)
g_spBx = add_gradients(grads=(g_spBx1, g_spBx2), system=system)

# update delays in g_refC1, g_refC2, rf_ref1 and rf_ref2 in case g_spAz1 is longer than g_ref1_pre
g_refC1.delay = g_refC1.delay + max(calc_duration(g_spAz1) - calc_duration(g_ref1_pre), 0)
g_refC2.delay = g_refC2.delay + max(calc_duration(g_spBy1) - calc_duration(g_ref2_pre), 0)
rf_ref1.delay = rf_ref1.delay + max(calc_duration(g_spAz1) - calc_duration(g_ref1_pre), 0)
rf_ref2.delay = rf_ref2.delay + max(calc_duration(g_spBy1) - calc_duration(g_ref2_pre), 0)

# end spoiler
end_sp_axes = ['x', 'y', 'z']
g_spEnd = []
for axis in end_sp_axes:
    g_spEnd.append(make_trapezoid(channel=axis, system=system, area=1 / 1e-4))

# Define delays and ADC events
delayTE1 = 1e-3
delayTE2 = np.round((TE / 2 - rf_ref1.shape_dur / 2 - g_ref1_post.shape_dur - rf_ref2.delay - rf_ref2.shape_dur / 2) / system.grad_raster_time) * system.grad_raster_time
assert delayTE2 >= 0
adc = make_adc(num_samples=Nx, duration=adcDur, system=system)

# WET water suppression
ws_fa = [89.2, 83.4, 160.8]
ws_rf_dur = 14.9e-3
ws_rf_bw = 60
ws_tau = 60e-3
ws_sp_axes = ['x', 'y', 'z']
ws_sp_area = 1 / 1e-4

rf_ws = []
g_ws = []
for i in range(3):
    rf_ws.append(make_gauss_pulse(flip_angle=ws_fa[i] * np.pi / 180, system=system, duration=ws_rf_dur, bandwidth=ws_rf_bw, use='saturation'))
    g_ws.append(make_trapezoid(channel=ws_sp_axes[i], system=system, delay=calc_duration(rf_ws[i]), area=ws_sp_area))

delay_ws = [ws_tau, ws_tau, ws_tau]
delay_ws[2] = np.round((ws_tau + rf_ws[2].delay + calc_rf_center(rf_ws[2])[0] - rf_ex.delay - calc_rf_center(rf_ex)[0]) / system.grad_raster_time) * system.grad_raster_time

delayTR = np.round((TR - max(calc_duration(g_ex), calc_duration(rf_ex)) - calc_duration(g_refC1, g_spAz, g_spAx) - delayTE1 - delayTE2 - calc_duration(g_refC2, g_spBy, g_spBx) - calc_duration(adc) - calc_duration(g_spEnd[0]) - sum(delay_ws)) / system.grad_raster_time) * system.grad_raster_time
assert delayTR >= 0

# Loop over repetitions and define sequence blocks
for i in range(1 - Ndummy, Nrep + 1):
    for w in range(3):
        seq.add_block(rf_ws[w], g_ws[w], make_delay(delay_ws[w]))

    seq.add_block(rf_ex, g_ex)
    seq.add_block(make_delay(delayTE1))
    seq.add_block(rf_ref1, g_refC1, g_spAz, g_spAx)
    seq.add_block(make_delay(delayTE2))
    seq.add_block(rf_ref2, g_refC2, g_spBy, g_spBx)

    if i > 0:
        seq.add_block(adc)
    else:
        seq.add_block(make_delay(calc_duration(adc)))

    seq.add_block(g_spEnd[0], g_spEnd[1], g_spEnd[2])
    seq.add_block(make_delay(delayTR))

# Timing validation
ok, error_report = seq.check_timing()

if ok:
    print('Timing check passed successfully')
else:
    print('Timing check failed! Error listing follows:')
    print(error_report)

seq.set_definition('FOV', voxel)
seq.set_definition('Name', 'press')
seq.set_definition('ReceiverGainHigh', 1)

# Export sequence
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '../demoSeq_pypulseq_results')
os.makedirs(RESULTS_DIR, exist_ok=True)
seq.write(os.path.join(RESULTS_DIR, 'PRESS_py.seq'))
