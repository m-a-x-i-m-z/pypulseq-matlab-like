
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
from pypulseq_matlab_like.make_extended_trapezoid_area import make_extended_trapezoid_area
from pypulseq_matlab_like.make_extended_trapezoid import make_extended_trapezoid
from pypulseq_matlab_like.make_delay import make_delay
from pypulseq_matlab_like.make_label import make_label
from pypulseq_matlab_like.make_soft_delay import make_soft_delay
from pypulseq_matlab_like.calc_duration import calc_duration
from pypulseq_matlab_like.calc_rf_center import calc_rf_center
from pypulseq_matlab_like.add_gradients import add_gradients
from pypulseq_matlab_like.scale_grad import scale_grad

# System limits
system = Opts(max_grad=24, grad_unit='mT/m', max_slew=50, slew_unit='T/m/s',
              rf_ringdown_time=20e-6, rf_dead_time=100e-6, adc_dead_time=20e-6, B0=2.89)

vendor = 'siemens'
# Sequence object
seq = Sequence(system)
adcDur = 2 * 2.56e-3
rfDur1 = 3e-3
rfDur2 = 8.8e-3
TR = 1400e-3
TE = 19e-3
spAx = 0
spAy = 0
spAz = 1500
ro_os = 2
sliceThickness = 5e-3
sliceGap = 5e-3
Nslices = 2
fov = 250e-3
Nx = 256
Ny = Nx
Ndummy = 1
Nrep = 1
sth_ex = 1
sth_ref = 1.25
max_TE = 120e-3

# Create 90 degree slice selection pulse and gradient
rf_ex, gz, gzr = make_slr_pulse(
    flip_angle=np.pi / 2,
    duration=rfDur1,
    dwell=rfDur1 / 500,
    time_bw_product=5,
    passband_ripple=1.0,
    stopband_ripple=1e-2,
    filter_type='ms',
    system=system,
    use='excitation',
    phase_offset=np.pi / 2,
    slice_thickness=sliceThickness * sth_ex,
    return_gz=True,
)

# Create refocusing pulse and gradient
rf_ref, g_ref, _ = make_slr_pulse(
    flip_angle=np.pi,
    duration=rfDur2,
    dwell=rfDur2 / 500,
    time_bw_product=6,
    passband_ripple=1.0,
    stopband_ripple=1e-2,
    filter_type='ms',
    system=system,
    use='refocusing',
    phase_offset=0,
    slice_thickness=sliceThickness * sth_ref,
    return_gz=True,
)


# Spoilers with slice selection pulses of refocusing gradients
g_ref_pre, _, _ = make_extended_trapezoid_area(channel=g_ref.channel, grad_start=0, grad_end=g_ref.amplitude, area=spAz + gzr.area, system=system)
g_ref_post, _, _ = make_extended_trapezoid_area(channel=g_ref.channel, grad_start=g_ref.amplitude, grad_end=0, area=spAz, system=system)
# Stitch
times_c = np.concatenate((g_ref_pre.tt, g_ref_post.tt + g_ref_pre.shape_dur + g_ref.flat_time))
amps_c = np.concatenate((g_ref_pre.waveform, g_ref_post.waveform))
g_refC = make_extended_trapezoid(channel=g_ref_pre.channel, system=system, times=times_c, amplitudes=amps_c)
rf_ref.delay = g_ref_pre.shape_dur

# Calculate spoiler gradients in x and y
g_SPx_pre = make_trapezoid(channel='x', area=spAx, system=system)
g_SPx_post = make_trapezoid(channel='x', area=spAx, system=system, delay=calc_duration(g_SPx_pre) + g_ref.flat_time)
g_SPx = add_gradients(grads=(g_SPx_pre, g_SPx_post), system=system)

g_SPy_pre = make_trapezoid(channel='y', area=spAy, system=system)
g_SPy_post = make_trapezoid(channel='y', area=spAy, system=system, delay=calc_duration(g_SPy_pre) + g_ref.flat_time)
g_SPy = add_gradients(grads=(g_SPy_pre, g_SPy_post), system=system)

# Update delays
if calc_duration(g_ref_pre) > calc_duration(g_SPx_pre, g_SPy_pre):
    g_SPx.delay = g_SPx.delay + calc_duration(g_ref_pre) - calc_duration(g_SPx_pre)
    g_SPy.delay = g_SPy.delay + calc_duration(g_ref_pre) - calc_duration(g_SPy_pre)
else:
    diff = calc_duration(g_SPx_pre, g_SPy_pre) - calc_duration(g_ref_pre)
    g_refC.delay = g_refC.delay + diff
    rf_ref.delay = rf_ref.delay + diff

deltak = 1 / fov
gr = make_trapezoid(channel='x', system=system, flat_area=Nx * deltak, flat_time=np.ceil(adcDur / system.grad_raster_time) * system.grad_raster_time)
adc = make_adc(num_samples=Nx * ro_os, system=system, duration=adcDur, delay=gr.rise_time)

grPredur = calc_duration(g_ref_post)
grPre = make_trapezoid(channel='x', system=system, area=(gr.area / 2 + deltak / 2), duration=grPredur)
phaseAreas = (np.arange(Ny) - Ny / 2) * deltak
PEscale = phaseAreas / max(abs(phaseAreas))
gyPre = make_trapezoid(channel='y', area=-max(abs(phaseAreas)), duration=grPredur, system=system)

gyPost = make_trapezoid(channel='y', area=max(abs(phaseAreas)), duration=grPredur, system=system)
gx_spoil = make_trapezoid(channel='x', area=spAx, system=system)
gz_spoil = make_trapezoid(channel='z', area=spAz, system=system)

# Slice positions
slicePositions = (sliceThickness + sliceGap) * (np.arange(Nslices) - (Nslices - 1) / 2)
# Reorder
indices = np.concatenate((np.arange(0, Nslices, 2), np.arange(1, Nslices, 2)))
slicePositions = slicePositions[indices]

delayTE1 = TE / 2 - (calc_duration(gz) - rf_ex.center - rf_ex.delay) - rf_ref.center - rf_ref.delay - calc_duration(grPre, gyPre)
delayTE2 = TE / 2 - (calc_duration(g_refC, g_SPx, g_SPy) - rf_ref.center - rf_ref.delay) - calc_duration(gr) / 2
delayTR = TR - Nslices * (rf_ex.delay + rf_ex.shape_dur / 2 + TE + calc_duration(gr) / 2 + calc_duration(gyPost, gx_spoil, gz_spoil) + max_TE - TE)
delayTR_1slice = np.ceil(delayTR / Nslices / system.block_duration_raster) * system.block_duration_raster

delayTE1 = np.round(delayTE1 / system.grad_raster_time) * system.grad_raster_time
delayTE2 = np.round(delayTE2 / system.grad_raster_time) * system.grad_raster_time
delayTR_1slice = np.round(delayTR_1slice / system.grad_raster_time) * system.grad_raster_time

assert delayTE1 >= 10e-6
assert delayTE2 >= 10e-6
assert delayTR_1slice > 0

# Orientation
if vendor.lower().startswith('s'):
    grPre = scale_grad(grPre, -1)
    g_SPx = scale_grad(g_SPx, -1)
    gr = scale_grad(gr, -1)
    gx_spoil = scale_grad(gx_spoil, -1)
elif vendor.lower().startswith('g'):
    gyPre = scale_grad(gyPre, -1)
    gyPost = scale_grad(gyPost, -1)
    g_SPy = scale_grad(g_SPy, -1)

# Loop
for r in range(Nrep):
    seq.add_block(make_label(label='LIN', type='SET', value=0))
    for i in range(1 - Ndummy, Ny + 1):
        seq.add_block(make_label(label='SLC', type='SET', value=0))
        for s in range(Nslices):
            # Python i matches value.

            adc.phase_offset = (i % 2) * np.pi
            rf_ex.freq_offset = gz.amplitude * slicePositions[s]
            rf_ref.freq_offset = g_ref.amplitude * slicePositions[s]

            if vendor.lower().startswith('g'):
                rf_ex.freq_offset = -rf_ex.freq_offset
                rf_ref.freq_offset = -rf_ref.freq_offset

            rf_ex.phase_offset = (i % 2) * np.pi + np.pi / 2 - 2 * np.pi * rf_ex.freq_offset * calc_rf_center(rf_ex)[0]
            rf_ref.phase_offset = -2 * np.pi * rf_ref.freq_offset * calc_rf_center(rf_ref)[0]

            seq.add_block(rf_ex, gz)
            seq.add_block(make_delay(delayTE1), make_soft_delay(default_duration=1e-5, hint='TE', offset=delayTE1 - TE / 2, factor=2))

            if i > 0:
                seq.add_block(grPre, scale_grad(gyPre, PEscale[i - 1])) # i-1 for 0-based index
                seq.add_block(rf_ref, g_refC, g_SPx, g_SPy)
            else:
                seq.add_block(grPre)
                seq.add_block(rf_ref, g_refC, g_SPx, g_SPy)

            seq.add_block(make_delay(delayTE2), make_soft_delay(default_duration=1e-5, hint='TE', offset=delayTE2 - TE / 2, factor=2))

            if i > 0:
                seq.add_block(adc, gr)
                seq.add_block(make_label(label='SLC', type='INC', value=1))
                seq.add_block(gx_spoil, scale_grad(gyPost, PEscale[i - 1]), gz_spoil)
            else:
                seq.add_block(gr)
                seq.add_block(gx_spoil, gz_spoil)

            seq.add_block(make_delay(max_TE - TE), make_soft_delay(default_duration=1e-5, hint='TE', factor=-1, offset=max_TE))

            if r == 0 and i == (1 - Ndummy) and s == 0:
                durPerSlc = np.sum(list(seq.block_durations.values()))
                # But here we just need offset.

            seq.add_block(make_delay(delayTR_1slice), make_soft_delay(default_duration=1e-5, hint='TR', offset=-durPerSlc, factor=Nslices))

        if i > 0:
            seq.add_block(make_label(label='LIN', type='INC', value=1))

    seq.add_block(make_label(label='REP', type='INC', value=1))

# Timing validation
ok, error_report = seq.check_timing()

if ok:
    print('Timing check passed successfully')
else:
    print('Timing check failed! Error listing follows:')
    print(error_report)

seq.set_definition('Name', 'QA_T1')
seq.set_definition('FOV', [fov, fov, max(slicePositions) - min(slicePositions) + sliceThickness])
seq.set_definition('SlicePositions', slicePositions)
seq.set_definition('SliceThickness', sliceThickness)
seq.set_definition('SliceGap', sliceGap)
seq.set_definition('ReadoutOversamplingFactor', ro_os)
seq.set_definition('ReceiverGainHigh', 1)

# Export sequence
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '../demoSeq_pypulseq_results')
os.makedirs(RESULTS_DIR, exist_ok=True)
seq.write(os.path.join(RESULTS_DIR, 'SpinEcho_softdelay_py.seq'))
