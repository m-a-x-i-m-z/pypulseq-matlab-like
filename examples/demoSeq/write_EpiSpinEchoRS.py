
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
from pypulseq_matlab_like.make_gauss_pulse import make_gauss_pulse
from pypulseq_matlab_like.make_delay import make_delay
from pypulseq_matlab_like.make_digital_output_pulse import make_digital_output_pulse
from pypulseq_matlab_like.make_extended_trapezoid import make_extended_trapezoid
from pypulseq_matlab_like.make_extended_trapezoid_area import make_extended_trapezoid_area
from pypulseq_matlab_like.calc_duration import calc_duration
from pypulseq_matlab_like.make_label import make_label
from pypulseq_matlab_like.split_gradient_at import split_gradient_at
from pypulseq_matlab_like.align import align
from pypulseq_matlab_like.add_gradients import add_gradients
from pypulseq_matlab_like.calc_rf_center import calc_rf_center
from pypulseq_matlab_like.scale_grad import scale_grad

# this is an experimenta high-performance EPI sequence
# which uses split gradients to overlap blips with the readout
# gradients combined with ramp-samping

# Sequence object
seq = Sequence()   # Create a new sequence object
fov = 250e-3
Nx = 64
Ny = 64
thickness = 3e-3
Nslices = 3
TE = 40e-3

pe_enable = 1
ro_os = 1
readoutTime = 4.2e-4
partFourierFactor = 0.75

tRFex = 2e-3
tRFref = 2e-3
spoilFactor = 1.5

# System limits
system = Opts(max_grad=32, grad_unit='mT/m', max_slew=130, slew_unit='T/m/s',
              rf_ringdown_time=30e-6, rf_dead_time=100e-6, adc_dead_time=10e-6)

# Create fat-sat pulse
B0 = 2.89
sat_ppm = -3.45
sat_freq = sat_ppm * 1e-6 * B0 * system.gamma
rf_fs = make_gauss_pulse(flip_angle=110 * np.pi / 180, system=system, duration=8e-3,
                         bandwidth=abs(sat_freq), freq_offset=sat_freq, use='saturation')
rf_fs.phase_offset = -2 * np.pi * rf_fs.freq_offset * calc_rf_center(rf_fs)[0]
gz_fs = make_trapezoid(channel='z', system=system, delay=calc_duration(rf_fs), area=1 / 1e-4) # spoil up to 0.1mm

# Create 90 degree slice selection pulse and gradient
rf, gz, gzReph = make_sinc_pulse(flip_angle=np.pi / 2, system=system, duration=tRFex,
                                 slice_thickness=thickness, apodization=0.5, time_bw_product=4, use='excitation', return_gz=True)

# Create 180 degree slice refocusing pulse and gradients
rf180, gz180, _ = make_sinc_pulse(flip_angle=np.pi, system=system, duration=tRFref,
                               slice_thickness=thickness, apodization=0.5, time_bw_product=4, phase_offset=np.pi / 2, use='refocusing', return_gz=True)

# Extended trapezoid area calculation
_, gzr1_t, gzr1_a = make_extended_trapezoid_area(channel='z', grad_start=0, grad_end=gz180.amplitude,
                                                 area=spoilFactor * gz.area, system=system)

_, gzr2_t, gzr2_a = make_extended_trapezoid_area(channel='z', grad_start=gz180.amplitude, grad_end=0,
                                                 area=-gzReph.area + spoilFactor * gz.area, system=system)

# Timing adjustments
# make_extended_trapezoid_area returns full shapes.
# gzr1_t[-1] is the duration.
duration_r1 = gzr1_t[-1]
rise_time_orig = gz180.rise_time

if gz180.delay > (duration_r1 - rise_time_orig):
    gz180.delay = gz180.delay - (duration_r1 - rise_time_orig)
else:
    rf180.delay = rf180.delay + (duration_r1 - rise_time_orig) - gz180.delay
    gz180.delay = 0

# Stitching
# amplitudes=[gzr1_a gzr2_a]
# make_extended_trapezoid requires distinct times?
# gzr2_t starts at 0.
# We append gzr2_t shifted by (duration_r1 + gz180.flat_time).
# And we take `gzr1_a` and `gzr2_a`.
# No, `gzr2_t[0]=0`. So shift+0 = shift.
# `gzr1_t[-1]` is the end.
# We need to bridge the flat time.
# Segment 1: `gzr1_t` (0 to duration_r1). Amps `gzr1_a`.
# Segment 2 (flat): duration `gz180.flat_time`. Amp `gz180.amplitude`.
# Segment 3: `gzr2_t` (0 to duration_r2). Amps `gzr2_a`.
# We construct times: `gzr1_t` , then `(gzr1_t[-1] + gz180.flat_time + gzr2_t)`.
# No, `makeExtendedTrapezoidArea` makes a trapezoid with start/end values.
# It connects 0 to `gz180.amplitude`.
# So `gzr1_a[-1]` is `gz180.amplitude`.
# `gzr2_a[0]` is `gz180.amplitude`.
# `gzr1_t` end is `T1`. `gzr2_t` start is 0.
# `T1 + flat + 0` = `T1 + flat`.
# So we have points at `T1` (from gzr1) and `T1+flat` (from shifted gzr2 start).
# We interpret this as a flat region between `T1` and `T1+flat`.
# So we keep `gzr1_t`, and `gzr2_t` shifted by `T1 + flat`.
# And amplitudes? `gzr1_a`, `gzr2_a`.
# Yes.
# So `times = np.concatenate((gzr1_t, gzr1_t[-1] + gz180.flat_time + gzr2_t))`
# `amplitudes = np.concatenate((gzr1_a, gzr2_a))`
# `make_extended_trapezoid` will see:
# `t_last_r1`, `amp`
# `t_first_r2 + shift`, `amp`
# Since `t_last_r1 != t_first_r2 + shift` (unless flat_time=0), it creates a line segment between them with amplitude `amp` -> `amp` (flat).
# This is correct.

times_stitched = np.concatenate((gzr1_t, gzr1_t[-1] + gz180.flat_time + gzr2_t)) + gz180.delay
amps_stitched = np.concatenate((gzr1_a, gzr2_a))

gz180n = make_extended_trapezoid(channel='z', system=system, times=times_stitched, amplitudes=amps_stitched)

# define the output trigger
trig = make_digital_output_pulse('osc0', duration=100e-6)

# Define other gradients and ADC events
deltak = 1 / fov
kWidth = Nx * deltak

# Phase blip
blip_dur = np.ceil(2 * np.sqrt(deltak / system.max_slew) / 10e-6 / 2) * 10e-6 * 2
gy = make_trapezoid(channel='y', system=system, area=-deltak, duration=blip_dur)

# readout gradient
extra_area = blip_dur / 2 * blip_dur / 2 * system.max_slew
gx = make_trapezoid(channel='x', system=system, area=kWidth + extra_area, duration=readoutTime + blip_dur)
actual_area = gx.area - gx.amplitude / gx.rise_time * blip_dur / 2 * blip_dur / 2 / 2 - gx.amplitude / gx.fall_time * blip_dur / 2 * blip_dur / 2 / 2
gx.amplitude = gx.amplitude / actual_area * kWidth
gx.area = gx.amplitude * (gx.flat_time + gx.rise_time / 2 + gx.fall_time / 2)
gx.flat_area = gx.amplitude * gx.flat_time

# calculate ADC
adcDwellNyquist = deltak / gx.amplitude / ro_os
adcDwell = np.floor(adcDwellNyquist * 1e7) * 1e-7
adcSamples = np.floor(readoutTime / adcDwell / 4) * 4
adc = make_adc(num_samples=int(adcSamples), dwell=adcDwell, delay=blip_dur / 2, system=system)
# realign the ADC with respect to the gradient
time_to_center = adc.dwell * ((adcSamples - 1) / 2 + 0.5)
adc.delay = np.round((gx.rise_time + gx.flat_time / 2 - time_to_center) * 1e6) * 1e-6

# split the blip
gy_parts = split_gradient_at(grad=gy, time_point=blip_dur / 2, system=system)
gy_blipup, gy_blipdown, _ = align(right=gy_parts[0], left=gy_parts[1], center=gx)
gy_blipdownup = add_gradients(grads=(gy_blipdown, gy_blipup), system=system)

# pe_enable support
gy_blipup.waveform = gy_blipup.waveform * pe_enable
gy_blipdown.waveform = gy_blipdown.waveform * pe_enable
gy_blipdownup.waveform = gy_blipdownup.waveform * pe_enable

# phase encoding and partial Fourier
Ny_pre = np.round(partFourierFactor * Ny / 2 - 1)
Ny_post = np.round(Ny / 2 + 1)
Ny_meas = int(Ny_pre + Ny_post)

# Pre-phasing gradients
gxPre = make_trapezoid(channel='x', system=system, area=-gx.area / 2)
gyPre = make_trapezoid(channel='y', system=system, area=Ny_pre * deltak)
gxPre, gyPre = align(right=gxPre, left=gyPre)
# relax the PE prepahser to reduce stimulation
gyPre = make_trapezoid(channel='y', system=system, area=gyPre.area, duration=calc_duration(gxPre, gyPre))
gyPre.amplitude = gyPre.amplitude * pe_enable

# Calculate delay times
durationToCenter = (Ny_pre + 0.5) * calc_duration(gx)
rfCenterInclDelay = rf.delay + calc_rf_center(rf)[0]
rf180centerInclDelay = rf180.delay + calc_rf_center(rf180)[0]
delayTE1 = np.ceil((TE / 2 - calc_duration(rf, gz) + rfCenterInclDelay - rf180centerInclDelay) / system.grad_raster_time) * system.grad_raster_time
delayTE2 = np.ceil((TE / 2 - calc_duration(rf180, gz180n) + rf180centerInclDelay - durationToCenter) / system.grad_raster_time) * system.grad_raster_time
assert delayTE1 >= 0

delayTE2 = delayTE2 + calc_duration(rf180, gz180n)
gxPre.delay = 0
gxPre.delay = delayTE2 - calc_duration(gxPre)
assert gxPre.delay >= calc_duration(rf180)
gyPre.delay = calc_duration(rf180)
assert calc_duration(gyPre) <= calc_duration(gxPre)

# Define sequence blocks
for s in range(Nslices):
    seq.add_block(rf_fs, gz_fs)
    rf.freq_offset = gz.amplitude * thickness * (s - (Nslices - 1) / 2)
    rf.phase_offset = -2 * np.pi * rf.freq_offset * calc_rf_center(rf)[0]
    rf180.freq_offset = gz180.amplitude * thickness * (s - (Nslices - 1) / 2)
    rf180.phase_offset = np.pi / 2 - 2 * np.pi * rf180.freq_offset * calc_rf_center(rf180)[0]
    seq.add_block(rf, gz, trig)
    seq.add_block(make_delay(delayTE1))

    # gxPre, gyPre, gz180n.
    # gxPre is on x, gyPre on y, gz180n on z. No overlap.
    # make_delay(delayTE2).
    # NOTE: delayTE2 was updated to include rf180 duration.
    # And gxPre.delay was set to `delayTE2 - duration`.
    # So `make_delay(delayTE2)` sets block duration to `delayTE2`.
    # `gxPre` lives in that block, ending at `delayTE2`.
    # `gz180n` lives in that block, starting at 0 (or delay).
    # `rf180` lives in that block.
    # `gyPre` starts at `rf180` duration.
    # This should work fine in add_block.
    seq.add_block(rf180, gz180n, make_delay(delayTE2), gxPre, gyPre)

    for i in range(Ny_meas):
        if i == 0:
            seq.add_block(gx, gy_blipup, adc)
        elif i == Ny_meas - 1:
            seq.add_block(gx, gy_blipdown, adc)
        else:
            seq.add_block(gx, gy_blipdownup, adc)
        gx.amplitude = -gx.amplitude

# Timing validation
ok, error_report = seq.check_timing()

if ok:
    print('Timing check passed successfully')
else:
    print('Timing check failed! Error listing follows:')
    print(error_report)

seq.set_definition('FOV', [fov, fov, thickness])
seq.set_definition('Name', 'epi')

# Export sequence
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '../demoSeq_pypulseq_results')
os.makedirs(RESULTS_DIR, exist_ok=True)
seq.write(os.path.join(RESULTS_DIR, 'EpiSpinEchoRS_py.seq'))