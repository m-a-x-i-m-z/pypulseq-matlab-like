
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
from pypulseq.make_gauss_pulse import make_gauss_pulse
from pypulseq.make_delay import make_delay
from pypulseq.make_digital_output_pulse import make_digital_output_pulse
from pypulseq.make_extended_trapezoid import make_extended_trapezoid
from pypulseq.make_extended_trapezoid_area import make_extended_trapezoid_area
from pypulseq.calc_duration import calc_duration
from pypulseq.make_label import make_label
from pypulseq.split_gradient_at import split_gradient_at
from pypulseq.align import align
from pypulseq.add_gradients import add_gradients
from pypulseq.calc_rf_center import calc_rf_center

# this is an experimental high-performance EPI sequence
# which uses split gradients to overlap blips with the readout
# gradients combined with ramp-samping
# it further features diffusion weighting using the standard
# Stejskal-Tanner scheme
#
# IMPORTANT NOTICE: be aware, that this sequence potentially uses very
# strong gradient that may overload your scanner!

# System limits
system = Opts(max_grad=38, grad_unit='mT/m', max_slew=180, slew_unit='T/m/s',
              rf_ringdown_time=10e-6, rf_dead_time=100e-6, B0=2.89)

# Sequence object
seq = Sequence(system)     # Create a new sequence object
fov = 224e-3
Nx = 112
Ny = Nx # Define FOV and resolution
thickness = 2e-3           # slice thinckness
Nslices = 3
bFactor = 1000 # s/mm^2
TE = 100e-3

pe_enable = 1              # a flag to quickly disable phase encoding (1/0) as needed for the delay calibration
ro_os = 1                  # oversampling factor (in contrast to the product sequence we don't really need it)
readoutTime = 6.3e-4       # this controls the readout bandwidth
partFourierFactor = 0.75   # partial Fourier factor: 1: full sampling 0: start with ky=0

tRFex = 3e-3
tRFref = 3e-3

# Create fat-sat pulse
sat_ppm = -3.45
sat_freq = sat_ppm * 1e-6 * system.B0 * system.gamma
rf_fs = make_gauss_pulse(flip_angle=110 * np.pi / 180, system=system, duration=8e-3,
                         bandwidth=abs(sat_freq), freq_offset=sat_freq, use='saturation')
gz_fs = make_trapezoid(channel='z', system=system, delay=calc_duration(rf_fs), area=1 / 1e-4) # spoil up to 0.1mm

# Create 90 degree slice selection pulse and gradient
rf, gz, gzReph = make_sinc_pulse(flip_angle=np.pi / 2, system=system, duration=tRFex,
                                 slice_thickness=thickness, apodization=0.5, time_bw_product=4,
                                 use='excitation', return_gz=True)

# Create 90 degree slice refocusing pulse and gradients
rf180, gz180, _ = make_sinc_pulse(flip_angle=np.pi, system=system, duration=tRFref,
                               slice_thickness=thickness, apodization=0.5, time_bw_product=4, phase_offset=np.pi / 2, use='refocusing', return_gz=True)

# Python make_extended_trapezoid_area returns (grad, times, amplitudes)
_, gzr_t, gzr_a = make_extended_trapezoid_area(channel='z', grad_start=gz180.amplitude, grad_end=0,
                                               area=-gzReph.area + 0.5 * gz180.amplitude * gz180.fall_time, system=system)
# Python make_extended_trapezoid uses arrays for times and amplitudes
# Times must be relative to start of block? No, relative to start of gradient event.
# gz180 has delay?
# construct new times
times = np.concatenate(([0, gz180.rise_time], gz180.rise_time + gz180.flat_time + gzr_t)) + gz180.delay
amplitudes = np.concatenate(([0, gz180.amplitude], gzr_a))

gz180n = make_extended_trapezoid(channel='z', system=system, times=times, amplitudes=amplitudes)

# define the output trigger to play out with every slice excitatuion
trig = make_digital_output_pulse('osc0', duration=100e-6) # possible channels: 'osc0','osc1','ext1'

# Define other gradients and ADC events
deltak = 1 / fov
kWidth = Nx * deltak

# Phase blip in shortest possible time
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
# Access [0] for calc_rf_center tuple return
rfCenterInclDelay = rf.delay + calc_rf_center(rf)[0]
rf180centerInclDelay = rf180.delay + calc_rf_center(rf180)[0]
delayTE1 = np.ceil((TE / 2 - calc_duration(rf, gz) + rfCenterInclDelay - rf180centerInclDelay) / system.grad_raster_time) * system.grad_raster_time
delayTE2tmp = np.ceil((TE / 2 - calc_duration(rf180, gz180n) + rf180centerInclDelay - durationToCenter) / system.grad_raster_time) * system.grad_raster_time
assert delayTE1 >= 0

gxPre.delay = 0
gyPre.delay = 0
delayTE2 = delayTE2tmp - calc_duration(gxPre, gyPre)
gxPre, gyPre = align(right=gxPre, left=gyPre)
assert delayTE2 >= 0

# diffusion weithting calculation
def bFactCalc(g, delta, DELTA):
    sigma = 1
    kappa_minus_lambda = 1 / 3 - 1 / 2
    b = (2 * np.pi * g * delta * sigma)**2 * (DELTA + 2 * kappa_minus_lambda * delta)
    return b

small_delta = delayTE2 - np.ceil(system.max_grad / system.max_slew / system.grad_raster_time) * system.grad_raster_time
big_delta = delayTE1 + calc_duration(rf180, gz180n)
g = np.sqrt(bFactor * 1e6 / bFactCalc(1, small_delta, big_delta))
gr = np.ceil(g / system.max_slew / system.grad_raster_time) * system.grad_raster_time
gDiff = make_trapezoid(channel='z', amplitude=g, rise_time=gr, flat_time=small_delta - gr, system=system)
assert calc_duration(gDiff) <= delayTE1
assert calc_duration(gDiff) <= delayTE2

lblPmcOn = make_label(label='PMC', type='SET', value=1)
lblPmcOff = make_label(label='PMC', type='SET', value=0)

# Define sequence blocks
for s in range(Nslices):
    seq.add_block(rf_fs, gz_fs)
    rf.freq_offset = gz.amplitude * thickness * (s - (Nslices - 1) / 2)
    rf180.freq_offset = gz180.amplitude * thickness * (s - (Nslices - 1) / 2)
    seq.add_block(rf, gz, trig, lblPmcOn)
    seq.add_block(make_delay(delayTE1), gDiff, lblPmcOff)
    rf180.freq_offset = gz180.amplitude * thickness * (s - (Nslices - 1) / 2)
    seq.add_block(rf180, gz180n)
    seq.add_block(make_delay(delayTE2), gDiff)
    seq.add_block(gxPre, gyPre)
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

seq.set_definition('FOV', [fov, fov, thickness * Nslices])
seq.set_definition('Name', 'epi-diff')

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '../demoSeq_pypulseq_results')
output_path = os.path.join(RESULTS_DIR, 'EpiDiffusionRS_PMC_py.seq')
print(f"Writing to: {os.path.abspath(output_path)}")
# Export sequence
os.makedirs(RESULTS_DIR, exist_ok=True)
seq.write(output_path)