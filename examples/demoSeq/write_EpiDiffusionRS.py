
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
from pypulseq_matlab_like.make_gauss_pulse import make_gauss_pulse
from pypulseq_matlab_like.make_digital_output_pulse import make_digital_output_pulse
from pypulseq_matlab_like.split_gradient_at import split_gradient_at
from pypulseq_matlab_like.align import align
from pypulseq_matlab_like.add_gradients import add_gradients
from pypulseq_matlab_like.calc_duration import calc_duration
from pypulseq_matlab_like.calc_rf_center import calc_rf_center
from pypulseq_matlab_like.make_extended_trapezoid_area import make_extended_trapezoid_area
from pypulseq_matlab_like.make_extended_trapezoid import make_extended_trapezoid

def b_fact_calc(g, delta, DELTA):
    # see DAVY SINNAEVE Concepts in Magnetic Resonance Part A, Vol. 40A(2) 39–65 (2012) DOI 10.1002/cmr.a
    # b = gamma^2  g^2 delta^2 sigma^2 (DELTA + 2 (kappa - lambda) delta)
    # in pulseq we don't need gamma as our gradinets are Hz/m
    # however, we do need 2pi as diffusion equations are all based on phase
    # for rect gradients: sigma=1 lambda=1/2 kappa=1/3
    sigma = 1
    # lambda=1/2;
    # kappa=1/3;
    kappa_minus_lambda = 1 / 3 - 1 / 2
    b = (2 * np.pi * g * delta * sigma) ** 2 * (DELTA + 2 * kappa_minus_lambda * delta)
    return b

# System limits
system = Opts(max_grad=38, grad_unit='mT/m', max_slew=180, slew_unit='T/m/s',
              rf_ringdown_time=10e-6, rf_dead_time=100e-6, adc_dead_time=10e-6, B0=2.89)

# Sequence object
seq = Sequence(system)  # Create a new sequence object
fov = 224e-3
Nx = 112
Ny = Nx  # Define FOV and resolution
thickness = 2e-3  # slice thinckness
Nslices = 3
bFactor = 1000  # s/mm^2
TE = 80e-3

pe_enable = 1  # a flag to quickly disable phase encoding (1/0) as needed for the delay calibration
ro_os = 1  # oversampling factor (in contrast to the product sequence we don't really need it)
readoutTime = 6.3e-4  # this controls the readout bandwidth
partFourierFactor = 0.5  # partial Fourier factor: 1: full sampling 0: start with ky=0

tRFex = 3e-3
tRFref = 3e-3

# Create fat-sat pulse
sat_ppm = -3.35
rf_fs = make_gauss_pulse(flip_angle=110 * np.pi / 180, system=system, duration=8e-3,
                         bandwidth=abs(sat_ppm * 1e-6 * system.B0 * system.gamma),
                         freq_ppm=sat_ppm, use='saturation')
rf_fs.phase_ppm = -2 * np.pi * rf_fs.freq_ppm * rf_fs.center  # compensate for the frequency-offset induced phase
# rf_fs.phase_offset = -2*pi*rf_fs.freq_ppm*1e-6*sys.gamma*sys.B0*rf_fs.center  # compensate for the frequency-offset induced phase
gz_fs = make_trapezoid(channel='z', system=system, delay=calc_duration(rf_fs), area=1 / 1e-4)  # spoil up to 0.1mm

# Create 90 degree slice selection pulse and gradient
rf, gz, gzReph = make_sinc_pulse(flip_angle=np.pi / 2, system=system, duration=tRFex,
                                 slice_thickness=thickness, phase_offset=np.pi / 2, apodization=0.5,
                                 time_bw_product=4, use='excitation', return_gz=True)

# Create 180 degree slice refocusing pulse and gradients
rf180, gz180, _ = make_sinc_pulse(flip_angle=np.pi, system=system, duration=tRFref,
                                  slice_thickness=thickness, apodization=0.5, time_bw_product=4, use='refocusing', return_gz=True)

_, gzr_t, gzr_a = make_extended_trapezoid_area(channel='z', grad_start=gz180.amplitude, grad_end=0,
                                               area=-gzReph.area + 0.5 * gz180.amplitude * gz180.fall_time,
                                               system=system)
times = np.concatenate(([0, gz180.rise_time], gz180.rise_time + gz180.flat_time + gzr_t)) + gz180.delay
amplitudes = np.concatenate(([0, gz180.amplitude], gzr_a))
gz180n = make_extended_trapezoid(channel='z', system=system, times=times, amplitudes=amplitudes)

# define the output trigger to play out with every slice excitatuion
trig = make_digital_output_pulse(channel='osc0', duration=100e-6)  # possible channels: 'osc0','osc1','ext1'

# Define other gradients and ADC events
deltak = 1 / fov
kWidth = Nx * deltak

# Phase blip in shortest possible time
blip_dur = np.ceil(2 * np.sqrt(deltak / system.max_slew) / 10e-6 / 2) * 10e-6 * 2  # we round-up the duration to 2x the gradient raster time
# the split code below fails if this really makes a trpezoid instead of a triangle...
gy = make_trapezoid(channel='y', system=system, area=-deltak, duration=blip_dur)  # we use negative blips to save one k-space line on our way towards the k-space center
# gy = mr.makeTrapezoid('y',sys,'amplitude',deltak/blip_dur*2,'riseTime',blip_dur/2, 'flatTime', 0);

# readout gradient is a truncated trapezoid with dead times at the beginnig
# and at the end each equal to a half of blip_dur
# the area between the blips should be defined by kWidth
# we do a two-step calculation: we first increase the area assuming maximum
# slewrate and then scale down the amlitude to fix the area
extra_area = blip_dur / 2 * blip_dur / 2 * system.max_slew  # check unit!;
gx = make_trapezoid(channel='x', system=system, area=kWidth + extra_area, duration=readoutTime + blip_dur)
actual_area = gx.area - gx.amplitude / gx.rise_time * blip_dur / 2 * blip_dur / 2 / 2 - gx.amplitude / gx.fall_time * blip_dur / 2 * blip_dur / 2 / 2
gx.amplitude = gx.amplitude / actual_area * kWidth
gx.area = gx.amplitude * (gx.flat_time + gx.rise_time / 2 + gx.fall_time / 2)
gx.flat_area = gx.amplitude * gx.flat_time

# calculate ADC
# we use ramp sampling, so we have to calculate the dwell time and the
# number of samples, which are will be qite different from Nx and
# readoutTime/Nx, respectively.
adcDwellNyquist = deltak / gx.amplitude / ro_os
# round-down dwell time to 100 ns
adcDwell = np.floor(adcDwellNyquist * 1e7) * 1e-7
adcSamples = int(np.floor(readoutTime / adcDwell / 4) * 4)  # on Siemens the number of ADC samples need to be divisible by 4
adc = make_adc(num_samples=adcSamples, dwell=adcDwell, delay=blip_dur / 2)
# realign the ADC with respect to the gradient
time_to_center = adc.dwell * ((adcSamples - 1) / 2 + 0.5)  # I've been told that Siemens samples in the center of the dwell period
adc.delay = np.round((gx.rise_time + gx.flat_time / 2 - time_to_center) * 1e6) * 1e-6  # we adjust the delay to align the trajectory with the gradient. We have to aligh the delay to 1us
# this rounding actually makes the sampling points on odd and even readouts
# to appear misalligned. However, on the real hardware this misalignment is
# much stronger anyways due to the grdient delays


# split the blip into two halves and produce a combined synthetic gradient
gy_parts = split_gradient_at(grad=gy, time_point=blip_dur / 2, system=system)
gy_blipup, gy_blipdown, _ = align(right=gy_parts[0], left=gy_parts[1], center=gx)
gy_blipdownup = add_gradients(grads=[gy_blipdown, gy_blipup], system=system)

# pe_enable support
gy_blipup.waveform = gy_blipup.waveform * pe_enable
gy_blipdown.waveform = gy_blipdown.waveform * pe_enable
gy_blipdownup.waveform = gy_blipdownup.waveform * pe_enable

# phase encoding and partial Fourier

Ny_pre = int(np.round(partFourierFactor * Ny / 2 - 1))  # PE steps prior to ky=0, excluding the central line
Ny_post = int(np.round(Ny / 2 + 1))  # PE lines after the k-space center including the central line
Ny_meas = Ny_pre + Ny_post

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
# Use system.grad_raster_time
delayTE1 = np.ceil((TE / 2 - calc_duration(rf, gz) + rfCenterInclDelay - rf180centerInclDelay) / system.grad_raster_time) * system.grad_raster_time
delayTE2tmp = np.ceil((TE / 2 - calc_duration(rf180, gz180n) + rf180centerInclDelay - durationToCenter) / system.grad_raster_time) * system.grad_raster_time
assert delayTE1 >= 0
# delayTE2=delayTE2tmp+mr.calcDuration(rf180,gz180n);
gxPre.delay = 0
gyPre.delay = 0
delayTE2 = delayTE2tmp - calc_duration(gxPre, gyPre)
gxPre, gyPre = align(right=gxPre, left=gyPre)
assert delayTE2 >= 0

# diffusion weithting calculation
# delayTE2 is our window for small_delta
# delayTE1+delayTE2-delayTE2 is our big delta
# we anticipate that we will use the maximum gradient amplitude, so we need
# to shorten delayTE2 by gmax/max_sr to accommodate the ramp down
small_delta = delayTE2 - np.ceil(system.max_grad / system.max_slew / system.grad_raster_time) * system.grad_raster_time
big_delta = delayTE1 + calc_duration(rf180, gz180n)
# we define bFactCalc function below to eventually calculate time-optimal
# gradients. for now we just abuse it with g=1 to give us the coefficient
g = np.sqrt(bFactor * 1e6 / b_fact_calc(1, small_delta, big_delta))  # for now it looks too large!
gr = np.ceil(g / system.max_slew / system.grad_raster_time) * system.grad_raster_time
gDiff = make_trapezoid(channel='z', system=system, amplitude=g, rise_time=gr, flat_time=small_delta - gr)

# assert(mr.calcDuration(gDiff)<=delayTE1); % not needed as we now use the
# new feature by setting the required block duration
# assert(mr.calcDuration(gDiff)<=delayTE2); % dito


# Define sequence blocks
for s in range(Nslices):
    seq.add_block(rf_fs, gz_fs)
    rf.freq_offset = gz.amplitude * thickness * (s - (Nslices - 1) / 2)
    rf.phase_offset = np.pi / 2 - 2 * np.pi * rf.freq_offset * calc_rf_center(rf)[0]  # compensate for the slice-offset induced phase
    rf180.freq_offset = gz180.amplitude * thickness * (s - (Nslices - 1) / 2)
    rf180.phase_offset = -2 * np.pi * rf180.freq_offset * calc_rf_center(rf180)[0]  # compensate for the slice-offset induced phase
    seq.add_block(rf, gz, trig)
    seq.add_block(make_delay(delayTE1), gDiff)
    seq.add_block(rf180, gz180n)
    seq.add_block(make_delay(delayTE2), gDiff)
    seq.add_block(gxPre, gyPre)
    for i in range(Ny_meas):
        if i == 0:
            seq.add_block(gx, gy_blipup, adc)  # Read the first line of k-space with a single half-blip at the end
        elif i == Ny_meas - 1:
            seq.add_block(gx, gy_blipdown, adc)  # Read the last line of k-space with a single half-blip at the beginning
        else:
            seq.add_block(gx, gy_blipdownup, adc)  # Read an intermediate line of k-space with a half-blip at the beginning and a half-blip at the end

        gx.amplitude = -gx.amplitude  # Reverse polarity of read gradient

# Timing validation
ok, error_report = seq.check_timing()

if ok:
    print('Timing check passed successfully')
else:
    print('Timing check failed! Error listing follows:')
    print(error_report)

# prepare sequence export
seq.set_definition('FOV', [fov, fov, thickness * Nslices])
seq.set_definition('Name', 'epi-diff')

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '../demoSeq_pypulseq_results')
output_path = os.path.join(RESULTS_DIR, 'EpiDiffusionRS_py.seq')
print(f"Writing to: {os.path.abspath(output_path)}")
# Export sequence
os.makedirs(RESULTS_DIR, exist_ok=True)
seq.write(output_path)