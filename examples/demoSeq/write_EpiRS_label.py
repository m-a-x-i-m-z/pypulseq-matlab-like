
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
from pypulseq_matlab_like.calc_duration import calc_duration
from pypulseq_matlab_like.make_label import make_label
from pypulseq_matlab_like.split_gradient_at import split_gradient_at
from pypulseq_matlab_like.align import align
from pypulseq_matlab_like.add_gradients import add_gradients
from pypulseq_matlab_like.calc_rf_center import calc_rf_center
from pypulseq_matlab_like.scale_grad import scale_grad

# System limits
system = Opts(max_grad=32, grad_unit='mT/m', max_slew=130, slew_unit='T/m/s',
              rf_ringdown_time=30e-6, rf_dead_time=100e-6, adc_dead_time=10e-6, B0=2.89)

# Sequence object
seq = Sequence(system)      # Create a new sequence object
fov = 220e-3
Nx = 96
Ny = Nx  # Define FOV and resolution
thickness = 3e-3            # slice thinckness in mm
sliceGap = 1.5e-3             # slice gap im mm
Nslices = 4
Nrep = 1
TR = 3000e-3

pe_enable = 1               # a flag to quickly disable phase encoding (1/0) as needed for the delay calibration
ro_os = 2                   # oversampling factor (in contrast to the product sequence we don't really need it)
readoutTime = 580e-6  # default value

readoutBW = 1 / readoutTime  # readout bandwidth
print(['Readout bandwidth = ', str(readoutBW), ' Hz/Px'])
partFourierFactor = 1       # partial Fourier factor: 1: full sampling 0: start with ky=0
Nnav = 3		   # navigator echoes for ghost supprerssion

# Create fat-sat pulse
sat_ppm = -3.35
sat_freq = sat_ppm * 1e-6 * system.B0 * system.gamma
rf_fs = make_gauss_pulse(flip_angle=110 * np.pi / 180, system=system, duration=8e-3,
                         bandwidth=abs(sat_freq), freq_ppm=sat_ppm, use='saturation')

rf_fs.phase_ppm = -2 * np.pi * rf_fs.freq_ppm * rf_fs.center

gz_fs = make_trapezoid(channel='z', system=system, delay=calc_duration(rf_fs), area=0.1 / 1e-4) # spoil up to 0.1mm

# Create 90 degree slice selection pulse and gradient
# Fix return_gz=True
rf, gz, gzReph = make_sinc_pulse(flip_angle=np.pi / 2, system=system, duration=2e-3,
                                 slice_thickness=thickness, apodization=0.42, time_bw_product=4, use='excitation', return_gz=True)

# define the output trigger to play out with every slice excitation
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
assert gx.amplitude <= system.max_grad
ESP = 1e3 * calc_duration(gx)
print(['echo spacing = ', str(ESP), ' ms'])

# calculate ADC
assert ro_os >= 2
adcSamples = Nx * ro_os
adcDwell = np.floor(readoutTime / adcSamples * 1e7) * 1e-7
print(['ADC bandwidth = ', str(1 / adcDwell / 1000), ' kHz'])

_phase_mod = 0.1 * np.random.RandomState(5489).rand(int(adcSamples))
adc = make_adc(num_samples=int(adcSamples), dwell=adcDwell, delay=blip_dur / 2, system=system,
               phase_modulation=_phase_mod)

# realign the ADC with respect to the gradient
time_to_center = adc.dwell * ((adcSamples - 1) / 2 + 0.5)
adc.delay = np.round((gx.rise_time + gx.flat_time / 2 - time_to_center) * 1e6) * 1e-6

# split the blip
gy_parts = split_gradient_at(grad=gy, time_point=blip_dur / 2, system=system)
gy_blipup, gy_blipdown, _ = align(right=gy_parts[0], left=gy_parts[1], center=gx)
# Fix add_gradients usage (grads=)
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
# Fix align usage
aligned_grads = align(right=[gxPre, gzReph], left=gyPre)
gxPre, gzReph, gyPre = aligned_grads

# relax the PE prephaser to reduce stimulation
max_dur_pre = calc_duration(gxPre, gyPre, gzReph)
gyPre = make_trapezoid(channel='y', system=system, area=gyPre.area, duration=max_dur_pre)
gyPre.amplitude = gyPre.amplitude * pe_enable

# slice positions
slicePositions = (thickness + sliceGap) * (np.arange(Nslices) - (Nslices - 1) / 2)
slicePositions = np.concatenate((slicePositions[0::2], slicePositions[1::2])) # reorder

# estimate sequence timing
minTR_1slice = calc_duration(gz_fs) + calc_duration(gz) + calc_duration(gzReph) + \
    Nnav * calc_duration(gx) + calc_duration(gyPre) + \
    Ny_meas * calc_duration(gx)

# calculate minimal TE
# Fix calc_rf_center access [0]
min_TE = rf.shape_dur - rf.center + max(rf.ringdown_time, gz.fall_time) + calc_duration(gzReph) + \
    Nnav * calc_duration(gx) + calc_duration(gyPre) + \
    Ny_pre * calc_duration(gx) + calc_duration(gx) / 2

TRdelay = TR - minTR_1slice * Nslices
TRdelay_perSlice = np.round(TRdelay / Nslices / system.block_duration_raster) * system.block_duration_raster
assert TRdelay_perSlice > 0

# change orientation to match the siemens product sequence
# initial readout gradient polarity
ROpolarity = np.sign(gx.amplitude)

# Manually track block duration sum for durPerSlc calculation
seq_duration = 0

for r in range(Nrep):
    seq.add_block(make_label(label='SLC', type='SET', value=0))
    for s in range(Nslices):
        seq.add_block(rf_fs, gz_fs)
        rf.freq_offset = gz.amplitude * slicePositions[s]
        # Fix calc_rf_center access [0]
        rf.phase_offset = -2 * np.pi * rf.freq_offset * rf.center
        seq.add_block(rf, gz, trig)

        if Nnav > 0:
            gxPre = scale_grad(gxPre, -1)
            gx = scale_grad(gx, -1)
            seq.add_block(gxPre, gzReph,
                          make_label(label='NAV', type='SET', value=1),
                          make_label(label='LIN', type='SET', value=int(np.floor(Ny / 2))))
            gxPre = scale_grad(gxPre, -1)

            for n in range(Nnav):
                seq.add_block(make_label(label='REV', type='SET', value= int(np.sign(gx.amplitude) != ROpolarity)),
                              make_label(label='SEG', type='SET', value= int(np.sign(gx.amplitude) != ROpolarity)),
                              make_label(label='AVG', type='SET', value= int(n == Nnav - 1)))
                seq.add_block(gx, adc)
                gx = scale_grad(gx, -1)

            # softdelay TE (skipped in this file, purely label file)
            seq.add_block(gyPre,
                          make_label(label='LIN', type='SET', value=-1),
                          make_label(label='NAV', type='SET', value=0),
                          make_label(label='AVG', type='SET', value=0))
        else:
            seq.add_block(gxPre, gyPre, gzReph,
                          make_label(label='LIN', type='SET', value=-1),
                          make_label(label='NAV', type='SET', value=0),
                          make_label(label='AVG', type='SET', value=0))

        for i in range(Ny_meas):
            lrev = make_label(label='REV', type='SET', value=int(np.sign(gx.amplitude) != ROpolarity))
            lseg = make_label(label='SEG', type='SET', value=int(np.sign(gx.amplitude) != ROpolarity))
            llin = make_label(label='LIN', type='INC', value=1)

            if i == 0:
                seq.add_block(gx, gy_blipup, adc, lrev, lseg, llin)
            elif i == Ny_meas - 1:
                seq.add_block(gx, gy_blipdown, adc, lrev, lseg, llin)
            else:
                seq.add_block(gx, gy_blipdownup, adc, lrev, lseg, llin)

            gx = scale_grad(gx, -1)

        seq.add_block(make_label(label='SLC', type='INC', value=1))

        if np.sign(gx.amplitude) != ROpolarity:
            gx = scale_grad(gx, -1)

        # if s == 0 and r == 0:
            # durPerSlc = np.sum(list(seq.block_durations.values()))

        seq.add_block(make_delay(TRdelay_perSlice))

    seq.add_block(make_label(label='REP', type='INC', value=1))

# Timing validation
ok, error_report = seq.check_timing()

if ok:
    print('Timing check passed successfully')
else:
    print('Timing check failed! Error listing follows:')
    print(error_report)

seq.set_definition('Name', 'epi')
seq.set_definition('FOV', [fov, fov, max(slicePositions) - min(slicePositions) + thickness])
seq.set_definition('SlicePositions', slicePositions)
seq.set_definition('SliceThickness', thickness)
seq.set_definition('SliceGap', sliceGap)
seq.set_definition('ReceiverGainHigh', 1)
seq.set_definition('ReadoutOversamplingFactor', ro_os)
seq.set_definition('TargetGriddedSamples', Nx * ro_os)
seq.set_definition('TrapezoidGriddingParameters',
                   [gx.rise_time, gx.flat_time, gx.fall_time, adc.delay - gx.delay, adc.dwell * adc.num_samples])

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '../demoSeq_pypulseq_results')
output_path = os.path.join(RESULTS_DIR, 'EpiRS_label_py.seq')
print(f"Writing to: {os.path.abspath(output_path)}")
# Export sequence
os.makedirs(RESULTS_DIR, exist_ok=True)
seq.write(output_path)
