import os
import sys

import numpy as np

# Add pypulseq source to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from pypulseq.TransformFOV.transform_fov import transform_fov
from pypulseq.Sequence.sequence import Sequence
from pypulseq.add_gradients import add_gradients
from pypulseq.align import align
from pypulseq.calc_duration import calc_duration
from pypulseq.calc_rf_center import calc_rf_center
from pypulseq.make_adc import make_adc
from pypulseq.make_digital_output_pulse import make_digital_output_pulse
from pypulseq.make_gauss_pulse import make_gauss_pulse
from pypulseq.make_label import make_label
from pypulseq.make_sinc_pulse import make_sinc_pulse
from pypulseq.make_trapezoid import make_trapezoid
from pypulseq.opts import Opts
from pypulseq.scale_grad import scale_grad
from pypulseq.split_gradient_at import split_gradient_at

# Set system limits
system = Opts(
    max_grad=32,
    grad_unit='mT/m',
    max_slew=130,
    slew_unit='T/m/s',
    rf_ringdown_time=30e-6,
    rf_dead_time=100e-6,
    adc_dead_time=10e-6,
    B0=2.89,
)

seq = Sequence(system)
fov = 220e-3
Nx = 96
Ny = Nx
thickness = 3e-3
sliceGap = 1.5e-3
Nslices = 48
Nrep = 1
TR = 3700e-3

pe_enable = 1
ro_os = 2
readoutTime = 580e-6

readoutBW = 1 / readoutTime
print(['Readout bandwidth = ', str(readoutBW), ' Hz/Px'])
partFourierFactor = 1
Nnav = 3

# Create fat-sat pulse
sat_ppm = -3.45
sat_freq = sat_ppm * 1e-6 * system.B0 * system.gamma
rf_fs = make_gauss_pulse(
    flip_angle=110 * np.pi / 180,
    system=system,
    duration=8e-3,
    dwell=10e-6,
    bandwidth=abs(sat_freq),
    freq_offset=sat_freq,
    use='saturation',
)
rf_fs.phase_offset = -2 * np.pi * rf_fs.freq_offset * calc_rf_center(rf_fs)[0]
gz_fs = make_trapezoid(channel='z', system=system, delay=calc_duration(rf_fs), area=0.1 / 1e-4)

# Create 90 degree slice selection pulse and gradient
rf, gz, gzReph = make_sinc_pulse(
    flip_angle=np.pi / 2,
    system=system,
    duration=2e-3,
    slice_thickness=thickness,
    apodization=0.42,
    time_bw_product=4,
    use='excitation',
    return_gz=True,
)

trig = make_digital_output_pulse('osc0', duration=100e-6)

# Define other gradients and ADC events
deltak = 1 / fov
kWidth = Nx * deltak

blip_dur = np.ceil(2 * np.sqrt(deltak / system.max_slew) / 10e-6 / 2) * 10e-6 * 2
gy = make_trapezoid(channel='y', system=system, area=-deltak, duration=blip_dur)

extra_area = blip_dur / 2 * blip_dur / 2 * system.max_slew
gx = make_trapezoid(channel='x', system=system, area=kWidth + extra_area, duration=readoutTime + blip_dur)
actual_area = (
    gx.area
    - gx.amplitude / gx.rise_time * blip_dur / 2 * blip_dur / 2 / 2
    - gx.amplitude / gx.fall_time * blip_dur / 2 * blip_dur / 2 / 2
)
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
print(
    'Actual RO oversampling factor is %g, Siemens recommends it to be above 1.3'
    % (deltak / gx.amplitude / adcDwell)
)
adc = make_adc(num_samples=int(adcSamples), dwell=adcDwell, delay=blip_dur / 2, system=system)

time_to_center = adc.dwell * ((adcSamples - 1) / 2 + 0.5)
adc.delay = np.round((gx.rise_time + gx.flat_time / 2 - time_to_center) * 1e6) * 1e-6

# Split the blip and create the combined PE gradient
gy_parts = split_gradient_at(grad=gy, time_point=blip_dur / 2, system=system)
gy_blipup, gy_blipdown, _ = align('right', gy_parts[0], 'left', gy_parts[1], gx)
gy_blipdownup = add_gradients(grads=(gy_blipdown, gy_blipup), system=system)

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
gxPre, gyPre, gzReph = align('right', gxPre, 'left', gyPre, gzReph)
gyPre = make_trapezoid(channel='y', system=system, area=gyPre.area, duration=calc_duration(gxPre, gyPre, gzReph))
gyPre.amplitude = gyPre.amplitude * pe_enable

# slice positions
slicePositions = (thickness + sliceGap) * (np.arange(Nslices) - (Nslices - 1) / 2)

minTR_1slice = (
    calc_duration(gz_fs)
    + calc_duration(gz)
    + calc_duration(gzReph)
    + Nnav * calc_duration(gx)
    + calc_duration(gyPre)
    + Ny_meas * calc_duration(gx)
)
TRdelay = TR - minTR_1slice * Nslices
TRdelay_perSlice = np.round(TRdelay / Nslices / system.block_duration_raster) * system.block_duration_raster
TRdelay_perSlice = 0
assert TRdelay_perSlice >= 0

TE = (
    rf.shape_dur / 2
    + rf.ringdown_time
    + calc_duration(gzReph)
    + Nnav * calc_duration(gx)
    + calc_duration(gyPre)
    + Ny_meas / 2 * calc_duration(gx)
    - calc_duration(gx) / 2
)
actualTR = (minTR_1slice + TRdelay_perSlice) * Nslices
print(['actual TR = ', str(actualTR * 1e3), ' ms', ', actual TE = ', str(1000 * TE), ' ms'])

ROpolarity = np.sign(gx.amplitude)

# Build one-slice kernel; additional slices are appended by transform_fov translation
for _ in range(Nrep):
    seq.add_block(make_label(label='SLC', type='SET', value=0))
    s = 0
    seq.add_block(rf_fs, gz_fs)
    rf.freq_offset = gz.amplitude * slicePositions[s]
    rf.phase_offset = -2 * np.pi * rf.freq_offset * calc_rf_center(rf)[0]
    seq.add_block(rf, gz, trig)

    if Nnav > 0:
        gxPre = scale_grad(gxPre, -1)
        gx = scale_grad(gx, -1)
        seq.add_block(
            gxPre,
            gzReph,
            make_label(label='NAV', type='SET', value=1),
            make_label(label='LIN', type='SET', value=int(np.floor(Ny / 2))),
        )
        gxPre = scale_grad(gxPre, -1)

        for n in range(Nnav):
            rev_seg = int(np.sign(gx.amplitude) != ROpolarity)
            seq.add_block(
                make_label(label='REV', type='SET', value=rev_seg),
                make_label(label='SEG', type='SET', value=rev_seg),
                make_label(label='AVG', type='SET', value=int(n == Nnav - 1)),
            )
            seq.add_block(gx, adc)
            gx = scale_grad(gx, -1)

        seq.add_block(
            gyPre,
            make_label(label='LIN', type='SET', value=-1),
            make_label(label='NAV', type='SET', value=0),
            make_label(label='AVG', type='SET', value=0),
        )
    else:
        seq.add_block(
            gxPre,
            gyPre,
            gzReph,
            make_label(label='LIN', type='SET', value=-1),
            make_label(label='NAV', type='SET', value=0),
            make_label(label='AVG', type='SET', value=0),
        )

    for i in range(Ny_meas):
        rev_seg = int(np.sign(gx.amplitude) != ROpolarity)
        lrev = make_label(label='REV', type='SET', value=rev_seg)
        lseg = make_label(label='SEG', type='SET', value=rev_seg)
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
    seq.add_block(TRdelay_perSlice)
    seq.add_block(make_label(label='REP', type='INC', value=1))

# do multislice
if Nslices > 1:
    nBlk = len(seq.block_durations)
    for s in range(1, Nslices):
        dz = slicePositions[s] - slicePositions[0]
        transformer = transform_fov(translation=[0, 0, dz])
        seq = transformer.apply_to_seq(seq, same_seq=True, block_range=[1, nBlk])

ok, error_report = seq.check_timing()
if ok:
    print('Timing check passed successfully')
else:
    print('Timing check failed! Error listing follows:')
    print(error_report)

# prepare the sequence output for the scanner (MATLAB-aligned definitions)
seq.set_definition('Name', 'epi')
seq.set_definition('FOV', [fov, fov, max(slicePositions) - min(slicePositions) + thickness])
seq.set_definition('SlicePositions', slicePositions)
seq.set_definition('SliceThickness', thickness)
seq.set_definition('SliceGap', sliceGap)
seq.set_definition('ReceiverGainHigh', 1)
seq.set_definition('ReadoutOversamplingFactor', ro_os)
seq.set_definition('TargetGriddedSamples', Nx * ro_os)
seq.set_definition(
    'TrapezoidGriddingParameters',
    [gx.rise_time, gx.flat_time, gx.fall_time, adc.delay - gx.delay, adc.dwell * adc.num_samples],
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '../demoSeq_pypulseq_results')
output_path = os.path.join(RESULTS_DIR, 'EpiRS_label_trans_py.seq')
print(f'Writing to: {os.path.abspath(output_path)}')
os.makedirs(RESULTS_DIR, exist_ok=True)
seq.write(output_path)
