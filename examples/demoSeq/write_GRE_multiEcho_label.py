
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
from pypulseq.calc_duration import calc_duration
from pypulseq.make_label import make_label

# this is a demo GRE sequence, which uses LABEL extension to produce raw
# data reconstuctable by the integrated image reconstruction on the scanner

# System limits
system = Opts(max_grad=28, grad_unit='mT/m', max_slew=150, slew_unit='T/m/s',
              rf_ringdown_time=20e-6, rf_dead_time=100e-6, adc_dead_time=10e-6)

# Sequence object
seq = Sequence(system)  # Create a new sequence object

fov = 224e-3
Nx = 256
Ny = Nx  # Define FOV and resolution
alpha = 15  # flip angle
thickness = 5e-3  # slice
Nslices = 1
TR = 20e-3
# TE=4.3e-3;
TE = np.array([4.38, 6.84, 12]) * 1e-3  # alternatively give a vector here to have multiple TEs (e.g. for field mapping)

# more in-depth parameters
rfSpoilingInc = 84  # RF spoiling increment
roDuration = 3.2e-3  # ADC duration

# Create alpha-degree slice selection pulse and gradient
# make_sinc_pulse returns (rf, gz, gzr) if return_gz=True.
rf, gz, _ = make_sinc_pulse(flip_angle=alpha * np.pi / 180, duration=3e-3,
                                 slice_thickness=thickness, apodization=0.42, time_bw_product=4,
                                 system=system, use='excitation', return_gz=True)

# Define other gradients and ADC events
deltak = 1 / fov
gx = make_trapezoid(channel='x', flat_area=Nx * deltak, flat_time=roDuration, system=system)
adc = make_adc(num_samples=Nx, duration=gx.flat_time, delay=gx.rise_time, system=system)
gxPre = make_trapezoid(channel='x', area=-gx.area / 2, duration=1e-3, system=system)

gzReph = make_trapezoid(channel='z', area=-gz.area / 2, duration=1e-3, system=system)

phaseAreas = -((np.arange(Ny)) - Ny / 2) * deltak  # phase area should be Kmax for clin=0 and -Kmax for clin=Ny... strange

# gradient spoiling
gxSpoil = make_trapezoid(channel='x', area=2 * Nx * deltak, system=system)
gzSpoil = make_trapezoid(channel='z', area=4 / thickness, system=system)

# Calculate timing
delayTE = np.ceil((TE - calc_duration(gxPre) - gz.fall_time - gz.flat_time / 2
                   - calc_duration(gx) / 2) / seq.grad_raster_time) * seq.grad_raster_time
delayTR = np.ceil((TR - calc_duration(gz) - calc_duration(gxPre)
                   - calc_duration(gx) - delayTE) / seq.grad_raster_time) * seq.grad_raster_time

assert np.all(delayTE >= 0)
assert np.all(delayTR >= calc_duration(gxSpoil, gzSpoil))

rf_phase = 0
rf_inc = 0

# all LABELS / counters an flags are automatically initialized to 0 in the beginning, no need to define initial 0's
# so we will just increment LIN after the ADC event (e.g. during the spoiler)

seq.add_block(make_label(label='REV', type='SET', value=1))  # left-right swap fix (needed for 1.4.0)

for r in range(2):
    seq.add_block(make_label(label='LIN', type='SET', value=0), make_label(label='SLC', type='SET', value=0))  # needed to make it compatible to multiple REPs

    # loop over slices
    for s in range(Nslices):
        rf.freq_offset = gz.amplitude * thickness * (s - (Nslices - 1) / 2)
        # loop over phase encodes and define sequence blocks
        for i in range(Ny):
            for c in range(len(TE)):
                rf.phase_offset = rf_phase / 180 * np.pi
                adc.phase_offset = rf_phase / 180 * np.pi
                rf_inc = (rf_inc + rfSpoilingInc) % 360.0
                rf_phase = (rf_phase + rf_inc) % 360.0
                #
                seq.add_block(rf, gz)
                gyPre = make_trapezoid(channel='y', area=phaseAreas[i], duration=calc_duration(gxPre), system=system)
                seq.add_block(gxPre, gyPre, gzReph)
                seq.add_block(make_delay(delayTE[c]))
                seq.add_block(gx, adc)
                gyPre.amplitude = -gyPre.amplitude

                spoil_block_contents = [make_delay(delayTR[c]), gxSpoil, gyPre, gzSpoil]
                # here we demonstrate the technique to combine variable counter-dependent content into the same block
                if c != len(TE) - 1:
                    spoil_block_contents.append(make_label(label='ECO', type='INC', value=1))
                else:
                    if len(TE) > 1:
                        spoil_block_contents.append(make_label(label='ECO', type='SET', value=0))
                    if i != Ny - 1:
                        spoil_block_contents.append(make_label(label='LIN', type='INC', value=1))
                    else:
                        spoil_block_contents.append(make_label(label='LIN', type='SET', value=0))
                        spoil_block_contents.append(make_label(label='SLC', type='INC', value=1))

                seq.add_block(*spoil_block_contents)

    seq.add_block(make_delay(5))
    seq.add_block(make_label(label='REP', type='INC', value=1))  # make the sequence compatible with multiple runs/repetitions

# Timing validation
ok, error_report = seq.check_timing()

if ok:
    print('Timing check passed successfully')
else:
    print('Timing check failed! Error listing follows:')
    print(error_report)

# prepare sequence export
seq.set_definition('FOV', [fov, fov, thickness * Nslices])
seq.set_definition('Name', 'gre_lbl')
seq.set_definition('TE', TE)  # 3 TE values
seq.set_definition('TR', TR)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '../demoSeq_pypulseq_results')
output_path = os.path.join(RESULTS_DIR, 'GRE_multiEcho_label_py.seq')
print(f"Writing to: {os.path.abspath(output_path)}")
# Export sequence
os.makedirs(RESULTS_DIR, exist_ok=True)
seq.write(output_path)
