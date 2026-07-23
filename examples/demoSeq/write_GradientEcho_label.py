
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
from pypulseq_matlab_like.calc_duration import calc_duration
from pypulseq_matlab_like.make_label import make_label

# System limits
system = Opts(max_grad=28, grad_unit='mT/m', max_slew=150, slew_unit='T/m/s',
              rf_ringdown_time=20e-6, rf_dead_time=100e-6, adc_dead_time=10e-6)

# Sequence object
seq = Sequence(system)

fov = 224e-3
Nx = 256
Ny = Nx
alpha = 15
thickness = 5e-3
Nslices = 1
TR = 10e-3
TE = [4.3e-3] # List for potential multi-echo

rfSpoilingInc = 84
roDuration = 3.2e-3

# Create alpha-degree slice selection pulse and gradient
rf, gz = make_sinc_pulse(flip_angle=alpha * np.pi / 180, system=system, duration=3e-3,
                         slice_thickness=thickness, apodization=0.42, time_bw_product=4, use='excitation', return_gz=True) [0:2]

# Define other gradients and ADC events
deltak = 1 / fov
gx = make_trapezoid(channel='x', flat_area=Nx * deltak, flat_time=roDuration, system=system)
adc = make_adc(num_samples=Nx, duration=gx.flat_time, delay=gx.rise_time, system=system)
gxPre = make_trapezoid(channel='x', area=-gx.area / 2, duration=1e-3, system=system)
gzReph = make_trapezoid(channel='z', area=-gz.area / 2, duration=1e-3, system=system)
phaseAreas = -((np.arange(Ny) - Ny / 2) * deltak)

# gradient spoiling
gxSpoil = make_trapezoid(channel='x', area=2 * Nx * deltak, system=system)
gzSpoil = make_trapezoid(channel='z', area=4 / thickness, system=system)

# Calculate timing
delayTE = [np.ceil((te - calc_duration(gxPre) - gz.fall_time - gz.flat_time / 2 - calc_duration(gx) / 2) / system.grad_raster_time) * system.grad_raster_time for te in TE]
# Note: calc_duration(gz) is used for TR calculation.
# `delayTR = TR - ... - delayTE`.
# If multiple TEs, we need to handle multi-echo timing.
# But for single TE 4.3e-3:
delayTE = delayTE[0]
delayTR = np.ceil((TR - calc_duration(gz) - calc_duration(gxPre) - calc_duration(gx) - delayTE) / system.grad_raster_time) * system.grad_raster_time
assert delayTE >= 0
assert delayTR >= calc_duration(gxSpoil, gzSpoil)

rf_phase = 0
rf_inc = 0

seq.add_block(make_label(label='REV', type='SET', value=1))

for r in range(2):
    seq.add_block(make_label(label='LIN', type='SET', value=0), make_label(label='SLC', type='SET', value=0))

    for s in range(Nslices):
        rf.freq_offset = gz.amplitude * thickness * (s - (Nslices - 1) / 2)

        for i in range(Ny):
            rf.phase_offset = rf_phase / 180 * np.pi
            adc.phase_offset = rf_phase / 180 * np.pi
            rf_inc = (rf_inc + rfSpoilingInc) % 360.0
            rf_phase = (rf_phase + rf_inc) % 360.0

            seq.add_block(rf, gz)
            gyPre = make_trapezoid(channel='y', area=phaseAreas[i], duration=calc_duration(gxPre), system=system)
            seq.add_block(gxPre, gyPre, gzReph)
            seq.add_block(make_delay(delayTE))
            seq.add_block(gx, adc)
            gyPre.amplitude = -gyPre.amplitude

            # Construct spoil block
            spoilBlockContents = [make_delay(delayTR), gxSpoil, gyPre, gzSpoil]
            if i != Ny - 1:
                spoilBlockContents.append(make_label(label='LIN', type='INC', value=1))
            else:
                spoilBlockContents.append(make_label(label='LIN', type='SET', value=0))
                spoilBlockContents.append(make_label(label='SLC', type='INC', value=1))

            seq.add_block(*spoilBlockContents)

    seq.add_block(make_delay(5.0))
    seq.add_block(make_label(label='REP', type='INC', value=1))

# Timing validation
ok, error_report = seq.check_timing()

if ok:
    print('Timing check passed successfully')
else:
    print('Timing check failed! Error listing follows:')
    print(error_report)

seq.set_definition('FOV', [fov, fov, thickness * Nslices])
seq.set_definition('Name', 'gre_lbl')

# Export sequence
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '../demoSeq_pypulseq_results')
os.makedirs(RESULTS_DIR, exist_ok=True)
seq.write(os.path.join(RESULTS_DIR, 'GradientEcho_label_py.seq'))
