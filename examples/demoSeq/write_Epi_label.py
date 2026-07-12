
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
from pypulseq.make_trigger import make_trigger
from pypulseq.calc_duration import calc_duration
from pypulseq.make_label import make_label
from pypulseq.calc_rf_center import calc_rf_center

# this is a demo low-performance EPI sequence;

# Sequence object
seq = Sequence()   # Create a new sequence object
fov = 220e-3
Nx = 96
Ny = Nx
thickness = 3e-3
Nslices = 7
sliceGap = 1e-3
Nreps = 4
Navigator = 3

# System limits
system = Opts(max_grad=32, grad_unit='mT/m', max_slew=130, slew_unit='T/m/s',
              rf_ringdown_time=30e-6, rf_dead_time=100e-6)

# Create 90 degree slice selection pulse and gradient
rf, gz, gzReph = make_sinc_pulse(flip_angle=np.pi / 2, system=system, duration=3e-3,
                         slice_thickness=thickness, apodization=0.5, time_bw_product=4, use='excitation', return_gz=True)

# define the trigger
trig = make_trigger('physio1', duration=2000e-6)

# Define other gradients and ADC events
deltak = 1 / fov
kWidth = Nx * deltak
dwellTime = 4e-6
readoutTime = Nx * dwellTime
flatTime = np.ceil(readoutTime * 1e5) * 1e-5
gx = make_trapezoid(channel='x', system=system, amplitude=kWidth / readoutTime, flat_time=flatTime)
adc = make_adc(num_samples=Nx, duration=readoutTime, delay=gx.rise_time + flatTime / 2 - (readoutTime - dwellTime) / 2, system=system)

# Pre-phasing gradients
preTime = 8e-4
gxPre = make_trapezoid(channel='x', system=system, area=-gx.area / 2, duration=preTime)
gzReph = make_trapezoid(channel='z', system=system, area=-gz.area / 2, duration=preTime)
gyPre = make_trapezoid(channel='y', system=system, area=Ny / 2 * deltak, duration=preTime)

# Phase blip
dur = np.ceil(2 * np.sqrt(deltak / system.max_slew) / 10e-6) * 10e-6
gy = make_trapezoid(channel='y', system=system, area=-deltak, duration=dur)

gz_spoil = make_trapezoid(channel='z', system=system, area=deltak * Nx * 4)

# slice positions
slicePositions = (thickness + sliceGap) * (np.arange(Nslices) - (Nslices - 1) / 2)
slicePositions = np.concatenate((slicePositions[0::2], slicePositions[1::2])) # reorder

# Define sequence blocks
for r in range(Nreps):
    seq.add_block(trig, make_label(label='SLC', type='SET', value=0))
    for s in range(Nslices):
        rf.freq_offset = gz.amplitude * thickness * (s - (Nslices - 1) / 2)
        rf.phase_offset = -2 * np.pi * rf.freq_offset * calc_rf_center(rf)[0]
        seq.add_block(rf, gz)
        seq.add_block(gxPre, gzReph,
                      make_label(label='NAV', type='SET', value=1),
                      make_label(label='LIN', type='SET', value=int(np.round(Ny / 2))))
        for n in range(Navigator):
            seq.add_block(gx, adc,
                          make_label(label='REV', type='SET', value=(gx.amplitude < 0)),
                          make_label(label='SEG', type='SET', value=(gx.amplitude < 0)),
                          make_label(label='AVG', type='SET', value=(n == 2)))
            if n != Navigator - 1:
                seq.add_block(make_delay(calc_duration(gy)))
            gx.amplitude = -gx.amplitude

        seq.add_block(gyPre,
                      make_label(label='LIN', type='SET', value=0),
                      make_label(label='NAV', type='SET', value=0),
                      make_label(label='AVG', type='SET', value=0))

        for i in range(Ny):
            seq.add_block(make_label(label='REV', type='SET', value=(gx.amplitude < 0)),
                          make_label(label='SEG', type='SET', value=(gx.amplitude < 0)))
            seq.add_block(gx, adc)
            seq.add_block(gy, make_label(label='LIN', type='INC', value=1))
            gx.amplitude = -gx.amplitude

        seq.add_block(gz_spoil, make_delay(0.1), make_label(label='SLC', type='INC', value=1))

        if (Navigator + Ny) % 2 != 0:
            gx.amplitude = -gx.amplitude

    seq.add_block(make_label(label='REP', type='INC', value=1))

# Timing validation
ok, error_report = seq.check_timing()

if ok:
    print('Timing check passed successfully')
else:
    print('Timing check failed! Error listing follows:')
    print(error_report)

seq.set_definition('FOV', [fov, fov, max(slicePositions) - min(slicePositions) + thickness])
seq.set_definition('Name', 'epi_lbl')
seq.set_definition('SlicePositions', slicePositions)
seq.set_definition('SliceThickness', thickness)
seq.set_definition('SliceGap', sliceGap)

# Export sequence
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '../demoSeq_pypulseq_results')
os.makedirs(RESULTS_DIR, exist_ok=True)
seq.write(os.path.join(RESULTS_DIR, 'Epi_label_py.seq'))