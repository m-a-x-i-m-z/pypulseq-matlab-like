
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
from pypulseq.make_block_pulse import make_block_pulse
from pypulseq.make_delay import make_delay
from pypulseq.calc_duration import calc_duration
from pypulseq.calc_rf_center import calc_rf_center

# Define FOV and resolution
fov = 256e-3
Nx = 64
Ny = 64
thickness = 3e-3

# System limits
system = Opts(max_grad=32, grad_unit='mT/m', max_slew=130, slew_unit='T/m/s',
              rf_ringdown_time=20e-6, rf_dead_time=100e-6, adc_dead_time=20e-6)

# Sequence object
seq = Sequence(system)

# Create 90 degree slice selection pulse and gradient
rf, gz, gzReph = make_sinc_pulse(flip_angle=np.pi / 2, system=system, duration=3e-3,
                         slice_thickness=thickness, apodization=0.5, time_bw_product=4, use='excitation', return_gz=True)

# Define other gradients and ADC events
deltak = 1 / fov
kWidth = Nx * deltak
readoutTime = 3.2e-4
gx = make_trapezoid(channel='x', system=system, flat_area=kWidth, flat_time=readoutTime)
adc = make_adc(num_samples=Nx, system=system, duration=gx.flat_time, delay=gx.rise_time)

# Pre-phasing gradients
preTime = 8e-4
gzReph = make_trapezoid(channel='z', system=system, area=-gz.area / 2, duration=preTime)
gxPre = make_trapezoid(channel='x', system=system, area=gx.area / 2 - deltak / 2, duration=preTime)
gyPre = make_trapezoid(channel='y', system=system, area=Ny / 2 * deltak, duration=preTime)

# Phase blip in shortest possible time
dur = np.ceil(2 * np.sqrt(deltak / system.max_slew) / 10e-6) * 10e-6
gy = make_trapezoid(channel='y', system=system, area=deltak, duration=dur)

# Refocusing pulse with spoiling gradients
rf180 = make_block_pulse(flip_angle=np.pi, system=system, duration=500e-6, use='refocusing')
gzSpoil = make_trapezoid(channel='z', system=system, area=gz.area * 2, duration=3 * preTime)

# Calculate delay time
TE = 60e-3
durationToCenter = (Nx / 2 + 0.5) * calc_duration(gx) + Ny / 2 * calc_duration(gy)
rfCenterInclDelay = rf.delay + calc_rf_center(rf)[0]
rf180centerInclDelay = rf180.delay + calc_rf_center(rf180)[0]
delayTE1 = TE / 2 - calc_duration(gz) + rfCenterInclDelay - preTime - calc_duration(gzSpoil) - rf180centerInclDelay
delayTE2 = TE / 2 - calc_duration(rf180) + rf180centerInclDelay - calc_duration(gzSpoil) - durationToCenter

# Define sequence blocks
seq.add_block(rf, gz)
seq.add_block(gxPre, gyPre, gzReph)
seq.add_block(make_delay(delayTE1))
seq.add_block(gzSpoil)
seq.add_block(rf180)
seq.add_block(gzSpoil)
seq.add_block(make_delay(delayTE2))

for i in range(Ny):
    seq.add_block(gx, adc)         # Read one line of k-space
    seq.add_block(gy)              # Phase blip
    gx.amplitude = -gx.amplitude   # Reverse polarity of read gradient

seq.add_block(make_delay(1e-4))

# Timing validation
ok, error_report = seq.check_timing()

if ok:
    print('Timing check passed successfully')
else:
    print('Timing check failed! Error listing follows:')
    print(error_report)

seq.set_definition('FOV', [fov, fov, thickness])
seq.set_definition('Name', 'epise')

# Export sequence
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '../demoSeq_pypulseq_results')
os.makedirs(RESULTS_DIR, exist_ok=True)
seq.write(os.path.join(RESULTS_DIR, 'EpiSpinEcho_py.seq'))