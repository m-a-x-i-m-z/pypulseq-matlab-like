
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

# this is a demo low-performance EPI sequence;
# it doesn't use ramp-samping and is only good for educational purposes.

# Sequence object
seq = Sequence()  # Create a new sequence object
fov = 220e-3
Nx = 64
Ny = 64  # Define FOV and resolution
thickness = 3e-3  # slice thinckness
Nslices = 3

# System limits
system = Opts(max_grad=32, grad_unit='mT/m', max_slew=130, slew_unit='T/m/s',
              rf_ringdown_time=30e-6, rf_dead_time=100e-6)

# Create 90 degree slice selection pulse and gradient
rf, gz, _ = make_sinc_pulse(flip_angle=np.pi / 2, system=system, duration=3e-3,
                            slice_thickness=thickness, apodization=0.5, time_bw_product=4,
                            use='excitation', return_gz=True)

# Define other gradients and ADC events
deltak = 1 / fov
kWidth = Nx * deltak
dwellTime = 4e-6  # I want it to be divisible by 2
readoutTime = Nx * dwellTime
flatTime = np.ceil(readoutTime * 1e5) * 1e-5  # round-up to the gradient raster
gx = make_trapezoid(channel='x', system=system, amplitude=kWidth / readoutTime, flat_time=flatTime)
adc = make_adc(num_samples=Nx, duration=readoutTime, delay=gx.rise_time + flatTime / 2 - (readoutTime - dwellTime) / 2)

# Pre-phasing gradients
preTime = 8e-4
gxPre = make_trapezoid(channel='x', system=system, area=-gx.area / 2, duration=preTime)
gzReph = make_trapezoid(channel='z', system=system, area=-gz.area / 2, duration=preTime)
gyPre = make_trapezoid(channel='y', system=system, area=-Ny / 2 * deltak, duration=preTime)

# Phase blip in shortest possible time
dur = np.ceil(2 * np.sqrt(deltak / system.max_slew) / 10e-6) * 10e-6
gy = make_trapezoid(channel='y', system=system, area=deltak, duration=dur)

# Define sequence blocks
TR_1slice = 0
for s in range(Nslices):
    rf.freq_offset = gz.amplitude * thickness * (s - (Nslices - 1) / 2)
    seq.add_block(rf, gz)
    seq.add_block(gxPre, gyPre, gzReph)
    for i in range(Ny):
        seq.add_block(gx, adc)  # Read one line of k-space
        seq.add_block(gy)  # Phase blip
        gx.amplitude = -gx.amplitude  # Reverse polarity of read gradient

    if s == 0:
        TR_1slice = seq.duration()

# Timing validation
ok, error_report = seq.check_timing()

if ok:
    print('Timing check passed successfully')
else:
    print('Timing check failed! Error listing follows:')
    print(error_report)

# seq.plot()

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '../demoSeq_pypulseq_results')
output_path = os.path.join(RESULTS_DIR, 'Epi_py.seq')
print(f"Writing to: {os.path.abspath(output_path)}")
# Export sequence
os.makedirs(RESULTS_DIR, exist_ok=True)
seq.write(output_path)