
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

# Define FOV and resolution
fov = 256e-3
sliceThickness = 5e-3
Nx = 128
Ny = Nx

# Define sequence parameters
TE = 8e-3
TR = 16e-3
alpha = 30

# System limits
system = Opts(max_grad=20, grad_unit='mT/m', max_slew=120, slew_unit='T/m/s',
              rf_ringdown_time=20e-6, rf_dead_time=100e-6)
sys_default = Opts()

# Create a new sequence object
# Sequence object
seq = Sequence(system)

# Create slice selection alpha-pulse and corresponding gradients
rf, gz, gzReph = make_sinc_pulse(flip_angle=alpha * np.pi / 180, duration=4e-3,
                                 slice_thickness=sliceThickness, apodization=0.5, time_bw_product=4,
                                 system=system, use='excitation', return_gz=True)

# Define other gradients and ADC events
deltak = 1 / fov
gx = make_trapezoid(channel='x', flat_area=Nx * deltak, flat_time=6.4e-3, system=sys_default)
adc = make_adc(num_samples=Nx, duration=gx.flat_time, delay=gx.rise_time, system=system)
gxPre = make_trapezoid(channel='x', area=-gx.area / 2, duration=2e-3, system=sys_default)
phaseAreas = ((np.arange(Ny)) - Ny / 2) * deltak

# Calculate timing
delayTE = np.round((TE - calc_duration(gxPre) - calc_duration(gz) / 2
                    - calc_duration(gx) / 2) / seq.grad_raster_time) * seq.grad_raster_time
delayTR = np.round((TR - calc_duration(gxPre) - calc_duration(gz)
                    - calc_duration(gx) - delayTE) / seq.grad_raster_time) * seq.grad_raster_time

# Loop over phase encodes and define sequence blocks
for i in range(Ny):
    seq.add_block(rf, gz)
    gyPre = make_trapezoid(channel='y', area=phaseAreas[i], duration=2e-3, system=sys_default)
    seq.add_block(gxPre, gyPre, gzReph)
    seq.add_block(make_delay(delayTE))
    seq.add_block(gx, adc)
    seq.add_block(make_delay(delayTR))

# Timing validation
ok, error_report = seq.check_timing()

if ok:
    print('Timing check passed successfully')
else:
    print('Timing check failed! Error listing follows:')
    print(error_report)

# export definitions
seq.set_definition('FOV', [fov, fov, sliceThickness])
seq.set_definition('Name', 'DEMO_gre0')

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '../demoSeq_pypulseq_results')
output_path = os.path.join(RESULTS_DIR, 'GRE_live_demo_step0_py.seq')
print(f"Writing to: {os.path.abspath(output_path)}")
# Export sequence
os.makedirs(RESULTS_DIR, exist_ok=True)
seq.write(output_path)
