
import numpy as np
import sys
import os

# Add pypulseq source to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from pypulseq.Sequence.sequence import Sequence
from pypulseq.opts import Opts
from pypulseq.make_block_pulse import make_block_pulse
from pypulseq.make_adc import make_adc
from pypulseq.make_delay import make_delay
from pypulseq.calc_duration import calc_duration

# System limits
system = Opts(rf_ringdown_time=20e-6, rf_dead_time=100e-6, adc_dead_time=20e-6)

# Sequence object
seq = Sequence(system)  # Create a new sequence object
Nx = 4096
Nrep = 16

# Create non-selective pulse
rf = make_block_pulse(flip_angle=np.pi / 2, duration=0.3e-3, system=system, use='excitation')

# Define delays and ADC events
adc = make_adc(num_samples=Nx, duration=512e-3, system=system, delay=system.adc_dead_time)
delayTE = 20e-3
delayTR = 5000e-3

#
assert delayTE >= calc_duration(rf)
assert delayTR >= calc_duration(adc)

# Loop over repetitions and define sequence blocks
for i in range(Nrep):
    seq.add_block(rf, make_delay(delayTE))
    seq.add_block(adc, make_delay(delayTR))

# Timing validation
ok, error_report = seq.check_timing()

if ok:
    print('Timing check passed successfully')
else:
    print('Timing check failed! Error listing follows:')
    print(error_report)

seq.set_definition('Name', 'fid')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '../demoSeq_pypulseq_results')
output_path = os.path.join(RESULTS_DIR, 'Fid_py.seq')
print(f"Writing to: {os.path.abspath(output_path)}")
# Export sequence
os.makedirs(RESULTS_DIR, exist_ok=True)
seq.write(output_path)