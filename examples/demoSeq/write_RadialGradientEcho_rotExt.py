
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
from pypulseq.make_rotation import make_rotation

fov = 260e-3
Nx = 320
alpha = 10
sliceThickness = 3e-3
TE = 8e-3
TR = 20e-3
Nr = 256
Ndummy = 20
delta = np.pi / Nr
rfSpoilingInc = 84

# System limits
system = Opts(max_grad=28, grad_unit='mT/m', max_slew=120, slew_unit='T/m/s',
              rf_ringdown_time=20e-6, rf_dead_time=100e-6, adc_dead_time=10e-6)

# Sequence object
seq = Sequence(system)

# Create alpha-degree slice selection pulse and gradient
rf, gz = make_sinc_pulse(flip_angle=alpha * np.pi / 180, duration=4e-3,
                         slice_thickness=sliceThickness, apodization=0.5, time_bw_product=4, system=system,
                         use='excitation', return_gz=True)[0:2]

# Define other gradients and ADC events
deltak = 1 / fov
gx = make_trapezoid(channel='x', flat_area=Nx * deltak, flat_time=6.4e-3 / 5, system=system)
adc = make_adc(num_samples=Nx, duration=gx.flat_time, delay=gx.rise_time, system=system)
gxPre = make_trapezoid(channel='x', area=-gx.area / 2 - deltak / 2, duration=2e-3, system=system)
gzReph = make_trapezoid(channel='z', area=-gz.area / 2, duration=2e-3, system=system)

# gradient spoiling
gxSpoil = make_trapezoid(channel='x', area=0.5 * Nx * deltak, system=system)
gzSpoil = make_trapezoid(channel='z', area=4 / sliceThickness, system=system)

# Calculate timing
delayTE = np.ceil((TE - calc_duration(gxPre) - gz.fall_time - gz.flat_time / 2 - calc_duration(gx) / 2) / system.grad_raster_time) * system.grad_raster_time
delayTR = np.ceil((TR - calc_duration(gxPre) - calc_duration(gz) - calc_duration(gx) - delayTE) / system.grad_raster_time) * system.grad_raster_time
assert delayTR >= calc_duration(gxSpoil, gzSpoil)

rf_phase = 0
rf_inc = 0

TE_list = [TE] if isinstance(TE, float) else TE

for i in range(-Ndummy, Nr + 1):
    for c in range(len(TE_list)):
        rf.phase_offset = rf_phase / 180 * np.pi
        adc.phase_offset = rf_phase / 180 * np.pi
        rf_inc = (rf_inc + rfSpoilingInc) % 360.0
        rf_phase = (rf_phase + rf_inc) % 360.0

        seq.add_block(rf, gz)
        phi = delta * (i - 1)
        rot = make_rotation(phi)

        seq.add_block(gxPre, gzReph, rot)
        seq.add_block(make_delay(delayTE))

        if i > 0:
            seq.add_block(gx, adc, rot)
        else:
            seq.add_block(gx, rot)

        seq.add_block(gxSpoil, gzSpoil, make_delay(delayTR), rot)

# Timing validation
ok, error_report = seq.check_timing()

if ok:
    print('Timing check passed successfully')
else:
    print('Timing check failed! Error listing follows:')
    print(error_report)

seq.set_definition('FOV', [fov, fov, sliceThickness])
seq.set_definition('Name', 'gre_rad')

# Export sequence
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '../demoSeq_pypulseq_results')
os.makedirs(RESULTS_DIR, exist_ok=True)
seq.write(os.path.join(RESULTS_DIR, 'RadialGradientEcho_rotExt_py.seq'))
