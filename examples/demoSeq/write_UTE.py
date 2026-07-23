
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

# System limits
system = Opts(max_grad=28, grad_unit='mT/m', max_slew=100, slew_unit='T/m/s',
              rf_ringdown_time=20e-6, rf_dead_time=100e-6, adc_dead_time=10e-6)

# Sequence object
seq = Sequence(system)
fov = 250e-3
Nx = 256
alpha = 10
sliceThickness = 3e-3
TR = 10e-3
Nr = 128
delta = 2 * np.pi / Nr
ro_duration = 2.56e-3
ro_os = 2
ro_asymmetry = 1
minRF_to_ADC_time = 50e-6
rfSpoilingInc = 84

# Create alpha-degree slice selection pulse and gradient
# Check make_sinc_pulse arguments.
# `center_pos` (0 to 1, default 0.5).
rf, gz, gzReph = make_sinc_pulse(flip_angle=alpha * np.pi / 180, duration=1e-3,
                                 slice_thickness=sliceThickness, apodization=0.5, time_bw_product=2,
                                 center_pos=1, system=system, use='excitation', return_gz=True)

# Align RO assymmetry to ADC samples
Nxo = int(round(ro_os * Nx))
ro_asymmetry = round(ro_asymmetry * Nxo / 2) / Nxo * 2

deltak = 1 / fov / (1 + ro_asymmetry)
ro_area = Nx * deltak
gx = make_trapezoid(channel='x', flat_area=ro_area, flat_time=ro_duration, system=system)
adc = make_adc(num_samples=Nxo, duration=gx.flat_time, delay=gx.rise_time, system=system)

# gxPre
gxPre_area = -(gx.area - ro_area) / 2 - gx.amplitude * adc.dwell / 2 - ro_area / 2 * (1 - ro_asymmetry)
gxPre = make_trapezoid(channel='x', area=gxPre_area, system=system)

# gradient spoiling
gxSpoil = make_trapezoid(channel='x', area=0.2 * Nx * deltak, system=system)

# Calculate timing
TE = gz.fall_time + calc_duration(gxPre, gzReph) + gx.rise_time + adc.dwell * Nxo / 2 * (1 - ro_asymmetry)
delayTR = np.ceil((TR - calc_duration(gxPre, gzReph) - calc_duration(gz) - calc_duration(gx)) / system.grad_raster_time) * system.grad_raster_time
assert delayTR >= calc_duration(gxSpoil)

print(f'TE= {round(TE * 1e6)} us')

# Align
if calc_duration(gzReph) > calc_duration(gxPre):
    gxPre.delay = calc_duration(gzReph) - calc_duration(gxPre)

rf_phase = 0
rf_inc = 0

for i in range(Nr):
    for c in range(2):
        rf.phase_offset = rf_phase / 180 * np.pi
        adc.phase_offset = rf_phase / 180 * np.pi
        rf_inc = (rf_inc + rfSpoilingInc) % 360.0
        rf_phase = (rf_phase + rf_inc) % 360.0

        gz.amplitude = -gz.amplitude
        gzReph.amplitude = -gzReph.amplitude

        seq.add_block(rf, gz)

        phi = delta * i

        # Rotate gradients
        # PyPulseq objects are mutable.
        # But `add_block` copies? No, it adds reference.
        # Use `copy` module or create new.
        # `gxPre` and `gx` and `gxSpoil` are base objects.
        # We need to create rotated versions for each block.
        # Actually, `seq.add_block` accepts objects.
        # We can create new objects with specific amplitudes.

        # Helper to scale/rotate
        import copy
        gpc = copy.deepcopy(gxPre)
        gps = copy.deepcopy(gxPre)
        gpc.amplitude = gxPre.amplitude * np.cos(phi)
        gps.amplitude = gxPre.amplitude * np.sin(phi)
        gps.channel = 'y'

        grc = copy.deepcopy(gx)
        grs = copy.deepcopy(gx)
        grc.amplitude = gx.amplitude * np.cos(phi)
        grs.amplitude = gx.amplitude * np.sin(phi)
        grs.channel = 'y'

        gsc = copy.deepcopy(gxSpoil)
        gss = copy.deepcopy(gxSpoil)
        gsc.amplitude = gxSpoil.amplitude * np.cos(phi)
        gss.amplitude = gxSpoil.amplitude * np.sin(phi)
        gss.channel = 'y'

        seq.add_block(gpc, gps, gzReph)
        seq.add_block(grc, grs, adc)
        seq.add_block(gsc, gss, make_delay(delayTR))

# Timing validation
ok, error_report = seq.check_timing()

if ok:
    print('Timing check passed successfully')
else:
    print('Timing check failed! Error listing follows:')
    print(error_report)

seq.set_definition('FOV', [fov, fov, sliceThickness])
seq.set_definition('Name', 'ute')

# Export sequence
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '../demoSeq_pypulseq_results')
os.makedirs(RESULTS_DIR, exist_ok=True)
seq.write(os.path.join(RESULTS_DIR, 'UTE_py.seq'))
