
import numpy as np
import sys
import os
from scipy.interpolate import interp1d
import copy

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
system = Opts(max_grad=28, grad_unit='mT/m', max_slew=170, slew_unit='T/m/s',
              rf_ringdown_time=0e-6, rf_dead_time=100e-6, adc_dead_time=10e-6)

# Sequence object
seq = Sequence(system)
fov = 240e-3
Nx = 240
alpha = 10
sliceThickness = 5e-3
TR = 20e-3
Nr = Nx * 2
Ndummy = 20
delta = 2 * np.pi / Nr
rf_duration = 0.5e-3
ro_duration = 0.720e-3
ro_os = 2
minRF_to_ADC_time = 70e-6
ro_discard = 0
ro_spoil = 1
rfSpoilingInc = 84

# Create alpha-degree slice selection pulse and gradient
rf, gz, _ = make_sinc_pulse(flip_angle=alpha * np.pi / 180, duration=rf_duration,
                         slice_thickness=sliceThickness, apodization=0.5, time_bw_product=2,
                         center_pos=1, system=system, use='excitation', return_gz=True)

# Resample RF pulse to the ramp
gza = np.array([0, 1, 1, 0])
gzt = np.cumsum([0, gz.rise_time, gz.flat_time, gz.fall_time])
gzt_interp = gzt + gz.delay
# Interpolate gradient amplitude at RF time points
# RF time points: rf.t + rf.delay
rf_t_abs = rf.t + rf.delay
gzas_0 = interp1d(gzt_interp, gza, kind='linear', fill_value='extrapolate')(rf_t_abs)

rft_1 = np.arange(system.rf_raster_time, rf_duration + 0.5 * gz.fall_time + system.rf_raster_time/2, system.rf_raster_time)
rft_1_interp = rft_1 + rf.delay + gz.fall_time * 0.5
gzas_1 = interp1d(gzt_interp, gza, kind='linear', fill_value='extrapolate')(rft_1_interp)
gzas_1[~np.isfinite(gzas_1)] = 0

kzs_0 = np.cumsum(gzas_0)
kzs_1 = np.cumsum(gzas_1)
kzs_0 = kzs_0 - np.max(kzs_0)
kzs_1 = kzs_1 - np.max(kzs_1)

# Resample RF signal
rf_sig_cum = np.cumsum(rf.signal)
# It might be monotonic?
# interp1(x, y, xi).
# gza is 0-1-1-0. Gradient is positive. Accumulation is increasing.
# kzs_0 is sorted.
rfs_1_cum = interp1d(kzs_0, rf_sig_cum, kind='linear', fill_value='extrapolate')(kzs_1)
rfs_1 = np.diff(np.concatenate(([0], rfs_1_cum)))

rf.t = rft_1[:-1]
rf.signal = rfs_1[:-1]
rf.shape_dur = len(rf.signal) * system.rf_raster_time
# Update gz flat time to align
gz.flat_time = np.ceil((gz.flat_time - gz.fall_time * 0.5) / system.grad_raster_time) * system.grad_raster_time
rf.delay = calc_duration(rf, gz) - rf.shape_dur
rf.center = rf.t[-1]
# But `make_sinc_pulse` creates standard object.
# We manually modified `rf.t`.

# Align RO asymmetry
Nxo = int(round(ro_os * Nx))
deltak = 1 / fov / 2
ro_area = Nx * deltak
gx = make_trapezoid(channel='x', flat_area=ro_area, flat_time=ro_duration, system=system)
# ADC duration
adc_dur = np.floor(gx.flat_time / Nxo * 1e7) * 1e-7 * Nxo
adc = make_adc(num_samples=Nxo, duration=adc_dur, system=system)

# RO-spoiling
gx.flat_time = gx.flat_time * ro_spoil

# Calculate timing
TE = np.ceil((minRF_to_ADC_time + adc.dwell * ro_discard) / system.grad_raster_time) * system.grad_raster_time
delayTR = np.ceil((TR - calc_duration(gz) - calc_duration(gx) - TE) / system.grad_raster_time) * system.grad_raster_time

print(f'TE= {round(TE * 1e6)} us; delay in TR:= {np.floor(delayTR * 1e6)} us')

# Set up timing
gx.delay = calc_duration(gz) + TE
adc.delay = np.floor((gx.delay - adc.dwell * 0.5 - adc.dwell * ro_discard) / system.grad_raster_time) * system.grad_raster_time

rf_phase = 0
rf_inc = 0

if Ndummy > 0:
    # PyPulseq `add_block` accepts single label.
    seq.add_block(make_label(label='ONCE', type='SET', value=1))

for i in range(-Ndummy + 1, Nr + 1):
    if Ndummy > 0 and i == 1:
        seq.add_block(make_label(label='ONCE', type='SET', value=0))

    for c in range(2):
        rf.phase_offset = rf_phase / 180 * np.pi
        adc.phase_offset = rf_phase / 180 * np.pi
        rf_inc = (rf_inc + rfSpoilingInc) % 360.0
        rf_phase = (rf_phase + rf_inc) % 360.0

        gz.amplitude = -gz.amplitude

        phi = delta * (i - 1)

        grc = copy.deepcopy(gx)
        grs = copy.deepcopy(gx)
        grc.amplitude = gx.amplitude * np.cos(phi)
        grs.amplitude = gx.amplitude * np.sin(phi)
        grs.channel = 'y'

        if i > 0:
            seq.add_block(rf, gz, grc, grs, adc)
        else:
            seq.add_block(rf, gz, grc, grs)

        seq.add_block(make_delay(delayTR))

# Timing validation
ok, error_report = seq.check_timing()

if ok:
    print('Timing check passed successfully')
else:
    print('Timing check failed! Error listing follows:')
    print(error_report)

seq.set_definition('FOV', [fov, fov, sliceThickness])
seq.set_definition('Name', 'ute_rs')

# Export sequence
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '../demoSeq_pypulseq_results')
os.makedirs(RESULTS_DIR, exist_ok=True)
seq.write(os.path.join(RESULTS_DIR, 'UTE_rs_py.seq'))
