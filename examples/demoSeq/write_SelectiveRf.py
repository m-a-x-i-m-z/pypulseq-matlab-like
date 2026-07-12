
import numpy as np
import sys
import os
from scipy.interpolate import interp1d

# Add pypulseq source to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from pypulseq.Sequence.sequence import Sequence
from pypulseq.opts import Opts
from pypulseq.make_trapezoid import make_trapezoid
from pypulseq.make_adc import make_adc
from pypulseq.make_sinc_pulse import make_sinc_pulse
from pypulseq.make_arbitrary_rf import make_arbitrary_rf
from pypulseq.make_arbitrary_grad import make_arbitrary_grad
from pypulseq.make_delay import make_delay
from pypulseq.calc_duration import calc_duration
from pypulseq.traj_to_grad import traj_to_grad
from pypulseq.add_ramps import add_ramps


def _colon(start, step, stop):
    count = int(round((stop - start) / step)) + 1
    split = count // 2
    values = np.empty(count)
    values[:split] = start + np.arange(split) * step
    values[split:] = stop - np.arange(count - split - 1, -1, -1) * step
    return values

# System limits
system = Opts(max_grad=32, grad_unit='mT/m', max_slew=130, slew_unit='T/m/s',
              rf_ringdown_time=30e-6, rf_dead_time=100e-6)

# Sequence object
seq = Sequence(system)
fov = 220e-3
Nx = 256
Ny = 256
foe = 200e-3
targetWidth = 22.5e-3
n = 8
T = 8e-3
gradOversampling = True

kMax = (2 * n) / foe / 2
if gradOversampling:
    dTG = system.grad_raster_time / 2
else:
    dTG = system.grad_raster_time

tk = _colon(0.0, dTG, T - dTG)

kx = kMax * (1 - tk / T) * np.cos(2 * np.pi * n * tk / T)
ky = kMax * (1 - tk / T) * np.sin(2 * np.pi * n * tk / T)

# Define RF pulse
tr = _colon(0.0, system.rf_raster_time, T - system.rf_raster_time)
kxRf = interp1d(tk, kx, kind='linear', fill_value='extrapolate')(tr)
kyRf = interp1d(tk, ky, kind='linear', fill_value='extrapolate')(tr)

beta = 2 * np.pi * kMax * targetWidth / 2 / np.sqrt(2)
signal0 = np.exp(-beta**2 * (1 - tr / T)**2) * np.sqrt((2 * np.pi * n * (1 - tr / T))**2 + 1)
signal = signal0 * (1 + np.exp(-1j * 2 * np.pi * 4e-2 * (kxRf + kyRf)))

# Add gradient ramps
kx_ramped, ky_ramped, signal_ramped = add_ramps(k=[kx, ky], rf=signal, system=system, oversampling=gradOversampling)

rf = make_arbitrary_rf(signal=signal_ramped, flip_angle=20 * np.pi / 180, system=system, use='excitation')

# Convert the ramped trajectories to gradient waveforms.
gx_grad_in, _ = traj_to_grad(kx_ramped[np.newaxis, :], raster_time=dTG)
gy_grad_in, _ = traj_to_grad(ky_ramped[np.newaxis, :], raster_time=dTG)
gx_grad_in = gx_grad_in[0]
gy_grad_in = gy_grad_in[0]
gxRf = make_arbitrary_grad(channel='x', waveform=gx_grad_in, system=system, first=0, last=0, oversampling=gradOversampling)
gyRf = make_arbitrary_grad(channel='y', waveform=gy_grad_in, system=system, first=0, last=0, oversampling=gradOversampling)


# Define other gradients and ADC events
deltak = 1 / fov
system_default = Opts()
gx = make_trapezoid(channel='x', flat_area=Nx * deltak, flat_time=6.4e-3, system=system_default)
adc = make_adc(num_samples=Nx, duration=gx.flat_time, delay=gx.rise_time, system=system_default)
gxPre = make_trapezoid(channel='x', area=-gx.area / 2, duration=2e-3, system=system_default)
phaseAreas = (np.arange(Ny) - Ny / 2) * deltak

# Refocusing pulse and spoiling gradients
rf180, gz = make_sinc_pulse(flip_angle=np.pi, system=system, duration=3e-3,
                            slice_thickness=5e-3, apodization=0.5, time_bw_product=4, use='refocusing', return_gz=True)[0:2]

gzSpoil = make_trapezoid(channel='z', area=gx.area, duration=2e-3, system=system_default)

# Calculate timing
delayTE1 = np.ceil((20e-3 / 2 - calc_duration(gzSpoil) - calc_duration(rf180) / 2) / system.grad_raster_time) * system.grad_raster_time
delayTE2 = delayTE1 - calc_duration(gxPre) - calc_duration(gx) / 2
delayTR = 500e-3 - 20e-3 - calc_duration(rf) - calc_duration(gx) / 2
# Loop
for i in range(Ny):
    seq.add_block(rf, gxRf, gyRf)
    seq.add_block(make_delay(delayTE1))
    seq.add_block(gzSpoil)
    seq.add_block(rf180, gz)
    seq.add_block(gzSpoil)
    seq.add_block(make_delay(delayTE2))
    gyPre = make_trapezoid(channel='y', area=phaseAreas[i], duration=2e-3, system=system_default)
    seq.add_block(gxPre, gyPre)
    seq.add_block(gx, adc)
    seq.add_block(make_delay(delayTR))

# Timing validation
ok, error_report = seq.check_timing()

if ok:
    print('Timing check passed successfully')
else:
    print('Timing check failed! Error listing follows:')
    print(error_report)

seq.auto_label(mirror_fourier=True)
seq.set_definition('FOV', [fov, fov, 5e-3])
seq.set_definition('Name', 'se_selRF')

# Export sequence
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '../demoSeq_pypulseq_results')
os.makedirs(RESULTS_DIR, exist_ok=True)
seq.write(os.path.join(RESULTS_DIR, 'SelectiveRf_py.seq'))
