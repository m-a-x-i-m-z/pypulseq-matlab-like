
import numpy as np
import sys
import os


def matlab_round(x):
    x = np.asarray(x)
    return np.where(x >= 0, np.floor(x + 0.5), np.ceil(x - 0.5))

# Add pypulseq source to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from pypulseq.Sequence.sequence import Sequence
from pypulseq.opts import Opts
from pypulseq.make_trapezoid import make_trapezoid
from pypulseq.make_adc import make_adc, calc_adc_segments
from pypulseq.make_gauss_pulse import make_gauss_pulse
from pypulseq.make_sinc_pulse import make_sinc_pulse
from pypulseq.make_arbitrary_grad import make_arbitrary_grad
from pypulseq.make_extended_trapezoid import make_extended_trapezoid
from pypulseq.make_delay import make_delay
from pypulseq.calc_duration import calc_duration
from pypulseq.calc_rf_center import calc_rf_center
from pypulseq.traj_to_grad import traj_to_grad
from pypulseq.rotate import rotate

fov = 256e-3
Nx = 256
Ny = Nx
sliceThickness = 3e-3
Nslices = 11
interleaves = 4
TRdelay = 1
adcOversampling = 2
phi = 0

# System limits
system = Opts(
    max_grad=40,
    grad_unit='mT/m',
    max_slew=200,
    slew_unit='T/m/s',
    rf_ringdown_time=20e-6,
    rf_dead_time=100e-6,
    adc_dead_time=10e-6,
    adc_samples_limit=8192,
)

# Sequence object
seq = Sequence(system)

# Create fat-sat pulse
B0 = 2.89
sat_ppm = -3.35
sat_freq = sat_ppm * 1e-6 * B0 * system.gamma
rf_fs = make_gauss_pulse(
    flip_angle=110 * np.pi / 180,
    system=system,
    duration=8e-3,
    dwell=10e-6,
    bandwidth=abs(sat_freq),
    freq_ppm=sat_ppm,
    use='saturation',
)
rf_fs.phase_ppm = -2 * np.pi * rf_fs.freq_ppm * rf_fs.center

gz_fs = make_trapezoid(channel='z', system=system, delay=calc_duration(rf_fs), area=1 / 1e-4)

# Create 90 degree slice selection pulse and gradient
rf, gz, gzReph = make_sinc_pulse(flip_angle=np.pi / 2, system=system, duration=3e-3,
                                 slice_thickness=sliceThickness, apodization=0.5, time_bw_product=4,
                                 use='excitation', return_gz=True)

# calculate a raw single-shot Archimedian spiral trajectory
deltak = 1 / fov
kRadius = int(matlab_round(Nx / 2))
kSamples = int(matlab_round(2 * np.pi * kRadius)) * adcOversampling
tos_calculation = 25
gradOversampling = True

cmax = kRadius * kSamples * tos_calculation / interleaves
slowStartingFactor = cmax
slowStarting = lambda c: c - slowStartingFactor * np.log(1 + c / slowStartingFactor)

ka = np.zeros(int(cmax) + 1, dtype=complex)

for c in range(int(cmax) + 1):
    slowStartingC = slowStarting(c) / slowStarting(cmax) * cmax
    r = deltak * slowStartingC * interleaves / kSamples / tos_calculation
    a = slowStartingC * 2 * np.pi / kSamples / tos_calculation
    ka[c] = r * np.exp(1j * a)

ka = np.array([ka.real, ka.imag])

dt = system.grad_raster_time / tos_calculation
# `pypulseq.traj_to_grad` usually returns `grad_waveform`.
# Wait, I need to check `traj_to_grad` signature from PyPulseq source or assume standard.
# Assuming it returns `grad` (complex or 2-row).
# But here `ka` is 2-row.
# `traj_to_grad` in python might handle 2D.
# `writeSpiral.m` uses `mr.traj2grad`.
# In Python, I can implement numerical differentiation of trajectory.
# `k = gamma * integral(g)`. `g = dk/dt / gamma`.
# Difference method.
ga, sa = traj_to_grad(ka, raster_time=dt, first_grad_step_half_raster=tos_calculation == 1,
                  conservative_slew_estimate=True, system=system)

# limit analysis
safety_margin = 0.97
g_abs = np.abs(ga[0] + 1j * ga[1])
s_abs = np.abs(sa[0] + 1j * sa[1])

dt_gabs = g_abs / (system.max_grad * safety_margin) * dt
dt_sabs = np.sqrt(s_abs / (system.max_slew * safety_margin)) * dt

dt_opt = np.maximum(dt_gabs, dt_sabs)

t_smooth = np.concatenate(([0], np.cumsum(dt_opt)))

dt_grad = system.grad_raster_time / (1 + int(gradOversampling)) # True -> 2

if gradOversampling:
    safety_factor_1st_timestep = 0.5
    t_end = t_smooth[-1] - safety_factor_1st_timestep * dt_grad
    t_grad = np.concatenate(([0], (safety_factor_1st_timestep + np.arange(np.floor(t_end / dt_grad) + 1)) * dt_grad))
else:
    t_end = t_smooth[-1] - 0.5 * dt_grad
    t_grad = np.concatenate(([0], (0.5 + np.arange(np.floor(t_end / dt_grad) + 1)) * dt_grad))

kopt = np.vstack((
    np.interp(t_grad, t_smooth, ka[0]),
    np.interp(t_grad, t_smooth, ka[1]),
))

# Calculate gradient from optimized k-space
gos, sos = traj_to_grad(kopt, raster_time=dt_grad, first_grad_step_half_raster=not gradOversampling,
                    system=system)

spiral_grad_shape = gos

# calculate the ADC readout
adcTime = dt_grad * spiral_grad_shape.shape[1]
adcSamplesDesired = kRadius * kSamples / interleaves
adcDwell = max(matlab_round(adcTime / adcSamplesDesired / system.adc_raster_time) * system.adc_raster_time, 1e-6)
adcSamplesDesired = int(np.ceil(adcTime / adcDwell))
adcSegments, adcSamplesPerSegment = calc_adc_segments(adcSamplesDesired, adcDwell, system=system, mode='shorten')

adcSamples = adcSegments * adcSamplesPerSegment
adc = make_adc(
    num_samples=adcSamples,
    dwell=adcDwell,
    delay=matlab_round((calc_duration(gzReph) - adcDwell / 2) / system.rf_raster_time)
    * system.rf_raster_time,
    system=system,
)

# Extend spiral to ensure odd length (Pulseq standard for oversampled shapes)
# Note: we don't end at zero here as it would violate slew rate; spoilers handle the ramp down.
if not gradOversampling:
    spiral_grad_shape = np.c_[spiral_grad_shape, spiral_grad_shape[:, -1]]
else:
    spiral_grad_shape = np.c_[spiral_grad_shape, spiral_grad_shape[:, -1], spiral_grad_shape[:, -1]]
    if spiral_grad_shape.shape[1] % 2 == 0:
        spiral_grad_shape = np.c_[spiral_grad_shape, spiral_grad_shape[:, -1]]

# readout grad
gx = make_arbitrary_grad(
    channel='x',
    waveform=spiral_grad_shape[0],
    delay=calc_duration(gzReph),
    first=0,
    last=spiral_grad_shape[0, -1],
    system=system,
    oversampling=gradOversampling,
)
gy = make_arbitrary_grad(
    channel='y',
    waveform=spiral_grad_shape[1],
    delay=calc_duration(gzReph),
    first=0,
    last=spiral_grad_shape[1, -1],
    system=system,
    oversampling=gradOversampling,
)

# spoilers
gz_spoil = make_trapezoid(channel='z', system=system, area=deltak * Nx * 4)
gx_spoil = make_extended_trapezoid(channel='x', times=np.array([0, calc_duration(gz_spoil)]),
                                 amplitudes=np.array([spiral_grad_shape[0, -1], 0]), system=system)
gy_spoil = make_extended_trapezoid(channel='y', times=np.array([0, calc_duration(gz_spoil)]),
                                 amplitudes=np.array([spiral_grad_shape[1, -1], 0]), system=system)

for s in range(Nslices):
    seq.add_block(rf_fs, gz_fs)
    rf.freq_offset = gz.amplitude * sliceThickness * (s - (Nslices - 1) / 2)
    seq.add_block(rf, gz)

    # Rotate
    # `seq.addBlock(mr.rotate('z',phi,gzReph,gx,gy,adc,'system',sys));`
    # We rotate gx, gy, gzReph, adc.
    # gzReph is on Z. Rotation axis Z. gzReph unchanged.
    # gx, gy rotate.
    # adc rotates (sets phase/dict).
    seq.add_block(*rotate(gzReph, gx, gy, adc, angle=phi, axis='z'))
    seq.add_block(*rotate(gx_spoil, gy_spoil, gz_spoil, angle=phi, axis='z'))

if interleaves > 1:
    from pypulseq.TransformFOV.transform_fov import transform_fov
    n_blocks_orig = len(seq.block_durations)
    for i in range(2, interleaves + 1):
        seq.add_block(make_delay(TRdelay))
        angle_deg = 360 / interleaves * (i - 1)
        transformer = transform_fov(rotation=np.array([[np.cos(np.deg2rad(angle_deg)), -np.sin(np.deg2rad(angle_deg)), 0],
                                                       [np.sin(np.deg2rad(angle_deg)),  np.cos(np.deg2rad(angle_deg)), 0],
                                                       [0, 0, 1]]))
        seq = transformer.apply_to_seq(seq, same_seq=True, block_range=[1, n_blocks_orig])

# Timing validation
ok, error_report = seq.check_timing()

if ok:
    print('Timing check passed successfully')
else:
    print('Timing check failed! Error listing follows:')
    print(error_report)

seq.set_definition('FOV', [fov, fov, sliceThickness])
seq.set_definition('Name', 'spiral')
seq.set_definition('ReceiverGainHigh', 1)
seq.set_definition('MaxAdcSegmentLength', adcSamplesPerSegment)

# Export sequence
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '../demoSeq_pypulseq_results')
os.makedirs(RESULTS_DIR, exist_ok=True)
seq.write(os.path.join(RESULTS_DIR, 'Spiral_py.seq'))
