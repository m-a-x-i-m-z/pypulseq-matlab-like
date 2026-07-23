
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
from pypulseq_matlab_like.make_trigger import make_trigger
from pypulseq_matlab_like.make_digital_output_pulse import make_digital_output_pulse
from pypulseq_matlab_like.calc_duration import calc_duration

# this is a very naiive and non-optimized cardiac cine GRE sequence

# Sequence object
seq = Sequence()  # Create a new sequence object
fov = 256e-3
Nx = 128
Ny = Nx  # Define FOV and resolution
alpha = 5  # flip angle
slice_thickness = 5e-3  # slice
# TE=[7.38 9.84]*1e-3;            % give a vector here to have multiple TEs (e.g. for field mapping)
TE = 4.92e-3
TR = 9e-3  # only a single value for now

# cardiac features
phases = 8
heartbeats = 15  # odd numbers of heartbeats / segments work better

# more in-depth parameters
rf_spoiling_inc = 84  # RF spoiling increment
rf_duration = 2e-3
adc_duration = 3.2e-3
pre_duration = 1e-3

# System limits
system = Opts(max_grad=28, grad_unit='mT/m', max_slew=150, slew_unit='T/m/s',
              rf_ringdown_time=20e-6, rf_dead_time=100e-6, adc_dead_time=10e-6)

# define the trigger to play out
trig = make_trigger(channel='physio1', duration=2000e-6)  # duration after
trig_out = make_digital_output_pulse(channel='ext1', duration=100e-6, delay=500e-6)

# Create alpha-degree slice selection pulse and gradient
rf, gz, _ = make_sinc_pulse(flip_angle=alpha * np.pi / 180, duration=rf_duration,
                            slice_thickness=slice_thickness, apodization=0.5, time_bw_product=4, system=system,
                            use='excitation', return_gz=True)

# Define other gradients and ADC events
deltak = 1 / fov
gx = make_trapezoid(channel='x', flat_area=Nx * deltak, flat_time=adc_duration, system=system)
adc = make_adc(num_samples=Nx, duration=gx.flat_time, delay=gx.rise_time, system=system)
gx_pre = make_trapezoid(channel='x', area=-gx.area / 2, duration=pre_duration, system=system)
gz_reph = make_trapezoid(channel='z', area=-gz.area / 2, duration=pre_duration, system=system)

lines_per_segment = round(Ny / heartbeats)
Ns = int(np.ceil(Ny / lines_per_segment))
Ny = Ns * lines_per_segment  # it can be that because of the rounding above we measure few more k-space lines...
phase_areas = (np.arange(Ny) - Ny / 2) * deltak
# now reverse the order in every second segment
phase_areas_seg = phase_areas.reshape(lines_per_segment, Ns, order='F')
phase_areas_seg[:, 1::2] = phase_areas_seg[::-1, 1::2]
phase_areas = phase_areas_seg.flatten(order='F')

# gradient spoiling
gx_spoil = make_trapezoid(channel='x', area=2 * Nx * deltak, system=system)
gz_spoil = make_trapezoid(channel='z', area=4 / slice_thickness, system=system)

# Calculate timing
delay_te = np.ceil((TE - calc_duration(gx_pre) - gz.fall_time - gz.flat_time / 2
                    - calc_duration(gx) / 2) / seq.grad_raster_time) * seq.grad_raster_time
delay_tr = np.ceil((TR - calc_duration(gx_pre) - calc_duration(gz)
                    - calc_duration(gx) - delay_te) / seq.grad_raster_time) * seq.grad_raster_time
assert np.all(delay_tr >= calc_duration(gx_spoil, gz_spoil))

print(f'the sequence will acquire {lines_per_segment} lines per segment resulting in a temporal resolution of {TR * lines_per_segment * 1e3:g} ms per phase')
print(f'cardiac acquisition window is: {TR * phases * lines_per_segment * 1e3:g} ms')

rf_phase = 0
rf_inc = 0

# Loop over phase encodes and define sequence blocks
for s in range(Ns):
    seq.add_block(trig)  # wait for the cardiac trigger
    for p in range(phases):
        for l in range(lines_per_segment):
            # restore the line counter
            i = s * lines_per_segment + l
            # seq.addBlock(rf_fs,gz_fs); % fat-sat
            rf.phase_offset = rf_phase / 180 * np.pi
            adc.phase_offset = rf_phase / 180 * np.pi
            rf_inc = (rf_inc + rf_spoiling_inc) % 360.0
            rf_phase = (rf_phase + rf_inc) % 360.0

            seq.add_block(rf, gz, trig_out)
            gy_pre = make_trapezoid(channel='y', area=phase_areas[i], duration=pre_duration, system=system)
            seq.add_block(gx_pre, gy_pre, gz_reph)
            if delay_te > 0:
                seq.add_block(make_delay(delay_te))

            seq.add_block(gx, adc)
            gy_pre.amplitude = -gy_pre.amplitude
            seq.add_block(make_delay(delay_tr), gx_spoil, gy_pre, gz_spoil)

# Timing validation
ok, error_report = seq.check_timing()

if ok:
    print('Timing check passed successfully')
else:
    print('Timing check failed! Error listing follows:')
    print(error_report)

# prepare sequence export
labels, aux = seq.auto_label(mirror_fourier=True, skip_apply=True, no_plots=True)
labels['PHS'] = labels['REP']
del labels['REP']
seq.auto_label(use_labels=labels, use_aux=aux)

seq.set_definition('FOV', [fov, fov, slice_thickness])
seq.set_definition('Name', 'cine-gre')

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '../demoSeq_pypulseq_results')
output_path = os.path.join(RESULTS_DIR, 'CineGradientEcho_py.seq')
print(f"Writing to: {os.path.abspath(output_path)}")
# Export sequence
os.makedirs(RESULTS_DIR, exist_ok=True)
seq.write(output_path)
