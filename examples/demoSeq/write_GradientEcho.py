
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
from pypulseq.scale_grad import scale_grad

# System limits
system = Opts(max_grad=22, grad_unit='mT/m', max_slew=120, slew_unit='T/m/s',
              rf_ringdown_time=20e-6, rf_dead_time=100e-6, adc_dead_time=10e-6)

# Sequence object
seq = Sequence(system)  # Create a new sequence object
fov = 256e-3
Nx = 128
Ny = Nx  # Define FOV and resolution
alpha = 10  # flip angle
slice_thickness = 3e-3  # slice
TR = 12e-3  # repetition time TR
TE = 5e-3  # echo time TE
# TE=[7.38 9.84]*1e-3;            % alternatively give a vector here to have multiple TEs (e.g. for field mapping)

# more in-depth parameters
rf_spoiling_inc = 84  # RF spoiling increment
ro_duration = 3.2e-3  # ADC duration

# Create alpha-degree slice selection pulse and gradient
rf, gz, _ = make_sinc_pulse(flip_angle=alpha * np.pi / 180, system=system, duration=3e-3,
                            slice_thickness=slice_thickness, apodization=0.42, time_bw_product=4,
                            use='excitation', return_gz=True)

# Define other gradients and ADC events
deltak = 1 / fov
gx = make_trapezoid(channel='x', system=system, flat_area=Nx * deltak, flat_time=ro_duration)
adc = make_adc(num_samples=Nx, system=system, duration=gx.flat_time, delay=gx.rise_time)
gx_pre = make_trapezoid(channel='x', system=system, area=-gx.area / 2, duration=1e-3)
gz_reph = make_trapezoid(channel='z', system=system, area=-gz.area / 2, duration=1e-3)
phase_areas = (np.arange(Ny) - Ny / 2) * deltak
gy_pre = make_trapezoid(channel='y', system=system, area=np.max(np.abs(phase_areas)), duration=calc_duration(gx_pre))
pe_scales = phase_areas / gy_pre.area

# gradient spoiling
gx_spoil = make_trapezoid(channel='x', system=system, area=2 * Nx * deltak)
gz_spoil = make_trapezoid(channel='z', system=system, area=4 / slice_thickness)

# Calculate timing
# Note: TE is scalar here, no loop needed if singular, but code has loop structure for potential vector TE
TEs = np.atleast_1d(TE)
delay_te = np.ceil((TEs - calc_duration(gx_pre) - gz.fall_time - gz.flat_time / 2
                    - calc_duration(gx) / 2) / seq.grad_raster_time) * seq.grad_raster_time
delay_tr = np.ceil((TR - calc_duration(gz) - calc_duration(gx_pre)
                    - calc_duration(gx) - delay_te) / seq.grad_raster_time) * seq.grad_raster_time
assert np.all(delay_te >= 0)
assert np.all(delay_tr >= calc_duration(gx_spoil, gz_spoil))

rf_phase = 0
rf_inc = 0

# Loop over phase encodes and define sequence blocks
for i in range(Ny):
    for c in range(len(TEs)):
        # seq.addBlock(rf_fs,gz_fs); % fat-sat
        rf.phase_offset = rf_phase / 180 * np.pi
        adc.phase_offset = rf_phase / 180 * np.pi
        rf_inc = (rf_inc + rf_spoiling_inc) % 360.0
        rf_phase = (rf_phase + rf_inc) % 360.0

        seq.add_block(rf, gz)
        seq.add_block(gx_pre, scale_grad(gy_pre, pe_scales[i]), gz_reph)
        seq.add_block(make_delay(delay_te[c]))
        seq.add_block(gx, adc)
        # gyPre.amplitude=-gyPre.amplitude;
        seq.add_block(make_delay(delay_tr[c]), gx_spoil, scale_grad(gy_pre, -pe_scales[i]), gz_spoil)

# Timing validation
ok, error_report = seq.check_timing()

if ok:
    print('Timing check passed successfully')
else:
    print('Timing check failed! Error listing follows:')
    print(error_report)

# prepare sequence export
seq.auto_label(mirror_fourier=True)
seq.set_definition('FOV', [fov, fov, slice_thickness])
seq.set_definition('Name', 'gre')

# Write to pulseq file
# Output to demoSeq_pypulseq_results/GradientEcho_py.seq
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '../demoSeq_pypulseq_results')
output_path = os.path.join(RESULTS_DIR, 'GradientEcho_py.seq')
print(f"Writing to: {os.path.abspath(output_path)}")
# Export sequence
os.makedirs(RESULTS_DIR, exist_ok=True)
seq.write(output_path)

# seq.plot()
# rep = seq.test_report()
# print(rep)
