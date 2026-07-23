
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
from pypulseq_matlab_like.make_label import make_label

# this is a demo GRE sequence, which uses LABEL extension to produce raw
# data reconstuctable by the integrated image reconstruction on the scanner

# System limits
system = Opts(max_grad=28, grad_unit='mT/m', max_slew=150, slew_unit='T/m/s',
              rf_ringdown_time=20e-6, rf_dead_time=100e-6, adc_dead_time=10e-6)

# Sequence object
seq = Sequence(system)  # Create a new sequence object

fov = 256e-3
Nx = 128
Ny = Nx  # Define FOV and resolution
phaseResolution = fov / Nx / (fov / Ny)
alpha = 10  # flip angle
thickness = 3.555e-3  # slice
Nslices = 3
sliceGap = 1.111e-3
TR = 30e-3
TE = 4.3e-3

# more in-depth parameters
rfSpoilingInc = 84  # RF spoiling increment
roDuration = 3.2e-3  # ADC duration

# Create alpha-degree slice selection pulse and gradient
# Fix: Ensure return_gz=True to get gradients, and ignore rephaser as we recreate it manually
rf, gz, _ = make_sinc_pulse(flip_angle=alpha * np.pi / 180, duration=3e-3,
                                 slice_thickness=thickness, apodization=0.42, time_bw_product=4,
                                 system=system, use='excitation', return_gz=True)

# Define other gradients and ADC events
deltak = 1 / fov
gx = make_trapezoid(channel='x', flat_area=Nx * deltak, flat_time=roDuration, system=system)
adc = make_adc(num_samples=Nx, duration=gx.flat_time, delay=gx.rise_time, system=system)
gxPre = make_trapezoid(channel='x', area=-gx.area / 2, duration=1e-3, system=system)
gzReph = make_trapezoid(channel='z', area=-gz.area / 2, duration=1e-3, system=system)
phaseAreas = -((np.arange(Ny)) - Ny / 2) * deltak  # phase area should be Kmax for clin=0 and -Kmax for clin=Ny... strange

# gradient spoiling
gxSpoil = make_trapezoid(channel='x', area=2 * Nx * deltak, system=system)
gzSpoil = make_trapezoid(channel='z', area=4 / thickness, system=system)

# Calculate timing
delayTE = np.ceil((TE - calc_duration(gxPre) - gz.fall_time - gz.flat_time / 2
                   - calc_duration(gx) / 2) / seq.grad_raster_time) * seq.grad_raster_time
delayTR = np.ceil((TR - calc_duration(gz) - calc_duration(gxPre)
                   - calc_duration(gx) - delayTE) / seq.grad_raster_time) * seq.grad_raster_time
assert np.all(delayTE >= 0)
assert np.all(delayTR >= calc_duration(gxSpoil, gzSpoil))

rf_phase = 0
rf_inc = 0

# implement GRAPPA pattern
# set ACS lines for GRAPPA simulation (fully sampled central k-space region)
accelFactorPE = 2
ACSnum = 32
centerLineIdx = np.floor(Ny / 2) + 1

centerLineIdx_mat = np.floor(Ny / 2) + 1
PEsamp_u = []
for i in range(1, Ny + 1):
    if (i - centerLineIdx_mat) % accelFactorPE == 0:
        PEsamp_u.append(i)

PEsamp_u = np.array(PEsamp_u)

minPATRefLineIdx = centerLineIdx_mat - ACSnum / 2
maxPATRefLineIdx = centerLineIdx_mat + np.floor(ACSnum - 1) / 2
PEsamp_ACS = np.arange(int(minPATRefLineIdx), int(maxPATRefLineIdx) + 1)

PEsamp = np.union1d(PEsamp_u, PEsamp_ACS).astype(int) # actually sampled lines (1-based indices)
nPEsamp = len(PEsamp)
# Or PEsamp(end) means the last element.
# If PEsamp = [1, 3, 5]. [1, 3, 5, 5]. Diff = [2, 2, 0].
# This creates nPEsamp increments.
PEsamp_INC = np.diff(np.concatenate((PEsamp, [PEsamp[-1]])))

# Set PAT scan flag
lblSetRefScan = make_label(label='REF', type='SET', value=True)
lblSetRefAndImaScan = make_label(label='IMA', type='SET', value=True)
lblResetRefScan = make_label(label='REF', type='SET', value=False)
lblResetRefAndImaScan = make_label(label='IMA', type='SET', value=False)

seq.register_label_event(lblSetRefScan)
seq.register_label_event(lblSetRefAndImaScan)
seq.register_label_event(lblResetRefScan)
seq.register_label_event(lblResetRefAndImaScan)

# Add noise scans.
seq.add_block(make_label(label='LIN', type='SET', value=0), make_label(label='SLC', type='SET', value=0))
seq.add_block(adc, make_label(label='NOISE', type='SET', value=True), lblResetRefScan, lblResetRefAndImaScan)
seq.add_block(make_label(label='NOISE', type='SET', value=False))

# slice positions
slicePositions = (thickness + sliceGap) * (np.arange(Nslices) - (Nslices - 1) / 2)
# reorder slices
if Nslices > 1:
    slicePositions = np.concatenate((slicePositions[0::2], slicePositions[1::2]))

for s in range(Nslices):
    rf.freq_offset = gz.amplitude * thickness * (s - (Nslices - 1) / 2) # Freq offset calculation
    # Loop over phase encodes
    for count in range(nPEsamp):
        idx_1based = PEsamp[count]
        idx_0based = idx_1based - 1 # Convert to 0-based for array access

        if idx_1based in PEsamp_ACS:
            if idx_1based in PEsamp_u:
                seq.add_block(lblSetRefAndImaScan, lblSetRefScan)
            else:
                seq.add_block(lblResetRefAndImaScan, lblSetRefScan)
        else:
            seq.add_block(lblResetRefAndImaScan, lblResetRefScan)

        rf.phase_offset = rf_phase / 180 * np.pi
        adc.phase_offset = rf_phase / 180 * np.pi
        rf_inc = (rf_inc + rfSpoilingInc) % 360.0
        rf_phase = (rf_phase + rf_inc) % 360.0

        seq.add_block(rf, gz)
        gyPre = make_trapezoid(channel='y', area=phaseAreas[idx_0based], duration=calc_duration(gxPre), system=system)
        seq.add_block(gxPre, gyPre, gzReph)
        seq.add_block(make_delay(delayTE))
        seq.add_block(gx, adc)
        gyPre.amplitude = -gyPre.amplitude

        seq.add_block(make_delay(delayTR), gxSpoil, gyPre, gzSpoil)
        seq.add_block(make_label(label='LIN', type='INC', value=int(PEsamp_INC[count]))) # make_label expects int value

    seq.add_block(make_label(label='LIN', type='SET', value=0), make_label(label='SLC', type='INC', value=1))

# Timing validation
ok, error_report = seq.check_timing()

if ok:
    print('Timing check passed successfully')
else:
    print('Timing check failed! Error listing follows:')
    print(error_report)

# prepare sequence export
seq.set_definition('FOV', [fov, fov, max(slicePositions) - min(slicePositions) + thickness])
seq.set_definition('Name', 'gre_p2')
seq.set_definition('SlicePositions', slicePositions)
seq.set_definition('SliceThickness', thickness)
seq.set_definition('SliceGap', sliceGap)
seq.set_definition('TE', TE)
seq.set_definition('TR', TR)
seq.set_definition('kSpaceCenterLine', int(centerLineIdx_mat - 1))
# If python user expects 0-based, centerLineIdx_mat - 1.

seq.set_definition('PhaseResolution', phaseResolution)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '../demoSeq_pypulseq_results')
output_path = os.path.join(RESULTS_DIR, 'GradientEcho_grappa_py.seq')
print(f"Writing to: {os.path.abspath(output_path)}")
# Export sequence
os.makedirs(RESULTS_DIR, exist_ok=True)
seq.write(output_path)
