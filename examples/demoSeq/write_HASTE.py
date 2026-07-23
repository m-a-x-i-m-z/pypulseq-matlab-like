
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
from pypulseq_matlab_like.make_extended_trapezoid import make_extended_trapezoid
from pypulseq_matlab_like.make_delay import make_delay
from pypulseq_matlab_like.calc_duration import calc_duration
from pypulseq_matlab_like.calc_rf_center import calc_rf_center
from pypulseq_matlab_like.scale_grad import scale_grad

# Create a TSE sequence and export for execution
dG = 250e-6
# System limits
system = Opts(max_grad=30, grad_unit='mT/m', max_slew=170, slew_unit='T/m/s',
              rf_ringdown_time=100e-6, rf_dead_time=100e-6, adc_dead_time=10e-6)

# Sequence object
seq = Sequence(system)

fov = 256e-3
Ny_pre = 8
Nx = 128
Ny = 128
necho = int(Ny / 2 + Ny_pre)
Nslices = 1
rflip = 180
if isinstance(rflip, (int, float)):
    rflip = [rflip] * necho

sliceThickness = 5e-3
TE = 12e-3
TR = 2000e-3
TEeff = 60e-3
k0 = int(round(TEeff / TE))
PEtype = 'linear'

samplingTime = 6.4e-3
readoutTime = samplingTime + 2 * system.adc_dead_time
tEx = 2.5e-3
tExwd = tEx + system.rf_ringdown_time + system.rf_dead_time
tRef = 2e-3
tRefwd = tRef + system.rf_ringdown_time + system.rf_dead_time
tSp = 0.5 * (TE - readoutTime - tRefwd)
tSpex = 0.5 * (TE - tExwd - tRefwd)
fspR = 1.0
fspS = 0.5

rfex_phase = np.pi / 2
rfref_phase = 0

# Base gradients
# Slice selection
flipex = 90 * np.pi / 180
rfex, gz = make_sinc_pulse(flip_angle=flipex, system=system, duration=tEx,
                           slice_thickness=sliceThickness, apodization=0.5, time_bw_product=4, phase_offset=rfex_phase,
                           use='excitation', return_gz=True)[0:2]
GSex = make_trapezoid(channel='z', system=system, amplitude=gz.amplitude, flat_time=tExwd, rise_time=dG)

flipref = rflip[0] * np.pi / 180
rfref, gz = make_sinc_pulse(flip_angle=flipref, system=system, duration=tRef,
                            slice_thickness=sliceThickness, apodization=0.5, time_bw_product=4, phase_offset=rfref_phase,
                            use='refocusing', return_gz=True)[0:2]
GSref = make_trapezoid(channel='z', system=system, amplitude=GSex.amplitude, flat_time=tRefwd, rise_time=dG)

AGSex = GSex.area / 2
GSspr = make_trapezoid(channel='z', system=system, area=AGSex * (1 + fspS), duration=tSp, rise_time=dG)
GSspex = make_trapezoid(channel='z', system=system, area=AGSex * fspS, duration=tSpex, rise_time=dG)

# Readout gradient
deltak = 1 / fov
kWidth = Nx * deltak

GRacq = make_trapezoid(channel='x', system=system, flat_area=kWidth, flat_time=readoutTime, rise_time=dG)
adc = make_adc(num_samples=Nx, duration=samplingTime, delay=system.adc_dead_time, system=system)
GRspr = make_trapezoid(channel='x', system=system, area=GRacq.area * fspR, duration=tSp, rise_time=dG)
GRspex = make_trapezoid(channel='x', system=system, area=GRacq.area * (1 + fspR), duration=tSpex, rise_time=dG)

AGRspr = GRspr.area
AGRpreph = GRacq.area / 2 + AGRspr
GRpreph = make_trapezoid(channel='x', system=system, area=AGRpreph, duration=tSpex, rise_time=dG)

# Phase encoding
nex = 1
PEorder = np.arange(-Ny_pre, Ny + 1)
# But `necho` was Ny/2 + Ny_pre = 64+8 = 72.
# `PEorder` is usually reordered (e.g. k0 in center).
# It iterates `necho` times.
# Ah, `necho` is `Ny/2 + Ny_pre`. That's 72.
# `PEorder` has `8 + 128 + 1` = 137 elements.
# But inside loop: `phaseArea=phaseAreas(kech,kex);`
# `phaseAreas = PEorder*deltak`.
# Wait, `PEorder` size vs `necho`.
# The loop goes up to `necho`. So it uses first `necho` entries of `PEorder`.
# That is indices 1 to 72.
# Which corresponds to -8 to 63.
# So it acquires part of k-space centrally (HASTE is half-fourier).
# And generate `phaseAreas` based on `PEorder`.

PEorder = np.arange(-Ny_pre, necho - Ny_pre) # Matches 1:necho indexing into -Ny_pre:Ny array
phaseAreas = PEorder * deltak

# split gradients and recombine into blocks
# Slice selection
GS1times = np.array([0, GSex.rise_time])
GS1amp = np.array([0, GSex.amplitude])
GS1 = make_extended_trapezoid(channel='z', times=GS1times, amplitudes=GS1amp, system=system)

GS2times = np.array([0, GSex.flat_time])
GS2amp = np.array([GSex.amplitude, GSex.amplitude])
GS2 = make_extended_trapezoid(channel='z', times=GS2times, amplitudes=GS2amp, system=system)

GS3times = np.array([0, GSspex.rise_time, GSspex.rise_time + GSspex.flat_time, GSspex.rise_time + GSspex.flat_time + GSspex.fall_time])
GS3amp = np.array([GSex.amplitude, GSspex.amplitude, GSspex.amplitude, GSref.amplitude])
GS3 = make_extended_trapezoid(channel='z', times=GS3times, amplitudes=GS3amp, system=system)

GS4times = np.array([0, GSref.flat_time])
GS4amp = np.array([GSref.amplitude, GSref.amplitude])
GS4 = make_extended_trapezoid(channel='z', times=GS4times, amplitudes=GS4amp, system=system)

GS5times = np.array([0, GSspr.rise_time, GSspr.rise_time + GSspr.flat_time, GSspr.rise_time + GSspr.flat_time + GSspr.fall_time])
GS5amp = np.array([GSref.amplitude, GSspr.amplitude, GSspr.amplitude, 0])
GS5 = make_extended_trapezoid(channel='z', times=GS5times, amplitudes=GS5amp, system=system)

GS7times = np.array([0, GSspr.rise_time, GSspr.rise_time + GSspr.flat_time, GSspr.rise_time + GSspr.flat_time + GSspr.fall_time])
GS7amp = np.array([0, GSspr.amplitude, GSspr.amplitude, GSref.amplitude])
GS7 = make_extended_trapezoid(channel='z', times=GS7times, amplitudes=GS7amp, system=system)

# Readout gradient
GR3 = GRpreph

GR5times = np.array([0, GRspr.rise_time, GRspr.rise_time+GRspr.flat_time, GRspr.rise_time+GRspr.flat_time+GRspr.fall_time])
GR5amp = np.array([0, GRspr.amplitude, GRspr.amplitude, GRacq.amplitude])
GR5 = make_extended_trapezoid(channel='x', times=GR5times, amplitudes=GR5amp, system=system)

GR6times = np.array([0, readoutTime])
GR6amp = np.array([GRacq.amplitude, GRacq.amplitude])
GR6 = make_extended_trapezoid(channel='x', times=GR6times, amplitudes=GR6amp, system=system)

GR7times = np.array([0, GRspr.rise_time, GRspr.rise_time+GRspr.flat_time, GRspr.rise_time+GRspr.flat_time+GRspr.fall_time])
GR7amp = np.array([GRacq.amplitude, GRspr.amplitude, GRspr.amplitude, 0])
GR7 = make_extended_trapezoid(channel='x', times=GR7times, amplitudes=GR7amp, system=system)

# filltimes
tex = GS1.shape_dur + GS2.shape_dur + GS3.shape_dur
tref = GS4.shape_dur + GS5.shape_dur + GS7.shape_dur + readoutTime
tend = GS4.shape_dur + GS5.shape_dur
tETrain = tex + necho * tref + tend
TRfill = (TR - Nslices * tETrain) / Nslices
TRfill = system.grad_raster_time * np.round(TRfill / system.grad_raster_time)
if TRfill < 0:
    TRfill = 1e-3
    print(f'Warning!!! TR too short, adapted to include all slices to : {1000 * Nslices * (tETrain + TRfill)} ms')
else:
    print(f'TRfill : {1000 * TRfill} ms')

delayTR = make_delay(TRfill)
delayEnd = make_delay(5.0)

# Define sequence blocks
for kex in range(nex):
    for s in range(Nslices):
        rfex.freq_offset = GSex.amplitude * sliceThickness * (s - (Nslices - 1) / 2)
        rfref.freq_offset = GSref.amplitude * sliceThickness * (s - (Nslices - 1) / 2)
        rfex.phase_offset = rfex_phase - 2 * np.pi * rfex.freq_offset * calc_rf_center(rfex)[0]
        rfref.phase_offset = rfref_phase - 2 * np.pi * rfref.freq_offset * calc_rf_center(rfref)[0]

        seq.add_block(GS1)
        seq.add_block(GS2, rfex)
        seq.add_block(GS3, GR3)

        for kech in range(necho):
            phaseArea = phaseAreas[kech]

            GPpre = make_trapezoid(channel='y', system=system, area=phaseArea, duration=tSp, rise_time=dG)
            GPrew = make_trapezoid(channel='y', system=system, area=-phaseArea, duration=tSp, rise_time=dG)

            seq.add_block(GS4, rfref)
            seq.add_block(GS5, GR5, GPpre)
            seq.add_block(GR6, adc)
            seq.add_block(GS7, GR7, GPrew)

        seq.add_block(GS4)
        seq.add_block(GS5)
        seq.add_block(delayTR)

seq.add_block(delayEnd)

# Timing validation
ok, error_report = seq.check_timing()

if ok:
    print('Timing check passed successfully')
else:
    print('Timing check failed! Error listing follows:')
    print(error_report)


seq.auto_label(mirror_fourier=True, sort_slices='descending')
seq.set_definition('FOV', [fov, fov, Nslices * sliceThickness])
seq.set_definition('Name', 'haste')

# Export sequence
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '../demoSeq_pypulseq_results')
os.makedirs(RESULTS_DIR, exist_ok=True)
seq.write(os.path.join(RESULTS_DIR, 'HASTE_py.seq'))