import os
import sys

import numpy as np

# Add pypulseq source to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from pypulseq_matlab_like.Sequence.sequence import Sequence
from pypulseq_matlab_like.TransformFOV.transform_fov import transform_fov
from pypulseq_matlab_like.calc_duration import calc_duration
from pypulseq_matlab_like.calc_rf_center import calc_rf_center
from pypulseq_matlab_like.make_adc import make_adc
from pypulseq_matlab_like.make_delay import make_delay
from pypulseq_matlab_like.make_extended_trapezoid import make_extended_trapezoid
from pypulseq_matlab_like.make_sinc_pulse import make_sinc_pulse
from pypulseq_matlab_like.make_trapezoid import make_trapezoid
from pypulseq_matlab_like.opts import Opts


def rotm_z(phi: float) -> np.ndarray:
    c = np.cos(phi)
    s = np.sin(phi)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)


# System limits
system = Opts(
    max_grad=30,
    grad_unit='mT/m',
    max_slew=170,
    slew_unit='T/m/s',
    rf_ringdown_time=100e-6,
    rf_dead_time=100e-6,
    adc_dead_time=10e-6,
)

# Sequence object
seq = Sequence(system)
fov = 256e-3
Nx = 128
Ny = 128
necho = 16
Nslices = 1
rflip = 180
rflip = [rflip] * necho if isinstance(rflip, (int, float)) else rflip

slice_thickness = 5e-3
TE1 = 12e-3
TR = 2000e-3
TEeff = 100e-3

sampling_time = 6.4e-3
readout_time = sampling_time + 2 * system.adc_dead_time
t_ex = 2.5e-3
t_exwd = t_ex + system.rf_ringdown_time + system.rf_dead_time
t_ref = 2e-3
t_refwd = t_ref + system.rf_ringdown_time + system.rf_dead_time
t_sp = 0.5 * (TE1 - readout_time - t_refwd)
t_spex = 0.5 * (TE1 - t_exwd - t_refwd)
fsp_r = 1.0
fsp_s = 0.5
d_g = 250e-6

rfex_phase = np.pi / 2
rfref_phase = 0.0

# Base gradients
flipex = np.pi / 2
rfex, gz, _ = make_sinc_pulse(
    flip_angle=flipex,
    system=system,
    duration=t_ex,
    slice_thickness=slice_thickness,
    apodization=0.5,
    time_bw_product=4,
    phase_offset=rfex_phase,
    use='excitation',
    return_gz=True,
)
gs_ex = make_trapezoid(channel='z', system=system, amplitude=gz.amplitude, flat_time=t_exwd, rise_time=d_g)

flipref = rflip[0] * np.pi / 180
rfref, _, _ = make_sinc_pulse(
    flip_angle=flipref,
    system=system,
    duration=t_ref,
    slice_thickness=slice_thickness,
    apodization=0.5,
    time_bw_product=4,
    phase_offset=rfref_phase,
    use='refocusing',
    return_gz=True,
)
gs_ref = make_trapezoid(channel='z', system=system, amplitude=gs_ex.amplitude, flat_time=t_refwd, rise_time=d_g)

agsex = gs_ex.area / 2
gs_spr = make_trapezoid(channel='z', system=system, area=agsex * (1 + fsp_s), duration=t_sp, rise_time=d_g)
gs_spex = make_trapezoid(channel='z', system=system, area=agsex * fsp_s, duration=t_spex, rise_time=d_g)

# Readout gradient
deltak = 1 / fov
k_width = Nx * deltak
gr_acq = make_trapezoid(channel='x', system=system, flat_area=k_width, flat_time=readout_time, rise_time=d_g)
adc = make_adc(num_samples=Nx, duration=sampling_time, delay=system.adc_dead_time, system=system)
gr_spr = make_trapezoid(channel='x', system=system, area=gr_acq.area * fsp_r, duration=t_sp, rise_time=d_g)
_ = make_trapezoid(channel='x', system=system, area=gr_acq.area * (1 + fsp_r), duration=t_spex, rise_time=d_g)

agr_spr = gr_spr.area
agr_preph = gr_acq.area / 2 + agr_spr
gr_preph = make_trapezoid(channel='x', system=system, area=agr_preph, duration=t_spex, rise_time=d_g)

# Phase encoding (propeller: one train per rotation angle + one dummy)
nex = 1
pe_steps = np.arange(1, necho * nex + 1) - 0.5 * necho * nex - 1
if necho % 2 == 0:
    pe_steps = np.roll(pe_steps, -int(np.round(nex / 2)))

i_pe_min = int(np.argmin(np.abs(pe_steps)))
k0curr = int(np.floor(i_pe_min / nex) + 1)  # MATLAB-style 1-based index
k0prescr = max(int(np.round(TEeff / TE1)), 1)
pe_order = np.roll(pe_steps.reshape((nex, necho), order='F').T, shift=k0prescr - k0curr, axis=0)
phase_areas = pe_order * deltak

# Split gradients and recombine
gs1 = make_extended_trapezoid(
    channel='z',
    times=np.array([0.0, gs_ex.rise_time]),
    amplitudes=np.array([0.0, gs_ex.amplitude]),
    system=system,
)
gs2 = make_extended_trapezoid(
    channel='z',
    times=np.array([0.0, gs_ex.flat_time]),
    amplitudes=np.array([gs_ex.amplitude, gs_ex.amplitude]),
    system=system,
)
gs3 = make_extended_trapezoid(
    channel='z',
    times=np.array([0.0, gs_spex.rise_time, gs_spex.rise_time + gs_spex.flat_time, gs_spex.rise_time + gs_spex.flat_time + gs_spex.fall_time]),
    amplitudes=np.array([gs_ex.amplitude, gs_spex.amplitude, gs_spex.amplitude, gs_ref.amplitude]),
    system=system,
)
gs4 = make_extended_trapezoid(
    channel='z',
    times=np.array([0.0, gs_ref.flat_time]),
    amplitudes=np.array([gs_ref.amplitude, gs_ref.amplitude]),
    system=system,
)
gs5 = make_extended_trapezoid(
    channel='z',
    times=np.array([0.0, gs_spr.rise_time, gs_spr.rise_time + gs_spr.flat_time, gs_spr.rise_time + gs_spr.flat_time + gs_spr.fall_time]),
    amplitudes=np.array([gs_ref.amplitude, gs_spr.amplitude, gs_spr.amplitude, 0.0]),
    system=system,
)
gs7 = make_extended_trapezoid(
    channel='z',
    times=np.array([0.0, gs_spr.rise_time, gs_spr.rise_time + gs_spr.flat_time, gs_spr.rise_time + gs_spr.flat_time + gs_spr.fall_time]),
    amplitudes=np.array([0.0, gs_spr.amplitude, gs_spr.amplitude, gs_ref.amplitude]),
    system=system,
)

gr3 = gr_preph
gr5 = make_extended_trapezoid(
    channel='x',
    times=np.array([0.0, gr_spr.rise_time, gr_spr.rise_time + gr_spr.flat_time, gr_spr.rise_time + gr_spr.flat_time + gr_spr.fall_time]),
    amplitudes=np.array([0.0, gr_spr.amplitude, gr_spr.amplitude, gr_acq.amplitude]),
    system=system,
)
gr6 = make_extended_trapezoid(
    channel='x',
    times=np.array([0.0, readout_time]),
    amplitudes=np.array([gr_acq.amplitude, gr_acq.amplitude]),
    system=system,
)
gr7 = make_extended_trapezoid(
    channel='x',
    times=np.array([0.0, gr_spr.rise_time, gr_spr.rise_time + gr_spr.flat_time, gr_spr.rise_time + gr_spr.flat_time + gr_spr.fall_time]),
    amplitudes=np.array([gr_acq.amplitude, gr_spr.amplitude, gr_spr.amplitude, 0.0]),
    system=system,
)

# Fill times
t_ex_tot = calc_duration(gs1) + calc_duration(gs2) + calc_duration(gs3)
t_ref_tot = calc_duration(gs4) + calc_duration(gs5) + calc_duration(gs7) + readout_time
t_end_tot = calc_duration(gs4) + calc_duration(gs5)
t_e_train = t_ex_tot + necho * t_ref_tot + t_end_tot
tr_fill = (TR - Nslices * t_e_train) / Nslices
tr_fill = system.grad_raster_time * np.round(tr_fill / system.grad_raster_time)
if tr_fill < 0:
    tr_fill = 1e-3
    print(f'Warning!!! TR too short, adapted to include all slices to: {1000 * Nslices * (t_e_train + tr_fill):.3f} ms')
else:
    print(f'TRfill: {1000 * tr_fill:.3f} ms')
delay_tr = make_delay(tr_fill)

# Build sequence
for kex in range(nex + 1):  # one dummy plus one propeller train
    for s in range(Nslices):
        rfex.freq_offset = gs_ex.amplitude * slice_thickness * (s - (Nslices - 1) / 2)
        rfref.freq_offset = gs_ref.amplitude * slice_thickness * (s - (Nslices - 1) / 2)
        rfex.phase_offset = rfex_phase - 2 * np.pi * rfex.freq_offset * calc_rf_center(rfex)[0]
        rfref.phase_offset = rfref_phase - 2 * np.pi * rfref.freq_offset * calc_rf_center(rfref)[0]

        seq.add_block(gs1)
        seq.add_block(gs2, rfex)
        seq.add_block(gs3, gr3)

        for kech in range(necho):
            if kex > 0:
                phase_area = phase_areas[kech, kex - 1]
            else:
                phase_area = 0.0
            gp_pre = make_trapezoid(channel='y', system=system, area=phase_area, duration=t_sp, rise_time=d_g)
            gp_rew = make_trapezoid(channel='y', system=system, area=-phase_area, duration=t_sp, rise_time=d_g)

            seq.add_block(gs4, rfref)
            seq.add_block(gs5, gr5, gp_pre)
            if kex > 0:
                seq.add_block(gr6, adc)
            else:
                seq.add_block(gr6)
            seq.add_block(gs7, gr7, gp_rew)

        seq.add_block(gs4)
        seq.add_block(gs5)
        seq.add_block(delay_tr)

# Propeller rotation stack (MATLAB writeTSEprop behavior)
n_blocks_orig = len(seq.block_durations)
n_prop = 14
for nr in range(1, n_prop):
    transformer = transform_fov(rotation=rotm_z(np.pi / n_prop * nr), use_rotation_extension=True)
    seq = transformer.apply_to_seq(seq, same_seq=True, block_range=[1, n_blocks_orig])

ok, error_report = seq.check_timing()
if ok:
    print('Timing check passed successfully')
else:
    print('Timing check failed! Error listing follows:')
    print(error_report)

print(seq.test_report())

results_dir = os.path.join(os.path.dirname(__file__), '../demoSeq_pypulseq_results')
os.makedirs(results_dir, exist_ok=True)
seq.write(os.path.join(results_dir, 'TSEprop_py.seq'))
