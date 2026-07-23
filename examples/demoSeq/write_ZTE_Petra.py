
import numpy as np
import sys
import os
import copy

# Add pypulseq source to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from pypulseq_matlab_like.Sequence.sequence import Sequence
from pypulseq_matlab_like.opts import Opts
from pypulseq_matlab_like.make_gauss_pulse import make_gauss_pulse
from pypulseq_matlab_like.make_adc import make_adc
from pypulseq_matlab_like.make_extended_trapezoid import make_extended_trapezoid
from pypulseq_matlab_like.make_delay import make_delay
from pypulseq_matlab_like.calc_duration import calc_duration
from pypulseq_matlab_like.calc_rf_bandwidth import calc_rf_bandwidth
from pypulseq_matlab_like.rotate import rotate
from pypulseq_matlab_like.align import align

def spherical_samples(Kr, dK, R):
    Ns = int(np.ceil(4 * np.pi * ((Kr / dK)**2) / R))
    np_idx = np.arange(Ns)
    alpha_gold = np.pi * (3 - np.sqrt(5))
    phi = np_idx * alpha_gold
    theta = np.arccos(1 - 2 * np_idx / (Ns - 1))

    xp = np.sin(theta) * np.cos(phi)
    yp = np.sin(theta) * np.sin(phi)
    zp = np.cos(theta)

    nm = int(np.floor(Ns / 2 + 0.5))
    sr = int(np.floor(np.sqrt(Ns) + 0.5))

    v0 = np.array([xp[nm], yp[nm], zp[nm]])
    v = np.array([xp[nm + 1:nm + sr + 1], yp[nm + 1:nm + sr + 1], zp[nm + 1:nm + sr + 1]])

    dKm = np.min(np.linalg.norm(v - v0[:, np.newaxis], axis=0))
    # im is unused in function but returned.
    im = int(np.argmin(np.linalg.norm(v - v0[:, np.newaxis], axis=0))) + 1

    return phi, theta, im

def populate_subsequence(sys, seq, rf, adc, phi, theta, im_step, Ns, Ag, TR, Tt, FN, FO):
    # im_step renamed from im to avoid confusion with imag
    # Azc = Ag * (TR - Tt) / (TR + Tt)
    Azc = Ag * (TR - Tt) / (TR + Tt)

    # Gr constant
    Gr = make_extended_trapezoid(channel='z', times=np.array([0, TR - Tt]), amplitudes=np.array([Ag, Ag]), system=system)

    if abs(Azc) > 1e-10:
        Tpr = max(2e-5, np.ceil(Azc / sys.max_slew / sys.grad_raster_time) * sys.grad_raster_time)
        assert Tpr <= TR
        dummy_grad = make_extended_trapezoid(channel='z', system=system, times=np.array([0, Tpr]), amplitudes=np.array([0, Azc]))
        delay_tr, dummy_grad = align(right=[make_delay(TR), dummy_grad])
        seq.add_block(delay_tr, dummy_grad) # Align right?

    # Python Loop
    # Loop over interleaves?
    # `im` from `spherical_samples` seems to be used as step? `for i=j:im:Ns`
    # Warning: `im` in `spherical_samples` output is index of nearest neighbor.
    # `population_subsequence` uses `im` as step.
    # So `im` roughly sqrt(Ns)?

    for j in range(1, im_step + 1): # 1-based to N
        Glast = {'x': 0, 'y': 0, 'z': Azc}
        for i_idx in range(j - 1, Ns, im_step):
            # i_idx is 0-based index for python arrays
            # Rotate
            # Gcr=mr.rotate('z',phi(i),mr.rotate('y',theta(i),Gr));
            # Chained rotation: First Y, then Z.
            Gr_y_rot = rotate(Gr, angle=theta[i_idx], axis='y')
            # Gr_y_rot is a tuple/list of gradients (x, y, z).
            # Pass *Gr_y_rot to rotate around Z.
            Gcr = rotate(*Gr_y_rot, angle=phi[i_idx], axis='z')

            Gcurr = {'x': 0, 'y': 0, 'z': 0}
            for g in Gcr:
                Gcurr[g.channel] = g.waveform[0]

            # Transition trapezoids
            # times=[0, Tt], amps=[Glast, Gcurr]
            seq.add_block(
                make_extended_trapezoid(channel='x', system=system, times=np.array([0, Tt]), amplitudes=np.array([Glast['x'], Gcurr['x']])),
                make_extended_trapezoid(channel='y', system=system, times=np.array([0, Tt]), amplitudes=np.array([Glast['y'], Gcurr['y']])),
                make_extended_trapezoid(channel='z', system=system, times=np.array([0, Tt]), amplitudes=np.array([Glast['z'], Gcurr['z']]))
            )

            # ADC blocks
            for f in range(-FN, FN + 1):
                rf.freq_offset = f * FO
                rf.phase_offset = -2 * np.pi * f * FO * rf.t[-1] / 2
                seq.add_block(rf, adc, *Gcr)

            Glast = Gcurr

        rf_aux = copy.deepcopy(rf)
        rf_aux.delay = rf.delay + Tt
        if j == im_step:
            Azc = 0

        seq.add_block(rf_aux,
                      make_extended_trapezoid(channel='x', system=system, times=np.array([0, TR]), amplitudes=np.array([Glast['x'], 0])),
                      make_extended_trapezoid(channel='y', system=system, times=np.array([0, TR]), amplitudes=np.array([Glast['y'], 0])),
                      make_extended_trapezoid(channel='z', system=system, times=np.array([0, TR]), amplitudes=np.array([Glast['z'], Azc]))
                     )

# Sequence Defs
fov = 256e-3
dx = 2e-3
alpha = 4
Nr = 300
R = 8
R_inner = 1
xSpoil = 3

Kmax = 1 / 2 / dx
dK = 1 / fov
rf_duration = 10e-6
ro_duration = 300e-6
minRF_to_ADC_time = 50e-6
rfSpoilingInc = 84

# System limits
system = Opts(max_grad=36, grad_unit='mT/m', max_slew=180, slew_unit='T/m/s',
              rf_ringdown_time=10e-6, rf_dead_time=100e-6, adc_dead_time=10e-6, gamma=42.576e6)

# Sequence object
seq = Sequence(system)

rf = make_gauss_pulse(flip_angle=alpha * np.pi / 180, duration=rf_duration, time_bw_product=3, system=system, use='excitation')

Tenc = rf_duration / 2 + minRF_to_ADC_time + ro_duration
Tg = system.rf_dead_time + rf_duration / 2 + Tenc + system.adc_dead_time
Tt = np.ceil((Tenc * (1 + xSpoil) - Tg) / system.grad_raster_time) * system.grad_raster_time
Ag = Kmax / Tenc
TR = Tg + Tt

adc = make_adc(num_samples=Nr, duration=ro_duration, delay=system.rf_dead_time + rf_duration + minRF_to_ADC_time, system=system)

# Generate samples
phi, theta, im_step = spherical_samples(Kmax, dK, R)
Ns = len(phi)
SamplesBookkeeping = [Ns]

print(f'Populating ZTE loop ({Ns} TRs)')
populate_subsequence(system, seq, rf, adc, phi, theta, im_step, Ns, Ag, TR, Tt, 0, 50e-3 * Ag)

# SPI Loop
KstartZTE = Ag * (rf_duration / 2 + minRF_to_ADC_time + adc.dwell)
nKspi = int(np.floor(KstartZTE / dK))
dKspi = KstartZTE / (nKspi + 1)
Tenc_spi = rf_duration / 2 + minRF_to_ADC_time + adc.dwell / 2

print(f'Populating SPI loop ({nKspi} spheres)')
for s in range(nKspi, 0, -1):
    phi_s, theta_s, im_step_s = spherical_samples(dKspi * s, dKspi, R_inner)
    Ns_s = len(phi_s)
    SamplesBookkeeping.append(Ns_s)
    Ag_s = dKspi * s / Tenc_spi

    print(f'Populating sphere {s} ({Ns_s} TRs)')
    populate_subsequence(system, seq, rf, adc, phi_s, theta_s, im_step_s, Ns_s, Ag_s, TR, Tt, 0, 50e-3 * Ag_s)

# Center of k-space
seq.add_block(make_delay(Tt))
rf_center = copy.deepcopy(rf)
rf_center.phase_offset = 0.0
rf_center.freq_offset = 0.0
seq.add_block(rf_center, adc, make_delay(TR - Tt))
SamplesBookkeeping.append(1)

# Timing validation
ok, error_report = seq.check_timing()

if ok:
    print('Timing check passed successfully')
else:
    print('Timing check failed! Error listing follows:')
    print(error_report)

seq.set_definition('FOV', [fov, fov, fov])
seq.set_definition('Name', 'petra')
seq.set_definition('SamplesPerShell', SamplesBookkeeping)

# Export sequence
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '../demoSeq_pypulseq_results')
os.makedirs(RESULTS_DIR, exist_ok=True)
seq.write(os.path.join(RESULTS_DIR, 'ZTE_Petra_py.seq'), remove_duplicates=False)
