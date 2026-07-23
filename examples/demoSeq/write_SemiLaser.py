
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
from pypulseq_matlab_like.make_adiabatic_pulse import make_adiabatic_pulse
from pypulseq_matlab_like.make_delay import make_delay
from pypulseq_matlab_like.make_extended_trapezoid_area import make_extended_trapezoid_area
from pypulseq_matlab_like.make_soft_delay import make_soft_delay
from pypulseq_matlab_like.calc_duration import calc_duration
from pypulseq_matlab_like.calc_rf_center import calc_rf_center
from pypulseq_matlab_like.split_gradient_at import split_gradient_at
from pypulseq_matlab_like.align import align
from pypulseq_matlab_like.add_gradients import add_gradients
from pypulseq_matlab_like.scale_grad import scale_grad
from pypulseq_matlab_like.make_extended_trapezoid import make_extended_trapezoid
from pypulseq_matlab_like.make_label import make_label
from pypulseq_matlab_like.make_slr_pulse import make_slr_pulse

# System limits
system = Opts(max_grad=10, grad_unit='mT/m', max_slew=50, slew_unit='T/m/s',
              rf_ringdown_time=20e-6, rf_dead_time=100e-6, adc_dead_time=10e-6)

alpha = 90
sliceThickness = 50e-3
Nx = 2048
Nrep = 1
TE = 30e-3
TR = 6000e-3

rf_90_duration = 2.6e-3
rf_180_duration = 4.6e-3

# Sequence object
seq = Sequence(system)

# Create excitation pulse
rf_90 = make_slr_pulse(
    flip_angle=np.pi / 2,
    duration=rf_90_duration,
    time_bw_product=7.88,
    dwell=rf_90_duration / 500,
    passband_ripple=1.0,
    stopband_ripple=1e-2,
    filter_type='ms',
    system=system,
    use='excitation',
)

# Create refocusing pulse and gradient
rf_180_1 = make_adiabatic_pulse(pulse_type='wurst', duration=rf_180_duration, bandwidth=6000,
                                dwell=rf_180_duration / 500, n_fac=20, use='refocusing', system=system)

timebwproduct_90 = 8
timebwproduct_180 = 22

grad_amplitude_90 = (timebwproduct_90 / 2.6e-3) / sliceThickness
grad_amplitude_180 = (timebwproduct_180 / 4.6e-3) / sliceThickness

gx = make_trapezoid(channel='x', flat_time=rf_90_duration, amplitude=grad_amplitude_90, system=system)
gy = make_trapezoid(channel='y', flat_time=rf_180_duration, amplitude=grad_amplitude_180, system=system)
gz = make_trapezoid(channel='z', flat_time=rf_180_duration, amplitude=grad_amplitude_180, system=system)
gz_2 = make_trapezoid(channel='z', flat_time=rf_180_duration, amplitude=grad_amplitude_180, system=system)

area_90_toRefocus = gx.amplitude * (0.5 * gx.fall_time + gx.flat_time - calc_rf_center(rf_90)[0])

# Gradient spoiling
spoilMoment = 10 / sliceThickness
gzSpoil = make_trapezoid(channel='z', area=spoilMoment, system=system)
gxSpoil = make_trapezoid(channel='x', duration=2 * calc_duration(gzSpoil), area=spoilMoment, system=system)
gzSpoil_2 = make_trapezoid(channel='z', duration=2 * calc_duration(gzSpoil), area=spoilMoment, system=system)

# Split x gradient
gx_parts = split_gradient_at(grad=gx, time_point=calc_duration(gx) - gx.fall_time + system.rf_ringdown_time, system=system)
gx_p1, rf_90, _ = align(right=[gx_parts[0], rf_90, make_delay(system.rf_dead_time + rf_90_duration + system.rf_ringdown_time)])
# `gx_p1` aligns to right.
# If 'right', it aligns endpoints.
# It returns the aligned objects.
# I'll rely on careful timing or use PyPulseq align.
# `gx_p1` is the gradient up to the split point.
gx_parts[1].delay = 0

# Split y gradient
gy_parts = split_gradient_at(grad=gy, time_point=calc_duration(gy) - gy.fall_time + system.rf_ringdown_time, system=system)
rf_180_1.delay = max(calc_duration(gy_parts[0]) - rf_180_duration - system.rf_ringdown_time, calc_duration(gzSpoil))
assert rf_180_1.delay >= system.rf_dead_time
gy_p1 = gy_parts[0]
gy_p1.delay = rf_180_1.delay + rf_180_duration + system.rf_ringdown_time - calc_duration(gy_p1)
gy_parts[1].delay = 0

# Combine y gradient
rf_180_2 = make_adiabatic_pulse(pulse_type='wurst', duration=rf_180_duration, bandwidth=6000, dwell=rf_180_duration / 500, n_fac=20, use='refocusing', system=system) # copy?
rf_180_2.delay = max(calc_duration(gxSpoil), calc_duration(gzSpoil_2))
gy_tmp = split_gradient_at(grad=gy, time_point=calc_duration(gy) - gy.fall_time, system=system)
gy_tmp[0].delay = rf_180_2.delay + rf_180_duration - calc_duration(gy_tmp[0])
gySpoil, _, _ = make_extended_trapezoid_area(channel='y', grad_start=gy.amplitude, grad_end=0, area=spoilMoment + 0.5 * gy.amplitude * gy.fall_time, system=system)
gySpoil.delay = calc_duration(gy_tmp[0])
gy_comb = add_gradients(grads=(gy_parts[1], gy_tmp[0], gySpoil), system=system)
gy_comb_parts = split_gradient_at(grad=gy_comb, time_point=rf_180_2.delay + rf_180_duration + system.rf_ringdown_time, system=system)
gy_comb_parts[1].delay = 0

#
gxSpoil_2 = make_trapezoid(channel='x', area=spoilMoment, system=system)
rf_180_3 = make_adiabatic_pulse(pulse_type='wurst', duration=rf_180_duration, bandwidth=6000, dwell=rf_180_duration / 500, n_fac=20, use='refocusing', system=system)
rf_180_3.delay = calc_duration(gxSpoil_2)
gz.delay = rf_180_3.delay - gz.rise_time

# Additional Spoiler gradients
gzSpoil_semiFinal, _, _ = make_extended_trapezoid_area(channel='z', grad_start=0, grad_end=gz.amplitude, area=spoilMoment, system=system)
gxSpoil_semiFinal = make_trapezoid(channel='x', area=spoilMoment + area_90_toRefocus, system=system)
gySpoil_semiFinal = make_trapezoid(channel='y', area=2 * spoilMoment, system=system)
# Align right?
max_dur_spf = calc_duration(gxSpoil_semiFinal, gySpoil_semiFinal, gzSpoil_semiFinal)
gxSpoil_semiFinal.delay = max_dur_spf - calc_duration(gxSpoil_semiFinal)
gySpoil_semiFinal.delay = max_dur_spf - calc_duration(gySpoil_semiFinal)
gzSpoil_semiFinal.delay = max_dur_spf - calc_duration(gzSpoil_semiFinal)

rf_180_4 = make_adiabatic_pulse(pulse_type='wurst', duration=rf_180_duration, bandwidth=6000, dwell=rf_180_duration / 500, n_fac=20, use='refocusing', system=system)
rf_180_4.delay = 0
gz_temp = split_gradient_at(grad=gz_2, time_point=gz_2.rise_time, system=system)
gz_temp[1].delay = 0
gz_parts = split_gradient_at(grad=gz_temp[1], time_point=rf_180_4.delay + rf_180_duration, system=system)
gz_parts[0].delay = calc_duration(gzSpoil_semiFinal)
rf_180_4.delay = calc_duration(gzSpoil_semiFinal)

gzSpoil_Final, _, _ = make_extended_trapezoid_area(channel='z', grad_start=gz.amplitude, grad_end=0, area=spoilMoment, system=system)
gxSpoil_Final = make_trapezoid(channel='x', area=spoilMoment, system=system)
gySpoil_Final = make_trapezoid(channel='y', area=spoilMoment, system=system)

gzSpoil_Final.delay = rf_180_4.delay + rf_180_duration
gz_comb = add_gradients(grads=(gzSpoil_semiFinal, gz_parts[0], gzSpoil_Final), system=system)

gxSpoil_Final.delay = rf_180_4.delay + rf_180_duration
gySpoil_Final.delay = rf_180_4.delay + rf_180_duration

gxSpoil_combi = add_gradients(grads=(gxSpoil_semiFinal, gxSpoil_Final), system=system)
gySpoil_combi = add_gradients(grads=(gySpoil_semiFinal, gySpoil_Final), system=system)

# Timing calculation
lTime1 = (rf_90_duration + rf_180_duration) / 2 + system.rf_ringdown_time + calc_duration(gzSpoil)
lTime2 = (rf_180_duration + rf_180_duration) / 2 + system.rf_ringdown_time + calc_duration(gxSpoil)
lTime3 = (rf_180_duration + rf_180_duration) / 2 + system.rf_ringdown_time + calc_duration(gxSpoil_2)

lTime4 = TE / 2 - lTime2
lTime5 = TE / 2 - lTime1 - lTime3

# ADC
adc = make_adc(num_samples=Nx, dwell=2e-4, system=system)
delayTE1 = lTime4 - (rf_180_duration + gz.fall_time + calc_duration(gySpoil_semiFinal))
delayTE2 = lTime5 - (rf_180_duration / 2 + calc_duration(gySpoil_Final) - gySpoil_Final.delay) - adc.dwell / 2
adc.delay = delayTE2

for i in range(1, Nrep + 1):
    seq.add_block(rf_90, gx_p1)
    seq.add_block(gzSpoil, gx_parts[1], rf_180_1, gy_p1)
    seq.add_block(gxSpoil, gzSpoil_2, rf_180_2, gy_comb_parts[0])
    seq.add_block(gxSpoil_2, gy_comb_parts[1], rf_180_3, gz)
    seq.add_block(make_delay(delayTE1))
    seq.add_block(gxSpoil_combi, gySpoil_combi, gz_comb, rf_180_4)
    seq.add_block(adc, make_delay(calc_duration(adc) + system.adc_dead_time))

    if i == 1:
        # seq.total_duration() maybe?
        # We can sum block durations.
        # But for new TR calculation:
        # `delayTR = TR - seq.duration()`
        # In PyPulseq, `seq.duration()` returns total duration of added blocks.
        current_dur = np.sum(list(seq.block_durations.values()))
        delayTR = TR - current_dur
        assert delayTR > 0

    seq.add_block(make_delay(delayTR))

# Timing validation
ok, error_report = seq.check_timing()

if ok:
    print('Timing check passed successfully')
else:
    print('Timing check failed! Error listing follows:')
    print(error_report)

seq.set_definition('FOV', [sliceThickness, sliceThickness, sliceThickness])
seq.set_definition('Name', 'semiLaser')

# Export sequence
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '../demoSeq_pypulseq_results')
os.makedirs(RESULTS_DIR, exist_ok=True)
seq.write(os.path.join(RESULTS_DIR, 'SemiLaser_py.seq'))