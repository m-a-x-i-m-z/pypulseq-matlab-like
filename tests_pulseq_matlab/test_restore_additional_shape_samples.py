import pytest

import numpy as np

## Add pypulseq source to path (for debugging)
#import sys
#import os
#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import pypulseq_matlab_like as pp


class TestRestoreAdditionalShapeSamples:
    def test_trapezoid_corners(self):
        system = pp.Opts()
        gradient = pp.make_trapezoid('x', area=5000, duration=4e-3)
        count = round(gradient.flat_time / system.grad_raster_time)
        waveform = gradient.amplitude * np.ones(count)
        times = (np.arange(1, count + 1) - 0.5) * system.grad_raster_time
        times_changed, waveform_changed = pp.restore_additional_shape_samples(times, waveform, 0, 0, system.grad_raster_time, 1)
        assert len(times_changed) > 0
        assert len(waveform_changed) > 0
        assert len(times_changed) == len(waveform_changed)

    def test_output_valid(self):
        system = pp.Opts()
        count = 10
        waveform = np.linspace(1000, 10000, count)
        times = (np.arange(1, count + 1) - 0.5) * system.grad_raster_time
        times_changed, waveform_changed = pp.restore_additional_shape_samples(times, waveform, 500, 10500, system.grad_raster_time, 1)
        assert np.all(np.isfinite(waveform_changed))
        assert np.all(np.isfinite(times_changed))
        assert np.all(np.diff(times_changed) >= 0)

    def test_constant_waveform(self):
        system = pp.Opts()
        count = 20
        waveform = 5000 * np.ones(count)
        times = (np.arange(1, count + 1) - 0.5) * system.grad_raster_time
        times_changed, waveform_changed = pp.restore_additional_shape_samples(times, waveform, 5000, 5000, system.grad_raster_time, 1)
        assert len(times_changed) == len(waveform_changed)
