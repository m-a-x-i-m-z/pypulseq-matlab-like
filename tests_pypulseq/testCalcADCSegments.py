import numpy as np
import pypulseq as pp
from pathlib import Path

from . import assert_equal


class TestCalcADCSegments:
    def testCalcADC(self):
        data = np.loadtxt(Path(__file__).resolve().parent / 'expected_output' / 'pulseq_calcAdcSeg.txt')
        data = data[::10]
        system = pp.Opts(adc_raster_time=1e-7, grad_raster_time=1e-5)
        for dwell, num_samples, adc_limit, adc_divisor, mode_number, expected_segments, expected_samples in data:
            system.adc_samples_limit = int(adc_limit)
            system.adc_samples_divisor = int(adc_divisor)
            mode = 'shorten' if mode_number == 1 else 'lengthen'
            segments, samples_per_segment = pp.calc_adc_segments(int(num_samples), dwell, system, mode)
            assert_equal(segments, expected_segments, abs_tol=1e-9)
            assert_equal(samples_per_segment, expected_samples, abs_tol=1e-9)
            assert samples_per_segment <= adc_limit
            segment_duration = samples_per_segment * dwell
            assert abs(round(segment_duration / system.grad_raster_time) - segment_duration / system.grad_raster_time) < 1e-9
            adc_duration = segment_duration * segments
            assert abs(round(adc_duration / system.grad_raster_time) - adc_duration / system.grad_raster_time) < 1e-9
