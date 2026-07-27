from itertools import combinations

import pypulseq_matlab_like as pp
import pytest


def _event_zoo():
    return (
        (pp.make_trapezoid('x', amplitude=1, duration=1), 1),
        (pp.make_trapezoid('x', amplitude=1, duration=1, delay=1), 2),
        (pp.make_delay(1), 1),
        (pp.make_delay(0), 0),
        (pp.make_block_pulse(0, duration=1), 1),
        (pp.make_block_pulse(10, duration=1), 1),
        (pp.make_block_pulse(10, duration=1, delay=1), 2),
        (pp.make_adc(1, duration=3), 3),
        (pp.make_adc(1, duration=3, delay=1), 4),
        (pp.make_digital_output_pulse('osc0', duration=42), 42),
        (pp.make_digital_output_pulse('osc1', duration=42, delay=1), 43),
        (pp.make_digital_output_pulse('osc1', duration=42, delay=9), 51),
        (pp.make_trigger('physio1', duration=59), 59),
        (pp.make_trigger('physio2', duration=59, delay=1), 60),
        (pp.make_label('SLC', 'SET', 0), 0),
    )


class TestCalcDuration:
    def testNoEvent(self):
        assert pp.calc_duration([]) == 0

    def testSingleEvents(self):
        for event, expected_duration in _event_zoo():
            assert pp.calc_duration(event) == expected_duration

    def testEventCombinations2(self):
        zoo = _event_zoo()
        for pair in combinations(zoo, 2):
            assert pp.calc_duration(*(event for event, _ in pair)) == max(expected for _, expected in pair)

    def testEventCombinations3(self):
        zoo = _event_zoo()
        for triple in combinations(zoo, 3):
            assert pp.calc_duration(*(event for event, _ in triple)) == max(expected for _, expected in triple)
