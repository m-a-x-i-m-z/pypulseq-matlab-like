import numpy as np
import pytest
import pypulseq as pp


def _events():
    return {
        'trap': pp.make_trapezoid('x', area=1000, duration=1e-3),
        'extended': pp.make_extended_trapezoid('x', times=[0, 1e-4, 2e-4], amplitudes=[0, 100000, 0]),
        'extended_delay': pp.make_extended_trapezoid('x', times=[1e-4, 2e-4, 3e-4], amplitudes=[0, 100000, 0]),
        'end_high': pp.make_extended_trapezoid('x', times=[0, 1e-4, 2e-4], amplitudes=[0, 100000, 100000]),
        'start_high': pp.make_extended_trapezoid('x', times=[0, 1e-4, 2e-4], amplitudes=[100000, 100000, 0]),
        'start_high2': pp.make_extended_trapezoid('x', times=[0, 1e-4, 2e-4], amplitudes=[200000, 100000, 0]),
        'all_high': pp.make_extended_trapezoid('x', times=[0, 1e-4, 2e-4], amplitudes=[100000, 100000, 100000]),
        'delay': pp.make_delay(1e-3),
    }


def _identity():
    return pp.make_rotation(np.eye(3))


def _rotation():
    return pp.make_rotation(np.array(((0, -1, 0), (1, 0, 0), (0, 0, 1))))


class TestBlock:
    def testGradientContinuity1(self):
        events = _events()
        sequence = pp.Sequence()
        sequence.add_block(events['trap'])
        sequence.add_block(events['extended'])
        sequence.add_block(events['trap'])

    def testGradientContinuity2(self):
        events = _events()
        sequence = pp.Sequence()
        sequence.add_block(events['trap'])
        with pytest.raises(Exception):
            sequence.add_block(events['start_high'])

    def testGradientContinuity3(self):
        with pytest.raises(Exception):
            pp.Sequence().add_block(_events()['start_high'])

    def testGradientContinuity4(self):
        events = _events()
        sequence = pp.Sequence()
        sequence.add_block(events['delay'])
        with pytest.raises(Exception):
            sequence.add_block(events['all_high'])

    def testGradientContinuity5(self):
        pp.Sequence().add_block(_events()['extended_delay'])

    def testGradientContinuity6(self):
        events = _events()
        sequence = pp.Sequence()
        sequence.add_block(events['delay'])
        with pytest.raises(Exception):
            sequence.add_block(events['start_high'])

    def testGradientContinuity7(self):
        events = _events()
        sequence = pp.Sequence()
        sequence.add_block(events['end_high'])
        with pytest.raises(Exception):
            sequence.add_block(events['start_high2'])

    def testGradientContinuityRot1(self):
        events = _events()
        sequence = pp.Sequence()
        sequence.add_block(events['trap'], _identity())
        sequence.add_block(events['extended'], _identity())
        sequence.add_block(events['trap'], _identity())

    def testGradientContinuityRot2(self):
        events = _events()
        sequence = pp.Sequence()
        sequence.add_block(events['trap'], _identity())
        with pytest.raises(Exception):
            sequence.add_block(events['start_high'], _identity())

    def testGradientContinuityRot3(self):
        with pytest.raises(Exception):
            pp.Sequence().add_block(_events()['start_high'], _identity())

    def testGradientContinuityRot4(self):
        events = _events()
        sequence = pp.Sequence()
        sequence.add_block(events['delay'])
        with pytest.raises(Exception):
            sequence.add_block(events['all_high'], _identity())

    def testGradientContinuityRot5(self):
        pp.Sequence().add_block(_events()['extended_delay'], _identity())

    def testGradientContinuityRot6(self):
        events = _events()
        sequence = pp.Sequence()
        sequence.add_block(events['delay'])
        with pytest.raises(Exception):
            sequence.add_block(events['start_high'], _identity())

    def testGradientContinuityRot7(self):
        events = _events()
        sequence = pp.Sequence()
        sequence.add_block(events['end_high'], _identity())
        with pytest.raises(Exception):
            sequence.add_block(events['start_high2'], _identity())

    def testGradientContinuityRot8(self):
        events = _events()
        sequence = pp.Sequence()
        sequence.add_block(events['end_high'], _rotation())
        sequence.add_block(events['start_high'], _rotation())

    def testGradientContinuityRot9(self):
        events = _events()
        sequence = pp.Sequence()
        sequence.add_block(events['end_high'], _identity())
        with pytest.raises(Exception):
            sequence.add_block(events['start_high'], _rotation())

    def testGradientContinuityRot10(self):
        events = _events()
        sequence = pp.Sequence()
        sequence.add_block(events['end_high'], _rotation())
        with pytest.raises(Exception):
            sequence.add_block(events['start_high'], _identity())