from types import SimpleNamespace

import pypulseq as pp


class TestBlock2events:
    def test_block_struct(self):
        rf = pp.make_block_pulse(3.141592653589793 / 2, duration=1e-3)
        gx = pp.make_trapezoid('x', area=1000, duration=1e-3)
        block = SimpleNamespace(rf=rf, gx=gx, gy=None, gz=None)
        events = pp.block_to_events(block)
        assert isinstance(events, tuple)
        assert len(events) == 2

    def test_cell_passthrough(self):
        gx = pp.make_trapezoid('x', area=1000, duration=1e-3)
        adc = pp.make_adc(128, duration=1e-3)
        events = pp.block_to_events([gx, adc])
        assert isinstance(events, tuple)
        assert len(events) == 2

    def test_nested_cell_unwrap(self):
        gx = pp.make_trapezoid('x', area=1000, duration=1e-3)
        events = pp.block_to_events([[gx]])
        assert isinstance(events, (tuple, list, SimpleNamespace))
        if isinstance(events, (tuple, list)):
            assert len(events) == 1

    def test_single_event_block(self):
        gx = pp.make_trapezoid('x', area=1000, duration=1e-3)
        events = pp.block_to_events(SimpleNamespace(rf=None, gx=gx, gy=None, gz=None))
        assert isinstance(events, tuple)
        assert len(events) == 1
