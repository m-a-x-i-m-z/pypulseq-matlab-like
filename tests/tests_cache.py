"""Compatibility and correctness tests for both independent Sequence caches."""

import gc
import subprocess
import sys
import weakref
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy

import numpy as np
import pytest

import pypulseq_matlab_like as pp
from pypulseq_matlab_like.Sequence import block
from pypulseq_matlab_like.Sequence.caches import BlockCache, EventRegistrationCache


def _rf(duration=1e-3, use='excitation'):
    return pp.make_block_pulse(
        np.pi / 2,
        duration=duration,
        use=use,
    )


def test_package_and_matlab_like_slr_do_not_eagerly_import_sigpy_or_numba():
    script = """
import sys
import numpy as np
import pypulseq_matlab_like as pp

def loaded():
    return [
        name for name in sys.modules
        if name == 'sigpy' or name.startswith('sigpy.')
        or name == 'numba' or name.startswith('numba.')
    ]

assert not loaded(), loaded()
pp.make_slr_pulse(
    np.pi / 6,
    duration=1e-3,
    filter_type='ls',
    return_gz=False,
)
assert not loaded(), loaded()
"""
    subprocess.run([sys.executable, '-c', script], check=True, timeout=15)


def test_both_caches_are_disabled_by_default_and_independently_configurable():
    default = pp.Sequence()
    matlab_mode = pp.Sequence(use_event_cache=False, use_block_cache=False)

    assert default.use_event_cache is False
    assert default.use_block_cache is False
    assert matlab_mode.use_event_cache is False
    assert matlab_mode.use_block_cache is False

    for use_event_cache, use_block_cache in (
        (False, False),
        (False, True),
        (True, False),
        (True, True),
    ):
        seq = pp.Sequence(
            use_event_cache=use_event_cache,
            use_block_cache=use_block_cache,
        )
        seq.add_block(_rf())
        seq.get_block(1)
        assert seq.event_cache_size == int(use_event_cache)
        assert seq.block_cache_size == int(use_block_cache)


def test_matlab_mode_registers_each_idless_event_and_decompresses_each_read(monkeypatch):
    rf = _rf()
    original = block.register_rf_event
    registration_count = 0

    def counted_registration(sequence, event):
        nonlocal registration_count
        registration_count += 1
        return original(sequence, event)

    monkeypatch.setattr(block, 'register_rf_event', counted_registration)
    seq = pp.Sequence(use_event_cache=False, use_block_cache=False)
    seq.add_block(rf)
    seq.add_block(rf)

    first = seq.get_block(1)
    second = seq.get_block(1)
    assert registration_count == 2
    assert first is not second
    assert first.rf is not second.rf
    assert seq.event_cache_size == 0
    assert seq.block_cache_size == 0
    assert len(seq.rf_library.data) == 1


def test_matlab_mode_honors_explicit_registered_event_ids(monkeypatch):
    seq = pp.Sequence(use_event_cache=False, use_block_cache=False)
    rf = _rf()
    trap = pp.make_trapezoid(channel='x', amplitude=1000, flat_time=1e-3)
    adc = pp.make_adc(num_samples=16, duration=1e-3)

    rf.id, rf.shape_IDs = seq.register_rf_event(rf)
    trap.id = seq.register_grad_event(trap)
    adc.id, adc.shape_id = seq.register_adc_event(adc)

    def unexpected_registration(*args, **kwargs):
        raise AssertionError('an event carrying a MATLAB-style explicit id must not be registered again')

    monkeypatch.setattr(block, 'register_rf_event', unexpected_registration)
    monkeypatch.setattr(block, 'register_grad_event', unexpected_registration)
    monkeypatch.setattr(block, 'register_adc_event', unexpected_registration)
    seq.add_block(rf, trap)
    seq.add_block(adc)

    assert int(seq.block_events[1][1]) == rf.id
    assert int(seq.block_events[1][2]) == trap.id
    assert int(seq.block_events[2][5]) == adc.id
    assert seq.event_cache_size == 0


def test_cache_modes_produce_byte_identical_sequence_files(tmp_path):
    rf = _rf()
    grad = pp.make_trapezoid(channel='x', amplitude=1000, flat_time=1e-3)
    adc = pp.make_adc(num_samples=16, duration=1e-3)
    cached = pp.Sequence(use_event_cache=True, use_block_cache=True)
    matlab_mode = pp.Sequence(use_event_cache=False, use_block_cache=False)

    for seq in (cached, matlab_mode):
        seq.add_block(rf, grad)
        seq.add_block(adc)
        seq.add_block(rf, grad)
        seq.get_block(1)

    cached_path = tmp_path / 'cached.seq'
    matlab_path = tmp_path / 'matlab-mode.seq'
    cached.write(str(cached_path))
    matlab_mode.write(str(matlab_path))

    assert cached_path.read_bytes() == matlab_path.read_bytes()
    assert cached.check_timing()[0] and matlab_mode.check_timing()[0]


def test_disabled_event_cache_does_not_build_registration_keys(monkeypatch):
    def unexpected_registration_key(*args, **kwargs):
        raise AssertionError('disabled event cache must not calculate registration keys')

    monkeypatch.setattr(block, 'make_registration_key', unexpected_registration_key)
    seq = pp.Sequence(use_event_cache=False)
    seq.add_block(_rf())


def test_event_cache_is_sequence_owned_and_avoids_registration_work(monkeypatch):
    rf = _rf()
    original = block.register_rf_event
    registration_count = 0

    def counted_registration(sequence, event):
        nonlocal registration_count
        registration_count += 1
        return original(sequence, event)

    monkeypatch.setattr(block, 'register_rf_event', counted_registration)
    first = pp.Sequence(use_event_cache=True)
    second = pp.Sequence(use_event_cache=True)
    for seq in (first, second):
        seq.add_block(rf)
        seq.add_block(rf)

    assert registration_count == 2
    assert first.event_cache_size == second.event_cache_size == 1
    assert not hasattr(rf, '_pypulseq_sequence_event_cache')
    assert len(first.rf_library.data) == len(second.rf_library.data) == 1


def test_repeated_sequence_writes_do_not_leak_event_ids_between_sequences(tmp_path):
    shared_rf = _rf(duration=1e-3, use='excitation')
    for sequence_index in range(12):
        seq = pp.Sequence(
            use_event_cache=True,
            use_block_cache=bool(sequence_index % 2),
        )
        expected_shared_id = 1
        if sequence_index % 2:
            seq.add_block(_rf(duration=0.5e-3, use='refocusing'))
            expected_shared_id = 2

        first_shared_block = seq.next_free_block_ID
        seq.add_block(shared_rf)
        seq.add_block(shared_rf)
        assert int(seq.block_events[first_shared_block][1]) == expected_shared_id
        assert int(seq.block_events[first_shared_block + 1][1]) == expected_shared_id
        assert len(seq.rf_library.data) == expected_shared_id

        first_path = tmp_path / f'sequence-{sequence_index}-first.seq'
        second_path = tmp_path / f'sequence-{sequence_index}-second.seq'
        seq.write(str(first_path), create_signature=False)
        assert seq.event_cache_size == 0
        seq.write(str(second_path), create_signature=False)
        assert first_path.read_bytes() == second_path.read_bytes()

        seq.add_block(shared_rf)
        assert int(seq.block_events[seq.next_free_block_ID - 1][1]) == expected_shared_id
        assert len(seq.rf_library.data) == expected_shared_id

        loaded = pp.Sequence()
        loaded.read(str(first_path))
        assert loaded.get_block(first_shared_block).rf.shape_dur == pytest.approx(1e-3)
        assert loaded.get_block(first_shared_block + 1).rf.shape_dur == pytest.approx(1e-3)

    assert not hasattr(shared_rf, '_pypulseq_sequence_event_cache')


def test_every_cached_event_id_is_local_to_its_sequence(tmp_path):
    shared_events = (
        (_rf(), ()),
        (pp.make_trapezoid('x', amplitude=1000, flat_time=1e-3), ()),
        (
            pp.make_arbitrary_grad(
                'x',
                np.array([0.0, 0.0, 1000.0, 0.0, 0.0]),
                first=0,
                last=0,
            ),
            (),
        ),
        (pp.make_adc(num_samples=16, duration=1e-3), ()),
        (pp.make_rotation(np.eye(3)), (1e-3,)),
    )
    dummy_events = (
        (_rf(duration=0.5e-3, use='refocusing'), ()),
        (pp.make_trapezoid('x', amplitude=2000, flat_time=0.5e-3), ()),
        (
            pp.make_arbitrary_grad(
                'x',
                np.array([0.0, 0.0, 2000.0, 0.0, 0.0]),
                first=0,
                last=0,
            ),
            (),
        ),
        (pp.make_adc(num_samples=8, duration=0.5e-3), ()),
        (pp.make_rotation(np.pi / 4), (1e-3,)),
    )

    offset = pp.Sequence(use_event_cache=True, use_block_cache=True)
    for event, extra_args in dummy_events:
        offset.add_block(event, *extra_args)
    offset_shared_start = offset.next_free_block_ID
    for event, extra_args in shared_events:
        offset.add_block(event, *extra_args)

    clean = pp.Sequence(use_event_cache=True, use_block_cache=True)
    for event, extra_args in shared_events:
        clean.add_block(event, *extra_args)

    def registered_ids(sequence, first_block):
        blocks = [sequence.get_block(first_block + index, add_ids=True) for index in range(5)]
        return (
            blocks[0].rf.id,
            blocks[1].gx.id,
            blocks[2].gx.id,
            blocks[3].adc.id,
            blocks[4].rotation.id,
        )

    assert registered_ids(offset, offset_shared_start) == (2, 3, 4, 2, 2)
    assert registered_ids(clean, 1) == (1, 1, 2, 1, 1)
    assert offset.event_cache_size == 10
    assert clean.event_cache_size == 5

    first_path = tmp_path / 'mixed-clean-first.seq'
    second_path = tmp_path / 'mixed-clean-second.seq'
    clean.write(str(first_path), create_signature=False)
    clean.write(str(second_path), create_signature=False)
    assert first_path.read_bytes() == second_path.read_bytes()

    loaded = pp.Sequence()
    loaded.read(str(first_path))
    assert registered_ids(loaded, 1) == (1, 1, 2, 1, 1)
    for event, _extra_args in shared_events:
        assert not hasattr(event, '_pypulseq_sequence_event_cache')


def test_all_cached_event_registration_branches_reuse_work(monkeypatch):
    counts = {'rf': 0, 'grad': 0, 'adc': 0, 'rotation': 0}
    original_rf = block.register_rf_event
    original_grad = block.register_grad_event
    original_adc = block.register_adc_event
    original_rotation = pp.Sequence.register_rotation_event

    def counted_rf(sequence, event):
        counts['rf'] += 1
        return original_rf(sequence, event)

    def counted_grad(sequence, event):
        counts['grad'] += 1
        return original_grad(sequence, event)

    def counted_adc(sequence, event):
        counts['adc'] += 1
        return original_adc(sequence, event)

    def counted_rotation(sequence, event):
        counts['rotation'] += 1
        return original_rotation(sequence, event)

    monkeypatch.setattr(block, 'register_rf_event', counted_rf)
    monkeypatch.setattr(block, 'register_grad_event', counted_grad)
    monkeypatch.setattr(block, 'register_adc_event', counted_adc)
    monkeypatch.setattr(pp.Sequence, 'register_rotation_event', counted_rotation)

    event_specs = (
        (_rf(), ()),
        (pp.make_trapezoid(channel='x', amplitude=1000, flat_time=1e-3), ()),
        (
            pp.make_arbitrary_grad(
                channel='x',
                waveform=np.array([0.0, 0.0, 1000.0, 0.0, 0.0]),
                first=0,
                last=0,
            ),
            (),
        ),
        (pp.make_adc(num_samples=16, duration=1e-3), ()),
        (pp.make_rotation(np.eye(3)), (1e-3,)),
    )
    seq = pp.Sequence(use_event_cache=True)
    for event, extra_args in event_specs:
        seq.add_block(event, *extra_args)
        seq.add_block(event, *extra_args)

    assert counts == {'rf': 1, 'grad': 2, 'adc': 1, 'rotation': 1}
    assert seq.event_cache_size == len(event_specs)


def test_event_mutation_changes_registration_and_cache_stays_bounded():
    rf = _rf()
    seq = pp.Sequence(use_event_cache=True)

    for phase_index in range(20):
        rf.phase_offset = phase_index * np.pi / 20
        seq.add_block(rf)
        assert seq.event_cache_size == 1

    phases = [seq.get_block(block_id).rf.phase_offset for block_id in seq.block_events]
    assert len(seq.rf_library.data) == 20
    assert phases == pytest.approx([phase_index * np.pi / 20 for phase_index in range(20)])

    rf.signal *= 0.5
    seq.add_block(rf)
    assert seq.event_cache_size == 1
    assert len(seq.rf_library.data) == 21


def test_rotation_mutation_changes_registration_key():
    rotation = pp.make_rotation(np.eye(3))
    seq = pp.Sequence(use_event_cache=True)
    seq.add_block(rotation, 1e-3)
    rotation.rot_quaternion = np.array([0.0, 1.0, 0.0, 0.0])
    seq.add_block(rotation, 1e-3)

    assert seq.event_cache_size == 1
    assert len(seq.rotation_library.data) == 2
    assert np.array_equal(seq.get_block(1).rotation.rot_quaternion, np.array([1.0, 0.0, 0.0, 0.0]))
    assert np.array_equal(seq.get_block(2).rotation.rot_quaternion, rotation.rot_quaternion)


def test_short_lived_and_parallel_sequences_cannot_share_event_registrations():
    short_rf = _rf(duration=0.5e-3, use='excitation')
    long_rf = _rf(duration=2e-3, use='refocusing')
    for index in range(200):
        temporary = pp.Sequence(use_event_cache=True)
        temporary.add_block(short_rf if index % 2 == 0 else long_rf)

    final = pp.Sequence(use_event_cache=True)
    final.add_block(short_rf)
    final.add_block(long_rf)
    assert [int(events[1]) for events in final.block_events.values()] == [1, 2]
    assert [final.get_block(i).rf.shape_dur for i in final.block_events] == pytest.approx([0.5e-3, 2e-3])

    shared_rf = _rf()

    def build_sequence(_):
        seq = pp.Sequence(use_event_cache=True)
        for _ in range(25):
            seq.add_block(shared_rf)
        return len(seq.rf_library.data), seq.event_cache_size, seq.check_timing()[0]

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(build_sequence, range(32)))
    assert results == [(1, 1, True)] * 32


def test_event_cache_retains_identity_until_clear_and_is_not_deepcopied():
    class Event:
        pass

    event = Event()
    event_ref = weakref.ref(event)
    cache = EventRegistrationCache()
    cache.put(event, ('content',), id=1)
    copied = deepcopy(cache)

    assert len(cache) == 1
    assert len(copied) == 0
    del event
    gc.collect()
    assert event_ref() is not None
    cache.clear()
    gc.collect()
    assert event_ref() is None


def test_block_cache_component_owns_copying_enablement_and_copy_lifecycle():
    block_cache = BlockCache()
    block = {'waveform': np.array([1.0, 2.0])}
    block_cache[7] = block
    block['waveform'][0] = 99

    first = block_cache[7]
    first['waveform'][1] = 88
    assert np.array_equal(block_cache[7]['waveform'], np.array([1.0, 2.0]))
    assert len(deepcopy(block_cache)) == 0

    block_cache.enabled = False
    assert block_cache == {}
    block_cache[8] = block
    assert block_cache == {}


def test_cached_blocks_are_returned_as_independent_values():
    seq = pp.Sequence(use_block_cache=True)
    seq.add_block(_rf())
    first = seq.get_block(1)
    original_phase = first.rf.phase_offset
    original_signal = first.rf.signal.copy()
    first.rf.phase_offset = 1.234
    first.rf.signal[:] = 0

    second = seq.get_block(1)
    assert second is not first
    assert second.rf is not first.rf
    assert second.rf.phase_offset == original_phase
    assert np.array_equal(second.rf.signal, original_signal)


def test_set_block_and_cache_toggle_cannot_return_stale_blocks():
    seq = pp.Sequence(use_block_cache=True)
    seq.add_block(_rf(duration=1e-3, use='excitation'))
    seq.add_block(_rf(duration=3e-3, use='refocusing'))
    seq.get_block(1)
    seq.get_block(2)
    assert set(seq.block_cache) == {1, 2}

    seq.set_block(1, _rf(duration=2e-3, use='refocusing'))
    assert set(seq.block_cache) == {2}
    assert seq.get_block(1).rf.shape_dur == 2e-3
    assert seq.get_block(2).rf.shape_dur == 3e-3

    seq.use_block_cache = False
    assert seq.block_cache == {}
    assert seq.get_block(1) is not seq.get_block(1)
    seq.use_block_cache = True
    assert seq.block_cache == {}
    assert seq.get_block(1).rf.shape_dur == 2e-3


def test_library_and_duration_mutations_invalidate_affected_cache_state():
    trap = pp.make_trapezoid(channel='x', amplitude=1000, flat_time=1e-3)
    seq = pp.Sequence(use_event_cache=True, use_block_cache=True)
    seq.add_block(trap)
    seq.get_block(1)
    assert seq.event_cache_size == seq.block_cache_size == 1

    seq.mod_grad_axis('x', 2)
    assert seq.event_cache_size == seq.block_cache_size == 0
    assert seq.get_block(1).gx.amplitude == pytest.approx(2000)

    # The caller-owned event is still the original 1000 Hz/m trapezoid.
    # It must be registered again instead of reusing the now-scaled library ID.
    seq.add_block(trap)
    assert int(seq.block_events[2][2]) != int(seq.block_events[1][2])
    assert seq.get_block(2).gx.amplitude == pytest.approx(1000)

    soft_seq = pp.Sequence(use_block_cache=True)
    soft_seq.add_block(pp.make_soft_delay('TE', default_duration=5e-3))
    assert soft_seq.get_block(1).block_duration == pytest.approx(5e-3)
    assert soft_seq.block_cache_size == 1

    soft_seq.apply_soft_delay(TE=8e-3)
    assert soft_seq.block_cache_size == 0
    refreshed = soft_seq.get_block(1)
    assert refreshed.block_duration == pytest.approx(8e-3)
    assert refreshed.soft_delay.default_duration == pytest.approx(8e-3)


def test_clear_context_io_and_deduplication_boundaries(tmp_path):
    rf = _rf()
    seq = pp.Sequence(use_event_cache=True, use_block_cache=True)
    seq.add_block(rf)
    seq.get_block(1)
    assert seq.event_cache_size == seq.block_cache_size == 1

    copied = seq.remove_duplicates()
    assert seq.event_cache_size == 1
    assert copied.event_cache_size == 0
    assert copied.block_cache == {}

    seq.remove_duplicates(in_place=True)
    assert seq.event_cache_size == 0
    assert seq.block_cache_size == 0

    seq.clear_caches()
    assert seq.event_cache_size == 0
    assert seq.block_cache_size == 0

    with pp.Sequence(use_event_cache=True, use_block_cache=True) as scoped:
        scoped.add_block(rf)
        scoped.get_block(1)
        assert scoped.event_cache_size == scoped.block_cache_size == 1
    assert scoped.event_cache_size == 0
    assert scoped.block_cache_size == 0

    binary_path = tmp_path / 'cache-boundary.bin'
    seq.add_block(rf)
    seq.write_binary(str(binary_path))
    assert seq.event_cache_size == 0
    seq.add_block(rf)
    seq.read_binary(str(binary_path))
    assert seq.event_cache_size == 0
    assert seq.block_cache_size == 0

    text_seq = pp.Sequence(use_event_cache=True, use_block_cache=True)
    text_seq.add_block(rf)
    text_seq.get_block(1)
    text_path = tmp_path / 'cache-boundary.seq'
    text_seq.write(str(text_path))
    assert text_seq.event_cache_size == 0
    text_seq.add_block(rf)
    text_seq.read(str(text_path))
    assert text_seq.event_cache_size == 0
    assert text_seq.block_cache_size == 0
