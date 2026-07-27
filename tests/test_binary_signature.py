from pathlib import Path
import tempfile

import numpy as np
import pypulseq_matlab_like as pp
import pytest


def _test_sequences():
    first = pp.Sequence()
    first.add_block(pp.make_trapezoid('x', area=1000, duration=1e-3), pp.make_adc(128, duration=1e-3))
    first.add_block(pp.make_delay(2e-3))

    second = pp.Sequence()
    rf, gz, _ = pp.make_sinc_pulse(np.pi / 2, duration=2e-3, slice_thickness=5e-3, apodization=0.5, time_bw_product=4, use='excitation', return_gz=True)
    second.add_block(rf, gz)
    second.add_block(pp.make_trapezoid('z', area=-gz.area / 2, duration=1e-3))
    second.add_block(pp.make_delay(1e-3))
    return first, second


class TestBinarySignature:
    def test_binary_signature_roundtrip_multiple_sequences(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            for index, sequence in enumerate(_test_sequences()):
                path = Path(temp_dir) / f'signature_{index}.bseq'
                sequence.write_binary(str(path))
                loaded = pp.Sequence()
                loaded.read_binary(str(path))
                valid, stored, computed = pp.verify_file_signature(path)
                assert loaded.signature_type.lower() == 'md5'
                assert loaded.signature_file.lower() == 'bin'
                assert valid
                assert stored == computed
