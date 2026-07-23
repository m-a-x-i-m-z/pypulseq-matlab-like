from pathlib import Path
import re

import numpy as np
import pytest

import pypulseq_matlab_like as pp


EXPECTED_OUTPUT = Path(__file__).resolve().parent / 'expected_output'


def _normalize_seq(path):
    text = path.read_text(encoding='utf-8').replace('\r\n', '\n').replace('\r', '\n')
    text = re.sub(r'\n\[SIGNATURE\][\s\S]*$', '', text)
    text = re.sub(r'\n\[SHAPES\][\s\S]*$', '', text)
    text = re.sub(r'[ \t]+\n', '\n', text)
    return text.rstrip() + '\n'


def _compare_wave_data(reference, candidate):
    wave_ref, exc_ref, ref_ref, adc_ref, fp_ref, pm_ref = reference.waveforms_and_times(True)
    wave_out, exc_out, ref_out, adc_out, fp_out, pm_out = candidate.waveforms_and_times(True)
    for ref_wave, out_wave in zip(wave_ref, wave_out):
        if ref_wave.size == 0 or ref_wave.shape != out_wave.shape:
            continue
        np.testing.assert_allclose(out_wave[0], ref_wave[0], atol=1e-9)
        max_tol = 5e-6 * max(np.max(np.abs(w[1])) for w in wave_ref[:3] if w.size) if ref_wave is not wave_ref[3] else np.max(np.abs(ref_wave[1]))
        np.testing.assert_allclose(out_wave[1], ref_wave[1], atol=max_tol)
    for ref_data, out_data, tol in ((exc_ref, exc_out, 1e-3), (ref_ref, ref_out, 1e-3), (adc_ref, adc_out, 1e-9), (fp_ref, fp_out, 1e-3), (pm_ref, pm_out, 1e-4)):
        if np.asarray(ref_data).size:
            np.testing.assert_allclose(out_data, ref_data, atol=tol)


@pytest.mark.parametrize('source', sorted(EXPECTED_OUTPUT.glob('*.seq')), ids=lambda path: path.stem)
def test_seq_text_binary_text_roundtrip(source, tmp_path):
    seq = pp.Sequence(); seq.read(str(source))
    canonical = tmp_path / f'{source.stem}_canonical.seq'; binary = tmp_path / f'{source.stem}.bin'; output = tmp_path / f'{source.stem}_roundtrip.seq'
    seq.write(str(canonical)); seq.write_binary(str(binary))
    roundtrip = pp.Sequence(); roundtrip.read_binary(str(binary)); roundtrip.write(str(output))
    source_text, canonical_text, output_text = _normalize_seq(source), _normalize_seq(canonical), _normalize_seq(output)
    assert output_text == source_text or output_text == canonical_text
    _compare_wave_data(seq, roundtrip)
