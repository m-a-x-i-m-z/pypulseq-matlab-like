import os
import subprocess
import sys
from pathlib import Path

import pytest

EXCLUDED_SEQUENCES = {'writeMPRAGE_4ge', 'just_some_nonexistant_sequence_to_test_multiple_excludes'}


def _get_demo_scripts() -> list[Path]:
    repository_root = Path(__file__).resolve().parents[1]
    demo_dir = repository_root / 'examples' / 'demoSeq'
    return sorted(script for script in demo_dir.glob('*.py') if script.stem not in EXCLUDED_SEQUENCES)


@pytest.mark.parametrize('script', _get_demo_scripts(), ids=lambda script: script.name)
def test_run_demo_seq(script: Path):
    repository_root = Path(__file__).resolve().parents[1]

    environment = os.environ.copy()
    environment.setdefault('MPLBACKEND', 'Agg')
    environment['PYTHONWARNINGS'] = 'default'
    python_path = environment.get('PYTHONPATH')
    environment['PYTHONPATH'] = os.pathsep.join(
        path for path in (str(repository_root / 'src'), python_path) if path
    )

    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=script.parent,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, f'{script.name} (exit {result.returncode}):\n{result.stdout}{result.stderr}'
