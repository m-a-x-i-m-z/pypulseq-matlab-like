import os
import subprocess
import sys
from pathlib import Path


EXCLUDED_SEQUENCES = {'writeMPRAGE_4ge', 'just_some_nonexistant_sequence_to_test_multiple_excludes'}


def test_run_all_demo_seqs():
    repository_root = Path(__file__).resolve().parents[1]
    demo_dir = repository_root / 'examples' / 'demoSeq'
    demo_scripts = sorted(script for script in demo_dir.glob('*.py') if script.stem not in EXCLUDED_SEQUENCES)

    environment = os.environ.copy()
    environment.setdefault('MPLBACKEND', 'Agg')
    environment['PYTHONWARNINGS'] = 'default'
    python_path = environment.get('PYTHONPATH')
    environment['PYTHONPATH'] = os.pathsep.join(
        path for path in (str(repository_root / 'src'), python_path) if path
    )

    failures = []
    for script in demo_scripts:
        result = subprocess.run(
            [sys.executable, str(script)],
            cwd=script.parent,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode:
            failures.append(f'{script.name} (exit {result.returncode}):\n{result.stdout}{result.stderr}')

    assert not failures, '\n'.join(failures)
