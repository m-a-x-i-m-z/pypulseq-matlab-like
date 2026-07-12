from __future__ import annotations

import sys
from pathlib import Path

import pytest


if __name__ == '__main__':
    root = Path(__file__).resolve().parent
    raise SystemExit(pytest.main([str(root), '-o', 'python_files=test*.py', '-W', 'default', *sys.argv[1:]]))
