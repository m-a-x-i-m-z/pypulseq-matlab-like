from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

_src = Path(__file__).resolve().parents[1] / 'src'
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))


def assert_equal(actual: Any, expected: Any, *, abs_tol: float | None = None, rel_tol: float | None = None) -> None:
    if abs_tol is not None or rel_tol is not None:
        np.testing.assert_allclose(actual, expected, atol=0.0 if abs_tol is None else abs_tol, rtol=0.0 if rel_tol is None else rel_tol, equal_nan=True)
    elif isinstance(actual, np.ndarray) or isinstance(expected, np.ndarray):
        np.testing.assert_equal(actual, expected)
    else:
        assert actual == expected


def quat_wxyz_to_xyzw(quaternion: Any) -> np.ndarray:
    q = np.asarray(quaternion, dtype=float).reshape(4)
    return np.array([q[1], q[2], q[3], q[0]], dtype=float)


def quat_xyzw_to_wxyz(quaternion: Any) -> np.ndarray:
    q = np.asarray(quaternion, dtype=float).reshape(4)
    return np.array([q[3], q[0], q[1], q[2]], dtype=float)