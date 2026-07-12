import numpy as np
from scipy.spatial.transform import Rotation

from . import assert_equal, quat_wxyz_to_xyzw, quat_xyzw_to_wxyz


def _normalise(q):
    q = np.asarray(q, dtype=float)
    return q / np.linalg.norm(q)


def _conjugate(q):
    q = np.asarray(q, dtype=float).copy()
    q[1:] *= -1
    return q


def _multiply(q, r):
    w1, x1, y1, z1 = q
    w2, x2, y2, z2 = r
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2, w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def _to_rotmat(q):
    return Rotation.from_quat(quat_wxyz_to_xyzw(_normalise(q))).as_matrix()


def _from_rotmat(matrix):
    return quat_xyzw_to_wxyz(Rotation.from_matrix(matrix).as_quat())


def _rotate(q, vector):
    return Rotation.from_quat(quat_wxyz_to_xyzw(_normalise(q))).apply(vector)


class TestQuaternion:
    def test_normalize_unit(self):
        assert_equal(np.linalg.norm(_normalise([1, 0, 0, 0])), 1, abs_tol=1e-12)

    def test_normalize_arbitrary(self):
        assert_equal(_normalise([2, 0, 0, 0]), [1, 0, 0, 0], abs_tol=1e-12)

    def test_normalize_general(self):
        q = _normalise([1, 1, 1, 1])
        assert_equal(np.linalg.norm(q), 1, abs_tol=1e-12)
        assert_equal(q, [0.5, 0.5, 0.5, 0.5], abs_tol=1e-12)

    def test_conjugate(self):
        assert_equal(_conjugate([1, 2, 3, 4]), [1, -2, -3, -4])

    def test_conjugate_identity(self):
        assert_equal(_conjugate([1, 0, 0, 0]), [1, 0, 0, 0])

    def test_multiply_identity(self):
        q = _normalise([1, 2, 3, 4])
        assert_equal(_multiply(q, [1, 0, 0, 0]), q, abs_tol=1e-12)

    def test_multiply_conjugate_gives_identity(self):
        q = _normalise([1, 2, 3, 4])
        assert_equal(_multiply(q, _conjugate(q)), [1, 0, 0, 0], abs_tol=1e-10)

    def test_multiply_associativity(self):
        q1, q2, q3 = _normalise([1, 1, 0, 0]), _normalise([1, 0, 1, 0]), _normalise([1, 0, 0, 1])
        assert_equal(_multiply(_multiply(q1, q2), q3), _multiply(q1, _multiply(q2, q3)), abs_tol=1e-10)

    def test_toRotMat_identity(self):
        assert_equal(_to_rotmat([1, 0, 0, 0]), np.eye(3), abs_tol=1e-12)

    def test_toRotMat_90deg_z(self):
        assert_equal(_to_rotmat([np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)]) @ [1, 0, 0], [0, 1, 0], abs_tol=1e-10)

    def test_toRotMat_180deg_x(self):
        assert_equal(_to_rotmat([0, 1, 0, 0]) @ [0, 1, 0], [0, -1, 0], abs_tol=1e-10)

    def test_toRotMat_orthogonal(self):
        matrix = _to_rotmat([1, 1, 1, 1])
        assert_equal(matrix @ matrix.T, np.eye(3), abs_tol=1e-10)
        assert_equal(np.linalg.det(matrix), 1, abs_tol=1e-10)

    def test_fromRotMat_identity(self):
        assert_equal(np.abs(_from_rotmat(np.eye(3))), [1, 0, 0, 0], abs_tol=1e-10)

    def test_fromRotMat_roundtrip(self):
        original = _normalise([1, 2, 3, 4])
        recovered = _from_rotmat(_to_rotmat(original))
        if np.dot(original, recovered) < 0:
            recovered = -recovered
        assert_equal(recovered, original, abs_tol=1e-10)

    def test_fromRotMat_90deg_x(self):
        matrix = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
        assert_equal(_to_rotmat(_from_rotmat(matrix)), matrix, abs_tol=1e-10)

    def test_rotate_identity(self):
        assert_equal(_rotate([1, 0, 0, 0], [1, 2, 3]), [1, 2, 3], abs_tol=1e-12)

    def test_rotate_90deg_z(self):
        assert_equal(_rotate([np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)], [1, 0, 0]), [0, 1, 0], abs_tol=1e-10)

    def test_rotate_matches_toRotMat(self):
        q, vector = _normalise([1, 2, 3, 4]), np.array([5, -3, 7])
        assert_equal(_rotate(q, vector), _to_rotmat(q) @ vector, abs_tol=1e-10)

    def test_rotate_preserves_norm(self):
        assert_equal(np.linalg.norm(_rotate(_normalise([3, 1, 4, 1]), [1, 2, 3])), np.linalg.norm([1, 2, 3]), abs_tol=1e-10)
