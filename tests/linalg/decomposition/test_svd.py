"""Tests for the singular value decomposition utilities ``svd`` and ``svd_k``."""

from __future__ import annotations

import numpy as np
import pytest

from cvx.linalg import svd, svd_k
from cvx.linalg.core.exceptions import InvalidComponentsError


def _matrix() -> np.ndarray:
    """Return a fixed 4x3 test matrix."""
    return np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 10.0],
            [11.0, 12.0, 13.0],
        ]
    )


# --- svd ----------------------------------------------------------------------


def test_svd_matches_numpy() -> None:
    """Test that svd returns the same compact decomposition as NumPy."""
    matrix = _matrix()

    u, s, vt = svd(matrix)
    u_np, s_np, vt_np = np.linalg.svd(matrix, full_matrices=False)

    assert u.shape == (4, 3)
    assert s.shape == (3,)
    assert vt.shape == (3, 3)
    assert np.allclose(s, s_np)
    assert np.allclose(np.abs(u), np.abs(u_np))
    assert np.allclose(np.abs(vt), np.abs(vt_np))


# --- svd_k --------------------------------------------------------------------


@pytest.mark.parametrize("k", [1, 2, 3])
def test_svd_k_matches_leading_components(k: int) -> None:
    """svd_k returns exactly the leading triplets of the full compact SVD."""
    matrix = _matrix()
    u, s, vt = svd_k(matrix, k)
    u_full, s_full, vt_full = np.linalg.svd(matrix, full_matrices=False)

    assert u.shape == (4, k)
    assert s.shape == (k,)
    assert vt.shape == (k, 3)
    assert np.allclose(s, s_full[:k])
    assert np.allclose(np.abs(u), np.abs(u_full[:, :k]))
    assert np.allclose(np.abs(vt), np.abs(vt_full[:k, :]))


def test_svd_k_full_rank_reconstructs() -> None:
    """At k = min(shape) the truncation reconstructs the matrix exactly."""
    matrix = _matrix()
    u, s, vt = svd_k(matrix, k=3)
    assert np.allclose(u @ np.diag(s) @ vt, matrix)


def test_svd_k_is_best_rank_k_approximation() -> None:
    """The rank-k truncation matches the Eckart-Young Frobenius error."""
    matrix = _matrix()
    _, s_full, _ = svd(matrix)
    for k in (1, 2):
        u, s, vt = svd_k(matrix, k)
        approx = u @ np.diag(s) @ vt
        # Frobenius error equals the norm of the discarded singular values.
        expected = np.sqrt(np.sum(s_full[k:] ** 2))
        assert np.isclose(np.linalg.norm(matrix - approx, "fro"), expected)


def test_svd_k_descending_singular_values() -> None:
    """Returned singular values are in descending order."""
    _, s, _ = svd_k(_matrix(), k=3)
    assert np.all(np.diff(s) <= 0)


@pytest.mark.parametrize("k", [0, -1, 4, 100])
def test_svd_k_rejects_out_of_range_k(k: int) -> None:
    """Reject k outside 1..min(shape) with InvalidComponentsError."""
    with pytest.raises(InvalidComponentsError):
        svd_k(_matrix(), k)
