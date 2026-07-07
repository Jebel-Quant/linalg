"""Tests for the raw singular value decomposition utility."""

from __future__ import annotations

import numpy as np

from cvx.linalg import svd


def test_svd_matches_numpy() -> None:
    """Test that svd returns the same compact decomposition as NumPy."""
    matrix = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 10.0],
            [11.0, 12.0, 13.0],
        ]
    )

    u, s, vt = svd(matrix)
    u_np, s_np, vt_np = np.linalg.svd(matrix, full_matrices=False)

    assert u.shape == (4, 3)
    assert s.shape == (3,)
    assert vt.shape == (3, 3)
    assert np.allclose(s, s_np)
    assert np.allclose(np.abs(u), np.abs(u_np))
    assert np.allclose(np.abs(vt), np.abs(vt_np))
