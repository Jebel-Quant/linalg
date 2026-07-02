"""Tests for the raw singular value decomposition utility."""

from __future__ import annotations

import importlib

import numpy as np

from cvx.linalg import pca, svd


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


def test_pca_delegates_to_svd(monkeypatch) -> None:
    """Test that pca uses the standalone svd helper internally."""
    pca_module = importlib.import_module("cvx.linalg.covariance.pca")
    returns = np.array(
        [
            [0.10, 0.05, -0.02],
            [0.01, 0.00, 0.03],
            [-0.02, -0.01, 0.04],
            [0.03, 0.02, 0.00],
        ]
    )
    called: dict[str, np.ndarray] = {}

    def fake_svd(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Record the matrix passed in and delegate to the real numpy SVD."""
        called["matrix"] = matrix.copy()
        return np.linalg.svd(matrix, full_matrices=False)

    monkeypatch.setattr(pca_module, "svd", fake_svd)

    pca(returns, n_components=2)

    assert "matrix" in called
    assert np.allclose(called["matrix"], returns - returns.mean(axis=0))
