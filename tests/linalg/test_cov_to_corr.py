"""Tests for cov_to_corr."""

from __future__ import annotations

import numpy as np
import pytest

from cvx.linalg import cov_to_corr


def test_diagonal_is_one() -> None:
    """Diagonal entries of the correlation matrix are 1.0."""
    cov = np.array([[4.0, 2.0], [2.0, 9.0]])
    corr = cov_to_corr(cov)
    assert np.allclose(np.diag(corr), 1.0)


def test_off_diagonal_value() -> None:
    """Off-diagonal entry matches the analytical correlation."""
    cov = np.array([[4.0, 2.0], [2.0, 9.0]])
    corr = cov_to_corr(cov)
    expected = 2.0 / (2.0 * 3.0)
    assert pytest.approx(corr[0, 1]) == expected
    assert pytest.approx(corr[1, 0]) == expected


def test_symmetry() -> None:
    """Output matrix is symmetric."""
    rng = np.random.default_rng(0)
    a = rng.standard_normal((5, 5))
    cov = a @ a.T + np.eye(5)
    corr = cov_to_corr(cov)
    assert np.allclose(corr, corr.T)


def test_values_clipped_to_minus_one_one() -> None:
    """All finite entries lie in [-1, 1]."""
    cov = np.array([[1.0, 1.1], [1.1, 1.0]])
    corr = cov_to_corr(cov)
    assert np.all(corr[np.isfinite(corr)] <= 1.0)
    assert np.all(corr[np.isfinite(corr)] >= -1.0)


def test_zero_variance_gives_nan() -> None:
    """Zero-variance rows/columns produce nan entries."""
    cov = np.array([[0.0, 0.0], [0.0, 4.0]])
    corr = cov_to_corr(cov)
    assert np.isnan(corr[0, 0])
    assert np.isnan(corr[0, 1])
    assert np.isnan(corr[1, 0])
    assert corr[1, 1] == pytest.approx(1.0)


def test_identity_cov_gives_identity_corr() -> None:
    """Identity covariance maps to identity correlation."""
    corr = cov_to_corr(np.eye(4))
    assert np.allclose(corr, np.eye(4))
