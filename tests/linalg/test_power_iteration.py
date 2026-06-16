"""Tests for the power-iteration dominant-eigenpair helper."""

from __future__ import annotations

import numpy as np
import pytest

from cvx.linalg import power_iteration, rand_cov
from cvx.linalg.exceptions import NonSquareMatrixError, NotAMatrixError


def test_power_iteration_diagonal() -> None:
    """Dominant eigenpair of a diagonal matrix is the largest entry."""
    matrix = np.diag([3.0, 2.0, 1.0])
    eigenvalue, eigenvector = power_iteration(matrix, seed=0)
    assert np.isclose(eigenvalue, 3.0)
    assert np.isclose(abs(eigenvector[0]), 1.0)
    assert np.isclose(np.linalg.norm(eigenvector), 1.0)


def test_power_iteration_matches_eigvalsh() -> None:
    """Estimate agrees with the largest-magnitude symmetric eigenvalue."""
    cov = rand_cov(10, seed=42)
    eigenvalue, eigenvector = power_iteration(cov, seed=0)
    eigenvalues = np.linalg.eigvalsh(cov)
    assert np.isclose(eigenvalue, eigenvalues[-1])
    # The iterate is an eigenvector: A v is parallel to v (relative residual).
    residual = cov @ eigenvector - eigenvalue * eigenvector
    assert np.linalg.norm(residual) / abs(eigenvalue) < 1e-3


def test_power_iteration_negative_dominant_eigenvalue() -> None:
    """The signed eigenvalue is recovered when the dominant one is negative."""
    matrix = np.diag([-5.0, 1.0, 2.0])
    eigenvalue, eigenvector = power_iteration(matrix, seed=1)
    assert np.isclose(eigenvalue, -5.0)
    assert np.isclose(abs(eigenvector[0]), 1.0)


def test_power_iteration_zero_matrix() -> None:
    """A zero matrix annihilates the iterate; eigenvalue is zero."""
    eigenvalue, eigenvector = power_iteration(np.zeros((4, 4)), seed=0)
    assert eigenvalue == 0.0
    assert np.isclose(np.linalg.norm(eigenvector), 1.0)


def test_power_iteration_reproducible() -> None:
    """The same seed yields the same result."""
    cov = rand_cov(6, seed=7)
    a = power_iteration(cov, seed=3)
    b = power_iteration(cov, seed=3)
    assert a[0] == b[0]
    assert np.array_equal(a[1], b[1])


def test_power_iteration_rejects_non_square() -> None:
    """A non-square matrix raises NonSquareMatrixError."""
    with pytest.raises(NonSquareMatrixError):
        power_iteration(np.ones((3, 2)))


def test_power_iteration_rejects_non_matrix() -> None:
    """A non-2-D input raises NotAMatrixError."""
    with pytest.raises(NotAMatrixError):
        power_iteration(np.ones((3, 3, 3)))


def test_power_iteration_respects_iteration_cap() -> None:
    """A tight spectral gap still terminates at the iteration cap."""
    # Two nearly-equal dominant eigenvalues: convergence is slow, so the
    # n_iter cap (not the tolerance) ends the loop.
    matrix = np.diag([2.0, 1.999, 0.5])
    eigenvalue, _ = power_iteration(matrix, n_iter=2, tol=0.0, seed=0)
    assert 1.9 < eigenvalue <= 2.0
