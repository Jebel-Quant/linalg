"""Tests for the power-iteration dominant-eigenpair helper."""

from __future__ import annotations

import numpy as np
import pytest

from cvx.linalg import power_iteration, rand_cov
from cvx.linalg.core.exceptions import NonSquareMatrixError, NotAMatrixError


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


# --- operator-aware power iteration -------------------------------------------


def test_power_iteration_on_symmetric_operator() -> None:
    """Runs matrix-free on a SymmetricOperator; matches the dense dominant eigenvalue."""
    from cvx.linalg import GramOperator

    rng = np.random.default_rng(0)
    m = rng.standard_normal((20, 6))
    lam, vec = power_iteration(GramOperator(m), seed=0)
    assert np.isclose(lam, np.linalg.eigvalsh(m.T @ m)[-1])
    assert vec.shape == (6,)


def test_power_iteration_on_sum_operator() -> None:
    """A composite (Gram + Factor) is handled matrix-free via its matvec."""
    from cvx.linalg import FactorOperator, GramOperator, SumOperator

    rng = np.random.default_rng(1)
    m = rng.standard_normal((25, 7))
    d = rng.uniform(0.5, 1.5, 7)
    u = rng.standard_normal((7, 2))
    delta = np.diag(rng.uniform(0.5, 2.0, 2))
    op = SumOperator([(0.7, GramOperator(m)), (0.3, FactorOperator(d, u, delta))])
    a = 0.7 * (m.T @ m) + 0.3 * (np.diag(d) + u @ delta @ u.T)
    lam, _ = power_iteration(op, seed=0)
    assert np.isclose(lam, np.linalg.eigvalsh(a)[-1])


def test_power_iteration_on_callable() -> None:
    """A bare matvec callable works when the dimension n is supplied."""
    rng = np.random.default_rng(2)
    a = rng.standard_normal((5, 5))
    a = a @ a.T + 5 * np.eye(5)
    lam, _ = power_iteration(lambda v: a @ v, n=5, seed=0)
    assert np.isclose(lam, np.linalg.eigvalsh(a)[-1])


def test_power_iteration_callable_requires_n() -> None:
    """A callable operator needs an explicit dimension."""
    with pytest.raises(ValueError, match="needs `n`"):
        power_iteration(lambda v: v, seed=0)
