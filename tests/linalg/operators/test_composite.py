"""Tests for the SumOperator (weighted sum of symmetric operators)."""

from __future__ import annotations

import numpy as np
import pytest

from cvx.linalg import DenseOperator, DimensionMismatchError, FactorOperator, GramOperator, SumOperator


def test_sum_operator_matches_weighted_sum() -> None:
    """Products equal the weighted sum of the dense terms (matvec/block_matvec/apply_free)."""
    rng = np.random.default_rng(0)
    p = rng.standard_normal((6, 6))
    p = p @ p.T + 6 * np.eye(6)
    q = rng.standard_normal((6, 6))
    q = q @ q.T + 6 * np.eye(6)
    op = SumOperator([(0.25, DenseOperator(p)), (2.0, DenseOperator(q))])
    a = 0.25 * p + 2.0 * q

    x = rng.standard_normal(6)
    assert np.allclose(op.matvec(x), a @ x)

    free = np.array([0, 2, 5])
    bound = np.array([1, 3, 4])
    vf = rng.standard_normal(3)
    assert np.allclose(op.apply_free(free, vf), a[np.ix_(free, free)] @ vf)
    assert np.allclose(op.block_matvec(bound, free, vf), a[np.ix_(bound, free)] @ vf)


def test_sum_operator_gram_plus_factor_matrix_free() -> None:
    """A data term plus a structured target composes without forming an n x n matrix."""
    rng = np.random.default_rng(1)
    m = rng.standard_normal((30, 8))
    d = rng.uniform(0.5, 1.5, 8)
    u = rng.standard_normal((8, 2))
    delta = np.diag(rng.uniform(0.5, 2.0, 2))
    alpha = 0.3
    op = SumOperator([(1.0 - alpha, GramOperator(m)), (alpha, FactorOperator(d, u, delta))])
    a = (1.0 - alpha) * (m.T @ m) + alpha * (np.diag(d) + u @ delta @ u.T)
    x = rng.standard_normal(8)
    assert np.allclose(op.matvec(x), a @ x)
    assert op.n == 8


def test_sum_operator_rejects_empty() -> None:
    """At least one term is required."""
    with pytest.raises(ValueError, match="at least one term"):
        SumOperator([])


def test_sum_operator_rejects_mismatched_dims() -> None:
    """All terms must share a dimension."""
    with pytest.raises(DimensionMismatchError):
        SumOperator([(1.0, DenseOperator(np.eye(3))), (1.0, DenseOperator(np.eye(4)))])


def test_sum_operator_solve_free_not_implemented() -> None:
    """A sum has no direct free-block solve."""
    op = SumOperator([(1.0, DenseOperator(np.eye(3)))])
    with pytest.raises(NotImplementedError, match="Krylov"):
        op.solve_free(np.array([0, 1]), np.ones(2))


def test_sum_operator_rcond_free_not_implemented() -> None:
    """A sum has no structural conditioning estimate."""
    op = SumOperator([(1.0, DenseOperator(np.eye(3)))])
    with pytest.raises(NotImplementedError):
        op.rcond_free(np.array([0, 1]))


def test_sum_operator_diag_is_weighted_sum() -> None:
    """Diag is the weighted sum of the terms' diagonals, matrix-free terms included."""
    rng = np.random.default_rng(2)
    m = rng.standard_normal((20, 7))
    d = rng.uniform(0.5, 1.5, 7)
    u = rng.standard_normal((7, 2))
    delta = np.diag(rng.uniform(0.5, 2.0, 2))
    alpha = 0.4
    op = SumOperator([(1.0 - alpha, GramOperator(m)), (alpha, FactorOperator(d, u, delta))])
    a = (1.0 - alpha) * (m.T @ m) + alpha * (np.diag(d) + u @ delta @ u.T)
    assert np.allclose(op.diag, np.diag(a))


def test_sum_operator_diag_propagates_missing_term_diag() -> None:
    """A term without a diagonal makes the sum's diag raise as well."""

    class _NoDiag(DenseOperator):
        """Dense backend with the diagonal override removed again."""

        @property
        def diag(self) -> np.ndarray:
            """Unavailable, as on the base class."""
            raise NotImplementedError("this operator does not expose its diagonal")

    op = SumOperator([(1.0, DenseOperator(np.eye(3))), (1.0, _NoDiag(np.eye(3)))])
    with pytest.raises(NotImplementedError, match="diagonal"):
        _ = op.diag
