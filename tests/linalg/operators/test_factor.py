"""Tests for ``FactorOperator``: the diag(d) + U Delta U.T low-rank backend."""

from __future__ import annotations

import numpy as np
import pytest

from cvx.linalg import (
    DimensionMismatchError,
    FactorOperator,
    NonSquareMatrixError,
    NotAMatrixError,
)

from ._helpers import check_against_dense, rcond_reference

# --- products & solves --------------------------------------------------------


def test_factor_operator_matches_dense() -> None:
    """FactorOperator reproduces diag(d) + U Delta U.T, Woodbury solve included."""
    rng = np.random.default_rng(2)
    n, r = 9, 3
    d = rng.uniform(1.0, 2.0, size=n)
    u = rng.standard_normal((n, r))
    c = rng.standard_normal((r, r))
    delta = c @ c.T + np.eye(r)  # SPD inner block
    a = np.diag(d) + u @ delta @ u.T
    check_against_dense(FactorOperator(d, u, delta), a, rng)


def test_factor_operator_inverts_inner_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """Delta^{-1} is computed on the first solve and reused by later ones."""
    rng = np.random.default_rng(11)
    d = rng.uniform(1.0, 2.0, 6)
    u = rng.standard_normal((6, 2))
    delta = np.array([[2.0, 0.3], [0.3, 1.0]])
    op = FactorOperator(d, u, delta)
    dense = np.diag(d) + u @ delta @ u.T

    calls = 0
    real_solve = np.linalg.solve

    def counting_solve(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Count solves against Delta, then delegate to NumPy."""
        nonlocal calls
        if a is delta:
            calls += 1
        return real_solve(a, b)

    monkeypatch.setattr(np.linalg, "solve", counting_solve)
    for free in (np.array([0, 1, 2]), np.array([1, 3, 4, 5])):
        rhs = rng.standard_normal(free.size)
        np.testing.assert_allclose(dense[np.ix_(free, free)] @ op.solve_free(free, rhs), rhs, atol=1e-10)
    assert calls == 1


def test_factor_operator_n_property() -> None:
    """FactorOperator reports the dimension it acts on."""
    assert FactorOperator(np.ones(5), np.ones((5, 2)), np.eye(2)).n == 5


def test_factor_operator_k_property() -> None:
    """FactorOperator reports its number of factors (columns of U)."""
    assert FactorOperator(np.ones(5), np.ones((5, 3)), np.eye(3)).k == 3


def test_factor_diag_matches_dense() -> None:
    """FactorOperator.diag equals the diagonal of diag(d) + U Delta U.T."""
    rng = np.random.default_rng(33)
    d = rng.uniform(1.0, 2.0, 6)
    u = rng.standard_normal((6, 2))
    c = rng.standard_normal((2, 2))
    delta = c @ c.T + np.eye(2)
    assert np.allclose(FactorOperator(d, u, delta).diag, np.diag(np.diag(d) + u @ delta @ u.T))


# --- rcond_free ---------------------------------------------------------------


def test_rcond_free_factor_is_lower_bound() -> None:
    """FactorOperator.rcond_free is a valid lower bound on the true reciprocal condition."""
    rng = np.random.default_rng(23)
    n, r = 9, 3
    d = rng.uniform(1.0, 2.0, size=n)
    u = rng.standard_normal((n, r))
    c = rng.standard_normal((r, r))
    delta = c @ c.T + np.eye(r)
    a = np.diag(d) + u @ delta @ u.T
    op = FactorOperator(d, u, delta)
    free = np.array([0, 3, 4, 8])
    bound = op.rcond_free(free)
    true = rcond_reference(a[np.ix_(free, free)])
    assert 0.0 < bound <= true + 1e-12


def test_rcond_free_factor_empty_is_one() -> None:
    """An empty free set is treated as perfectly conditioned."""
    assert FactorOperator(np.ones(3), np.ones((3, 2)), np.eye(2)).rcond_free(np.array([], dtype=int)) == 1.0


# --- input validation ---------------------------------------------------------


def test_factor_operator_rejects_mismatched_shapes() -> None:
    """Loadings and inner block must have matching factor rank."""
    with pytest.raises(DimensionMismatchError):
        FactorOperator(np.ones(5), np.ones((5, 2)), np.eye(3))


def test_factor_operator_rejects_non_positive_diagonal() -> None:
    """FactorOperator requires a strictly positive diagonal."""
    with pytest.raises(ValueError, match="strictly positive"):
        FactorOperator(np.array([1.0, 0.0, 2.0]), np.ones((3, 2)), np.eye(2))

    with pytest.raises(ValueError, match="strictly positive"):
        FactorOperator(np.array([1.0, -1.0, 2.0]), np.ones((3, 2)), np.eye(2))


def test_factor_operator_rejects_non_1d_diagonal() -> None:
    """The diagonal must be one-dimensional."""
    with pytest.raises(ValueError, match="1-D"):
        FactorOperator(np.eye(3), np.ones((3, 2)), np.eye(2))


def test_factor_operator_rejects_non_matrix_loadings() -> None:
    """The loadings must be a 2-D matrix."""
    with pytest.raises(NotAMatrixError):
        FactorOperator(np.ones(3), np.ones(3), np.eye(1))


def test_factor_operator_rejects_loadings_length_mismatch() -> None:
    """Loadings must have one row per diagonal entry."""
    with pytest.raises(DimensionMismatchError):
        FactorOperator(np.ones(3), np.ones((4, 2)), np.eye(2))


def test_factor_operator_rejects_non_matrix_inner() -> None:
    """The inner block must be a 2-D matrix."""
    with pytest.raises(NotAMatrixError):
        FactorOperator(np.ones(3), np.ones((3, 2)), np.ones(2))


def test_factor_operator_rejects_non_square_inner() -> None:
    """The inner block must be square."""
    with pytest.raises(NonSquareMatrixError):
        FactorOperator(np.ones(3), np.ones((3, 2)), np.ones((2, 3)))
