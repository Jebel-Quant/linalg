"""Tests for ``GramOperator``: the matrix-free A = M.T M backend and its ridge/regularised variants."""

from __future__ import annotations

import numpy as np
import pytest

from cvx.linalg import (
    DimensionMismatchError,
    GramOperator,
    NotAMatrixError,
)

from ._helpers import check_against_dense, rcond_reference

# --- products & solves --------------------------------------------------------


def test_gram_operator_matches_gram() -> None:
    """GramOperator reproduces A = M.T @ M without forming it."""
    rng = np.random.default_rng(1)
    m = rng.standard_normal((30, 8))
    a = m.T @ m
    check_against_dense(GramOperator(m), a, rng)


def test_gram_ridge_matches_dense() -> None:
    """GramOperator with a ridge reproduces A = M.T @ M + ridge * I (m >= n path)."""
    rng = np.random.default_rng(11)
    m = rng.standard_normal((30, 8))
    ridge = 0.7
    a = m.T @ m + ridge * np.eye(8)
    check_against_dense(GramOperator(m, ridge=ridge), a, rng)


def test_gram_ridge_tspace_woodbury_solve() -> None:
    """With m < len(free) the ridge solve uses row-space Woodbury and stays exact."""
    rng = np.random.default_rng(12)
    m = rng.standard_normal((4, 12))  # under-determined: M.T M has rank <= 4
    ridge = 0.5
    op = GramOperator(m, ridge=ridge)
    free = np.arange(12)  # len(free) = 12 > 4 rows -> Woodbury branch
    a = m.T @ m + ridge * np.eye(12)
    rhs = rng.standard_normal(12)
    assert np.allclose(a[np.ix_(free, free)] @ op.solve_free(free, rhs), rhs)
    rhs_mat = rng.standard_normal((12, 3))
    assert np.allclose(a[np.ix_(free, free)] @ op.solve_free(free, rhs_mat), rhs_mat)


def test_gram_ridge_woodbury_matches_normal_equations() -> None:
    """The row-space Woodbury branch agrees with the direct normal-equations solve."""
    rng = np.random.default_rng(13)
    m = rng.standard_normal((5, 9))
    ridge = 0.3
    free = np.array([0, 2, 3, 5, 7, 8])  # len 6 > m = 5 -> Woodbury branch
    rhs = rng.standard_normal(len(free))
    block = m[:, free].T @ m[:, free] + ridge * np.eye(len(free))
    assert np.allclose(GramOperator(m, ridge=ridge).solve_free(free, rhs), np.linalg.solve(block, rhs))


def test_gram_regularized_matches_convex_combination() -> None:
    """GramOperator.regularized realises (1 - alpha) M.T M + alpha R.T R."""
    rng = np.random.default_rng(3)
    m = rng.standard_normal((5, 8))  # under-determined: M.T M is singular
    alpha = 0.3
    root = np.eye(8)
    a = (1 - alpha) * (m.T @ m) + alpha * (root.T @ root)
    op = GramOperator.regularized(m, alpha, root)
    check_against_dense(op, a, rng)


def test_gram_regularized_restores_full_rank_free_solve() -> None:
    """The ridge makes free-block solves well posed even when M is rank-deficient."""
    rng = np.random.default_rng(4)
    m = rng.standard_normal((3, 10))  # rank <= 3
    op = GramOperator.regularized(m, 0.5, np.eye(10))
    free = np.arange(10)  # far larger than the data rank
    rhs = rng.standard_normal(10)
    a = 0.5 * (m.T @ m) + 0.5 * np.eye(10)
    assert np.allclose(a @ op.solve_free(free, rhs), rhs)


def test_gram_operator_n_property() -> None:
    """GramOperator reports the dimension it acts on (columns of M)."""
    assert GramOperator(np.ones((7, 4))).n == 4


# --- diag ---------------------------------------------------------------------


def test_gram_diag_matches_dense() -> None:
    """GramOperator.diag equals the diagonal of M.T M (+ ridge)."""
    rng = np.random.default_rng(32)
    m = rng.standard_normal((4, 6))
    ridge = 0.3
    assert np.allclose(GramOperator(m).diag, np.diag(m.T @ m))
    assert np.allclose(GramOperator(m, ridge=ridge).diag, np.diag(m.T @ m + ridge * np.eye(6)))


def test_gram_diag_free_slice_is_free_block_diagonal() -> None:
    """diag[free] is the diagonal of the free block (the Jacobi preconditioner for CG)."""
    rng = np.random.default_rng(31)
    m = rng.standard_normal((5, 8))
    op = GramOperator(m, ridge=0.2)
    free = np.array([1, 4, 6])
    a = m.T @ m + 0.2 * np.eye(8)
    assert np.allclose(op.diag[free], np.diag(a[np.ix_(free, free)]))


# --- rcond_free ---------------------------------------------------------------


def test_rcond_free_gram_matches_eigvals() -> None:
    """GramOperator.rcond_free equals the reciprocal condition of M_F.T M_F + ridge I."""
    rng = np.random.default_rng(21)
    m = rng.standard_normal((25, 8))
    for ridge in (0.0, 0.4):
        op = GramOperator(m, ridge=ridge)
        free = np.array([1, 2, 4, 7])
        block = m[:, free].T @ m[:, free] + ridge * np.eye(len(free))
        assert np.isclose(op.rcond_free(free), rcond_reference(block))


def test_rcond_free_gram_rank_deficient_drives_to_zero() -> None:
    """Without a ridge, an over-large free set is singular (rcond 0); a ridge lifts it."""
    rng = np.random.default_rng(22)
    m = rng.standard_normal((3, 10))  # rank <= 3
    free = np.arange(10)
    assert GramOperator(m).rcond_free(free) == 0.0
    assert GramOperator(m, ridge=0.5).rcond_free(free) > 0.0


def test_gram_rcond_free_zero_factor_is_zero() -> None:
    """An all-zero factor block has lambda_max = 0, so the un-ridged rcond is 0."""
    op = GramOperator(np.zeros((3, 4)))
    assert op.rcond_free(np.array([0, 1, 2, 3])) == 0.0


def test_rcond_free_gram_empty_is_one() -> None:
    """An empty free set is treated as perfectly conditioned."""
    assert GramOperator(np.ones((4, 3))).rcond_free(np.array([], dtype=int)) == 1.0


# --- input validation ---------------------------------------------------------


def test_gram_rejects_negative_ridge() -> None:
    """A negative ridge is rejected."""
    with pytest.raises(ValueError, match="ridge"):
        GramOperator(np.ones((4, 3)), ridge=-1.0)


def test_regularized_rejects_bad_alpha() -> None:
    """The regularisation intensity must lie in [0, 1]."""
    with pytest.raises(ValueError, match="alpha"):
        GramOperator.regularized(np.ones((4, 3)), 1.5, np.eye(3))


def test_regularized_rejects_mismatched_root() -> None:
    """The target root must have as many columns as the factor."""
    with pytest.raises(DimensionMismatchError):
        GramOperator.regularized(np.ones((4, 3)), 0.5, np.eye(2))


def test_gram_operator_rejects_non_matrix() -> None:
    """A non-2D factor is rejected."""
    with pytest.raises(NotAMatrixError):
        GramOperator(np.ones(4))


def test_regularized_rejects_non_matrix_factor() -> None:
    """A non-2D factor is rejected by the regularised constructor."""
    with pytest.raises(NotAMatrixError):
        GramOperator.regularized(np.ones(3), 0.5, np.eye(3))


def test_regularized_rejects_non_matrix_root() -> None:
    """A non-2D target root is rejected by the regularised constructor."""
    with pytest.raises(NotAMatrixError):
        GramOperator.regularized(np.ones((4, 3)), 0.5, np.ones(3))
