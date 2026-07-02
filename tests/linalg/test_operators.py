"""Tests for the symmetric operator abstraction and its backends."""

from __future__ import annotations

import numpy as np
import pytest

from cvx.linalg import (
    DenseOperator,
    DimensionMismatchError,
    FactorOperator,
    GramOperator,
    NonSquareMatrixError,
    NotAMatrixError,
    SymmetricOperator,
)


def _check_against_dense(op: SymmetricOperator, a: np.ndarray, rng: np.random.Generator) -> None:
    """Every backend must reproduce the products of the dense matrix ``a``."""
    n = a.shape[0]
    x = rng.standard_normal(n)
    assert np.allclose(op.matvec(x), a @ x)

    perm = rng.permutation(n)
    free = np.sort(perm[: n // 2])
    bound = np.setdiff1d(np.arange(n), free)

    # Free-block forward product (drives a matrix-free CG solve).
    vf = rng.standard_normal(len(free))
    assert np.allclose(op.apply_free(free, vf), a[np.ix_(free, free)] @ vf)

    # Cross product on disjoint sets (the reduced gradient at bound indices).
    assert np.allclose(op.block_matvec(bound, free, vf), a[np.ix_(bound, free)] @ vf)

    # Direct free-block solve, vector and matrix right-hand sides.
    rhs = rng.standard_normal(len(free))
    x_free = op.solve_free(free, rhs)
    assert np.allclose(a[np.ix_(free, free)] @ x_free, rhs)

    rhs_mat = rng.standard_normal((len(free), 3))
    x_mat = op.solve_free(free, rhs_mat)
    assert np.allclose(a[np.ix_(free, free)] @ x_mat, rhs_mat)


def test_dense_operator_matches_matrix() -> None:
    """DenseOperator reproduces the products of its backing matrix."""
    rng = np.random.default_rng(0)
    b = rng.standard_normal((8, 8))
    a = b @ b.T + 8 * np.eye(8)  # SPD
    _check_against_dense(DenseOperator(a), a, rng)


def test_gram_operator_matches_gram() -> None:
    """GramOperator reproduces A = M.T @ M without forming it."""
    rng = np.random.default_rng(1)
    m = rng.standard_normal((30, 8))
    a = m.T @ m
    _check_against_dense(GramOperator(m), a, rng)


def test_factor_operator_matches_dense() -> None:
    """FactorOperator reproduces diag(d) + U Delta U.T, Woodbury solve included."""
    rng = np.random.default_rng(2)
    n, r = 9, 3
    d = rng.uniform(1.0, 2.0, size=n)
    u = rng.standard_normal((n, r))
    c = rng.standard_normal((r, r))
    delta = c @ c.T + np.eye(r)  # SPD inner block
    a = np.diag(d) + u @ delta @ u.T
    _check_against_dense(FactorOperator(d, u, delta), a, rng)


def test_gram_regularized_matches_convex_combination() -> None:
    """GramOperator.regularized realises (1 - alpha) M.T M + alpha R.T R."""
    rng = np.random.default_rng(3)
    m = rng.standard_normal((5, 8))  # under-determined: M.T M is singular
    alpha = 0.3
    root = np.eye(8)
    a = (1 - alpha) * (m.T @ m) + alpha * (root.T @ root)
    op = GramOperator.regularized(m, alpha, root)
    _check_against_dense(op, a, rng)


def test_gram_regularized_restores_full_rank_free_solve() -> None:
    """The ridge makes free-block solves well posed even when M is rank-deficient."""
    rng = np.random.default_rng(4)
    m = rng.standard_normal((3, 10))  # rank <= 3
    op = GramOperator.regularized(m, 0.5, np.eye(10))
    free = np.arange(10)  # far larger than the data rank
    rhs = rng.standard_normal(10)
    a = 0.5 * (m.T @ m) + 0.5 * np.eye(10)
    assert np.allclose(a @ op.solve_free(free, rhs), rhs)


def test_apply_free_is_block_matvec_diagonal() -> None:
    """apply_free equals block_matvec with equal row and column sets."""
    rng = np.random.default_rng(5)
    a = np.diag(rng.uniform(1, 2, 6)) + 0.1 * np.ones((6, 6))
    op = DenseOperator(a)
    free = np.array([1, 3, 5])
    v = rng.standard_normal(3)
    assert np.allclose(op.apply_free(free, v), op.block_matvec(free, free, v))


def test_n_property() -> None:
    """Each backend reports the dimension it acts on."""
    assert DenseOperator(np.eye(4)).n == 4
    assert GramOperator(np.ones((7, 4))).n == 4
    assert FactorOperator(np.ones(5), np.ones((5, 2)), np.eye(2)).n == 5


def test_symmetric_operator_is_abstract() -> None:
    """The protocol base class cannot be instantiated directly."""
    with pytest.raises(TypeError):
        SymmetricOperator()  # type: ignore[abstract]


def test_dense_operator_rejects_non_square() -> None:
    """A non-square backing matrix is rejected."""
    with pytest.raises(NonSquareMatrixError):
        DenseOperator(np.ones((3, 4)))


def test_factor_operator_rejects_mismatched_shapes() -> None:
    """Loadings and inner block must have matching factor rank."""
    with pytest.raises(DimensionMismatchError):
        FactorOperator(np.ones(5), np.ones((5, 2)), np.eye(3))


def test_regularized_rejects_bad_alpha() -> None:
    """The regularisation intensity must lie in [0, 1]."""
    with pytest.raises(ValueError, match="alpha"):
        GramOperator.regularized(np.ones((4, 3)), 1.5, np.eye(3))


def test_regularized_rejects_mismatched_root() -> None:
    """The target root must have as many columns as the factor."""
    with pytest.raises(DimensionMismatchError):
        GramOperator.regularized(np.ones((4, 3)), 0.5, np.eye(2))


def test_block_matvec_rejects_duplicate_indices() -> None:
    """An index set with repeated positions is rejected."""
    op = DenseOperator(np.eye(4))
    with pytest.raises(ValueError, match="duplicate"):
        op.block_matvec(np.array([0, 2, 2]), np.array([1, 3]), np.ones(2))


def test_solve_free_rejects_duplicate_indices() -> None:
    """The free set passed to a solve must not contain duplicates."""
    op = DenseOperator(np.eye(4))
    with pytest.raises(ValueError, match="duplicate"):
        op.solve_free(np.array([1, 1]), np.ones(2))


def test_index_rejects_non_integer_dtype() -> None:
    """A non-integer index set is rejected."""
    op = DenseOperator(np.eye(4))
    with pytest.raises(ValueError, match="integer positions"):
        op.apply_free(np.array([0.0, 1.0]), np.ones(2))


def test_index_rejects_boolean_dtype() -> None:
    """A boolean mask is not accepted as an integer index set."""
    op = DenseOperator(np.eye(4))
    with pytest.raises(ValueError, match="integer positions"):
        op.apply_free(np.array([True, False, True, False]), np.ones(2))


def test_index_rejects_non_1d() -> None:
    """An index set must be one-dimensional."""
    op = DenseOperator(np.eye(4))
    with pytest.raises(ValueError, match="1-D"):
        op.apply_free(np.array([[0, 1], [2, 3]]), np.ones(2))


def test_factor_operator_rejects_non_positive_diagonal() -> None:
    """FactorOperator requires a strictly positive diagonal."""
    with pytest.raises(ValueError, match="strictly positive"):
        FactorOperator(np.array([1.0, 0.0, 2.0]), np.ones((3, 2)), np.eye(2))

    with pytest.raises(ValueError, match="strictly positive"):
        FactorOperator(np.array([1.0, -1.0, 2.0]), np.ones((3, 2)), np.eye(2))


def test_dense_operator_rejects_non_matrix() -> None:
    """A non-2D backing array is rejected."""
    with pytest.raises(NotAMatrixError):
        DenseOperator(np.ones(4))


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
