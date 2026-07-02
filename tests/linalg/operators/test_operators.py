"""Tests for the symmetric operator abstraction and its backends."""

from __future__ import annotations

import numpy as np
import pytest

from cvx.linalg import (
    DenseOperator,
    DimensionMismatchError,
    FactorOperator,
    GramOperator,
    IncrementalDenseOperator,
    NonSquareMatrixError,
    NotAMatrixError,
    SymmetricOperator,
)


def _rcond_reference(block: np.ndarray) -> float:
    """Reference reciprocal 2-norm condition number from the symmetric eigenvalues."""
    if block.shape[0] == 0:
        return 1.0
    eig = np.linalg.eigvalsh(block)
    lam_max = float(eig[-1])
    if lam_max <= 0.0:
        return 0.0
    return max(float(eig[0]), 0.0) / lam_max


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


def test_gram_ridge_matches_dense() -> None:
    """GramOperator with a ridge reproduces A = M.T @ M + ridge * I (m >= n path)."""
    rng = np.random.default_rng(11)
    m = rng.standard_normal((30, 8))
    ridge = 0.7
    a = m.T @ m + ridge * np.eye(8)
    _check_against_dense(GramOperator(m, ridge=ridge), a, rng)


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


def test_gram_rejects_negative_ridge() -> None:
    """A negative ridge is rejected."""
    with pytest.raises(ValueError, match="ridge"):
        GramOperator(np.ones((4, 3)), ridge=-1.0)


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


def test_factor_operator_k_property() -> None:
    """FactorOperator reports its number of factors (columns of U)."""
    assert FactorOperator(np.ones(5), np.ones((5, 3)), np.eye(3)).k == 3


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


# --- rcond_free ---------------------------------------------------------------


def test_rcond_free_dense_matches_eigvals() -> None:
    """DenseOperator.rcond_free equals the block's reciprocal condition number."""
    rng = np.random.default_rng(20)
    b = rng.standard_normal((7, 7))
    a = b @ b.T + np.eye(7)
    op = DenseOperator(a)
    free = np.array([0, 2, 5, 6])
    assert np.isclose(op.rcond_free(free), _rcond_reference(a[np.ix_(free, free)]))


def test_rcond_free_gram_matches_eigvals() -> None:
    """GramOperator.rcond_free equals the reciprocal condition of M_F.T M_F + ridge I."""
    rng = np.random.default_rng(21)
    m = rng.standard_normal((25, 8))
    for ridge in (0.0, 0.4):
        op = GramOperator(m, ridge=ridge)
        free = np.array([1, 2, 4, 7])
        block = m[:, free].T @ m[:, free] + ridge * np.eye(len(free))
        assert np.isclose(op.rcond_free(free), _rcond_reference(block))


def test_rcond_free_gram_rank_deficient_drives_to_zero() -> None:
    """Without a ridge, an over-large free set is singular (rcond 0); a ridge lifts it."""
    rng = np.random.default_rng(22)
    m = rng.standard_normal((3, 10))  # rank <= 3
    free = np.arange(10)
    assert GramOperator(m).rcond_free(free) == 0.0
    assert GramOperator(m, ridge=0.5).rcond_free(free) > 0.0


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
    true = _rcond_reference(a[np.ix_(free, free)])
    assert 0.0 < bound <= true + 1e-12


def test_rcond_free_empty_is_one() -> None:
    """An empty free set is treated as perfectly conditioned by every backend."""
    empty = np.array([], dtype=int)
    assert DenseOperator(np.eye(3)).rcond_free(empty) == 1.0
    assert GramOperator(np.ones((4, 3))).rcond_free(empty) == 1.0
    assert FactorOperator(np.ones(3), np.ones((3, 2)), np.eye(2)).rcond_free(empty) == 1.0


# --- IncrementalDenseOperator -------------------------------------------------


def test_incremental_matches_dense_on_flip_sequence() -> None:
    """The maintained inverse tracks fresh solves across single-index insert/delete flips."""
    rng = np.random.default_rng(24)
    b = rng.standard_normal((10, 10))
    a = b @ b.T + 10 * np.eye(10)
    inc = IncrementalDenseOperator(a)
    ref = DenseOperator(a)
    free_sets = [
        [0, 1, 2, 3],
        [0, 1, 2, 3, 7],  # insert 7
        [0, 2, 3, 7],  # delete 1
        [0, 2, 3, 5, 7],  # insert 5
        [0, 2, 3, 5, 7, 9],  # insert 9
        [0, 2, 3, 5, 9],  # delete 7
    ]
    for fs in free_sets:
        free = np.array(fs)
        rhs = rng.standard_normal(len(fs))
        assert np.allclose(inc.solve_free(free, rhs), ref.solve_free(free, rhs))


def test_incremental_handles_multi_index_change() -> None:
    """A change of more than one index falls back to a fresh inverse and stays correct."""
    rng = np.random.default_rng(25)
    b = rng.standard_normal((8, 8))
    a = b @ b.T + 8 * np.eye(8)
    inc = IncrementalDenseOperator(a)
    ref = DenseOperator(a)
    for fs in ([0, 1, 2], [3, 4, 5, 6]):  # disjoint jump -> refactor path
        free = np.array(fs)
        rhs = rng.standard_normal(len(fs))
        assert np.allclose(inc.solve_free(free, rhs), ref.solve_free(free, rhs))


def test_incremental_matrix_rhs_and_delegated_products() -> None:
    """Matrix RHS solves, and matvec/block_matvec/rcond_free match the plain dense backend."""
    rng = np.random.default_rng(26)
    b = rng.standard_normal((6, 6))
    a = b @ b.T + 6 * np.eye(6)
    inc = IncrementalDenseOperator(a)
    ref = DenseOperator(a)
    free = np.array([0, 2, 4])
    rhs_mat = rng.standard_normal((3, 2))
    assert np.allclose(inc.solve_free(free, rhs_mat), ref.solve_free(free, rhs_mat))
    x = rng.standard_normal(6)
    assert np.allclose(inc.matvec(x), ref.matvec(x))
    v = rng.standard_normal(3)
    assert np.allclose(inc.block_matvec(np.array([1, 3]), free, v), ref.block_matvec(np.array([1, 3]), free, v))
    assert np.isclose(inc.rcond_free(free), ref.rcond_free(free))


def test_incremental_empty_free_set() -> None:
    """An empty free set returns an empty solution without error."""
    inc = IncrementalDenseOperator(np.eye(4))
    out = inc.solve_free(np.array([], dtype=int), np.array([]))
    assert out.shape == (0,)
