"""Tests for ``DenseOperator`` and the incremental ``IncrementalDenseOperator``."""

from __future__ import annotations

import numpy as np
import pytest

import cvx.linalg.operators.dense as dense_module
from cvx.linalg import (
    DenseOperator,
    DimensionMismatchError,
    IncrementalDenseOperator,
    NonSquareMatrixError,
    NotAMatrixError,
)

from ._helpers import check_against_dense, rcond_reference

# --- DenseOperator ------------------------------------------------------------


def test_dense_operator_matches_matrix() -> None:
    """DenseOperator reproduces the products of its backing matrix."""
    rng = np.random.default_rng(0)
    b = rng.standard_normal((8, 8))
    a = b @ b.T + 8 * np.eye(8)  # SPD
    check_against_dense(DenseOperator(a), a, rng)


def test_dense_operator_n_property() -> None:
    """DenseOperator reports the dimension it acts on."""
    assert DenseOperator(np.eye(4)).n == 4


def test_dense_operator_rejects_non_square() -> None:
    """A non-square backing matrix is rejected."""
    with pytest.raises(NonSquareMatrixError):
        DenseOperator(np.ones((3, 4)))


def test_dense_operator_rejects_non_matrix() -> None:
    """A non-2D backing array is rejected."""
    with pytest.raises(NotAMatrixError):
        DenseOperator(np.ones(4))


def test_dense_diag_matches_dense() -> None:
    """The dense/incremental backends' diag equals the matrix diagonal."""
    rng = np.random.default_rng(30)
    b = rng.standard_normal((6, 6))
    a = b @ b.T + 6 * np.eye(6)
    assert np.allclose(DenseOperator(a).diag, np.diag(a))
    assert np.allclose(IncrementalDenseOperator(a).diag, np.diag(a))


def test_rcond_free_dense_matches_eigvals() -> None:
    """DenseOperator.rcond_free equals the block's reciprocal condition number."""
    rng = np.random.default_rng(20)
    b = rng.standard_normal((7, 7))
    a = b @ b.T + np.eye(7)
    op = DenseOperator(a)
    free = np.array([0, 2, 5, 6])
    assert np.isclose(op.rcond_free(free), rcond_reference(a[np.ix_(free, free)]))


def test_rcond_free_dense_empty_is_one() -> None:
    """An empty free set is treated as perfectly conditioned."""
    assert DenseOperator(np.eye(3)).rcond_free(np.array([], dtype=int)) == 1.0


def test_dense_rcond_free_non_positive_block_is_zero() -> None:
    """A free block with a non-positive top eigenvalue is reported as singular (rcond 0)."""
    # A negative-definite block: lambda_max <= 0, so rcond_free short-circuits to 0.0.
    op = DenseOperator(-np.eye(3))
    assert op.rcond_free(np.array([0, 1])) == 0.0


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


def test_incremental_insert_bad_pivot_falls_back() -> None:
    """A single-index insert with a non-positive Schur pivot recomputes and stays exact."""
    # diag([1, -1, 2]) is symmetric but indefinite: every principal block is invertible,
    # yet inserting index 1 into free {0} yields a Schur complement of -1 (<= 0), which
    # forces the bordered update to bail out to a fresh refactorisation.
    a = np.diag([1.0, -1.0, 2.0])
    inc = IncrementalDenseOperator(a)
    ref = DenseOperator(a)
    rng = np.random.default_rng(27)
    inc.solve_free(np.array([0]), rng.standard_normal(1))  # seed the cache with free {0}
    free = np.array([0, 1])  # insert 1 -> Schur = -1 -> bad pivot -> refactor
    rhs = rng.standard_normal(2)
    assert np.allclose(inc.solve_free(free, rhs), ref.solve_free(free, rhs))


def test_incremental_delete_bad_pivot_falls_back() -> None:
    """A single-index delete with a non-positive inverse pivot recomputes and stays exact."""
    # From free {0, 1} the maintained inverse is diag([1, -1]); removing index 1 reads a
    # pivot of -1 (<= 0), so the deletion update bails out to a fresh refactorisation.
    a = np.diag([1.0, -1.0, 2.0])
    inc = IncrementalDenseOperator(a)
    ref = DenseOperator(a)
    rng = np.random.default_rng(28)
    inc.solve_free(np.array([0, 1]), rng.standard_normal(2))  # cache inverse of diag([1, -1])
    free = np.array([0])  # delete 1 -> pivot -1 -> bad pivot -> refactor
    rhs = rng.standard_normal(1)
    assert np.allclose(inc.solve_free(free, rhs), ref.solve_free(free, rhs))


def _spd(n: int, seed: int) -> np.ndarray:
    """Return a well-conditioned random symmetric positive-definite matrix."""
    b = np.random.default_rng(seed).standard_normal((n, n))
    return b @ b.T + n * np.eye(n)


@pytest.mark.parametrize("have_scipy", [True, False])
def test_incremental_long_sweep_matches_dense(monkeypatch: pytest.MonkeyPatch, have_scipy: bool) -> None:
    """A long random insert/delete sweep tracks fresh solves on both the BLAS and the NumPy path."""
    monkeypatch.setattr(dense_module, "_HAVE_SCIPY", have_scipy)
    rng = np.random.default_rng(29)
    a = _spd(30, 29)
    inc = IncrementalDenseOperator(a)
    ref = DenseOperator(a)
    free: list[int] = []
    for _ in range(200):
        out = [i for i in range(30) if i not in free]
        if free and (not out or rng.random() < 0.4):
            free.remove(free[rng.integers(len(free))])  # delete from any slot, the last one included
        else:
            free.append(out[rng.integers(len(out))])
        idx = np.array(sorted(free), dtype=int)
        rhs = rng.standard_normal((idx.size, 2))
        np.testing.assert_allclose(inc.solve_free(idx, rhs), ref.solve_free(idx, rhs), atol=1e-10)


def test_incremental_accepts_unsorted_free_indices() -> None:
    """Free indices may come in any order; rhs and the solution follow that order."""
    rng = np.random.default_rng(30)
    a = _spd(8, 30)
    inc = IncrementalDenseOperator(a)
    ref = DenseOperator(a)
    for fs in ([5, 1, 3], [5, 1, 3, 0], [0, 3, 5], [7, 0, 5, 3]):
        free = np.array(fs)
        rhs = rng.standard_normal(len(fs))
        assert np.allclose(inc.solve_free(free, rhs), ref.solve_free(free, rhs))


def test_incremental_normalises_negative_indices() -> None:
    """Negative indices name the same entries as their non-negative counterparts."""
    a = _spd(5, 31)
    inc = IncrementalDenseOperator(a)
    rhs = np.array([1.0, 2.0])
    expected = DenseOperator(a).solve_free(np.array([1, 4]), rhs)
    assert np.allclose(inc.solve_free(np.array([1, -1]), rhs), expected)
    assert np.allclose(inc.solve_free(np.array([1, 4]), rhs), expected)


def test_incremental_out_of_range_index_raises() -> None:
    """An index outside the operator raises IndexError, as for DenseOperator."""
    with pytest.raises(IndexError):
        IncrementalDenseOperator(np.eye(3)).solve_free(np.array([0, 3]), np.ones(2))


@pytest.mark.parametrize("rhs", [np.ones(3), np.float64(1.0)])
def test_incremental_rhs_length_mismatch_raises(rhs: np.ndarray) -> None:
    """A right-hand side not aligned to the free set is rejected rather than broadcast."""
    with pytest.raises(DimensionMismatchError):
        IncrementalDenseOperator(np.eye(3)).solve_free(np.array([0, 1]), rhs)


def test_incremental_unchanged_free_set_reuses_inverse(monkeypatch: pytest.MonkeyPatch) -> None:
    """Solving twice on the same free set reuses the cached inverse without refactorising."""
    a = _spd(6, 32)
    inc = IncrementalDenseOperator(a)
    free = np.array([0, 2, 4])
    inc.solve_free(free, np.ones(3))

    def fail(_cur: np.ndarray) -> None:
        """Stand in for ``_refactor``; any call fails the test."""
        raise AssertionError("refactorised an unchanged free set")  # noqa: TRY003

    monkeypatch.setattr(inc, "_refactor", fail)
    rhs = np.array([1.0, -2.0, 0.5])
    assert np.allclose(inc.solve_free(free, rhs), DenseOperator(a).solve_free(free, rhs))


def test_incremental_singular_refactor_keeps_cache() -> None:
    """A refactorisation that raises leaves the previous cache usable."""
    a = np.ones((2, 2))  # singular; the 1x1 blocks are fine
    inc = IncrementalDenseOperator(a)
    inc.solve_free(np.array([0]), np.array([2.0]))
    with pytest.raises(np.linalg.LinAlgError):
        inc.solve_free(np.array([0, 1]), np.ones(2))  # Schur complement 0 -> refactor -> singular
    assert np.allclose(inc.solve_free(np.array([0]), np.array([2.0])), [2.0])


def test_incremental_jump_to_empty_free_set() -> None:
    """Dropping several indices at once down to an empty free set refactorises cleanly."""
    a = _spd(4, 33)
    inc = IncrementalDenseOperator(a)
    inc.solve_free(np.array([0, 1, 2]), np.ones(3))
    assert inc.solve_free(np.array([], dtype=int), np.array([])).shape == (0,)
    rhs = np.array([1.0, 2.0])
    assert np.allclose(inc.solve_free(np.array([1, 3]), rhs), DenseOperator(a).solve_free(np.array([1, 3]), rhs))
