"""Tests for the ``SymmetricOperator`` base class: abstract contract, index validation, and shared defaults."""

from __future__ import annotations

import numpy as np
import pytest

from cvx.linalg import (
    DenseOperator,
    FactorOperator,
    GramOperator,
    Matrix,
    SumOperator,
    SymmetricOperator,
    Vector,
)


def _operators(rng: np.random.Generator) -> list[tuple[SymmetricOperator, np.ndarray]]:
    """Return ``(operator, dense equivalent)`` pairs covering every backend."""
    n, m, r = 7, 5, 2
    factor = rng.standard_normal((m, n))
    gram_a = factor.T @ factor + 0.5 * np.eye(n)

    dense_a = rng.standard_normal((n, n))
    dense_a = dense_a @ dense_a.T + np.eye(n)

    d = rng.uniform(1.0, 2.0, size=n)
    u = rng.standard_normal((n, r))
    delta = np.diag(rng.uniform(0.5, 1.5, size=r))
    factor_a = np.diag(d) + u @ delta @ u.T

    pairs: list[tuple[SymmetricOperator, np.ndarray]] = [
        (GramOperator(factor, ridge=0.5), gram_a),
        (DenseOperator(dense_a), dense_a),
        (FactorOperator(d, u, delta), factor_a),
    ]
    pairs.append(
        (
            SumOperator([(0.25, pairs[0][0]), (2.0, pairs[2][0])]),
            0.25 * gram_a + 2.0 * factor_a,
        )
    )
    return pairs


class _NoDiagOperator(SymmetricOperator):
    """Minimal backend that leaves the base class's default ``diag`` in place."""

    @property
    def n(self) -> int:
        """Fixed dimension of 2."""
        return 2

    def matvec(self, x: Vector | Matrix) -> Vector | Matrix:
        """Identity product."""
        return x

    def block_matvec(self, rows: object, cols: object, v: Vector | Matrix) -> Vector | Matrix:
        """Identity sub-block product."""
        return v

    def solve_free(self, free: object, rhs: Vector | Matrix) -> Vector | Matrix:
        """Identity solve."""
        return rhs

    def rcond_free(self, free: object) -> float:
        """Perfectly conditioned."""
        return 1.0


# --- abstract contract & defaults ---------------------------------------------


def test_symmetric_operator_is_abstract() -> None:
    """The protocol base class cannot be instantiated directly."""
    with pytest.raises(TypeError):
        SymmetricOperator()  # type: ignore[abstract]


def test_diag_default_raises() -> None:
    """A backend that does not override ``diag`` raises NotImplementedError."""
    with pytest.raises(NotImplementedError, match="diagonal"):
        _ = _NoDiagOperator().diag


def test_apply_free_is_block_matvec_diagonal() -> None:
    """apply_free equals block_matvec with equal row and column sets."""
    rng = np.random.default_rng(5)
    a = np.diag(rng.uniform(1, 2, 6)) + 0.1 * np.ones((6, 6))
    op = DenseOperator(a)
    free = np.array([1, 3, 5])
    v = rng.standard_normal(3)
    assert np.allclose(op.apply_free(free, v), op.block_matvec(free, free, v))


# --- index validation (shared _as_index helper) -------------------------------


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


# --- restricted (default + backend overrides) ---------------------------------


@pytest.mark.parametrize("free", [np.array([0, 2, 5]), np.array([1]), np.arange(7)])
def test_restricted_matches_free_block(free: np.ndarray) -> None:
    """``restricted(free).matvec`` reproduces the dense free block product."""
    rng = np.random.default_rng(0)
    for op, a in _operators(rng):
        block = a[np.ix_(free, free)]
        sub = op.restricted(free)
        assert sub.n == len(free)
        v = rng.standard_normal(len(free))
        assert np.allclose(sub.matvec(v), block @ v)


def test_restricted_matches_apply_free() -> None:
    """``restricted(free).matvec`` and ``apply_free(free, .)`` agree on every backend."""
    rng = np.random.default_rng(1)
    free = np.array([1, 3, 4, 6])
    for op, _ in _operators(rng):
        sub = op.restricted(free)
        v = rng.standard_normal(len(free))
        assert np.allclose(sub.matvec(v), op.apply_free(free, v))


def test_restricted_default_raises() -> None:
    """A backend that does not override ``restricted`` raises NotImplementedError."""

    class _NoRestrictedOperator(SymmetricOperator):
        """Minimal backend leaving the base class's default ``restricted`` in place."""

        @property
        def n(self) -> int:
            """Fixed dimension of 2."""
            return 2

        def matvec(self, x):
            """Identity product."""
            return x

        def block_matvec(self, rows, cols, v):
            """Identity sub-block product."""
            return v

        def solve_free(self, free, rhs):
            """Identity solve."""
            return rhs

        def rcond_free(self, free) -> float:
            """Perfect conditioning."""
            return 1.0

    with pytest.raises(NotImplementedError, match="restricted"):
        _NoRestrictedOperator().restricted(np.array([0]))


def test_restricted_is_itself_restrictable() -> None:
    """Restriction composes: restricting the restricted operator stays consistent."""
    rng = np.random.default_rng(3)
    for op, a in _operators(rng):
        outer = np.array([0, 2, 3, 5])
        inner = np.array([1, 3])
        sub = op.restricted(outer).restricted(inner)
        block = a[np.ix_(outer[inner], outer[inner])]
        v = rng.standard_normal(2)
        assert np.allclose(sub.matvec(v), block @ v)
