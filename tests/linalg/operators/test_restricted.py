"""Tests for :meth:`SymmetricOperator.restricted`: pre-sliced free-block operators."""

from __future__ import annotations

import numpy as np
import pytest

from cvx.linalg import (
    DenseOperator,
    FactorOperator,
    GramOperator,
    SumOperator,
    SymmetricOperator,
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

        def matvec(self, x):  # noqa: ANN001, ANN201
            """Identity product."""
            return x

        def block_matvec(self, rows, cols, v):  # noqa: ANN001, ANN201
            """Identity sub-block product."""
            return v

        def solve_free(self, free, rhs):  # noqa: ANN001, ANN201
            """Identity solve."""
            return rhs

        def rcond_free(self, free) -> float:  # noqa: ANN001
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
