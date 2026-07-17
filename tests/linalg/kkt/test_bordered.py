"""Tests for the bordered KKT solve (``bordered_solve``)."""

from __future__ import annotations

import numpy as np

from cvx.linalg import (
    DenseOperator,
    GramOperator,
    bordered_solve,
)


def _spd(n: int, rng: np.random.Generator) -> np.ndarray:
    """Return a random ``n``-by-``n`` symmetric positive-definite matrix."""
    b = rng.standard_normal((n, n))
    return b @ b.T + n * np.eye(n)


def test_bordered_solve_no_constraints_matches_solve_free() -> None:
    """With mc == 0 the result is the plain free-block solve and nu is empty."""
    rng = np.random.default_rng(0)
    a = _spd(6, rng)
    op = DenseOperator(a)
    free = np.array([0, 2, 3, 5])
    rhs = rng.standard_normal(free.size)
    c_free = np.zeros((0, free.size))
    x, nu = bordered_solve(op, free, c_free, rhs, np.zeros(0))
    assert np.allclose(x, op.solve_free(free, rhs))
    assert nu.shape == (0,)


def test_bordered_solve_matches_assembled_kkt() -> None:
    """With constraints, (x, nu) solve the assembled dense bordered KKT system."""
    rng = np.random.default_rng(1)
    a = _spd(7, rng)
    op = DenseOperator(a)
    free = np.array([0, 1, 3, 4, 6])
    nf = free.size
    c_free = rng.standard_normal((2, nf))
    rhs = rng.standard_normal(nf)
    d = rng.standard_normal(2)
    x, nu = bordered_solve(op, free, c_free, rhs, d)
    block = a[np.ix_(free, free)]
    kkt = np.block([[block, c_free.T], [c_free, np.zeros((2, 2))]])
    sol = np.linalg.solve(kkt, np.concatenate([rhs, d]))
    assert np.allclose(x, sol[:nf])
    assert np.allclose(nu, sol[nf:])


def test_bordered_solve_multi_rhs_matches_columnwise() -> None:
    """A multi-column rhs/d solves each column independently."""
    rng = np.random.default_rng(2)
    a = _spd(6, rng)
    op = DenseOperator(a)
    free = np.array([0, 1, 2, 4, 5])
    nf = free.size
    c_free = rng.standard_normal((1, nf))
    rhs = rng.standard_normal((nf, 3))
    d = rng.standard_normal((1, 3))
    x, nu = bordered_solve(op, free, c_free, rhs, d)
    assert x.shape == (nf, 3)
    assert nu.shape == (1, 3)
    for k in range(3):
        xk, nuk = bordered_solve(op, free, c_free, rhs[:, k], d[:, k])
        assert np.allclose(x[:, k], xk)
        assert np.allclose(nu[:, k], nuk)


def test_bordered_solve_matrix_free_gram_backend() -> None:
    """bordered_solve works through a matrix-free GramOperator (never forms A)."""
    rng = np.random.default_rng(3)
    m = rng.standard_normal((25, 6))
    op = GramOperator(m)
    a = m.T @ m
    free = np.array([0, 2, 3, 5])
    nf = free.size
    c_free = rng.standard_normal((1, nf))
    rhs = rng.standard_normal(nf)
    d = rng.standard_normal(1)
    x, nu = bordered_solve(op, free, c_free, rhs, d)
    block = a[np.ix_(free, free)]
    kkt = np.block([[block, c_free.T], [c_free, np.zeros((1, 1))]])
    sol = np.linalg.solve(kkt, np.concatenate([rhs, d]))
    assert np.allclose(x, sol[:nf])
    assert np.allclose(nu, sol[nf:])
