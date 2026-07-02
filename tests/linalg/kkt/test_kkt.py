"""Tests for the bordered KKT solve and the affine projection."""

from __future__ import annotations

import numpy as np
import pytest

from cvx.linalg import (
    AffineProjection,
    DenseOperator,
    DimensionMismatchError,
    GramOperator,
    NotAMatrixError,
    bordered_solve,
)


def _spd(n: int, rng: np.random.Generator) -> np.ndarray:
    b = rng.standard_normal((n, n))
    return b @ b.T + n * np.eye(n)


# --- bordered_solve -----------------------------------------------------------


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


# --- AffineProjection ---------------------------------------------------------


def test_affine_projection_lands_on_set() -> None:
    """The projection satisfies the equality constraints."""
    rng = np.random.default_rng(4)
    c = rng.standard_normal((3, 8))
    d = rng.standard_normal(3)
    proj = AffineProjection(c, d)
    p = proj.project(rng.standard_normal(8))
    assert np.allclose(c @ p, d)


def test_affine_projection_is_nearest_point() -> None:
    """The residual x - P(x) is orthogonal to the constraint nullspace (nearest point)."""
    rng = np.random.default_rng(5)
    c = rng.standard_normal((2, 6))
    d = rng.standard_normal(2)
    proj = AffineProjection(c, d)
    x = rng.standard_normal(6)
    residual = x - proj.project(x)
    # A nullspace direction z (C z = 0) must be orthogonal to the residual.
    _, _, vh = np.linalg.svd(c)
    z = vh[-1]  # right-singular vector for the smallest singular value (in the nullspace)
    assert np.linalg.norm(c @ z) < 1e-10
    assert abs(float(residual @ z)) < 1e-10


def test_affine_projection_idempotent_and_fixes_feasible() -> None:
    """Projection is idempotent and leaves an already-feasible point unchanged."""
    rng = np.random.default_rng(6)
    c = rng.standard_normal((2, 5))
    d = rng.standard_normal(2)
    proj = AffineProjection(c, d)
    p = proj.project(rng.standard_normal(5))
    assert np.allclose(proj.project(p), p)


def test_affine_projection_matrix_input() -> None:
    """A stack of points is projected column-wise."""
    rng = np.random.default_rng(7)
    c = rng.standard_normal((2, 6))
    d = rng.standard_normal(2)
    proj = AffineProjection(c, d)
    xs = rng.standard_normal((6, 4))
    ps = proj.project(xs)
    assert ps.shape == (6, 4)
    assert np.allclose(c @ ps, d[:, None])
    for k in range(4):
        assert np.allclose(ps[:, k], proj.project(xs[:, k]))


def test_affine_projection_dims() -> None:
    """Report the constraint count and ambient dimension through m and n."""
    proj = AffineProjection(np.ones((2, 5)), np.zeros(2))
    assert proj.m == 2
    assert proj.n == 5


def test_affine_projection_rejects_non_matrix() -> None:
    """A 1-D constraint array is rejected."""
    with pytest.raises(NotAMatrixError):
        AffineProjection(np.ones(3), np.zeros(1))


def test_affine_projection_rejects_length_mismatch() -> None:
    """The target must have one entry per constraint row."""
    with pytest.raises(DimensionMismatchError):
        AffineProjection(np.ones((2, 4)), np.zeros(3))
