"""Tests for the affine projection (``AffineProjection``)."""

from __future__ import annotations

import numpy as np
import pytest

from cvx.linalg import (
    AffineProjection,
    DimensionMismatchError,
    NotAMatrixError,
)


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
