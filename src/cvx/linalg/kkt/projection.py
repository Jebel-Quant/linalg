"""Euclidean projection onto an affine set ``{x : C x = d}``."""

from __future__ import annotations

from typing import cast

import numpy as np

from ..core.exceptions import DimensionMismatchError, NotAMatrixError
from ..core.types import Matrix, Vector


class AffineProjection:
    """Euclidean projection onto the affine set ``{x : C x = d}``.

    The projection of ``x`` is the nearest point (in the 2-norm) satisfying the
    equality constraints,

    ``P(x) = x - C.T @ (C C.T)^{-1} @ (C x - d)``.

    The Gram matrix ``C C.T`` is formed once at construction (the ``O(mc**2 * n)``
    cost) and reused, so repeated projections -- for instance an alternating
    box / affine projection loop -- pay only an ``mc x mc`` solve each. ``C`` should
    have full row rank.

    Args:
        c: Constraint matrix ``C`` of shape ``(mc, n)``.
        d: Constraint target ``d`` of shape ``(mc,)``.

    Example:
        >>> import numpy as np
        >>> from cvx.linalg import AffineProjection
        >>> proj = AffineProjection(np.ones((1, 3)), np.array([1.0]))
        >>> p = proj.project(np.array([1.0, 1.0, 1.0]))
        >>> bool(np.isclose(np.ones(3) @ p, 1.0))     # lands on the affine set
        True
        >>> np.allclose(p, [1 / 3, 1 / 3, 1 / 3])     # nearest point with sum 1
        True
    """

    def __init__(self, c: Matrix, d: Vector) -> None:
        """Store ``C`` and ``d`` and precompute the Gram matrix ``C C.T``."""
        c = np.asarray(c, dtype=np.float64)
        d = np.asarray(d, dtype=np.float64)
        if c.ndim != 2:
            raise NotAMatrixError(c.ndim, func="AffineProjection")
        if d.ndim != 1 or d.shape[0] != c.shape[0]:
            raise DimensionMismatchError(d.shape[0] if d.ndim == 1 else d.size, c.shape[0])
        self._c = c
        self._d = d
        self._gram = c @ c.T

    @property
    def m(self) -> int:
        """Number of constraints (rows of ``C``)."""
        return int(self._c.shape[0])

    @property
    def n(self) -> int:
        """Ambient dimension (columns of ``C``)."""
        return int(self._c.shape[1])

    def project(self, x: Vector | Matrix) -> Vector | Matrix:
        """Return the Euclidean projection of ``x`` onto ``{x : C x = d}``.

        Args:
            x: Point of shape ``(n,)`` or a stack of points ``(n, k)``.

        Returns:
            The projected point(s), same shape as ``x``.
        """
        x = np.asarray(x, dtype=np.float64)
        target = self._d if x.ndim == 1 else self._d[:, None]
        residual = self._c @ x - target
        correction = self._c.T @ np.linalg.solve(self._gram, residual)
        return cast("Vector | Matrix", x - correction)
