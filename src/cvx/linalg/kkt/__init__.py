"""Equality-constrained linear algebra: bordered KKT solves and affine projections.

Two operator- and constraint-level building blocks for active-set and path-tracing
solvers:

- :func:`bordered_solve` -- the range-space (Schur complement) solution of a
  bordered KKT system ``[[H_FF, C.T], [C, 0]] @ [x; nu] = [rhs; d]``, reaching the
  Hessian block only through a :class:`~cvx.linalg.SymmetricOperator`.
- :class:`AffineProjection` -- Euclidean projection onto ``{x : C x = d}``, caching
  the Gram matrix ``C C.T`` for repeated use.
"""

from .bordered import bordered_solve as bordered_solve
from .projection import AffineProjection as AffineProjection

__all__ = [
    "AffineProjection",
    "bordered_solve",
]
