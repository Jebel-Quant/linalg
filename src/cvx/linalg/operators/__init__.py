"""Symmetric linear operators for active-set and Krylov solvers.

Active-set and Krylov solvers never need a symmetric matrix ``A`` as an explicit
array. They touch it through only a handful of products: a full matrix-vector
product ``A x``, an arbitrary sub-block product ``A[rows, cols] v``, and a direct
solve ``A[free, free]^{-1} rhs`` against a principal ("free") sub-block. This
subpackage captures exactly that contract as :class:`SymmetricOperator` -- together
with :meth:`SymmetricOperator.rcond_free`, a conditioning check an active-set
solver uses to detect a rank-deficient free block -- and provides four backends
that implement it at very different cost:

* :class:`DenseOperator` wraps an explicit ``n x n`` matrix.
* :class:`GramOperator` represents ``A = M.T @ M`` from a factor ``M`` and never
  forms the ``n x n`` Gram matrix.
* :class:`FactorOperator` represents a diagonal-plus-low-rank
  ``A = diag(d) + U @ Delta @ U.T`` and inverts free blocks by the Woodbury
  identity.
* :class:`IncrementalDenseOperator` is a :class:`DenseOperator` that maintains the
  free-block inverse across single-index changes for active-set sweeps.

The protocol and the shared index/conditioning helpers live in :mod:`~.base`;
each backend lives in its own module (:mod:`~.dense`, :mod:`~.gram`,
:mod:`~.factor`).

Two solving strategies compose with the same protocol. A direct solver consumes
:meth:`SymmetricOperator.solve_free`; a matrix-free conjugate-gradient solver
consumes :meth:`SymmetricOperator.apply_free` (the free-block forward product)
and does its own iterating. Because a solver reaches ``A`` only through the
protocol, switching backend requires no change to the solver.

The two products a solver needs beyond the free-block solve are the free-block
forward product (:meth:`SymmetricOperator.apply_free`, i.e. ``block_matvec`` with
``rows == cols``) and a cross product ``A[bound, free] x_free`` -- the reduced
gradient at the bound indices -- obtained from :meth:`block_matvec` on the
disjoint index sets. ``block_matvec`` is defined for arbitrary ``rows`` and
``cols``, so both regimes and everything between are covered.

Example:
    >>> import numpy as np
    >>> from cvx.linalg import GramOperator
    >>> rng = np.random.default_rng(0)
    >>> M = rng.standard_normal((20, 5))
    >>> op = GramOperator(M)
    >>> A = M.T @ M
    >>> np.allclose(op.matvec(np.ones(5)), A @ np.ones(5))
    True
    >>> free = np.array([0, 2, 4])
    >>> rhs = np.array([1.0, 2.0, 3.0])
    >>> x = op.solve_free(free, rhs)
    >>> np.allclose(A[np.ix_(free, free)] @ x, rhs)
    True
"""

from .base import SymmetricOperator as SymmetricOperator
from .composite import SumOperator as SumOperator
from .dense import DenseOperator as DenseOperator
from .dense import IncrementalDenseOperator as IncrementalDenseOperator
from .factor import FactorOperator as FactorOperator
from .gram import GramOperator as GramOperator

__all__ = [
    "DenseOperator",
    "FactorOperator",
    "GramOperator",
    "IncrementalDenseOperator",
    "SumOperator",
    "SymmetricOperator",
]
