"""Symmetric linear operators for active-set and Krylov solvers.

Active-set and Krylov solvers never need a symmetric matrix ``A`` as an explicit
array. They touch it through only a handful of products: a full matrix-vector
product ``A x``, an arbitrary sub-block product ``A[rows, cols] v``, and a direct
solve ``A[free, free]^{-1} rhs`` against a principal ("free") sub-block. This
module captures exactly that contract as :class:`SymmetricOperator` and provides
three backends that implement it at very different cost:

* :class:`DenseOperator` wraps an explicit ``n x n`` matrix.
* :class:`GramOperator` represents ``A = M.T @ M`` from a factor ``M`` and never
  forms the ``n x n`` Gram matrix.
* :class:`FactorOperator` represents a diagonal-plus-low-rank
  ``A = diag(d) + U @ Delta @ U.T`` and inverts free blocks by the Woodbury
  identity.

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

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from .cholesky import cholesky_solve
from .exceptions import DimensionMismatchError, NonSquareMatrixError, NotAMatrixError
from .types import Matrix, Vector

_ALPHA_RANGE_MESSAGE = "alpha must lie in the interval [0, 1]"
_INDEX_NDIM_MESSAGE = "index set must be a 1-D array of integer positions"
_DIAGONAL_NDIM_MESSAGE = "diagonal must be a 1-D array"

__all__ = [
    "DenseOperator",
    "FactorOperator",
    "GramOperator",
    "SymmetricOperator",
]


def _as_index(indices: object) -> np.ndarray:
    """Coerce an index set to a 1-D integer array."""
    idx = np.asarray(indices, dtype=np.intp)
    if idx.ndim != 1:
        raise ValueError(_INDEX_NDIM_MESSAGE)
    return idx


class SymmetricOperator(ABC):
    """A symmetric operator accessed only through block products and a free solve.

    Subclasses implement :meth:`matvec`, :meth:`block_matvec`, and
    :meth:`solve_free`; :meth:`apply_free` is derived. Index sets passed to the
    block methods are 1-D arrays of integer positions into ``range(n)``.
    """

    @property
    @abstractmethod
    def n(self) -> int:
        """Dimension of the operator (it acts on vectors of length ``n``)."""

    @abstractmethod
    def matvec(self, x: Vector | Matrix) -> Vector | Matrix:
        """Return ``A @ x``."""

    @abstractmethod
    def block_matvec(self, rows: object, cols: object, v: Vector | Matrix) -> Vector | Matrix:
        """Return ``A[rows, cols] @ v``, the product of a sub-block with *v*.

        Here ``A[rows, cols]`` is the sub-matrix with row indices *rows* and
        column indices *cols* (as by ``A[np.ix_(rows, cols)]``); *v* has length
        ``len(cols)`` (or that many rows) and the result has length
        ``len(rows)``.
        """

    @abstractmethod
    def solve_free(self, free: object, rhs: Vector | Matrix) -> Vector | Matrix:
        """Return ``A[free, free]^{-1} @ rhs``, a direct solve on a principal block.

        ``A[free, free]`` must be positive definite; a Krylov solver that would
        rather iterate should call :meth:`apply_free` instead.
        """

    def apply_free(self, free: object, v: Vector | Matrix) -> Vector | Matrix:
        """Return ``A[free, free] @ v``, the free-block forward product.

        This is the single operation a matrix-free conjugate-gradient solve on
        the free block requires; it is :meth:`block_matvec` with equal row and
        column sets.
        """
        free = _as_index(free)
        return self.block_matvec(free, free, v)


class DenseOperator(SymmetricOperator):
    """Symmetric operator backed by an explicit dense matrix.

    Args:
        matrix: A symmetric ``n x n`` matrix. It is stored by reference, not
            copied or symmetrised.

    Example:
        >>> import numpy as np
        >>> from cvx.linalg import DenseOperator
        >>> A = np.array([[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]])
        >>> op = DenseOperator(A)
        >>> free = np.array([0, 2])
        >>> np.allclose(op.apply_free(free, np.array([1.0, 1.0])), A[np.ix_(free, free)] @ np.ones(2))
        True
    """

    def __init__(self, matrix: Matrix) -> None:
        """Store the backing matrix after checking it is square."""
        matrix = np.asarray(matrix, dtype=np.float64)
        if matrix.ndim != 2:
            raise NotAMatrixError(matrix.ndim, func="DenseOperator")
        if matrix.shape[0] != matrix.shape[1]:
            raise NonSquareMatrixError(matrix.shape[0], matrix.shape[1])
        self._a = matrix

    @property
    def n(self) -> int:
        """Dimension of the operator."""
        return int(self._a.shape[0])

    def matvec(self, x: Vector | Matrix) -> Vector | Matrix:
        """Return ``A @ x`` by dense multiplication."""
        return self._a @ x

    def block_matvec(self, rows: object, cols: object, v: Vector | Matrix) -> Vector | Matrix:
        """Return ``A[rows, cols] @ v`` by slicing the dense matrix."""
        rows = _as_index(rows)
        cols = _as_index(cols)
        result: Vector | Matrix = self._a[np.ix_(rows, cols)] @ v
        return result

    def solve_free(self, free: object, rhs: Vector | Matrix) -> Vector | Matrix:
        """Solve the free block by Cholesky (LU fallback via :func:`cholesky_solve`)."""
        free = _as_index(free)
        return cholesky_solve(self._a[np.ix_(free, free)], rhs)


class GramOperator(SymmetricOperator):
    """Symmetric operator ``A = M.T @ M`` represented by its factor ``M``.

    The ``n x n`` Gram matrix is never formed: products go through *M* directly,
    at ``O(m * len(cols))`` per :meth:`block_matvec` and ``O(m * n)`` memory
    rather than ``O(n**2)``. This is the least-squares / normal-equations
    setting, where ``A = M.T @ M`` for an ``m x n`` factor *M*.

    :meth:`solve_free` forms the small ``len(free) x len(free)`` free Gram block
    and solves it directly; when ``M`` has fewer rows than ``free`` has entries
    that block is singular. Regularise via :meth:`regularized` (a ridge or an
    arbitrary positive-definite target) to restore positive definiteness.

    Args:
        factor: The ``m x n`` factor ``M``.

    Example:
        >>> import numpy as np
        >>> from cvx.linalg import GramOperator
        >>> M = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])
        >>> op = GramOperator(M)
        >>> bound, free = np.array([0]), np.array([1, 2])
        >>> A = M.T @ M
        >>> np.allclose(op.block_matvec(bound, free, np.array([1.0, 1.0])), A[np.ix_(bound, free)] @ np.ones(2))
        True
    """

    def __init__(self, factor: Matrix) -> None:
        """Store the ``m x n`` factor ``M`` (the Gram matrix is never formed)."""
        factor = np.asarray(factor, dtype=np.float64)
        if factor.ndim != 2:
            raise NotAMatrixError(factor.ndim, func="GramOperator")
        self._m = factor

    @classmethod
    def regularized(cls, factor: Matrix, alpha: float, root: Matrix) -> GramOperator:
        """Build the regularised operator ``(1 - alpha) M.T M + alpha R.T R``.

        The convex combination is realised as the Gram matrix of the stacked
        factor ``[sqrt(1 - alpha) M; sqrt(alpha) R]``, so the result is again a
        :class:`GramOperator` and every product stays matrix-free. With *root*
        an ``r x n`` square root of a positive-definite target ``R.T @ R`` (e.g.
        ``sqrt(lam) I`` for a scaled-identity ridge), the stacked factor has full
        column rank whenever ``alpha > 0``, so :meth:`solve_free` is well posed
        on any free set.

        Args:
            factor: The ``m x n`` factor ``M``.
            alpha: Regularisation intensity in ``[0, 1]``.
            root: An ``r x n`` matrix ``R`` with ``R.T @ R`` the target.

        Returns:
            A :class:`GramOperator` for ``(1 - alpha) M.T M + alpha R.T R``.

        Example:
            >>> import numpy as np
            >>> from cvx.linalg import GramOperator
            >>> M = np.array([[1.0, 2.0], [3.0, 4.0]])
            >>> R = np.eye(2)
            >>> op = GramOperator.regularized(M, 0.25, R)
            >>> A = 0.75 * (M.T @ M) + 0.25 * (R.T @ R)
            >>> np.allclose(op.matvec(np.array([1.0, -1.0])), A @ np.array([1.0, -1.0]))
            True
        """
        factor = np.asarray(factor, dtype=np.float64)
        root = np.asarray(root, dtype=np.float64)
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(_ALPHA_RANGE_MESSAGE)
        if root.ndim != 2:
            raise NotAMatrixError(root.ndim, func="GramOperator.regularized")
        if root.shape[1] != factor.shape[1]:
            raise DimensionMismatchError(root.shape[1], factor.shape[1])
        stacked = np.vstack([np.sqrt(1.0 - alpha) * factor, np.sqrt(alpha) * root])
        return cls(stacked)

    @property
    def n(self) -> int:
        """Dimension of the operator (columns of the factor ``M``)."""
        return int(self._m.shape[1])

    def matvec(self, x: Vector | Matrix) -> Vector | Matrix:
        """Return ``A @ x = M.T @ (M @ x)`` without forming ``A``."""
        return self._m.T @ (self._m @ x)

    def block_matvec(self, rows: object, cols: object, v: Vector | Matrix) -> Vector | Matrix:
        """Return ``A[rows, cols] @ v = M[:, rows].T @ (M[:, cols] @ v)``."""
        rows = _as_index(rows)
        cols = _as_index(cols)
        return self._m[:, rows].T @ (self._m[:, cols] @ v)

    def solve_free(self, free: object, rhs: Vector | Matrix) -> Vector | Matrix:
        """Form the small free Gram block ``M_F.T M_F`` and solve it directly."""
        free = _as_index(free)
        mf = self._m[:, free]
        return cholesky_solve(mf.T @ mf, rhs)


class FactorOperator(SymmetricOperator):
    """Diagonal-plus-low-rank operator ``A = diag(d) + U @ Delta @ U.T``.

    Free-block solves use the Woodbury identity, costing ``O(len(free) r**2 +
    r**3)`` for a rank-``r`` factor rather than ``O(len(free)**3)``, and no
    ``n x n`` matrix is formed (memory ``O(n r)``). With a strictly positive
    diagonal *d* and positive-definite *Delta* every principal block is positive
    definite, so :meth:`solve_free` is always well posed.

    Args:
        diagonal: The strictly positive diagonal ``d`` of length ``n``.
        loadings: The ``n x r`` factor loadings ``U``.
        inner: The ``r x r`` positive-definite inner matrix ``Delta``.

    Example:
        >>> import numpy as np
        >>> from cvx.linalg import FactorOperator
        >>> d = np.array([2.0, 3.0, 4.0])
        >>> U = np.array([[1.0], [0.5], [-1.0]])
        >>> Delta = np.array([[2.0]])
        >>> op = FactorOperator(d, U, Delta)
        >>> A = np.diag(d) + U @ Delta @ U.T
        >>> free, rhs = np.array([0, 2]), np.array([1.0, 1.0])
        >>> np.allclose(A[np.ix_(free, free)] @ op.solve_free(free, rhs), rhs)
        True
    """

    def __init__(self, diagonal: Vector, loadings: Matrix, inner: Matrix) -> None:
        """Store the diagonal, loadings, and inner block after shape checks."""
        d = np.asarray(diagonal, dtype=np.float64)
        u = np.asarray(loadings, dtype=np.float64)
        delta = np.asarray(inner, dtype=np.float64)
        if d.ndim != 1:
            raise ValueError(_DIAGONAL_NDIM_MESSAGE)
        if np.any(d <= 0.0):
            raise ValueError("diagonal entries must be strictly positive")
        if u.ndim != 2:
            raise NotAMatrixError(u.ndim, func="FactorOperator")
        if u.shape[0] != d.shape[0]:
            raise DimensionMismatchError(u.shape[0], d.shape[0])
        if delta.ndim != 2:
            raise NotAMatrixError(delta.ndim, func="FactorOperator")
        if delta.shape[0] != delta.shape[1]:
            raise NonSquareMatrixError(delta.shape[0], delta.shape[1])
        if delta.shape[0] != u.shape[1]:
            raise DimensionMismatchError(delta.shape[0], u.shape[1])
        self._d = d
        self._u = u
        self._delta = delta

    @property
    def n(self) -> int:
        """Dimension of the operator (length of the diagonal ``d``)."""
        return int(self._d.shape[0])

    def matvec(self, x: Vector | Matrix) -> Vector | Matrix:
        """Return ``A @ x = d * x + U @ (Delta @ (U.T @ x))``."""
        return (self._d * x.T).T + self._u @ (self._delta @ (self._u.T @ x))

    def block_matvec(self, rows: object, cols: object, v: Vector | Matrix) -> Vector | Matrix:
        """Return ``A[rows, cols] @ v`` from the low-rank term and the diagonal overlap."""
        rows = _as_index(rows)
        cols = _as_index(cols)
        low_rank = self._u[rows] @ (self._delta @ (self._u[cols].T @ v))
        # Diagonal couples only positions where a row index equals a column index.
        common, r_idx, c_idx = np.intersect1d(rows, cols, return_indices=True)
        diag = np.zeros_like(low_rank)
        diag[r_idx] = (self._d[common] * np.asarray(v)[c_idx].T).T
        result: Vector | Matrix = low_rank + diag
        return result

    def solve_free(self, free: object, rhs: Vector | Matrix) -> Vector | Matrix:
        """Solve the free block by the Woodbury identity on the ``r x r`` capacitance matrix."""
        free = _as_index(free)
        df = self._d[free]
        uf = self._u[free]
        # Woodbury: A_FF^{-1} = D^{-1} - D^{-1} U W^{-1} U.T D^{-1},
        # with W = Delta^{-1} + U.T D^{-1} U.
        dinv_rhs = (np.asarray(rhs, dtype=np.float64).T / df).T
        w = np.linalg.inv(self._delta) + uf.T @ ((uf.T / df).T)
        inner = cholesky_solve(w, uf.T @ dinv_rhs)
        correction = (uf @ inner).T / df
        result: Vector | Matrix = dinv_rhs - correction.T
        return result
