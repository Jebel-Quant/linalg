"""Dense-matrix backends: :class:`DenseOperator` and :class:`IncrementalDenseOperator`.

:class:`DenseOperator` wraps an explicit ``n x n`` matrix and slices it directly.
:class:`IncrementalDenseOperator` specialises it for active-set sweeps, maintaining
the free-block inverse across single-index changes with rank-one bordered / deletion
updates instead of refactorising each step.
"""

from __future__ import annotations

import numpy as np

from ..core.exceptions import NonSquareMatrixError, NotAMatrixError
from ..core.types import Matrix, Vector
from ..decomposition.cholesky import cholesky_solve
from .base import SymmetricOperator, _as_index, _rcond_symmetric


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

    @property
    def diag(self) -> Vector:
        """The diagonal of the backing matrix (a read-only view)."""
        result: Vector = np.diagonal(self._a)
        return result

    def matvec(self, x: Vector | Matrix) -> Vector | Matrix:
        """Return ``A @ x`` by dense multiplication."""
        return self._a @ x

    def restricted(self, free: object) -> DenseOperator:
        """Return ``DenseOperator(A[free, free])``: the free block, pre-sliced."""
        free = _as_index(free)
        return DenseOperator(self._a[np.ix_(free, free)])

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

    def rcond_free(self, free: object) -> float:
        """Reciprocal condition number of the free block from its symmetric eigenvalues."""
        free = _as_index(free)
        return _rcond_symmetric(self._a[np.ix_(free, free)])


class IncrementalDenseOperator(DenseOperator):
    """Dense operator that maintains ``A[free, free]^{-1}`` across single-index flips.

    A drop-in :class:`DenseOperator` whose :meth:`solve_free` reuses the previous
    free-block inverse when the free set changed by exactly one index since the last
    call, updating it with a rank-one bordered (index added) or deletion (index
    removed) formula at ``O(len(free)**2)`` instead of refactorising at
    ``O(len(free)**3)``. Any other change -- the first solve, a multi-index change,
    or a non-positive/non-finite pivot -- recomputes the inverse from scratch.

    This suits an active-set loop that changes its free set one index at a time. The
    free index arrays must be **ascending** (as produced by ``np.flatnonzero`` of a
    boolean mask) and *rhs* aligned to that order; the maintained inverse and the
    returned solution follow the same order. :meth:`matvec`, :meth:`block_matvec`, and
    :meth:`rcond_free` are the plain dense ones -- only :meth:`solve_free` differs.

    A maintained inverse accumulates rounding over the ``O(n)`` updates of a sweep, so
    on ill-conditioned problems the plain :class:`DenseOperator` (a clean solve each
    step) is the safer choice.

    Example:
        >>> import numpy as np
        >>> from cvx.linalg import IncrementalDenseOperator
        >>> op = IncrementalDenseOperator(np.eye(3))
        >>> np.allclose(op.solve_free(np.array([0, 1]), np.array([1.0, 2.0])), [1.0, 2.0])
        True
    """

    def __init__(self, matrix: Matrix) -> None:
        """Wrap ``matrix`` (validated as in :class:`DenseOperator`) and start with no cache."""
        super().__init__(matrix)
        self._free_idx: np.ndarray | None = None
        self._inv: Matrix | None = None

    def solve_free(self, free: object, rhs: Vector | Matrix) -> Vector | Matrix:
        """Solve ``A[free, free] @ y = rhs`` using the maintained (incrementally updated) inverse."""
        cur = _as_index(free)
        inv = self._inverse_for(cur)
        self._free_idx, self._inv = cur, inv
        return inv @ rhs

    def _inverse_for(self, cur: np.ndarray) -> Matrix:
        """Return ``A[cur, cur]^{-1}``, updating the cache incrementally when possible."""
        prev, prev_inv = self._free_idx, self._inv
        if prev is None or prev_inv is None:
            return self._refactor(cur)
        added = np.setdiff1d(cur, prev, assume_unique=True)
        removed = np.setdiff1d(prev, cur, assume_unique=True)
        if added.size == 1 and removed.size == 0:
            updated = self._insert(prev, prev_inv, int(added[0]), cur)
        elif removed.size == 1 and added.size == 0:
            updated = self._delete(prev, prev_inv, int(removed[0]))
        else:
            updated = None  # not a single-index flip; recompute
        return updated if updated is not None else self._refactor(cur)

    def _refactor(self, cur: np.ndarray) -> Matrix:
        """Invert ``A[cur, cur]`` from scratch."""
        if cur.size == 0:
            return np.zeros((0, 0))
        inv: Matrix = np.linalg.inv(self._a[np.ix_(cur, cur)])
        return inv

    def _insert(self, prev: np.ndarray, prev_inv: Matrix, asset: int, cur: np.ndarray) -> Matrix | None:
        """Rank-one bordered update for one index entering the free set (``None`` if the pivot is bad)."""
        c = self._a[prev, asset]
        v = prev_inv @ c
        schur = float(self._a[asset, asset] - c @ v)
        if not np.isfinite(schur) or schur <= 0.0:
            return None
        k = prev.shape[0]
        aug = np.empty((k + 1, k + 1))
        aug[:k, :k] = prev_inv + np.outer(v, v) / schur
        aug[:k, k] = -v / schur
        aug[k, :k] = -v / schur
        aug[k, k] = 1.0 / schur
        # ``aug`` is ordered [prev..., asset]; permute to ascending ``cur`` order.
        perm = np.empty(k + 1, dtype=np.intp)
        is_new = cur == asset
        perm[is_new] = k
        perm[~is_new] = np.searchsorted(prev, cur[~is_new])
        permuted: Matrix = aug[np.ix_(perm, perm)]
        return permuted

    def _delete(self, prev: np.ndarray, prev_inv: Matrix, asset: int) -> Matrix | None:
        """Rank-one deletion update for one index leaving the free set (``None`` if the pivot is bad)."""
        p = int(np.searchsorted(prev, asset))
        pivot = float(prev_inv[p, p])
        if not np.isfinite(pivot) or pivot <= 0.0:
            return None
        mask = np.ones(prev.shape[0], dtype=bool)
        mask[p] = False
        col = prev_inv[mask, p]
        updated: Matrix = prev_inv[np.ix_(mask, mask)] - np.outer(col, col) / pivot
        return updated
