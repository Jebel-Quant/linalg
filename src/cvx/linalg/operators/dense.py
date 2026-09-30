"""Dense-matrix backends: :class:`DenseOperator` and :class:`IncrementalDenseOperator`.

:class:`DenseOperator` wraps an explicit ``n x n`` matrix and slices it directly.
:class:`IncrementalDenseOperator` specialises it for active-set sweeps, maintaining
the free-block inverse across single-index changes with in-place rank-one bordered /
deletion updates instead of refactorising each step.
"""

from __future__ import annotations

import numpy as np

from ..core.exceptions import DimensionMismatchError, NonSquareMatrixError, NotAMatrixError
from ..core.types import Matrix, Vector
from ..decomposition.cholesky import cholesky_solve
from .base import SymmetricOperator, as_index, rcond_symmetric

try:  # BLAS ``ger`` applies the rank-one updates in place, with no k x k temporary.
    from scipy.linalg.blas import get_blas_funcs as _get_blas_funcs  # type: ignore[import-untyped]

    _HAVE_SCIPY = True
except ImportError:  # pragma: no cover - depends on the environment; the fallback is tested by patching the flag
    _HAVE_SCIPY = False


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
        free = as_index(free)
        return DenseOperator(self._a[np.ix_(free, free)])

    def block_matvec(self, rows: object, cols: object, v: Vector | Matrix) -> Vector | Matrix:
        """Return ``A[rows, cols] @ v`` by slicing the dense matrix."""
        rows = as_index(rows)
        cols = as_index(cols)
        result: Vector | Matrix = self._a[np.ix_(rows, cols)] @ v
        return result

    def solve_free(self, free: object, rhs: Vector | Matrix) -> Vector | Matrix:
        """Solve the free block by Cholesky (LU fallback via :func:`cholesky_solve`)."""
        free = as_index(free)
        return cholesky_solve(self._a[np.ix_(free, free)], rhs)

    def rcond_free(self, free: object) -> float:
        """Reciprocal condition number of the free block from its symmetric eigenvalues."""
        free = as_index(free)
        return rcond_symmetric(self._a[np.ix_(free, free)])


class IncrementalDenseOperator(DenseOperator):
    """Dense operator that maintains ``A[free, free]^{-1}`` across single-index flips.

    A drop-in :class:`DenseOperator` whose :meth:`solve_free` reuses the previous
    free-block inverse when the free set changed by at most one index since the last
    call, updating it with a rank-one bordered (index added) or deletion (index
    removed) formula at ``O(n * len(free))`` instead of refactorising at
    ``O(len(free)**3)``. Any other change -- a multi-index change, or a
    non-positive/non-finite pivot -- recomputes the inverse from scratch.

    The inverse is kept in *insertion order* in the leading block of a preallocated
    ``n x n`` Fortran-ordered buffer (allocated on the first solve), so an update
    never copies or permutes the whole block: an insert appends a border row and
    column, and a delete swaps the leaving slot with the last one. With SciPy
    installed the rank-one term is applied in place by BLAS ``ger``; without it, by
    NumPy with one ``k x k`` temporary.

    This suits an active-set loop that changes its free set one index at a time. The
    free indices may come in any order; *rhs* is aligned to that order, and so is the
    returned solution. :meth:`matvec`, :meth:`block_matvec`, and :meth:`rcond_free`
    are the plain dense ones -- only :meth:`solve_free` differs.

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
        n = self.n
        self._range = np.arange(n, dtype=np.intp)  # bounds-checks and normalises free indices
        self._order = np.empty(n, dtype=np.intp)  # index held in each slot
        self._slot = np.full(n, -1, dtype=np.intp)  # slot of each index, -1 when not free
        self._k = 0  # number of occupied slots
        self._buf: Matrix = np.empty((0, 0), order="F")  # inverse in slot order in its leading k x k block

    def solve_free(self, free: object, rhs: Vector | Matrix) -> Vector | Matrix:
        """Solve ``A[free, free] @ y = rhs`` using the maintained (incrementally updated) inverse."""
        cur = self._range[as_index(free)]
        rhs = np.asarray(rhs)
        if rhs.ndim == 0 or rhs.shape[0] != cur.size:
            raise DimensionMismatchError(rhs.shape[0] if rhs.ndim else rhs.size, cur.size)
        if self._buf.shape[0] != self.n:  # allocated on the first solve
            self._buf = np.zeros((self.n, self.n), order="F")
        if not self._update(cur):
            self._refactor(cur)
        pos = self._slot[cur]
        rhs_slot = np.empty(rhs.shape, dtype=np.result_type(rhs, self._buf))
        rhs_slot[pos] = rhs
        solution: Vector | Matrix = (self._buf[: self._k, : self._k] @ rhs_slot)[pos]
        return solution

    def _update(self, cur: np.ndarray) -> bool:
        """Bring the cache from the held free set to *cur* by at most one update (``False`` if it cannot)."""
        n_held = int(np.count_nonzero(self._slot[cur] >= 0))
        n_added, n_removed = cur.size - n_held, self._k - n_held
        if n_added == 0 and n_removed == 0:
            return True
        if n_added == 1 and n_removed == 0:
            return self._insert(int(cur[self._slot[cur] < 0][0]))
        if n_added == 0 and n_removed == 1:
            held = self._order[: self._k]
            keep = np.zeros(self.n, dtype=bool)
            keep[cur] = True
            return self._delete(int(held[~keep[held]][0]))
        return False  # not a single-index flip; recompute

    def _refactor(self, cur: np.ndarray) -> None:
        """Invert ``A[cur, cur]`` from scratch into slots ``0..len(cur)-1``, in *cur* order."""
        k = cur.size
        inv = np.linalg.inv(self._a[np.ix_(cur, cur)])  # may raise: leave the cache intact
        self._slot[self._order[: self._k]] = -1
        self._order[:k] = cur
        self._slot[cur] = self._range[:k]
        self._k = k
        self._block()[:] = inv

    def _block(self) -> Matrix:
        """The leading ``k x k`` block of the buffer: the maintained inverse in slot order."""
        block: Matrix = self._buf[: self._k, : self._k]
        return block

    def _rank_one(self, alpha: float, x: Vector) -> None:
        """Add ``alpha * outer(x, x)`` to the leading ``len(x) x len(x)`` block of the buffer, in place."""
        m = x.shape[0]
        buf = self._buf
        if m == 0:
            return
        if _HAVE_SCIPY:
            # buf[:, :m] is Fortran-contiguous, so ger updates it in place; buf[:m, :m] is
            # not, and ger would copy it. Zero-padding x leaves rows m.. unchanged.
            padded = np.zeros(self.n)
            padded[:m] = x
            ger = _get_blas_funcs("ger", (buf,))
            ger(alpha, padded, x, a=buf[:, :m], overwrite_a=True)
        else:
            block = buf[:m, :m]
            block += alpha * np.outer(x, x)

    def _insert(self, asset: int) -> bool:
        """Rank-one bordered update for one index entering the free set (``False`` if the pivot is bad)."""
        k = self._k
        c = self._a[self._order[:k], asset]
        v = self._block() @ c
        schur = float(self._a[asset, asset] - c @ v)
        if not np.isfinite(schur) or schur <= 0.0:
            return False
        self._rank_one(1.0 / schur, v)
        buf = self._buf
        buf[:k, k] = buf[k, :k] = -v / schur
        buf[k, k] = 1.0 / schur
        self._order[k], self._slot[asset], self._k = asset, k, k + 1
        return True

    def _delete(self, asset: int) -> bool:
        """Rank-one deletion update for one index leaving the free set (``False`` if the pivot is bad)."""
        block = self._block()
        p, last = int(self._slot[asset]), self._k - 1
        pivot = float(block[p, p])
        if not np.isfinite(pivot) or pivot <= 0.0:
            return False
        if p != last:  # move the index in the last slot into slot p
            block[[p, last], :] = block[[last, p], :]
            block[:, [p, last]] = block[:, [last, p]]
            moved = self._order[last]
            self._order[p], self._slot[moved] = moved, p
        self._rank_one(-1.0 / pivot, block[:last, last].copy())
        self._slot[asset], self._k = -1, last
        return True
