"""Symmetric linear operators for active-set and Krylov solvers.

Active-set and Krylov solvers never need a symmetric matrix ``A`` as an explicit
array. They touch it through only a handful of products: a full matrix-vector
product ``A x``, an arbitrary sub-block product ``A[rows, cols] v``, and a direct
solve ``A[free, free]^{-1} rhs`` against a principal ("free") sub-block. This
module captures exactly that contract as :class:`SymmetricOperator` -- together
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

from ..core.exceptions import DimensionMismatchError, NonSquareMatrixError, NotAMatrixError
from ..core.types import Matrix, Vector
from ..decomposition.cholesky import cholesky_solve

_ALPHA_RANGE_MESSAGE = "alpha must lie in the interval [0, 1]"
_RIDGE_NEGATIVE_MESSAGE = "ridge must be non-negative"
_INDEX_NDIM_MESSAGE = "index set must be a 1-D array of integer positions"
_INDEX_DUPLICATE_MESSAGE = "index set must not contain duplicates"
_DIAGONAL_NDIM_MESSAGE = "diagonal must be a 1-D array"
_DIAGONAL_POSITIVE_MESSAGE = "diagonal entries must be strictly positive"

__all__ = [
    "DenseOperator",
    "FactorOperator",
    "GramOperator",
    "IncrementalDenseOperator",
    "SymmetricOperator",
]


def _as_index(indices: object) -> np.ndarray:
    """Coerce an index set to a 1-D integer array."""
    arr = np.asarray(indices)
    if arr.ndim != 1:
        raise ValueError(_INDEX_NDIM_MESSAGE)
    if arr.dtype == np.bool_ or not np.issubdtype(arr.dtype, np.integer):
        raise ValueError(_INDEX_NDIM_MESSAGE)
    idx = arr.astype(np.intp, copy=False)
    if np.unique(idx).size != idx.size:
        raise ValueError(_INDEX_DUPLICATE_MESSAGE)
    return idx


def _rcond_symmetric(block: Matrix) -> float:
    """Reciprocal 2-norm condition number of a symmetric positive-(semi)definite block.

    Returns a value in ``[0, 1]`` read off the symmetric eigenvalues
    (``lambda_min / lambda_max``): ``1`` is perfectly conditioned, ``0`` numerically
    singular. Deterministic and BLAS-independent. An empty block is well conditioned.
    """
    if block.shape[0] == 0:
        return 1.0
    eigenvalues = np.linalg.eigvalsh(block)
    lam_max = float(eigenvalues[-1])
    if lam_max <= 0.0:
        return 0.0
    return max(float(eigenvalues[0]), 0.0) / lam_max


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

    @abstractmethod
    def rcond_free(self, free: object) -> float:
        """Return the reciprocal 2-norm condition number of ``A[free, free]``.

        A value in ``[0, 1]`` (``1`` well conditioned, ``0`` numerically singular).
        An active-set solver uses this to detect a rank-deficient free block before
        trusting :meth:`solve_free`. Backends compute it from their structure
        without materialising the ``len(free) x len(free)`` block.
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

    def rcond_free(self, free: object) -> float:
        """Reciprocal condition number of the free block from its symmetric eigenvalues."""
        free = _as_index(free)
        return _rcond_symmetric(self._a[np.ix_(free, free)])


class GramOperator(SymmetricOperator):
    """Symmetric operator ``A = M.T @ M + ridge * I`` represented by its factor ``M``.

    The ``n x n`` Gram matrix is never formed: products go through *M* directly,
    at ``O(m * len(cols))`` per :meth:`block_matvec` and ``O(m * n)`` memory
    rather than ``O(n**2)``. This is the least-squares / normal-equations
    setting, where ``A = M.T @ M`` for an ``m x n`` factor *M*.

    An optional Tikhonov ``ridge >= 0`` adds ``ridge * I`` to the Gram matrix. It
    matters because ``M.T @ M`` has rank at most ``m``: when ``M`` has fewer rows
    than the free set has entries the un-ridged free block is singular. A positive
    ridge makes every free block positive definite, and :meth:`solve_free` then
    switches to the Woodbury identity in the ``m``-dimensional row space whenever
    that is cheaper than the ``len(free) x len(free)`` normal equations -- i.e.
    when ``m < len(free)``.

    For a general (non-scaled-identity) positive-definite target, use
    :meth:`regularized` instead, which forms the convex combination
    ``(1 - alpha) M.T M + alpha R.T R`` as a stacked-factor Gram matrix.

    Args:
        factor: The ``m x n`` factor ``M``.
        ridge: A non-negative diagonal loading ``ridge`` added as ``ridge * I``.

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

    def __init__(self, factor: Matrix, ridge: float = 0.0) -> None:
        """Store the ``m x n`` factor ``M`` and ridge (the Gram matrix is never formed)."""
        factor = np.asarray(factor, dtype=np.float64)
        if factor.ndim != 2:
            raise NotAMatrixError(factor.ndim, func="GramOperator")
        if ridge < 0.0:
            raise ValueError(_RIDGE_NEGATIVE_MESSAGE)
        self._m = factor
        self._ridge = float(ridge)

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
        if factor.ndim != 2:
            raise NotAMatrixError(factor.ndim, func="GramOperator.regularized")
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
        """Return ``A @ x = M.T @ (M @ x) + ridge * x`` without forming ``A``."""
        product = self._m.T @ (self._m @ x)
        return product + self._ridge * x if self._ridge else product

    def block_matvec(self, rows: object, cols: object, v: Vector | Matrix) -> Vector | Matrix:
        """Return ``A[rows, cols] @ v = M[:, rows].T @ (M[:, cols] @ v)`` plus the ridge overlap."""
        rows = _as_index(rows)
        cols = _as_index(cols)
        product: Vector | Matrix = self._m[:, rows].T @ (self._m[:, cols] @ v)
        if not self._ridge:
            return product
        # ridge * I couples only positions where a row index equals a column index.
        _common, r_idx, c_idx = np.intersect1d(rows, cols, return_indices=True)
        diag = np.zeros_like(product)
        diag[r_idx] = (self._ridge * np.asarray(v)[c_idx].T).T
        return product + diag

    def solve_free(self, free: object, rhs: Vector | Matrix) -> Vector | Matrix:
        """Solve the free block ``M_F.T M_F + ridge * I``, by Woodbury in row space when ``m < len(free)``."""
        free = _as_index(free)
        mf = self._m[:, free]
        m_rows, n_free = mf.shape
        if self._ridge and m_rows < n_free:
            # Woodbury in the m-dimensional row space:
            # (ridge I + M_F.T M_F)^{-1} rhs = (rhs - M_F.T (ridge I_m + M_F M_F.T)^{-1} (M_F rhs)) / ridge.
            rhs_arr = np.asarray(rhs, dtype=np.float64)
            capacitance = self._ridge * np.eye(m_rows) + mf @ mf.T
            correction = mf.T @ cholesky_solve(capacitance, mf @ rhs_arr)
            return (rhs_arr - correction) / self._ridge
        block = mf.T @ mf
        if self._ridge:
            block = block + self._ridge * np.eye(n_free)
        return cholesky_solve(block, rhs)

    def rcond_free(self, free: object) -> float:
        """Reciprocal condition number of the free block from the SVD of ``M[:, free]``.

        The eigenvalues of ``M_F.T M_F + ridge I`` are ``sigma_i**2 + ridge`` for the
        singular values ``sigma_i`` of ``M_F``, with extra ``ridge`` eigenvalues when
        the free set outgrows the rank of ``M_F``. Reads the conditioning off the
        ``(m, len(free))`` factor block without forming the ``len(free)`` square block,
        so a rank-deficient free set correctly drives the result toward zero.
        """
        free = _as_index(free)
        n_free = free.size
        if n_free == 0:
            return 1.0
        singular_values = np.linalg.svd(self._m[:, free], compute_uv=False)
        eig = singular_values**2
        lam_max = (float(eig[0]) if eig.size else 0.0) + self._ridge
        if lam_max <= 0.0:
            return 0.0
        lam_min = self._ridge if n_free > singular_values.shape[0] else float(eig[-1]) + self._ridge
        return max(lam_min, 0.0) / lam_max


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
        >>> (op.n, op.k)                          # 3 assets, 1 factor
        (3, 1)
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
            raise ValueError(_DIAGONAL_POSITIVE_MESSAGE)
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

    @property
    def k(self) -> int:
        """Number of factors (rank ``r`` of the low-rank term; columns of ``U``)."""
        return int(self._u.shape[1])

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
        delta_inv = np.linalg.solve(self._delta, np.eye(self._delta.shape[0]))
        w = delta_inv + uf.T @ ((uf.T / df).T)
        inner = cholesky_solve(w, uf.T @ dinv_rhs)
        correction = (uf @ inner).T / df
        result: Vector | Matrix = dinv_rhs - correction.T
        return result

    def rcond_free(self, free: object) -> float:
        """Lower bound on the free block's reciprocal condition number, via Weyl's inequalities.

        The free block ``diag(d_F) + U_F Delta U_F.T`` is positive definite (the
        positive diagonal keeps it full rank). Rather than form it, bound
        ``lambda_min >= min(d_F)`` and
        ``lambda_max <= max(d_F) + ||U_F||_2^2 * lambda_max(Delta)``; their ratio is
        a guaranteed lower bound on the true reciprocal condition number, at
        ``O(len(free) r**2 + r**3)`` and without an ``n x n`` matrix.
        """
        free = _as_index(free)
        if free.size == 0:
            return 1.0
        d_free = self._d[free]
        u_free = self._u[free]
        u_spectral_norm = float(np.linalg.svd(u_free, compute_uv=False)[0])
        delta_max = float(np.linalg.eigvalsh(self._delta)[-1])
        lam_max_upper = float(np.max(d_free)) + u_spectral_norm**2 * max(delta_max, 0.0)
        return float(np.min(d_free)) / lam_max_upper


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
