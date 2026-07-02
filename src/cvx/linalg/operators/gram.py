""":class:`GramOperator`: ``A = M.T @ M + ridge I`` represented matrix-free by its factor ``M``."""

from __future__ import annotations

import numpy as np

from ..core.exceptions import DimensionMismatchError, NotAMatrixError
from ..core.types import Matrix, Vector
from ..decomposition.cholesky import cholesky_solve
from .base import SymmetricOperator, _as_index

_ALPHA_RANGE_MESSAGE = "alpha must lie in the interval [0, 1]"
_RIDGE_NEGATIVE_MESSAGE = "ridge must be non-negative"


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
