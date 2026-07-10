""":class:`FactorOperator`: diagonal-plus-low-rank ``A = diag(d) + U @ Delta @ U.T``."""

from __future__ import annotations

import numpy as np

from ..core.exceptions import DimensionMismatchError, NonSquareMatrixError, NotAMatrixError
from ..core.types import Matrix, Vector
from ..decomposition.cholesky import cholesky_solve
from .base import SymmetricOperator, _as_index

_DIAGONAL_NDIM_MESSAGE = "diagonal must be a 1-D array"
_DIAGONAL_POSITIVE_MESSAGE = "diagonal entries must be strictly positive"


def _validate_diagonal(d: Vector) -> None:
    """Check the diagonal is a 1-D, strictly positive vector."""
    if d.ndim != 1:
        raise ValueError(_DIAGONAL_NDIM_MESSAGE)
    if np.any(d <= 0.0):
        raise ValueError(_DIAGONAL_POSITIVE_MESSAGE)


def _validate_loadings(u: Matrix, d: Vector) -> None:
    """Check the loadings are an ``n x r`` matrix whose rows match the diagonal."""
    if u.ndim != 2:
        raise NotAMatrixError(u.ndim, func="FactorOperator")
    if u.shape[0] != d.shape[0]:
        raise DimensionMismatchError(u.shape[0], d.shape[0])


def _validate_inner(delta: Matrix, u: Matrix) -> None:
    """Check the inner block is a square ``r x r`` matrix matching the loadings' rank."""
    if delta.ndim != 2:
        raise NotAMatrixError(delta.ndim, func="FactorOperator")
    if delta.shape[0] != delta.shape[1]:
        raise NonSquareMatrixError(delta.shape[0], delta.shape[1])
    if delta.shape[0] != u.shape[1]:
        raise DimensionMismatchError(delta.shape[0], u.shape[1])


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
        _validate_diagonal(d)
        _validate_loadings(u, d)
        _validate_inner(delta, u)
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

    @property
    def diag(self) -> Vector:
        """The diagonal ``d_i + U[i] @ Delta @ U[i]``, at ``O(n r**2)`` without forming ``A``."""
        result: Vector = self._d + np.einsum("ij,ij->i", self._u @ self._delta, self._u)
        return result

    def matvec(self, x: Vector | Matrix) -> Vector | Matrix:
        """Return ``A @ x = d * x + U @ (Delta @ (U.T @ x))``."""
        return (self._d * x.T).T + self._u @ (self._delta @ (self._u.T @ x))

    def restricted(self, free: object) -> FactorOperator:
        """Return ``FactorOperator(d[free], U[free], Delta)``: the free block, pre-sliced."""
        free = _as_index(free)
        return FactorOperator(self._d[free], np.ascontiguousarray(self._u[free, :]), self._delta)

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
