"""The :class:`SymmetricOperator` protocol and shared index/conditioning helpers.

This module holds the abstract contract every backend implements together with
the two structure-agnostic helpers the backends share: :func:`_as_index`, which
normalises an index set, and :func:`_rcond_symmetric`, which reads the reciprocal
2-norm condition number off a symmetric block's eigenvalues.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..core.types import Matrix, Vector

_INDEX_NDIM_MESSAGE = "index set must be a 1-D array of integer positions"
_INDEX_DUPLICATE_MESSAGE = "index set must not contain duplicates"


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
