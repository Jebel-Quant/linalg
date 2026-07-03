"""Weighted sum of symmetric operators."""

from __future__ import annotations

from collections.abc import Sequence

from ..core.exceptions import DimensionMismatchError
from ..core.types import Matrix, Vector
from .base import SymmetricOperator

_EMPTY_TERMS_MESSAGE = "SumOperator needs at least one term"
_NO_SOLVE_MESSAGE = "SumOperator has no direct free-block solve; run a Krylov method (e.g. CG) over apply_free"
_NO_RCOND_MESSAGE = "SumOperator has no structural rcond_free estimate"


class SumOperator(SymmetricOperator):
    """A weighted sum of symmetric operators, ``A = sum_i coeff_i * op_i``.

    Composes several :class:`SymmetricOperator` terms that act on the same
    dimension into one operator whose products are the (weighted) sums of the
    terms' products. This expresses forms a single backend cannot -- e.g. a
    data term plus a structured target,
    ``(1 - alpha) * GramOperator(M) + alpha * FactorOperator(...)`` -- while
    every product stays matrix-free (each term applies its own).

    **Forward-only.** A general sum has no cheap direct free-block solve or
    structural conditioning estimate, so :meth:`solve_free` and
    :meth:`rcond_free` raise :class:`NotImplementedError`. Feed
    :meth:`apply_free` to a Krylov solver (conjugate gradients) instead -- which
    is exactly how such a composite is used. :attr:`diag` *is* available (the
    weighted sum of the terms' diagonals, provided every term exposes one) and
    supplies the Jacobi preconditioner for that solve.

    Args:
        terms: A non-empty sequence of ``(coeff, operator)`` pairs. All operators
            must share the same dimension ``n``.

    Example:
        >>> import numpy as np
        >>> from cvx.linalg import DenseOperator, SumOperator
        >>> P = DenseOperator(np.array([[2.0, 0.0], [0.0, 1.0]]))
        >>> Q = DenseOperator(np.array([[1.0, 1.0], [1.0, 1.0]]))
        >>> op = SumOperator([(0.5, P), (2.0, Q)])
        >>> A = 0.5 * np.array([[2.0, 0.0], [0.0, 1.0]]) + 2.0 * np.ones((2, 2))
        >>> np.allclose(op.matvec(np.array([1.0, -1.0])), A @ np.array([1.0, -1.0]))
        True
        >>> free = np.array([0, 1])
        >>> np.allclose(op.apply_free(free, np.array([1.0, 1.0])), A @ np.ones(2))
        True
        >>> np.allclose(op.diag, np.diag(A))                # Jacobi preconditioner
        True
    """

    def __init__(self, terms: Sequence[tuple[float, SymmetricOperator]]) -> None:
        """Store the ``(coeff, operator)`` terms after checking they share a dimension."""
        terms = list(terms)
        if not terms:
            raise ValueError(_EMPTY_TERMS_MESSAGE)
        dims = {op.n for _, op in terms}
        if len(dims) != 1:
            raise DimensionMismatchError(min(dims), max(dims))
        self._terms: list[tuple[float, SymmetricOperator]] = [(float(c), op) for c, op in terms]
        self._n = dims.pop()

    @property
    def n(self) -> int:
        """Dimension shared by the terms."""
        return int(self._n)

    @property
    def diag(self) -> Vector:
        """The weighted sum of the terms' diagonals.

        Raises:
            NotImplementedError: If any term does not expose its diagonal.
        """
        coeff, op = self._terms[0]
        result: Vector = coeff * op.diag
        for coeff, op in self._terms[1:]:
            result = result + coeff * op.diag
        return result

    def matvec(self, x: Vector | Matrix) -> Vector | Matrix:
        """Return ``A @ x = sum_i coeff_i * op_i.matvec(x)``."""
        coeff, op = self._terms[0]
        result = coeff * op.matvec(x)
        for coeff, op in self._terms[1:]:
            result = result + coeff * op.matvec(x)
        return result

    def block_matvec(self, rows: object, cols: object, v: Vector | Matrix) -> Vector | Matrix:
        """Return ``A[rows, cols] @ v`` as the weighted sum of the terms' sub-block products."""
        coeff, op = self._terms[0]
        result = coeff * op.block_matvec(rows, cols, v)
        for coeff, op in self._terms[1:]:
            result = result + coeff * op.block_matvec(rows, cols, v)
        return result

    def solve_free(self, free: object, rhs: Vector | Matrix) -> Vector | Matrix:
        """Not available: a sum has no cheap direct free-block solve.

        Raises:
            NotImplementedError: Always. Run a Krylov solver over :meth:`apply_free`.
        """
        raise NotImplementedError(_NO_SOLVE_MESSAGE)

    def rcond_free(self, free: object) -> float:
        """Not available: a sum has no structural conditioning estimate.

        Raises:
            NotImplementedError: Always. Estimate conditioning from the assembled
                free block or a Krylov method if needed.
        """
        raise NotImplementedError(_NO_RCOND_MESSAGE)
