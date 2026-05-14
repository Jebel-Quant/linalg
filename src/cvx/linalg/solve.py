"""Linear-system helpers that ignore matrix rows and columns with non-finite diagonals."""

from __future__ import annotations

import numpy as np

from .valid import valid


def solve(matrix: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Solve a linear system restricted to the valid submatrix.

    Args:
        matrix: Square coefficient matrix.
        rhs: Right-hand side vector.

    Returns:
        A solution vector with the same shape as ``rhs``. Entries mapped to invalid
        rows or columns are returned as ``NaN``.

    Raises:
        AssertionError: If the matrix is not square or is incompatible with ``rhs``.

    """
    if matrix.shape[0] != matrix.shape[1]:
        raise AssertionError

    if rhs.size != matrix.shape[0]:
        raise AssertionError

    solution = np.nan * np.ones(rhs.size)
    mask, submatrix = valid(matrix)

    if mask.any():
        solution[mask] = np.linalg.solve(submatrix, rhs[mask])

    return solution
