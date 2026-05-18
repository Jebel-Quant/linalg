"""Eigenvalue utilities for general square matrices."""

from __future__ import annotations

import numpy as np

from .exceptions import NonSquareMatrixError


def eigvals(matrix: np.ndarray) -> np.ndarray:
    """Return the eigenvalues of a square matrix.

    This routine supports general (non-symmetric) square matrices and may
    return complex eigenvalues.

    Args:
        matrix: Square input matrix.

    Returns:
        Eigenvalues of ``matrix`` as returned by ``numpy.linalg.eigvals``.

    Raises:
        TypeError: If *matrix* is not two-dimensional.
        NonSquareMatrixError: If *matrix* is not square.
    """
    if matrix.ndim != 2:
        raise TypeError(
            f"eigvals() expected a 2-D matrix, got {matrix.ndim}-D input"
        )

    if matrix.shape[0] != matrix.shape[1]:
        raise NonSquareMatrixError(matrix.shape[0], matrix.shape[1])

    return np.linalg.eigvals(matrix)
