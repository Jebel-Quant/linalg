"""Eigenvalue utilities for general square matrices."""

from __future__ import annotations

from typing import cast

import numpy as np
import numpy.typing as npt

from .exceptions import NonSquareMatrixError, NotAMatrixError
from .types import Matrix, Vector


def eigvals(matrix: Matrix) -> Vector | npt.NDArray[np.complex128]:
    """Return the eigenvalues of a square matrix.

    This routine supports general (non-symmetric) square matrices and may
    return complex eigenvalues.

    Args:
        matrix: Square input matrix.

    Returns:
        Eigenvalues of ``matrix`` as returned by ``numpy.linalg.eigvals``.

    Raises:
        NotAMatrixError: If *matrix* is not two-dimensional.
        NonSquareMatrixError: If *matrix* is not square.
    """
    if matrix.ndim != 2:
        raise NotAMatrixError(matrix.ndim, func="eigvals")

    if matrix.shape[0] != matrix.shape[1]:
        raise NonSquareMatrixError(matrix.shape[0], matrix.shape[1])

    return cast("Vector | npt.NDArray[np.complex128]", np.linalg.eigvals(matrix))
