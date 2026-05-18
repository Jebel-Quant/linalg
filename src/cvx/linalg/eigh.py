"""Symmetric/Hermitian eigendecomposition helpers with NaN-aware filtering."""

from __future__ import annotations

import numpy as np

from .valid import valid


def eigh(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute eigendecomposition of a real symmetric or Hermitian matrix.

    Rows and columns with non-finite diagonal entries are excluded before
    decomposition.

    Args:
        matrix: Square symmetric/Hermitian input matrix.

    Returns:
        A tuple ``(eigenvalues, eigenvectors)`` as returned by ``np.linalg.eigh``
        on the valid submatrix. Eigenvalues are sorted in ascending order.
    """
    _, submatrix = valid(matrix)
    return np.linalg.eigh(submatrix)


def eigvalsh(matrix: np.ndarray) -> np.ndarray:
    """Return eigenvalues of a real symmetric or Hermitian matrix.

    Rows and columns with non-finite diagonal entries are excluded before
    decomposition.

    Args:
        matrix: Square symmetric/Hermitian input matrix.

    Returns:
        Eigenvalues of the valid submatrix in ascending order.
    """
    eigenvalues, _ = eigh(matrix)
    return eigenvalues
