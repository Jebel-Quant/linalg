"""Raw singular value decomposition utilities."""

from __future__ import annotations

import numpy as np

from .types import Matrix


def svd(matrix: Matrix) -> tuple[Matrix, Matrix, Matrix]:
    """Compute the compact singular value decomposition of a matrix.

    This is a thin wrapper around ``numpy.linalg.svd`` with
    ``full_matrices=False``.

    Args:
        matrix: Input matrix of shape ``(m, n)``.

    Returns:
        Tuple ``(u, s, vt)`` such that ``matrix == u @ np.diag(s) @ vt``.
    """
    return np.linalg.svd(matrix, full_matrices=False)
