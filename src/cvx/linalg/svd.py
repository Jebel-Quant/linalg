"""Raw singular value decomposition utilities."""

from __future__ import annotations

import numpy as np

from .exceptions import InvalidComponentsError
from .types import Matrix, Vector


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


def svd_k(matrix: Matrix, k: int) -> tuple[Matrix, Vector, Matrix]:
    """Compute the truncated rank-``k`` singular value decomposition.

    Returns the ``k`` leading singular triplets — the best rank-``k``
    approximation of *matrix* in both the spectral and Frobenius norms
    (Eckart-Young). This is an *exact* truncation: the full compact SVD is
    computed and sliced, so the result is deterministic and matches
    :func:`numpy.linalg.svd` on the leading components. Like :func:`svd`, it is
    a raw decomposition and is **not** NaN-aware; clean non-finite entries
    first.

    Args:
        matrix: Input matrix of shape ``(m, n)``.
        k: Number of leading singular triplets to keep. Must be between 1 and
            ``min(m, n)``.

    Returns:
        Tuple ``(u, s, vt)`` with shapes ``(m, k)``, ``(k,)`` and ``(k, n)``,
        such that ``u @ np.diag(s) @ vt`` is the best rank-``k`` approximation
        of *matrix*. Singular values are in descending order.

    Raises:
        InvalidComponentsError: If *k* is smaller than 1 or larger than
            ``min(m, n)``.

    Example:
        >>> import numpy as np
        >>> from cvx.linalg import svd_k
        >>> matrix = np.diag([3.0, 2.0, 1.0]) @ np.ones((3, 4))
        >>> u, s, vt = svd_k(matrix, k=1)
        >>> u.shape, s.shape, vt.shape
        ((3, 1), (1,), (1, 4))

        The leading triplets agree with the full SVD:

        >>> u_full, s_full, vt_full = np.linalg.svd(matrix, full_matrices=False)
        >>> bool(np.allclose(s, s_full[:1]))
        True

        ``svd_k(matrix, min(m, n))`` reconstructs the matrix exactly:

        >>> u, s, vt = svd_k(matrix, k=3)
        >>> bool(np.allclose(u @ np.diag(s) @ vt, matrix))
        True
    """
    max_components = min(matrix.shape)
    if not 1 <= k <= max_components:
        raise InvalidComponentsError(k, max_components)

    u, s, vt = np.linalg.svd(matrix, full_matrices=False)
    return u[:, :k], s[:k], vt[:k, :]
