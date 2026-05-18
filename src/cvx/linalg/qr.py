"""QR decomposition helpers."""

from __future__ import annotations

import numpy as np

from .exceptions import DimensionMismatchError


def qr(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute the reduced QR decomposition of a 2-D matrix.

    Args:
        matrix: Input matrix with shape ``(m, n)``.

    Returns:
        A tuple ``(Q, R)`` matching ``np.linalg.qr(matrix, mode="reduced")``.

    Raises:
        DimensionMismatchError: If ``matrix`` is not two-dimensional.

    Example:
        >>> import numpy as np
        >>> from cvx.linalg import qr
        >>> q, r = qr(np.array([[1.0, 2.0], [3.0, 4.0]]))
        >>> np.allclose(q @ r, np.array([[1.0, 2.0], [3.0, 4.0]]))
        True
    """
    if matrix.ndim != 2:
        raise DimensionMismatchError(matrix.ndim, 2)

    return np.linalg.qr(matrix, mode="reduced")
