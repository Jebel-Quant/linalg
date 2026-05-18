"""Matrix determinant with NaN-aware submatrix handling."""

from __future__ import annotations

import numpy as np

from .exceptions import NonSquareMatrixError
from .exceptions import check_and_warn_condition as _check_and_warn_condition
from .solve import _DEFAULT_COND_THRESHOLD
from .valid import valid

_SENTINEL = float("nan")


def det(
    matrix: np.ndarray,
    cond_threshold: float = _DEFAULT_COND_THRESHOLD,
) -> float:
    """Return the determinant of a square matrix.

    Rows and columns with non-finite diagonal entries are excluded before the
    computation; when no valid rows or columns remain the function returns
    ``nan``.  When the condition number of the valid sub-matrix exceeds
    *cond_threshold*, an ``IllConditionedMatrixWarning`` is emitted.

    Args:
        matrix: Square input matrix.
        cond_threshold: Condition-number threshold above which a warning is
            emitted. Defaults to ``1e12``.

    Returns:
        The determinant of the valid sub-matrix, or ``nan`` when no valid
        entries exist.

    Raises:
        NonSquareMatrixError: If the matrix is not square.

    Example:
        >>> import numpy as np
        >>> from cvx.linalg import det
        >>> det(np.eye(3))
        1.0

        NaN-masked entries are skipped:

        >>> matrix = np.array([[2.0, 0.0], [0.0, np.nan]])
        >>> det(matrix)
        2.0
    """
    if matrix.shape[0] != matrix.shape[1]:
        raise NonSquareMatrixError(matrix.shape[0], matrix.shape[1])

    mask, submatrix = valid(matrix)

    if not mask.any():
        return _SENTINEL

    _check_and_warn_condition(submatrix, cond_threshold)
    return float(np.linalg.det(submatrix))
