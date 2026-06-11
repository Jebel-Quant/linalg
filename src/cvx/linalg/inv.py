"""Matrix inversion helpers that ignore rows and columns with non-finite diagonals."""

from __future__ import annotations

import numpy as np

from .exceptions import (
    DEFAULT_COND_THRESHOLD,
    NonSquareMatrixError,
    SingularMatrixError,
)
from .exceptions import (
    check_and_warn_condition as _check_and_warn_condition,
)
from .types import Matrix
from .valid import valid


def inv(
    matrix: Matrix,
    cond_threshold: float = DEFAULT_COND_THRESHOLD,
) -> Matrix:
    """Invert a matrix restricted to the valid submatrix.

    Rows and columns with non-finite diagonal entries are excluded from the
    inversion; the corresponding rows and columns in the result are set to NaN.
    When the condition number of the valid sub-matrix exceeds *cond_threshold*,
    an ``IllConditionedMatrixWarning`` is emitted.

    Args:
        matrix: Square matrix to invert.
        cond_threshold: Condition-number threshold above which a warning is
            emitted. Defaults to ``1e12``.

    Returns:
        An inverted matrix with the same shape as *matrix*. Rows and columns
        mapped to invalid entries are returned as ``NaN``.

    Raises:
        NonSquareMatrixError: If the matrix is not square.
        SingularMatrixError: If the valid sub-matrix is singular.

    Example:
        >>> import numpy as np
        >>> from cvx.linalg import inv
        >>> np.allclose(inv(np.eye(2)), np.eye(2))
        True

        NaN-masked entries are skipped:

        >>> matrix = np.array([[4.0, 0.0], [0.0, np.nan]])
        >>> result = inv(matrix)
        >>> float(result[0, 0])
        0.25
        >>> bool(np.isnan(result[0, 1]) and np.isnan(result[1, 0]) and np.isnan(result[1, 1]))
        True
    """
    if matrix.shape[0] != matrix.shape[1]:
        raise NonSquareMatrixError(matrix.shape[0], matrix.shape[1])

    n = matrix.shape[0]
    result = np.full((n, n), np.nan)
    mask, submatrix = valid(matrix)

    if mask.any():
        _check_and_warn_condition(submatrix, cond_threshold)
        try:
            sub_inv = np.linalg.inv(submatrix)
        except np.linalg.LinAlgError as exc:
            raise SingularMatrixError(str(exc)) from exc

        idx = np.where(mask)[0]
        result[np.ix_(idx, idx)] = sub_inv

    return result
