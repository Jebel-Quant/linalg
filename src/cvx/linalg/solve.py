"""Linear-system helpers that ignore matrix rows and columns with non-finite diagonals."""

from __future__ import annotations

import numpy as np

from .cholesky import cholesky as _cholesky
from .exceptions import (
    DimensionMismatchError,
    NonSquareMatrixError,
    SingularMatrixError,
)
from .exceptions import (
    check_and_warn_condition as _check_and_warn_condition,
)
from .valid import valid

_DEFAULT_COND_THRESHOLD: float = 1e12
"""Default condition-number threshold above which a warning is emitted."""


def solve(
    matrix: np.ndarray,
    rhs: np.ndarray,
    cond_threshold: float = _DEFAULT_COND_THRESHOLD,
) -> np.ndarray:
    """Solve a linear system restricted to the valid submatrix.

    Rows and columns with non-finite diagonal entries are excluded from the
    solve; the corresponding positions in the result are set to NaN.  Cholesky
    decomposition is attempted first for numerical stability and falls back to
    LU decomposition for non-positive-definite matrices.  When the condition
    number of the valid sub-matrix exceeds *cond_threshold*, an
    ``IllConditionedMatrixWarning`` is emitted.

    Args:
        matrix: Square coefficient matrix.
        rhs: Right-hand side vector.
        cond_threshold: Condition-number threshold above which a warning is
            emitted. Defaults to ``1e12``.

    Returns:
        A solution vector with the same shape as ``rhs``. Entries mapped to
        invalid rows or columns are returned as ``NaN``.

    Raises:
        NonSquareMatrixError: If the matrix is not square.
        DimensionMismatchError: If ``rhs`` size does not match the matrix dimension.
        SingularMatrixError: If the valid sub-matrix is singular.

    Example:
        >>> import numpy as np
        >>> from cvx.linalg import solve
        >>> solve(np.eye(2), np.array([1.0, 2.0])).tolist()
        [1.0, 2.0]

        NaN-masked entries are skipped:

        >>> matrix = np.array([[4.0, 0.0], [0.0, np.nan]])
        >>> solve(matrix, np.array([8.0, 1.0])).tolist()
        [2.0, nan]
    """
    if matrix.shape[0] != matrix.shape[1]:
        raise NonSquareMatrixError(matrix.shape[0], matrix.shape[1])

    if rhs.size != matrix.shape[0]:
        raise DimensionMismatchError(rhs.size, matrix.shape[0])

    solution = np.nan * np.ones(rhs.size)
    mask, submatrix = valid(matrix)

    if mask.any():
        _check_and_warn_condition(submatrix, cond_threshold)
        try:
            solution[mask] = _cholesky(submatrix, rhs[mask])
        except np.linalg.LinAlgError as exc:
            raise SingularMatrixError(str(exc)) from exc

    return solution
