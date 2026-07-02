"""Linear-system helpers that ignore matrix rows and columns with non-finite diagonals."""

from __future__ import annotations

import numpy as np

from ..core.exceptions import (
    DEFAULT_COND_THRESHOLD,
    DimensionMismatchError,
    NonSquareMatrixError,
    SingularMatrixError,
)
from ..core.exceptions import (
    check_and_warn_condition as _check_and_warn_condition,
)
from ..core.types import Matrix, Vector
from ..core.valid import valid
from ..decomposition.cholesky import cholesky_solve as _cholesky_solve


def solve(
    matrix: Matrix,
    rhs: Vector | Matrix,
    cond_threshold: float = DEFAULT_COND_THRESHOLD,
) -> Vector | Matrix:
    """Solve a linear system restricted to the valid submatrix.

    Rows and columns with non-finite diagonal entries are excluded from the
    solve; the corresponding positions in the result are set to NaN.  Cholesky
    decomposition is attempted first for numerical stability and falls back to
    LU decomposition for non-positive-definite matrices.  When the condition
    number of the valid sub-matrix exceeds *cond_threshold*, an
    ``IllConditionedMatrixWarning`` is emitted.

    Args:
        matrix: Square coefficient matrix of shape ``(n, n)``.
        rhs: Right-hand side vector of length ``n`` or matrix of shape ``(n, k)``.
        cond_threshold: Condition-number threshold above which a warning is
            emitted. Defaults to ``1e12``.

    Returns:
        A solution array with the same shape as ``rhs``. Entries mapped to
        invalid rows or columns are returned as ``NaN``.

    Raises:
        NonSquareMatrixError: If the matrix is not square.
        DimensionMismatchError: If the leading dimension of ``rhs`` does not
            match the matrix dimension.
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

        Matrix right-hand sides are supported:

        >>> solve(np.eye(2), np.array([[1.0, 2.0], [3.0, 4.0]])).tolist()
        [[1.0, 2.0], [3.0, 4.0]]
    """
    if matrix.shape[0] != matrix.shape[1]:
        raise NonSquareMatrixError(matrix.shape[0], matrix.shape[1])

    if rhs.shape[0] != matrix.shape[0]:
        raise DimensionMismatchError(rhs.shape[0], matrix.shape[0])

    solution = np.full(rhs.shape, np.nan)
    mask, submatrix = valid(matrix)

    if mask.any():
        _check_and_warn_condition(submatrix, cond_threshold)
        try:
            solution[mask] = _cholesky_solve(submatrix, rhs[mask])
        except np.linalg.LinAlgError as exc:
            raise SingularMatrixError(str(exc)) from exc

    return solution
