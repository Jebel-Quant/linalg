"""Norm utilities for vectors with optional NaN-aware matrix weighting."""

from __future__ import annotations

import numpy as np

from .cholesky import cholesky_solve as _cholesky_solve
from .exceptions import (
    DimensionMismatchError,
    NonSquareMatrixError,
    SingularMatrixError,
)
from .exceptions import (
    check_and_warn_condition as _check_and_warn_condition,
)
from .solve import _DEFAULT_COND_THRESHOLD
from .valid import valid


def a_norm(vector: np.ndarray, matrix: np.ndarray | None = None) -> float:
    """Calculate the generalized norm of a vector with respect to a matrix.

    Args:
        vector: The input vector.
        matrix: Optional square matrix defining the quadratic form.

    Returns:
        The Euclidean norm of the finite vector entries, or ``sqrt(v.T @ A @ v)``
        after dropping rows and columns whose diagonal entries are not finite.

    Raises:
        NonSquareMatrixError: If the matrix is not square.
        DimensionMismatchError: If the vector length does not match the matrix dimension.

    Example:
        >>> import numpy as np
        >>> from cvx.linalg import a_norm
        >>> a_norm(np.array([3.0, 4.0]))
        5.0
    """
    if matrix is None:
        return float(np.linalg.norm(vector[np.isfinite(vector)], 2))

    if matrix.shape[0] != matrix.shape[1]:
        raise NonSquareMatrixError(matrix.shape[0], matrix.shape[1])

    if vector.size != matrix.shape[0]:
        raise DimensionMismatchError(vector.size, matrix.shape[0])

    mask, submatrix = valid(matrix)
    if mask.any():
        filtered_vector = vector[mask]
        return float(np.sqrt(filtered_vector @ submatrix @ filtered_vector))

    return float("nan")


def inv_a_norm(
    vector: np.ndarray,
    matrix: np.ndarray | None = None,
    cond_threshold: float = _DEFAULT_COND_THRESHOLD,
) -> float:
    """Calculate the inverse A-norm of a vector using an optional matrix.

    If ``matrix`` is ``None``, returns the Euclidean norm of finite entries.
    Otherwise computes ``sqrt(v.T @ A^{-1} @ v)`` on the valid submatrix,
    attempting Cholesky decomposition first for numerical stability and
    falling back to LU decomposition for non-positive-definite matrices.
    When the condition number of the valid sub-matrix exceeds *cond_threshold*,
    an ``IllConditionedMatrixWarning`` is emitted.

    Args:
        vector: The input vector.
        matrix: Optional square matrix defining the quadratic form.
        cond_threshold: Condition-number threshold above which a warning is
            emitted. Defaults to ``1e12``.

    Returns:
        The Euclidean norm of the finite vector entries, or
        ``sqrt(v.T @ A^{-1} @ v)`` after dropping rows and columns whose
        diagonal entries are not finite. Returns ``nan`` when no valid entries
        exist.

    Raises:
        NonSquareMatrixError: If the matrix is not square.
        DimensionMismatchError: If the vector length does not match the matrix dimension.
        SingularMatrixError: If the valid sub-matrix is singular.

    Example:
        >>> import numpy as np
        >>> from cvx.linalg import inv_a_norm
        >>> inv_a_norm(np.array([3.0, 4.0]))
        5.0
    """
    if matrix is None:
        return float(np.linalg.norm(vector[np.isfinite(vector)], 2))

    if matrix.shape[0] != matrix.shape[1]:
        raise NonSquareMatrixError(matrix.shape[0], matrix.shape[1])

    if vector.size != matrix.shape[0]:
        raise DimensionMismatchError(vector.size, matrix.shape[0])

    mask, submatrix = valid(matrix)
    if mask.any():
        _check_and_warn_condition(submatrix, cond_threshold)
        filtered_vector = vector[mask]
        try:
            solved = _cholesky_solve(submatrix, filtered_vector)
        except np.linalg.LinAlgError as exc:
            raise SingularMatrixError(str(exc)) from exc
        return float(np.sqrt(np.dot(filtered_vector, solved)))

    return float("nan")
