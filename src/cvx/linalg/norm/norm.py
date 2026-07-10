"""Norm utilities for vectors with optional NaN-aware matrix weighting."""

from __future__ import annotations

from typing import Literal

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


def _validate_square(vector: Vector, matrix: Matrix) -> None:
    """Check *matrix* is square and its dimension matches *vector*'s length."""
    if matrix.shape[0] != matrix.shape[1]:
        raise NonSquareMatrixError(matrix.shape[0], matrix.shape[1])
    if vector.size != matrix.shape[0]:
        raise DimensionMismatchError(vector.size, matrix.shape[0])


def norm(
    x: Vector | Matrix,
    ord: int | float | Literal["fro", "nuc"] | None = None,  # noqa: A002  # mirrors np.linalg.norm public API
) -> float:
    """Compute the norm of a vector or matrix, ignoring non-finite entries.

    Non-finite entries (NaN, inf) are treated as zero before computing the norm,
    so they contribute nothing to the result. Supports all ``ord`` values accepted
    by ``np.linalg.norm``.

    Args:
        x: Input array (1-D vector or 2-D matrix).
        ord: Order of the norm. See ``np.linalg.norm`` for valid values.
            Common choices: ``None`` (default 2-norm for vectors, Frobenius for
            matrices), ``1``, ``2``, ``np.inf``, ``'fro'``, ``'nuc'``.

    Returns:
        The norm as a float.

    Example:
        >>> import numpy as np
        >>> from cvx.linalg import norm
        >>> norm(np.array([3.0, np.nan, 4.0]))
        5.0
        >>> norm(np.array([[1.0, np.nan], [np.nan, 1.0]]), ord='fro')
        1.4142135623730951
    """
    return float(np.linalg.norm(np.where(np.isfinite(x), x, 0.0), ord=ord))


def a_norm(vector: Vector, matrix: Matrix | None = None) -> float:
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

    _validate_square(vector, matrix)

    mask, submatrix = valid(matrix)
    if mask.any():
        filtered_vector = vector[mask]
        return float(np.sqrt(filtered_vector @ submatrix @ filtered_vector))

    return float("nan")


def inv_a_norm(
    vector: Vector,
    matrix: Matrix | None = None,
    cond_threshold: float = DEFAULT_COND_THRESHOLD,
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

    _validate_square(vector, matrix)

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
