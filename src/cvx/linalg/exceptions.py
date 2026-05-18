"""Domain-specific exceptions and warnings for cvx.linalg."""

from __future__ import annotations

import warnings

import numpy as np


class NonSquareMatrixError(ValueError):
    """Raised when a square matrix is required but the input is not square.

    Args:
        rows: Number of rows in the offending matrix.
        cols: Number of columns in the offending matrix.

    Examples:
        >>> raise NonSquareMatrixError(3, 2)
        Traceback (most recent call last):
            ...
        cvx.linalg.exceptions.NonSquareMatrixError: Matrix must be square, got shape (3, 2).
    """

    def __init__(self, rows: int, cols: int) -> None:
        """Initialize with the offending matrix shape."""
        super().__init__(f"Matrix must be square, got shape ({rows}, {cols}).")
        self.rows = rows
        self.cols = cols


class DimensionMismatchError(ValueError):
    """Raised when vector and matrix dimensions are incompatible.

    Args:
        vector_size: Length of the offending vector.
        matrix_size: Expected dimension inferred from the matrix.

    Examples:
        >>> raise DimensionMismatchError(3, 2)
        Traceback (most recent call last):
            ...
        cvx.linalg.exceptions.DimensionMismatchError: Vector length 3 does not match matrix dimension 2.
    """

    def __init__(self, vector_size: int, matrix_size: int) -> None:
        """Initialize with the offending vector and matrix sizes."""
        super().__init__(f"Vector length {vector_size} does not match matrix dimension {matrix_size}.")
        self.vector_size = vector_size
        self.matrix_size = matrix_size


class SingularMatrixError(ValueError):
    """Raised when a matrix is (numerically) singular and cannot be inverted.

    Args:
        detail: Optional extra detail string to append to the message.

    Examples:
        >>> raise SingularMatrixError()
        Traceback (most recent call last):
            ...
        cvx.linalg.exceptions.SingularMatrixError: Matrix is singular and cannot be solved.
    """

    def __init__(self, detail: str = "") -> None:
        """Initialize with an optional extra detail string."""
        msg = "Matrix is singular and cannot be solved."
        if detail:
            msg = f"{msg} {detail}"
        super().__init__(msg)


class IllConditionedMatrixWarning(UserWarning):
    """Emitted when a matrix condition number exceeds a configurable threshold.

    Examples:
        >>> import warnings
        >>> warnings.warn("condition number 1e13", IllConditionedMatrixWarning)
    """


def cond(matrix: np.ndarray, p: int | float | str | None = None) -> float:
    """Return the condition number of a matrix.

    Returns ``nan`` if the matrix contains any non-finite (NaN or inf) entries.
    Otherwise delegates to :func:`numpy.linalg.cond`.

    Args:
        matrix: Input matrix.
        p: Order of the norm used to compute the condition number.
            Accepts the same values as :func:`numpy.linalg.cond`
            (``None``, ``1``, ``-1``, ``2``, ``-2``, ``numpy.inf``,
            ``-numpy.inf``, ``'fro'``).  Defaults to ``None`` which
            corresponds to the 2-norm (largest singular value divided by
            the smallest).

    Returns:
        The condition number as a ``float``, or ``nan`` when the matrix
        contains non-finite entries.

    Examples:
        >>> import numpy as np
        >>> cond(np.eye(3))
        1.0
        >>> import math
        >>> math.isnan(cond(np.array([[float('nan'), 1.0], [1.0, 2.0]])))
        True
        >>> cond(np.diag([1.0, 1e10]), p=1)
        10000000000.0
    """
    if not np.all(np.isfinite(matrix)):
        return float("nan")
    return float(np.linalg.cond(matrix, p=p))


def check_and_warn_condition(matrix: np.ndarray, threshold: float) -> None:
    """Emit IllConditionedMatrixWarning when the condition number exceeds threshold.

    Args:
        matrix: Square matrix whose condition number is checked.
        threshold: Upper bound before a warning is issued.

    Example:
        >>> import numpy as np
        >>> import warnings
        >>> with warnings.catch_warnings(record=True) as w:
        ...     warnings.simplefilter("always")
        ...     check_and_warn_condition(np.eye(2), 0.5)
        ...     len(w)
        1
    """
    c = cond(matrix)
    if c > threshold:
        warnings.warn(
            f"Matrix condition number {c:.3e} exceeds threshold {threshold:.3e}; "
            "results may be numerically unreliable.",
            IllConditionedMatrixWarning,
            stacklevel=3,
        )
