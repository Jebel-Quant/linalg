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
    cond = float(np.linalg.cond(matrix))
    if cond > threshold:
        warnings.warn(
            f"Matrix condition number {cond:.3e} exceeds threshold {threshold:.3e}; "
            "results may be numerically unreliable.",
            IllConditionedMatrixWarning,
            stacklevel=3,
        )
