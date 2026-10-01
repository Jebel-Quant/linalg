"""Domain-specific exceptions and warnings for cvx.linalg."""

from __future__ import annotations


class NotAMatrixError(TypeError):
    """Raised when a 2-D matrix is required but the input has a different number of dimensions.

    Args:
        ndim: Actual number of dimensions of the offending array.
        func: Name of the function that rejected the input.

    Examples:
        >>> raise NotAMatrixError(3)
        Traceback (most recent call last):
            ...
        cvx.linalg.core.exceptions.NotAMatrixError: eigvals() expected a 2-D matrix, got 3-D input.
        >>> raise NotAMatrixError(3, func="qr")
        Traceback (most recent call last):
            ...
        cvx.linalg.core.exceptions.NotAMatrixError: qr() expected a 2-D matrix, got 3-D input.
    """

    def __init__(self, ndim: int, func: str = "eigvals") -> None:
        """Initialize with the actual number of dimensions and the rejecting function."""
        super().__init__(f"{func}() expected a 2-D matrix, got {ndim}-D input.")
        self.ndim = ndim
        self.func = func


class NonSquareMatrixError(ValueError):
    """Raised when a square matrix is required but the input is not square.

    Args:
        rows: Number of rows in the offending matrix.
        cols: Number of columns in the offending matrix.

    Examples:
        >>> raise NonSquareMatrixError(3, 2)
        Traceback (most recent call last):
            ...
        cvx.linalg.core.exceptions.NonSquareMatrixError: Matrix must be square, got shape (3, 2).
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
        cvx.linalg.core.exceptions.DimensionMismatchError: Vector length 3 does not match matrix dimension 2.
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
        cvx.linalg.core.exceptions.SingularMatrixError: Matrix is singular and cannot be solved.
    """

    def __init__(self, detail: str = "") -> None:
        """Initialize with an optional extra detail string."""
        msg = "Matrix is singular and cannot be solved."
        if detail:
            msg = f"{msg} {detail}"
        super().__init__(msg)


class NegativeWarmupError(ValueError):
    """Raised when a negative warmup period is requested.

    Args:
        warmup: The offending warmup value.

    Examples:
        >>> raise NegativeWarmupError(-3)
        Traceback (most recent call last):
            ...
        cvx.linalg.core.exceptions.NegativeWarmupError: warmup must be non-negative, got -3.
    """

    def __init__(self, warmup: int | None = None) -> None:
        """Initialize with the offending warmup value."""
        msg = "warmup must be non-negative."
        if warmup is not None:
            msg = f"warmup must be non-negative, got {warmup}."
        super().__init__(msg)
        self.warmup = warmup


class NonIntegerWarmupError(TypeError):
    """Raised when warmup is not an integer (booleans are rejected as well).

    Args:
        value: The offending warmup value.

    Examples:
        >>> raise NonIntegerWarmupError(True)
        Traceback (most recent call last):
            ...
        cvx.linalg.core.exceptions.NonIntegerWarmupError: warmup must be an integer, got bool.
    """

    def __init__(self, value: object) -> None:
        """Initialize with the offending warmup value."""
        super().__init__(f"warmup must be an integer, got {type(value).__name__}.")
        self.value = value


class InvalidComponentsError(ValueError):
    """Raised when the requested number of principal components is out of range.

    Args:
        n_components: The requested number of components.
        max_components: The largest number of components supported by the data.

    Examples:
        >>> raise InvalidComponentsError(10, 5)
        Traceback (most recent call last):
            ...
        cvx.linalg.core.exceptions.InvalidComponentsError: n_components must be between 1 and 5, got 10.
    """

    def __init__(self, n_components: int, max_components: int) -> None:
        """Initialize with the requested and maximum number of components."""
        super().__init__(f"n_components must be between 1 and {max_components}, got {n_components}.")
        self.n_components = n_components
        self.max_components = max_components


class IllConditionedMatrixWarning(UserWarning):
    """Emitted when a matrix condition number exceeds a configurable threshold.

    Examples:
        >>> import warnings
        >>> with warnings.catch_warnings(record=True) as w:
        ...     warnings.simplefilter("always")
        ...     warnings.warn("condition number 1e13", IllConditionedMatrixWarning)
        ...     issubclass(w[-1].category, IllConditionedMatrixWarning)
        True
    """
