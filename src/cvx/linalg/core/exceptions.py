"""Domain-specific exceptions and warnings for cvx.linalg."""

from __future__ import annotations

import warnings
from typing import Literal

import numpy as np

from .types import Matrix

try:  # SciPy lets check_and_warn_condition skip the SVD for well-conditioned SPD input.
    from scipy.linalg import cho_factor as _cho_factor  # type: ignore[import-untyped]
    from scipy.linalg.lapack import get_lapack_funcs as _get_lapack_funcs  # type: ignore[import-untyped]

    _HAVE_SCIPY = True
except ImportError:  # pragma: no cover - depends on the environment; the fallback is tested by patching the flag
    _HAVE_SCIPY = False

_SCREEN_MARGIN: float = 10.0
"""Safety factor on the LAPACK condition estimate before it may stand in for the exact 2-norm value."""

DEFAULT_COND_THRESHOLD: float = 1e12
"""Default condition-number threshold above which an IllConditionedMatrixWarning is emitted."""


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


def cond(matrix: Matrix, p: int | float | Literal["fro", "nuc"] | None = None) -> float:
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


def warn_ill_conditioned(cond_value: float, threshold: float | None, stacklevel: int = 3) -> None:
    """Emit IllConditionedMatrixWarning when *cond_value* exceeds *threshold*.

    Args:
        cond_value: Condition number to compare against the threshold.
        threshold: Upper bound before a warning is issued, or ``None`` to never warn.
        stacklevel: Stack level passed to :func:`warnings.warn` so the warning
            points at the caller of the public API. Defaults to ``3``.

    Example:
        >>> import warnings
        >>> with warnings.catch_warnings(record=True) as w:
        ...     warnings.simplefilter("always")
        ...     warn_ill_conditioned(2.0, 0.5)
        ...     len(w)
        1
    """
    if threshold is not None and cond_value > threshold:
        warnings.warn(
            f"Matrix condition number {cond_value:.3e} exceeds threshold {threshold:.3e}; "
            "results may be numerically unreliable.",
            IllConditionedMatrixWarning,
            stacklevel=stacklevel,
        )


def check_and_warn_condition(matrix: Matrix, threshold: float | None) -> None:
    """Emit IllConditionedMatrixWarning when the condition number exceeds threshold.

    Args:
        matrix: Square matrix whose condition number is checked.
        threshold: Upper bound before a warning is issued, or ``None`` to skip
            the check entirely.

    The condition number is the 2-norm one, as from :func:`cond`. With SciPy
    installed, a symmetric positive-definite matrix is first screened by a
    LAPACK estimate from its Cholesky factor; when that proves the matrix well
    within *threshold*, the full SVD is skipped.

    Example:
        >>> import numpy as np
        >>> import warnings
        >>> with warnings.catch_warnings(record=True) as w:
        ...     warnings.simplefilter("always")
        ...     check_and_warn_condition(np.eye(2), 0.5)
        ...     len(w)
        1
    """
    if threshold is None or _certainly_within(matrix, threshold):
        return
    warn_ill_conditioned(cond(matrix), threshold, stacklevel=4)


def _certainly_within(matrix: Matrix, threshold: float) -> bool:
    """Return True when a cheap estimate proves the 2-norm condition number is at most *threshold*.

    For a symmetric matrix ``||A||_2 <= ||A||_1``, so the 1-norm condition
    number bounds the 2-norm one from above. LAPACK ``pocon`` estimates the
    former from a Cholesky factor for about the cost of one more solve, whereas
    the exact 2-norm value needs a full SVD. The estimate can fall short of the
    true 1-norm value, hence the :data:`_SCREEN_MARGIN` safety factor.

    ``False`` means only "not proven": SciPy is missing, or the matrix is not
    symmetric positive-definite, or the estimate is too close to the threshold.
    The caller then computes the exact condition number.

    Args:
        matrix: Square matrix whose condition number is screened.
        threshold: Upper bound the condition number is tested against.

    Returns:
        ``True`` if the 2-norm condition number is certainly at most *threshold*.
    """
    if not _HAVE_SCIPY or not np.array_equal(matrix, matrix.T):
        return False
    try:
        factor, lower = _cho_factor(matrix)
    except (np.linalg.LinAlgError, ValueError):  # not positive-definite, or not finite
        return False
    (pocon,) = _get_lapack_funcs(("pocon",), (factor,))
    rcond, _info = pocon(factor, np.linalg.norm(matrix, 1), uplo="L" if lower else "U")
    return bool(rcond * threshold >= _SCREEN_MARGIN)
