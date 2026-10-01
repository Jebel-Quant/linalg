"""Condition-number helpers: the NaN-aware ``cond`` and the ill-conditioning warning checks."""

from __future__ import annotations

import warnings
from typing import Literal

import numpy as np

from .exceptions import IllConditionedMatrixWarning
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
