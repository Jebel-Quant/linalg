"""Least-squares solver with NaN-aware row filtering."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from ..core.exceptions import DEFAULT_COND_THRESHOLD, DimensionMismatchError
from ..core.exceptions import warn_ill_conditioned as _warn_ill_conditioned
from ..core.types import Matrix, Vector


def _condition_number(sv: npt.NDArray[np.floating]) -> float:
    """Condition number ``sv[0] / sv[-1]`` from descending singular values.

    Returns ``inf`` when the smallest singular value is zero and ``1.0`` when
    there are no singular values (an empty valid sub-matrix).
    """
    if sv.size == 0:
        return 1.0
    if sv[-1] > 0:
        return float(sv[0] / sv[-1])
    return float("inf")


def lstsq(
    matrix: Matrix,
    rhs: Vector,
    cond_threshold: float = DEFAULT_COND_THRESHOLD,
) -> tuple[Vector, Vector, int, Vector]:
    """Solve an overdetermined or underdetermined system in the least-squares sense.

    Rows where any entry in *matrix* or the corresponding entry in *rhs* is
    non-finite are excluded before solving.  The returned solution vector
    always has length equal to the number of columns in *matrix*.  When the
    effective condition number of the valid sub-matrix exceeds
    *cond_threshold*, an ``IllConditionedMatrixWarning`` is emitted.

    Args:
        matrix: Coefficient matrix of shape ``(m, n)``.
        rhs: Right-hand side vector of length ``m``.
        cond_threshold: Condition-number threshold above which a warning is
            emitted. Defaults to ``1e12``.

    Returns:
        A four-tuple ``(x, residuals, rank, sv)`` matching the convention of
        :func:`numpy.linalg.lstsq`:

        - ``x`` — least-squares solution of shape ``(n,)``.
        - ``residuals`` — sum of squared residuals; empty when the solution is
          not unique or all rows are invalid.
        - ``rank`` — effective rank of the valid sub-matrix.
        - ``sv`` — singular values of the valid sub-matrix in descending order.

    Raises:
        DimensionMismatchError: If ``rhs`` length does not match the number of
            rows in *matrix*.

    Example:
        >>> import numpy as np
        >>> from cvx.linalg import lstsq
        >>> A = np.array([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]])
        >>> b = np.array([6.0, 5.0, 7.0])
        >>> x, res, rank, sv = lstsq(A, b)
        >>> int(rank)
        2

        NaN rows are silently dropped:

        >>> A_nan = np.array([[1.0, 1.0], [np.nan, 2.0], [1.0, 3.0]])
        >>> b_nan = np.array([6.0, 5.0, 7.0])
        >>> x2, _, rank2, _ = lstsq(A_nan, b_nan)
        >>> int(rank2)
        2
    """
    if rhs.shape[0] != matrix.shape[0]:
        raise DimensionMismatchError(rhs.shape[0], matrix.shape[0])

    n_cols = matrix.shape[1]

    # Filter rows that contain any non-finite value in matrix or rhs.
    row_mask = np.isfinite(matrix).all(axis=1) & np.isfinite(rhs)
    sub_matrix = matrix[row_mask]
    sub_rhs = rhs[row_mask]

    if sub_matrix.shape[0] == 0:
        return np.full(n_cols, np.nan), np.array([]), 0, np.array([])

    x, residuals, rank, sv = np.linalg.lstsq(sub_matrix, sub_rhs, rcond=None)

    _warn_ill_conditioned(_condition_number(sv), cond_threshold)

    return (
        x.astype(np.float64, copy=False),
        residuals.astype(np.float64, copy=False),
        int(rank),
        sv.astype(np.float64, copy=False),
    )
