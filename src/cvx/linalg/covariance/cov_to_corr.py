"""Convert a covariance matrix to a correlation matrix."""

from __future__ import annotations

import numpy as np

from ..core.types import Matrix


def cov_to_corr(cov: Matrix, min_var: float = 1e-14) -> Matrix:
    """Convert a covariance matrix to a correlation matrix.

    Off-diagonal entries are symmetrised by averaging the upper and lower
    triangles, so floating-point asymmetry in *cov* does not propagate.
    Diagonal entries are set to ``1.0`` when the variance is above *min_var*
    and to ``nan`` otherwise.  All entries are clipped to ``[-1, 1]``.

    Args:
        cov: Square covariance matrix of shape ``(N, N)``.
        min_var: Threshold below which a variance is treated as zero;
            the corresponding row and column are filled with ``nan``.
            Defaults to ``1e-14``.

    Returns:
        Symmetrised correlation matrix of shape ``(N, N)`` with diagonal
        entries in ``{1.0, nan}``.

    Example:
        >>> import numpy as np
        >>> from cvx.linalg import cov_to_corr
        >>> cov = np.array([[4.0, 2.0], [2.0, 9.0]])
        >>> corr = cov_to_corr(cov)
        >>> np.allclose(np.diag(corr), [1.0, 1.0])
        True
        >>> float(round(corr[0, 1], 6))
        0.333333
    """
    var = np.diag(cov)
    denom = np.sqrt(np.outer(var, var))
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = np.where(denom > min_var, cov / denom, np.nan)
    corr = np.clip(corr, -1.0, 1.0)
    n = len(var)
    idx = np.arange(n)
    corr[idx, idx] = np.where(var > min_var, 1.0, np.nan)
    tril_i, tril_j = np.tril_indices(n, k=-1)
    avg = 0.5 * (corr[tril_i, tril_j] + corr[tril_j, tril_i])
    corr[tril_i, tril_j] = avg
    corr[tril_j, tril_i] = avg
    return corr
