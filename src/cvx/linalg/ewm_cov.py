"""Exponentially weighted covariance matrix computation."""

from __future__ import annotations

from collections.abc import Hashable

import numpy as np


class NegativeWarmupError(ValueError):
    """Raised when warmup is a negative integer."""


def _ewm_mean_1d(x: np.ndarray, alpha: float, min_samples: int) -> np.ndarray:
    """EWM mean (adjust=True) of a 1-D array, skipping NaN values."""
    n_obs = len(x)
    result = np.full(n_obs, np.nan)
    numer = denom = 0.0
    count = 0
    one_minus = 1.0 - alpha
    for t in range(n_obs):
        v = x[t]
        if not np.isnan(v):
            numer = alpha * v + one_minus * numer
            denom = alpha + one_minus * denom
            count += 1
        else:
            numer *= one_minus
            denom *= one_minus
        if count >= min_samples:
            result[t] = numer / denom
    return result


def ewm_covariance(
    returns: np.ndarray,
    index: list | np.ndarray | None = None,
    window: int = 30,
    is_halflife: bool = False,
    warmup: int = 0,
) -> dict[Hashable, np.ndarray]:
    """Compute the exponentially weighted covariance matrix of returns.

    Uses the identity ``Cov(X, Y) = EWM(X*Y) - EWM(X)*EWM(Y)`` on the
    common non-null observations of each pair, equivalent to
    ``pandas.DataFrame.ewm(span).cov(bias=True)``.

    Args:
        returns: Array of shape ``(T, N)`` where rows are time steps and
            columns are assets.  Use ``np.nan`` for missing observations.
            A 1-D array is treated as a single asset.
        index: Optional length-``T`` sequence of index keys (e.g. dates or
            integers).  Defaults to ``range(T)``.
        window: Span (default) or half-life (when *is_halflife* is ``True``)
            of the exponential decay.  Defaults to ``30``.
        is_halflife: When ``True`` *window* is interpreted as the half-life;
            otherwise it is the EWMA span.  Defaults to ``False``.
        warmup: Minimum number of common observations required before a
            pair's cell is non-NaN.  Defaults to ``0``.

    Returns:
        Dictionary keyed by index value mapping to a square symmetric
        ``numpy.ndarray`` of shape ``(N, N)``.  Rows with no finite cell
        are omitted.

    """
    if isinstance(warmup, bool) or not isinstance(warmup, int):
        raise TypeError
    if warmup < 0:
        raise NegativeWarmupError

    arr = np.asarray(returns, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, None]
    n_time, n_assets = arr.shape

    if index is None:
        index = range(n_time)

    alpha = 1.0 - np.exp(-np.log(2.0) / window) if is_halflife else 2.0 / (window + 1)
    min_samples = 1 if warmup == 0 else warmup

    cube = np.full((n_time, n_assets, n_assets), np.nan)
    for i in range(n_assets):
        for j in range(i, n_assets):
            xi = arr[:, i]
            xj = arr[:, j]
            xi_masked = np.where(np.isnan(xj), np.nan, xi)
            xj_masked = np.where(np.isnan(xi), np.nan, xj)
            cov = _ewm_mean_1d(xi_masked * xj_masked, alpha, min_samples) - _ewm_mean_1d(
                xi_masked, alpha, min_samples
            ) * _ewm_mean_1d(xj_masked, alpha, min_samples)
            cube[:, i, j] = cov
            cube[:, j, i] = cov

    has_data = ~np.all(np.isnan(cube), axis=(1, 2))
    return {k: cube[t] for t, k in enumerate(index) if has_data[t]}
