"""Tests for the Principal Component Analysis (PCA) utility."""

from __future__ import annotations

import numpy as np
import pytest

from cvx.linalg import InvalidComponentsError, pca


@pytest.fixture
def returns(resource_dir) -> np.ndarray:
    """Pytest fixture that provides stock return data for testing.

    Loads stock prices from CSV, computes percentage returns, and fills
    the first row of NaNs (from pct_change) with zeros.

    Args:
        resource_dir: Pytest fixture providing the path to the test resources directory

    Returns:
        numpy.ndarray: Array containing stock returns

    """
    prices = np.genfromtxt(
        resource_dir / "stock_prices.csv",
        delimiter=",",
        skip_header=1,
        usecols=range(1, 21),
    )
    ret = np.zeros_like(prices)
    ret[1:] = prices[1:] / prices[:-1] - 1.0
    return np.nan_to_num(ret, nan=0.0)


def test_pca(returns: np.ndarray) -> None:
    """Test that the pca function correctly calculates the principal components.

    This test verifies that:
    1. The pca function can process returns data
    2. The explained variance ratios match the expected values

    Args:
        returns: Pytest fixture providing stock return data

    """
    xxx = pca(returns)

    assert np.allclose(
        xxx.explained_variance,
        np.array(
            [
                0.33383825,
                0.19039608,
                0.11567561,
                0.07965253,
                0.06379108,
                0.04580062,
                0.03461307,
                0.02205145,
                0.01876345,
                0.01757195,
            ]
        ),
    )


def test_pca_too_many_components_raises(returns: np.ndarray) -> None:
    """Test that requesting more components than assets raises InvalidComponentsError."""
    with pytest.raises(InvalidComponentsError):
        pca(returns, n_components=returns.shape[1] + 1)


def test_pca_zero_components_raises(returns: np.ndarray) -> None:
    """Test that requesting zero components raises InvalidComponentsError."""
    with pytest.raises(InvalidComponentsError):
        pca(returns, n_components=0)


def test_pca_single_component_cov_is_2d(returns: np.ndarray) -> None:
    """Test that the factor covariance matrix is 2-D even for a single component."""
    result = pca(returns, n_components=1)
    assert result.cov.shape == (1, 1)


def test_pca_result_type_and_exposure() -> None:
    """pca() returns a namedtuple called PCA whose exposure reconstructs the fit."""
    rng = np.random.default_rng(0)
    returns = rng.standard_normal((30, 4))
    result = pca(returns, n_components=2)
    assert type(result).__name__ == "PCA"
    assert result.exposure.shape == (2, 4)
    np.testing.assert_allclose(
        result.factors @ result.exposure + returns.mean(axis=0),
        result.systematic,
        atol=1e-12,
    )


def test_pca_centers_returns_by_column_mean() -> None:
    """pca() centers returns by the per-column mean before decomposing.

    Adding a per-column constant to every row of the input must leave the
    mean-centered structure — explained variance and idiosyncratic residuals —
    unchanged, while ``systematic`` shifts by exactly that constant. This pins
    the centering behaviour through observable outputs alone, without reaching
    into how pca performs the decomposition.
    """
    rng = np.random.default_rng(1)
    returns = rng.standard_normal((40, 5))
    shift = rng.standard_normal(5)  # constant added to every row

    base = pca(returns, n_components=2)
    shifted = pca(returns + shift, n_components=2)

    np.testing.assert_allclose(shifted.explained_variance, base.explained_variance, atol=1e-10)
    np.testing.assert_allclose(shifted.idiosyncratic, base.idiosyncratic, atol=1e-10)
    np.testing.assert_allclose(shifted.systematic, base.systematic + shift, atol=1e-10)
