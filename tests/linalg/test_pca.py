"""Tests for the Principal Component Analysis (PCA) utility."""

from __future__ import annotations

import numpy as np
import pytest

from cvx.linalg import pca


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
