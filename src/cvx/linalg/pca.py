"""PCA analysis (pure NumPy implementation).

This module provides Principal Component Analysis (PCA) for dimensionality
reduction of return data. PCA is commonly used to construct factor models
for portfolio optimization.

Example:
    Perform PCA on stock returns:

    >>> import numpy as np
    >>> from cvx.linalg import pca
    >>> np.random.seed(42)
    >>> returns = np.random.randn(100, 5)
    >>> result = pca(returns, n_components=3)
    >>> len(result.explained_variance)
    3
    >>> result.factors.shape
    (100, 3)
    >>> result.exposure.shape
    (3, 5)

"""

from __future__ import annotations

from collections import namedtuple

import numpy as np
import numpy.typing as npt

Matrix = npt.NDArray[np.float64]

PCA = namedtuple(
    "PCA",
    ["explained_variance", "factors", "exposure", "cov", "systematic", "idiosyncratic"],
)
"""Named tuple containing the results of PCA analysis.

Attributes:
    explained_variance: Explained variance ratio for each component.
        Shape (n_components,).
    factors: Factor returns (principal components). Shape (n_samples, n_components).
    exposure: Factor exposures (loadings). Shape (n_components, n_assets).
    cov: Covariance matrix of the factors. Shape (n_components, n_components).
    systematic: Returns explained by the factors. Shape (n_samples, n_assets).
    idiosyncratic: Residual returns not explained by factors. Shape (n_samples, n_assets).

Example:
    >>> import numpy as np
    >>> from cvx.linalg import pca
    >>> np.random.seed(42)
    >>> returns = np.random.randn(50, 4)
    >>> result = pca(returns, n_components=2)
    >>> result.explained_variance.sum() < 1
    True
    >>> np.allclose(result.systematic + result.idiosyncratic, returns, atol=1e-10)
    True

"""


def pca(returns: Matrix, n_components: int = 10) -> PCA:
    """Compute the first n principal components for a return matrix using SVD.

    Args:
        returns: Array of asset returns with shape (n_samples, n_assets).
        n_components: Number of principal components to extract. Defaults to 10.

    Returns:
        PCA named tuple containing:
            - explained_variance: Ratio of variance explained by each component
            - factors: Factor returns (scores)
            - exposure: Factor exposures (loadings)
            - cov: Factor covariance matrix
            - systematic: Returns explained by factors
            - idiosyncratic: Residual returns

    Example:
        >>> import numpy as np
        >>> from cvx.linalg import pca
        >>> np.random.seed(42)
        >>> returns = np.random.randn(100, 10)
        >>> result = pca(returns, n_components=3)
        >>> bool(result.explained_variance[0] > result.explained_variance[1])
        True
        >>> factor_corr = np.corrcoef(result.factors.T)
        >>> bool(np.allclose(factor_corr, np.eye(3), atol=0.1))
        True
        >>> VtV = result.exposure @ result.exposure.T
        >>> bool(np.allclose(VtV, np.eye(3), atol=1e-10))
        True
        >>> all(result.explained_variance[i] >= result.explained_variance[i+1]
        ...     for i in range(len(result.explained_variance)-1))
        True
        >>> reconstructed = result.factors @ result.exposure
        >>> centered_systematic = result.systematic - returns.mean(axis=0)
        >>> bool(np.allclose(reconstructed, centered_systematic, atol=1e-10))
        True

    """
    x_mean = returns.mean(axis=0)
    x_centered = returns - x_mean

    u, s_full, vt = np.linalg.svd(x_centered, full_matrices=False)

    u = u[:, :n_components]
    s = s_full[:n_components]
    vt = vt[:n_components, :]

    factors: Matrix = u * s
    exposure: Matrix = vt
    explained_variance: Matrix = (s**2) / np.sum(s_full**2)
    cov: Matrix = np.cov(factors.T)
    systematic: Matrix = factors @ vt + x_mean
    idiosyncratic: Matrix = x_centered - factors @ vt

    return PCA(
        explained_variance=explained_variance,
        factors=factors,
        exposure=exposure,
        cov=cov,
        systematic=systematic,
        idiosyncratic=idiosyncratic,
    )
