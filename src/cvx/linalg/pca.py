"""PCA analysis (pure NumPy implementation).

This module provides Principal Component Analysis (PCA) for dimensionality
reduction of return data. PCA is commonly used to construct factor models
for portfolio optimization.

Example:
    Perform PCA on stock returns:

    >>> import numpy as np
    >>> import polars as pl
    >>> from cvx.linalg import pca
    >>> # Create sample returns data
    >>> np.random.seed(42)
    >>> returns = pl.DataFrame(
    ...     np.random.randn(100, 5),
    ...     schema=['A', 'B', 'C', 'D', 'E']
    ... )
    >>> # Compute PCA with 3 components
    >>> result = pca(returns, n_components=3)
    >>> # Access explained variance
    >>> len(result.explained_variance)
    3
    >>> # Access factors (principal components)
    >>> result.factors.shape
    (100, 3)
    >>> # Access factor exposures (loadings)
    >>> result.exposure.shape
    (3, 5)

"""

from __future__ import annotations

from collections import namedtuple

import numpy as np
import polars as pl

PCA = namedtuple(
    "PCA",
    ["explained_variance", "factors", "exposure", "cov", "systematic", "idiosyncratic"],
)
"""Named tuple containing the results of PCA analysis.

Attributes:
    explained_variance: Explained variance ratio for each component.
        An array of shape (n_components,) where each element represents
        the proportion of total variance explained by that component.
    factors: Factor returns (principal components) as a DataFrame.
        Shape is (n_samples, n_components). Each column is a factor.
    exposure: Factor exposures (loadings) for each asset as a DataFrame.
        Shape is (n_components, n_assets). Each row contains the loadings
        of one component on all assets.
    cov: Covariance matrix of the factors as a DataFrame.
        Shape is (n_components, n_components).
    systematic: Systematic returns explained by the factors as a DataFrame.
        Shape is (n_samples, n_assets). This is the part of returns
        explained by the factor model.
    idiosyncratic: Idiosyncratic returns not explained by factors as a DataFrame.
        Shape is (n_samples, n_assets). This is the residual part of returns.

Example:
    >>> import numpy as np
    >>> import polars as pl
    >>> from cvx.linalg import pca
    >>> np.random.seed(42)
    >>> returns = pl.DataFrame(np.random.randn(50, 4))
    >>> result = pca(returns, n_components=2)
    >>> # Check explained variance sums to less than 1
    >>> result.explained_variance.sum() < 1
    True
    >>> # Systematic + idiosyncratic approximately equals original
    >>> np.allclose(
    ...     result.systematic.to_numpy() + result.idiosyncratic.to_numpy(),
    ...     returns.to_numpy(),
    ...     atol=1e-10
    ... )
    True

"""


def pca(returns: pl.DataFrame, n_components: int = 10) -> PCA:
    """Compute the first n principal components for a return matrix using SVD.

    This function performs Principal Component Analysis on asset returns to
    extract the main sources of variance. The results can be used to construct
    a factor model for portfolio optimization.

    Args:
        returns: DataFrame of asset returns with shape (n_samples, n_assets).
            Rows represent time periods, columns represent assets.
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
        Basic PCA on synthetic returns:

        >>> import numpy as np
        >>> import polars as pl
        >>> from cvx.linalg import pca
        >>> np.random.seed(42)
        >>> # Create returns with 100 periods and 10 assets
        >>> returns = pl.DataFrame(np.random.randn(100, 10))
        >>> result = pca(returns, n_components=3)
        >>> # First component explains most variance
        >>> bool(result.explained_variance[0] > result.explained_variance[1])
        True
        >>> # Factors are orthogonal
        >>> factor_corr = np.corrcoef(result.factors.to_numpy().T)
        >>> bool(np.allclose(factor_corr, np.eye(3), atol=0.1))
        True

        Verifying variance decomposition (systematic + idiosyncratic = total):

        >>> np.random.seed(123)
        >>> returns = pl.DataFrame(np.random.randn(50, 5))
        >>> result = pca(returns, n_components=3)
        >>> # Systematic variance + idiosyncratic variance ≈ total variance
        >>> total_var = np.var(returns.to_numpy(), axis=0, ddof=1).sum()
        >>> systematic_var = np.var(result.systematic.to_numpy(), axis=0, ddof=1).sum()
        >>> idio_var = np.var(result.idiosyncratic.to_numpy(), axis=0, ddof=1).sum()
        >>> # Note: small differences due to demeaning
        >>> bool(np.isclose(systematic_var + idio_var, total_var, rtol=0.1))
        True

        Exposure matrix has orthonormal rows (loadings are orthogonal):

        >>> np.random.seed(42)
        >>> returns = pl.DataFrame(np.random.randn(100, 6))
        >>> result = pca(returns, n_components=3)
        >>> # V^T @ V should be identity (orthonormal loadings)
        >>> VtV = result.exposure.to_numpy() @ result.exposure.to_numpy().T
        >>> bool(np.allclose(VtV, np.eye(3), atol=1e-10))
        True

        Explained variance is ordered (first component explains most):

        >>> all(result.explained_variance[i] >= result.explained_variance[i+1]
        ...     for i in range(len(result.explained_variance)-1))
        True

        Reconstructing returns from factors and exposures:

        >>> # systematic = factors @ exposure (plus mean)
        >>> reconstructed = result.factors.to_numpy() @ result.exposure.to_numpy()
        >>> # Should match systematic (centered part)
        >>> centered_systematic = result.systematic.to_numpy() - returns.to_numpy().mean(axis=0)
        >>> bool(np.allclose(reconstructed, centered_systematic, atol=1e-10))
        True

    """
    # Demean the returns
    x = returns.to_numpy()
    x_mean = x.mean(axis=0)
    x_centered = x - x_mean

    # Singular Value Decomposition
    # x = u s V^T, where columns of V are principal axes
    u, s_full, vt = np.linalg.svd(x_centered, full_matrices=False)

    # Take only the first n components
    u = u[:, :n_components]
    s = s_full[:n_components]
    vt = vt[:n_components, :]

    pc_names = [f"PC{i + 1}" for i in range(n_components)]

    # Factor exposures (loadings): each component's weight per asset
    exposure = pl.DataFrame(vt, schema=returns.columns)

    # Factor returns (scores): projection of data onto components
    factors = pl.DataFrame(u * s, schema=pc_names)

    # Explained variance ratio (normalize by total variance across ALL components)
    explained_variance = (s**2) / np.sum(s_full**2)

    # Covariance of factor returns
    cov = pl.DataFrame(np.cov((u * s).T), schema=pc_names)

    # Systematic + Idiosyncratic returns
    systematic = pl.DataFrame(
        (u * s) @ vt + x_mean,
        schema=returns.columns,
    )
    idiosyncratic = pl.DataFrame(
        x_centered - (u * s) @ vt,
        schema=returns.columns,
    )

    return PCA(
        explained_variance=explained_variance,
        factors=factors,
        exposure=exposure,
        cov=cov,
        systematic=systematic,
        idiosyncratic=idiosyncratic,
    )
