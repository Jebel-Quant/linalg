"""Random covariance matrix generation utilities.

This module provides functions for generating random positive semi-definite
covariance matrices. These are useful for testing and simulation purposes.

Example:
    Generate a random covariance matrix:

    >>> import numpy as np
    >>> from cvx.linalg import rand_cov
    >>> # Generate a 5x5 random covariance matrix
    >>> cov = rand_cov(5, seed=42)
    >>> cov.shape
    (5, 5)
    >>> # Verify it's symmetric
    >>> bool(np.allclose(cov, cov.T))
    True
    >>> # Verify it's positive semi-definite
    >>> bool(np.all(np.linalg.eigvals(cov) >= -1e-10))
    True

"""

from __future__ import annotations

import numpy as np

from ..core.types import Matrix


def rand_cov(n: int, seed: int | None = None) -> Matrix:
    """Construct a random positive semi-definite covariance matrix of size n x n.

    The matrix is constructed as A^T @ A where A is a random n x n matrix with
    elements drawn from a standard normal distribution. This ensures the result
    is symmetric and positive semi-definite.

    Args:
        n: Size of the covariance matrix (n x n).
        seed: Random seed for reproducibility. If None, uses the current
            random state.

    Returns:
        A random positive semi-definite n x n covariance matrix.

    Example:
        Generate a reproducible random covariance matrix:

        >>> import numpy as np
        >>> from cvx.linalg import rand_cov
        >>> cov1 = rand_cov(3, seed=42)
        >>> cov2 = rand_cov(3, seed=42)
        >>> np.allclose(cov1, cov2)
        True

        Verify positive definiteness via Cholesky decomposition:

        >>> cov = rand_cov(5, seed=123)
        >>> # If Cholesky succeeds without error, matrix is positive definite
        >>> L = np.linalg.cholesky(cov)
        >>> bool(np.allclose(L @ L.T, cov))
        True

        Eigenvalue verification:

        >>> cov = rand_cov(3, seed=99)
        >>> eigenvalues = np.linalg.eigvalsh(cov)
        >>> # All eigenvalues should be positive for PD matrix
        >>> bool(np.all(eigenvalues > 0))
        True

        Different seeds produce different matrices:

        >>> cov1 = rand_cov(3, seed=1)
        >>> cov2 = rand_cov(3, seed=2)
        >>> bool(not np.allclose(cov1, cov2))
        True

        Without seed, consecutive calls may differ (random state):

        >>> # These may or may not be equal depending on random state
        >>> cov_a = rand_cov(2, seed=None)
        >>> cov_b = rand_cov(2, seed=None)
        >>> cov_a.shape == cov_b.shape == (2, 2)
        True

    Note:
        The generated matrix is guaranteed to be positive semi-definite because
        it is constructed as A^T @ A. In practice, it will typically be positive
        definite (all eigenvalues strictly positive) unless n is very large.

    """
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((n, n))
    return np.transpose(a) @ a
