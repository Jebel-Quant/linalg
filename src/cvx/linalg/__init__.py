"""Linear algebra utilities for risk models.

This subpackage provides linear algebra utilities commonly used in risk modeling,
including Cholesky decomposition, Principal Component Analysis, and matrix
validation.

Example:
    >>> import numpy as np
    >>> from cvx.linalg import cholesky, pca, rand_cov, valid
    >>> # Cholesky decomposition
    >>> cov = np.array([[4.0, 2.0], [2.0, 5.0]])
    >>> R = cholesky(cov)
    >>> np.allclose(R.T @ R, cov)
    True

Functions:
    cholesky: Compute upper triangular Cholesky decomposition
    pca: Compute principal components of return data
    rand_cov: Generate a random positive semi-definite covariance matrix
    valid: Extract valid submatrix from a matrix with NaN values

"""

from .cholesky import cholesky as cholesky
from .pca import pca as pca
from .rand_cov import rand_cov as rand_cov
from .valid import valid as valid
