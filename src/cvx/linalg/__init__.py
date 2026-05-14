"""Linear algebra utilities for risk models.

This subpackage provides linear algebra utilities commonly used in risk modeling,
including Cholesky decomposition, Principal Component Analysis, and matrix
validation.

Example:
    >>> import numpy as np
    >>> from cvx.linalg import a_norm, cholesky, pca, rand_cov, solve, valid
    >>> # Cholesky decomposition
    >>> cov = np.array([[4.0, 2.0], [2.0, 5.0]])
    >>> R = cholesky(cov)
    >>> np.allclose(R.T @ R, cov)
    True

Functions:
    a_norm: Compute the matrix norm of a vector
    cholesky: Compute upper triangular Cholesky decomposition
    pca: Compute principal components of return data
    rand_cov: Generate a random positive semi-definite covariance matrix
    solve: Solve linear systems with NaN-aware matrix filtering
    valid: Extract valid submatrix from a matrix with NaN values

"""

from .cholesky import cholesky as cholesky
from .norm import a_norm as a_norm
from .norm import inv_a_norm as inv_a_norm
from .pca import pca as pca
from .rand_cov import rand_cov as rand_cov
from .solve import solve as solve
from .types import Matrix as Matrix
from .valid import valid as valid
