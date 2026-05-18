"""Linear algebra utilities for risk models.

This subpackage provides linear algebra utilities commonly used in risk modeling,
including Cholesky decomposition, Principal Component Analysis, matrix norms,
linear-system solving, matrix validation, and EWM covariance estimation.

Example:
    >>> import numpy as np
    >>> from cvx.linalg import a_norm, cholesky, inv_a_norm, pca, rand_cov, solve, svd, valid
    >>> # Cholesky decomposition
    >>> cov = np.array([[4.0, 2.0], [2.0, 5.0]])
    >>> R = cholesky(cov)
    >>> np.allclose(R.T @ R, cov)
    True

Functions:
    a_norm: Compute the matrix norm of a vector
    cholesky: Compute upper triangular Cholesky decomposition
    det: Compute the determinant of a square matrix
    ewm_covariance: Compute exponentially weighted covariance matrices
    inv: Invert a matrix with NaN-aware matrix filtering and condition-number guarding
    inv_a_norm: Compute the inverse matrix norm of a vector
    norm: Compute the norm of a vector or matrix, ignoring non-finite entries
    pca: Compute principal components of return data
    rand_cov: Generate a random positive semi-definite covariance matrix
    solve: Solve linear systems with NaN-aware matrix filtering
    svd: Compute compact singular value decomposition
    valid: Extract valid submatrix from a matrix with NaN values

"""

from .cholesky import cholesky as cholesky
from .cholesky import is_positive_definite as is_positive_definite
from .det import det as det
from .ewm_cov import NegativeWarmupError as NegativeWarmupError
from .ewm_cov import ewm_covariance as ewm_covariance
from .exceptions import DimensionMismatchError as DimensionMismatchError
from .exceptions import IllConditionedMatrixWarning as IllConditionedMatrixWarning
from .exceptions import NonSquareMatrixError as NonSquareMatrixError
from .exceptions import SingularMatrixError as SingularMatrixError
from .exceptions import check_and_warn_condition as check_and_warn_condition
from .inv import inv as inv
from .norm import a_norm as a_norm
from .norm import inv_a_norm as inv_a_norm
from .norm import norm as norm
from .pca import pca as pca
from .rand_cov import rand_cov as rand_cov
from .solve import solve as solve
from .svd import svd as svd
from .types import Matrix as Matrix
from .valid import valid as valid
