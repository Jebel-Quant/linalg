"""Linear algebra utilities for risk models.

This subpackage provides linear algebra utilities commonly used in risk modeling,
including Cholesky decomposition, Principal Component Analysis, matrix norms,
linear-system solving, and matrix validation.

Example:
    >>> import numpy as np
    >>> from cvx.linalg import a_norm, cholesky, eigh, inv_a_norm, pca, qr, rand_cov, solve, svd, valid
    >>> # Cholesky decomposition
    >>> cov = np.array([[4.0, 2.0], [2.0, 5.0]])
    >>> R = cholesky(cov)
    >>> np.allclose(R.T @ R, cov)
    True

Functions:
    a_norm: Compute the matrix norm of a vector
    cholesky: Compute upper triangular Cholesky decomposition
    cholesky_solve: Solve a linear system via Cholesky with LU fallback
    cond: Return the condition number of a matrix (NaN-aware)
    cov_to_corr: Convert a covariance matrix to a correlation matrix
    det: Compute the determinant of a square matrix
    eigvals: Compute eigenvalues of a general square matrix
    eigh: Compute eigenvalues and eigenvectors of a symmetric/Hermitian matrix
    eigvalsh: Compute eigenvalues of a symmetric/Hermitian matrix
    inv: Invert a matrix with NaN-aware matrix filtering and condition-number guarding
    inv_a_norm: Compute the inverse matrix norm of a vector
    lstsq: Solve least-squares problems with NaN-aware row filtering
    norm: Compute the norm of a vector or matrix, ignoring non-finite entries
    pca: Compute principal components of return data
    power_iteration: Estimate the dominant eigenpair of a symmetric matrix
    qr: Compute reduced QR decomposition of a matrix
    rand_cov: Generate a random positive semi-definite covariance matrix
    solve: Solve linear systems with NaN-aware matrix filtering
    svd: Compute compact singular value decomposition
    svd_k: Compute the truncated rank-k singular value decomposition
    valid: Extract valid submatrix from a matrix with NaN values

"""

from importlib.metadata import PackageNotFoundError, version

from .cholesky import cholesky as cholesky
from .cholesky import cholesky_solve as cholesky_solve
from .cholesky import is_positive_definite as is_positive_definite
from .cov_to_corr import cov_to_corr as cov_to_corr
from .det import det as det
from .eigh import eigh as eigh
from .eigh import eigvalsh as eigvalsh
from .eigvals import eigvals as eigvals
from .exceptions import DEFAULT_COND_THRESHOLD as DEFAULT_COND_THRESHOLD
from .exceptions import DimensionMismatchError as DimensionMismatchError
from .exceptions import IllConditionedMatrixWarning as IllConditionedMatrixWarning
from .exceptions import InvalidComponentsError as InvalidComponentsError
from .exceptions import NegativeWarmupError as NegativeWarmupError
from .exceptions import NonIntegerWarmupError as NonIntegerWarmupError
from .exceptions import NonSquareMatrixError as NonSquareMatrixError
from .exceptions import NotAMatrixError as NotAMatrixError
from .exceptions import SingularMatrixError as SingularMatrixError
from .exceptions import check_and_warn_condition as check_and_warn_condition
from .exceptions import cond as cond
from .exceptions import warn_ill_conditioned as warn_ill_conditioned
from .inv import inv as inv
from .lstsq import lstsq as lstsq
from .norm import a_norm as a_norm
from .norm import inv_a_norm as inv_a_norm
from .norm import norm as norm
from .pca import pca as pca
from .power_iteration import power_iteration as power_iteration
from .qr import qr as qr
from .rand_cov import rand_cov as rand_cov
from .solve import solve as solve
from .svd import svd as svd
from .svd import svd_k as svd_k
from .types import Matrix as Matrix
from .types import Vector as Vector
from .valid import valid as valid

__all__ = [
    "DEFAULT_COND_THRESHOLD",
    "DimensionMismatchError",
    "IllConditionedMatrixWarning",
    "InvalidComponentsError",
    "Matrix",
    "NegativeWarmupError",
    "NonIntegerWarmupError",
    "NonSquareMatrixError",
    "NotAMatrixError",
    "SingularMatrixError",
    "Vector",
    "a_norm",
    "check_and_warn_condition",
    "cholesky",
    "cholesky_solve",
    "cond",
    "cov_to_corr",
    "det",
    "eigh",
    "eigvals",
    "eigvalsh",
    "inv",
    "inv_a_norm",
    "is_positive_definite",
    "lstsq",
    "norm",
    "pca",
    "power_iteration",
    "qr",
    "rand_cov",
    "solve",
    "svd",
    "svd_k",
    "valid",
    "warn_ill_conditioned",
]

try:
    __version__ = version("cvx-linalg")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"
