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
    check_and_warn_condition: Compute the condition number and warn if ill-conditioned
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
    is_positive_definite: Test whether a matrix is positive definite
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
    warn_ill_conditioned: Emit an IllConditionedMatrixWarning for a condition number

Operators:
    SymmetricOperator: Protocol exposing a symmetric matrix through block products and a free-block solve
    DenseOperator: SymmetricOperator backed by an explicit dense matrix
    GramOperator: SymmetricOperator A = M.T @ M represented by its factor M (matrix-free)
    FactorOperator: Diagonal-plus-low-rank SymmetricOperator with Woodbury free-block solves
    IncrementalDenseOperator: DenseOperator maintaining the free-block inverse across single-index flips

Types:
    Matrix: Type alias for a 2-D NumPy array
    Vector: Type alias for a 1-D NumPy array

Exceptions:
    DimensionMismatchError: Raised when operand dimensions are incompatible
    InvalidComponentsError: Raised when the requested number of components is invalid
    NegativeWarmupError: Raised when an EWM warmup period is negative
    NonIntegerWarmupError: Raised when an EWM warmup period is not an integer
    NonSquareMatrixError: Raised when a square matrix is required but not given
    NotAMatrixError: Raised when an input is not a 2-D matrix
    SingularMatrixError: Raised when a matrix is singular and cannot be inverted/solved

Warnings:
    IllConditionedMatrixWarning: Emitted when a matrix is ill-conditioned

Constants:
    DEFAULT_COND_THRESHOLD: Default condition-number threshold for ill-conditioning checks

"""

from importlib.metadata import PackageNotFoundError, version

from .core import DEFAULT_COND_THRESHOLD as DEFAULT_COND_THRESHOLD
from .core import DimensionMismatchError as DimensionMismatchError
from .core import IllConditionedMatrixWarning as IllConditionedMatrixWarning
from .core import InvalidComponentsError as InvalidComponentsError
from .core import Matrix as Matrix
from .core import NegativeWarmupError as NegativeWarmupError
from .core import NonIntegerWarmupError as NonIntegerWarmupError
from .core import NonSquareMatrixError as NonSquareMatrixError
from .core import NotAMatrixError as NotAMatrixError
from .core import SingularMatrixError as SingularMatrixError
from .core import Vector as Vector
from .core import check_and_warn_condition as check_and_warn_condition
from .core import cond as cond
from .core import valid as valid
from .core import warn_ill_conditioned as warn_ill_conditioned
from .covariance import cov_to_corr as cov_to_corr
from .covariance import pca as pca
from .covariance import rand_cov as rand_cov
from .decomposition import cholesky as cholesky
from .decomposition import cholesky_solve as cholesky_solve
from .decomposition import eigh as eigh
from .decomposition import eigvals as eigvals
from .decomposition import eigvalsh as eigvalsh
from .decomposition import is_positive_definite as is_positive_definite
from .decomposition import power_iteration as power_iteration
from .decomposition import qr as qr
from .decomposition import svd as svd
from .decomposition import svd_k as svd_k
from .norm import a_norm as a_norm
from .norm import inv_a_norm as inv_a_norm
from .norm import norm as norm
from .operators import DenseOperator as DenseOperator
from .operators import FactorOperator as FactorOperator
from .operators import GramOperator as GramOperator
from .operators import IncrementalDenseOperator as IncrementalDenseOperator
from .operators import SymmetricOperator as SymmetricOperator
from .solve import det as det
from .solve import inv as inv
from .solve import lstsq as lstsq
from .solve import solve as solve

__all__ = [
    "DEFAULT_COND_THRESHOLD",
    "DenseOperator",
    "DimensionMismatchError",
    "FactorOperator",
    "GramOperator",
    "IllConditionedMatrixWarning",
    "IncrementalDenseOperator",
    "InvalidComponentsError",
    "Matrix",
    "NegativeWarmupError",
    "NonIntegerWarmupError",
    "NonSquareMatrixError",
    "NotAMatrixError",
    "SingularMatrixError",
    "SymmetricOperator",
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
