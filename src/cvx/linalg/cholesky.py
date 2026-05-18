"""Cholesky decomposition utilities for covariance matrices.

This module provides a function to compute the upper triangular Cholesky
decomposition of a positive definite covariance matrix.

Example:
    Compute the Cholesky decomposition of a covariance matrix:

    >>> import numpy as np
    >>> from cvx.linalg import cholesky
    >>> # Create a positive definite matrix
    >>> cov = np.array([[4.0, 2.0], [2.0, 5.0]])
    >>> # Compute upper triangular Cholesky factor
    >>> R = cholesky(cov)
    >>> # Verify: R.T @ R = cov
    >>> np.allclose(R.T @ R, cov)
    True

"""

from __future__ import annotations

import numpy as np
from numpy.linalg import cholesky as _cholesky


def cholesky(cov: np.ndarray) -> np.ndarray:
    """Compute the upper triangular part of the Cholesky decomposition.

    This function computes the Cholesky decomposition of a positive definite matrix
    and returns the upper triangular matrix R such that R^T @ R = cov.

    The Cholesky decomposition is useful in portfolio optimization because it
    provides an efficient way to compute portfolio risk as ||R @ w||_2, where
    w is the portfolio weights vector.

    Args:
        cov: A positive definite covariance matrix of shape (n, n).

    Returns:
        The upper triangular Cholesky factor R of shape (n, n).

    Example:
        Basic usage with a simple covariance matrix:

        >>> import numpy as np
        >>> from cvx.linalg import cholesky
        >>> # Identity matrix
        >>> cov = np.eye(3)
        >>> R = cholesky(cov)
        >>> np.allclose(R, np.eye(3))
        True

        With a more complex covariance matrix:

        >>> cov = np.array([[1.0, 0.5, 0.0],
        ...                 [0.5, 1.0, 0.5],
        ...                 [0.0, 0.5, 1.0]])
        >>> R = cholesky(cov)
        >>> np.allclose(R.T @ R, cov)
        True

        Verify upper triangular structure:

        >>> R = cholesky(np.array([[4.0, 2.0], [2.0, 5.0]]))
        >>> # R is upper triangular (zeros below diagonal)
        >>> bool(np.allclose(R[1, 0], 0.0))
        True
        >>> bool(R[0, 0] > 0 and R[1, 1] > 0)  # Positive diagonal
        True

        Portfolio risk computation via Cholesky factor:

        >>> cov = np.array([[0.04, 0.01], [0.01, 0.09]])
        >>> R = cholesky(cov)
        >>> w = np.array([0.6, 0.4])
        >>> # Risk via Cholesky: ||R @ w||_2
        >>> risk_chol = np.linalg.norm(R @ w)
        >>> # Risk via covariance: sqrt(w^T @ cov @ w)
        >>> risk_cov = np.sqrt(w @ cov @ w)
        >>> bool(np.isclose(risk_chol, risk_cov))
        True

        Relationship between upper (R) and lower (L) triangular factors:

        >>> cov = np.array([[9.0, 3.0], [3.0, 5.0]])
        >>> R = cholesky(cov)
        >>> L = np.linalg.cholesky(cov)  # numpy returns lower triangular
        >>> # R = L^T
        >>> np.allclose(R, L.T)
        True
        >>> # Both reconstruct the covariance
        >>> np.allclose(L @ L.T, cov)
        True
        >>> np.allclose(R.T @ R, cov)
        True

    Note:
        This function returns the upper triangular factor (R), whereas
        numpy.linalg.cholesky returns the lower triangular factor (L).
        The relationship is: L @ L^T = cov and R^T @ R = cov, where R = L^T.

    """
    return _cholesky(cov).transpose()


def is_positive_definite(matrix: np.ndarray) -> bool:
    """Return True if *matrix* is symmetric positive-definite, False otherwise.

    The check is performed via an attempted Cholesky decomposition — the most
    numerically reliable way to test positive-definiteness for symmetric matrices.

    This function is side-effect-free: it raises no exceptions and emits no
    warnings.  It is suitable for use as a guard before passing a matrix to a
    linear solver.

    Args:
        matrix: Square matrix to test.

    Returns:
        ``True`` if the matrix is positive-definite, ``False`` otherwise.

    Example:
        >>> import numpy as np
        >>> from cvx.linalg import is_positive_definite
        >>> is_positive_definite(np.eye(3))
        True
        >>> is_positive_definite(np.array([[1.0, 2.0], [2.0, 1.0]]))
        False
        >>> is_positive_definite(np.array([[1.0, 0.5], [0.5, 1.0]]))
        True
    """
    try:
        cholesky(matrix)
    except np.linalg.LinAlgError:
        return False
    else:
        return True


def cholesky_solve(matrix: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Solve *matrix* x = *rhs* using Cholesky, falling back to LU if needed.

    Cholesky decomposition is attempted first as it is numerically more stable
    for positive-definite matrices.  If the decomposition fails the matrix is
    not positive-definite and a standard LU-based solve is used instead.

    Args:
        matrix: Square coefficient matrix.
        rhs: Right-hand side vector or matrix.

    Returns:
        Solution x such that ``matrix @ x ≈ rhs``.

    Raises:
        np.linalg.LinAlgError: If both Cholesky and LU-based solves fail.

    Example:
        >>> import numpy as np
        >>> from cvx.linalg import cholesky_solve
        >>> cholesky_solve(np.eye(2), np.array([1.0, 2.0])).tolist()
        [1.0, 2.0]
        >>> cholesky_solve(np.array([[4.0, 0.0], [0.0, 9.0]]), np.array([8.0, 27.0])).tolist()
        [2.0, 3.0]
    """
    try:
        upper = cholesky(matrix)
        return np.linalg.solve(upper, np.linalg.solve(upper.T, rhs))
    except np.linalg.LinAlgError:
        return np.linalg.solve(matrix, rhs)
