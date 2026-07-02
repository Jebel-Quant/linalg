"""Cholesky decomposition utilities for covariance matrices."""

from __future__ import annotations

import warnings
from typing import cast

import numpy as np
from numpy.linalg import cholesky as _cholesky

from ..core.types import Matrix, Vector


def cholesky(cov: Matrix, rhs: Vector | Matrix | None = None) -> Vector | Matrix:
    """Compute the upper triangular Cholesky factor of a covariance matrix.

    Returns the upper triangular factor R such that R.T @ R = cov.

    Args:
        cov: A positive definite covariance matrix of shape (n, n).
        rhs: Deprecated. When provided the system ``cov @ x = rhs`` is solved
            and *x* is returned; use :func:`cholesky_solve` instead. This
            parameter will be removed in 1.0.

    Returns:
        The upper triangular Cholesky factor R when *rhs* is ``None``, or the
        solution x to ``cov @ x = rhs`` otherwise (deprecated).

    Raises:
        np.linalg.LinAlgError: When *rhs* is ``None`` and *cov* is not
            positive-definite, or when *rhs* is given and both Cholesky and
            LU-based solves fail.

    Warns:
        DeprecationWarning: When *rhs* is given.

    Example:
        >>> import numpy as np
        >>> from cvx.linalg import cholesky
        >>> cov = np.array([[4.0, 2.0], [2.0, 5.0]])
        >>> R = cholesky(cov)
        >>> np.allclose(R.T @ R, cov)
        True
    """
    if rhs is None:
        return cast("Matrix", _cholesky(cov).transpose())
    warnings.warn(
        "Passing 'rhs' to cholesky() is deprecated and will be removed in 1.0; use cholesky_solve(cov, rhs) instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return cholesky_solve(cov, rhs)


def cholesky_solve(cov: Matrix, rhs: Vector | Matrix) -> Vector | Matrix:
    """Solve ``cov @ x = rhs`` using the Cholesky decomposition.

    The Cholesky factorisation is attempted first for numerical stability;
    when *cov* is not positive-definite the solve falls back to LU
    decomposition.

    Args:
        cov: A positive definite covariance matrix of shape (n, n).
        rhs: Right-hand side vector of length n or matrix of shape (n, k).

    Returns:
        The solution x to ``cov @ x = rhs`` with the same shape as *rhs*.

    Raises:
        np.linalg.LinAlgError: When both the Cholesky and LU-based solves fail.

    Example:
        >>> import numpy as np
        >>> from cvx.linalg import cholesky_solve
        >>> cholesky_solve(np.eye(2), np.array([1.0, 2.0])).tolist()
        [1.0, 2.0]
        >>> cholesky_solve(np.array([[4.0, 0.0], [0.0, 9.0]]), np.array([8.0, 27.0])).tolist()
        [2.0, 3.0]
    """
    try:
        upper = _cholesky(cov).transpose()
        return cast("Vector | Matrix", np.linalg.solve(upper, np.linalg.solve(upper.T, rhs)))
    except np.linalg.LinAlgError:
        return cast("Vector | Matrix", np.linalg.solve(cov, rhs))


def is_positive_definite(matrix: Matrix) -> bool:
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
