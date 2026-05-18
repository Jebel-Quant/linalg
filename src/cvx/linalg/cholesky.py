"""Cholesky decomposition utilities for covariance matrices."""

from __future__ import annotations

import numpy as np
from numpy.linalg import cholesky as _cholesky


def cholesky(cov: np.ndarray, rhs: np.ndarray | None = None) -> np.ndarray:
    """Compute the upper triangular Cholesky factor, or solve a linear system.

    When called without *rhs*, returns the upper triangular factor R such that
    R.T @ R = cov.  When *rhs* is given, solves ``cov @ x = rhs`` using the
    Cholesky decomposition for numerical stability, falling back to LU
    decomposition if *cov* is not positive-definite.

    Args:
        cov: A positive definite covariance matrix of shape (n, n).
        rhs: Optional right-hand side vector or matrix. When provided the
            system ``cov @ x = rhs`` is solved and *x* is returned.

    Returns:
        The upper triangular Cholesky factor R when *rhs* is ``None``, or the
        solution x to ``cov @ x = rhs`` otherwise.

    Raises:
        np.linalg.LinAlgError: When *rhs* is ``None`` and *cov* is not
            positive-definite, or when *rhs* is given and both Cholesky and
            LU-based solves fail.

    Example:
        Decomposition (no rhs):

        >>> import numpy as np
        >>> from cvx.linalg import cholesky
        >>> cov = np.array([[4.0, 2.0], [2.0, 5.0]])
        >>> R = cholesky(cov)
        >>> np.allclose(R.T @ R, cov)
        True

        Solve (with rhs):

        >>> cholesky(np.eye(2), np.array([1.0, 2.0])).tolist()
        [1.0, 2.0]
        >>> cholesky(np.array([[4.0, 0.0], [0.0, 9.0]]), np.array([8.0, 27.0])).tolist()
        [2.0, 3.0]
    """
    if rhs is None:
        return _cholesky(cov).transpose()
    try:
        upper = _cholesky(cov).transpose()
        return np.linalg.solve(upper, np.linalg.solve(upper.T, rhs))
    except np.linalg.LinAlgError:
        return np.linalg.solve(cov, rhs)


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
