"""Cholesky decomposition utilities for covariance matrices."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import cast

import numpy as np
from numpy.linalg import cholesky as _cholesky

from ..core.types import Matrix, Vector

try:  # SciPy's triangular solves keep a factored solve at O(n^2) per right-hand side.
    from scipy.linalg import cho_factor as _cho_factor  # type: ignore[import-untyped]
    from scipy.linalg import cho_solve as _cho_solve

    _HAVE_SCIPY = True
except ImportError:  # pragma: no cover - depends on the environment; the fallback is tested by patching the flag
    _HAVE_SCIPY = False


def cholesky(cov: Matrix, rhs: Vector | Matrix | None = None) -> Vector | Matrix:
    """Compute the upper triangular Cholesky factor of a covariance matrix.

    Returns the upper triangular factor R such that R.T @ R = cov.

    Args:
        cov: A positive definite covariance matrix of shape (n, n).
        rhs: Deprecated. When provided the system ``cov @ x = rhs`` is solved
            and *x* is returned; use :func:`cholesky_solve` instead. This
            parameter will be removed in 2.0.

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
        "Passing 'rhs' to cholesky() is deprecated and will be removed in 2.0; use cholesky_solve(cov, rhs) instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return cholesky_solve(cov, rhs)


def cholesky_solve(cov: Matrix, rhs: Vector | Matrix) -> Vector | Matrix:
    """Solve ``cov @ x = rhs`` using the Cholesky decomposition.

    The Cholesky factorisation is attempted first for numerical stability;
    when *cov* is not positive-definite the solve falls back to LU
    decomposition. With SciPy installed (the ``scipy`` extra) the factor is
    applied by two triangular solves, so the whole solve costs one
    factorisation. Without it, NumPy has no triangular solve, and a successful
    Cholesky is followed by a single LU solve.

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
    return _factored_solver(cov)(rhs)


def _factored_solver(cov: Matrix) -> Callable[[Vector | Matrix], Vector | Matrix]:
    """Factorise *cov* once and return a function solving ``cov @ x = rhs``.

    The returned solver follows :func:`cholesky_solve` -- Cholesky when *cov* is
    positive-definite, LU otherwise -- but pays for the factorisation a single
    time, so repeated solves against a fixed matrix cost ``O(n**2)`` each with
    SciPy installed. Without SciPy each call is still an LU solve. A matrix that
    is not positive-definite falls back to LU at call time, so a singular *cov*
    raises :class:`numpy.linalg.LinAlgError` from the solver, not from here.

    Args:
        cov: A covariance matrix of shape (n, n).

    Returns:
        A function mapping a right-hand side of shape (n,) or (n, k) to the
        solution of the same shape.
    """
    try:
        if _HAVE_SCIPY:
            factor = _cho_factor(cov)
            # cho_factor has already checked cov; skipping the rhs scan lets NaNs propagate.
            return lambda rhs: cast("Vector | Matrix", _cho_solve(factor, rhs, check_finite=False))
        _cholesky(cov)  # raises LinAlgError unless cov is positive-definite
    except np.linalg.LinAlgError:
        pass
    return lambda rhs: cast("Vector | Matrix", np.linalg.solve(cov, rhs))


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
