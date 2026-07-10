"""Dominant eigenpair estimation via power iteration."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from ..core.exceptions import NonSquareMatrixError, NotAMatrixError
from ..core.types import Matrix, SupportsMatvec, Vector

_CALLABLE_NEEDS_N_MESSAGE = "power_iteration needs `n` when `operator` is a bare callable"


def power_iteration(
    operator: Matrix | SupportsMatvec | Callable[[Vector], Vector],
    *,
    n: int | None = None,
    n_iter: int = 1000,
    tol: float = 1e-9,
    seed: int | None = None,
) -> tuple[float, Vector]:
    """Estimate the dominant eigenpair of a real symmetric operator.

    Repeatedly applies *operator* to a random unit vector, renormalizing each
    step, until the Rayleigh-quotient eigenvalue estimate stops changing. The
    iterate converges to the eigenvector whose eigenvalue is largest in
    magnitude; the matching eigenvalue (returned with its sign) comes from the
    Rayleigh quotient ``v.T @ (operator @ v)``.

    *operator* may be given three ways, so the leading eigenvalue can be
    estimated **matrix-free** (e.g. the Lipschitz constant of a gradient step):

    * a dense symmetric ``(n, n)`` array;
    * any :class:`~cvx.linalg.SymmetricOperator` (its :meth:`matvec` and ``n``
      drive the iteration, so no ``n x n`` matrix is formed); or
    * a callable ``v -> A @ v`` together with the dimension *n*.

    Each iteration costs a single application, so this is far cheaper than a full
    :func:`~cvx.linalg.eigh` when only the leading eigenpair is needed and there
    is a clear spectral gap. Symmetry is assumed; only the result's
    interpretation as an eigenpair relies on it. Like :func:`~cvx.linalg.svd`,
    this is a raw primitive and is **not** NaN-aware.

    Args:
        operator: A symmetric matrix, a :class:`~cvx.linalg.SymmetricOperator`,
            or a callable applying it to a vector.
        n: Dimension of the operator. Required when *operator* is a bare callable;
            ignored otherwise (taken from the array shape or the operator's ``n``).
        n_iter: Maximum number of iterations. Defaults to 1000.
        tol: Convergence tolerance on the relative change of the eigenvalue
            estimate between iterations. Defaults to ``1e-9``.
        seed: Seed for the random starting vector, for reproducibility. If
            ``None``, uses the current NumPy random state.

    Returns:
        Tuple ``(eigenvalue, eigenvector)`` where *eigenvalue* is the signed
        dominant eigenvalue estimate (a ``float``) and *eigenvector* is the
        corresponding unit-norm eigenvector. The eigenvector sign is arbitrary.

    Raises:
        NotAMatrixError: If *operator* is an array that is not 2-D.
        NonSquareMatrixError: If *operator* is an array that is not square.
        ValueError: If *operator* is a bare callable and *n* is not given.

    Example:
        >>> import numpy as np
        >>> from cvx.linalg import power_iteration
        >>> matrix = np.diag([3.0, 2.0, 1.0])
        >>> eigenvalue, eigenvector = power_iteration(matrix, seed=0)
        >>> bool(np.isclose(eigenvalue, 3.0))
        True
        >>> bool(np.isclose(abs(eigenvector[0]), 1.0))
        True

        It runs matrix-free on a :class:`~cvx.linalg.SymmetricOperator`, so the
        leading eigenvalue of ``M.T @ M`` needs no ``n x n`` matrix:

        >>> from cvx.linalg import GramOperator
        >>> rng = np.random.default_rng(0)
        >>> M = rng.standard_normal((20, 5))
        >>> lam, _ = power_iteration(GramOperator(M), seed=0)
        >>> bool(np.isclose(lam, np.linalg.eigvalsh(M.T @ M)[-1]))
        True
    """
    apply, dim = _resolve(operator, n)

    rng = np.random.default_rng(seed)
    v = rng.standard_normal((dim,))
    v = v / np.linalg.norm(v)

    eigenvalue = float(v @ apply(v))
    for _ in range(n_iter):
        w = apply(v)
        norm = float(np.linalg.norm(w))
        if norm < 1e-15:
            # operator annihilates the iterate: the dominant eigenvalue is zero.
            return 0.0, v
        v = w / norm
        new_eigenvalue = float(v @ apply(v))
        if abs(new_eigenvalue - eigenvalue) <= tol * max(1.0, abs(new_eigenvalue)):
            return new_eigenvalue, v
        eigenvalue = new_eigenvalue

    return eigenvalue, v


def _resolve(
    operator: Matrix | SupportsMatvec | Callable[[Vector], Vector],
    n: int | None,
) -> tuple[Callable[[Vector], Vector], int]:
    """Return ``(apply, dimension)`` for an array, a SymmetricOperator, or a callable.

    The operator is matched structurally against :class:`~cvx.linalg.core.types.SupportsMatvec`
    (a lower layer than :mod:`~cvx.linalg.operators`), so this needs no import of the
    operator backends and closes no import cycle.
    """
    if isinstance(operator, SupportsMatvec):
        return operator.matvec, operator.n
    if not isinstance(operator, np.ndarray) and callable(operator):
        if n is None:
            raise ValueError(_CALLABLE_NEEDS_N_MESSAGE)
        return operator, int(n)
    return _array_apply(operator)


def _array_apply(
    operator: Matrix | SupportsMatvec | Callable[[Vector], Vector],
) -> tuple[Callable[[Vector], Vector], int]:
    """Return ``(v -> A @ v, n)`` for a dense square array, validating its shape."""
    array = np.asarray(operator)
    if array.ndim != 2:
        raise NotAMatrixError(array.ndim, func="power_iteration")
    rows, cols = array.shape
    if rows != cols:
        raise NonSquareMatrixError(rows, cols)
    return lambda v: array @ v, cols
