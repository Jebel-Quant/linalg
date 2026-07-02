"""Dominant eigenpair estimation via power iteration."""

from __future__ import annotations

import numpy as np

from ..core.exceptions import NonSquareMatrixError, NotAMatrixError
from ..core.types import Matrix, Vector


def power_iteration(
    matrix: Matrix,
    *,
    n_iter: int = 1000,
    tol: float = 1e-9,
    seed: int | None = None,
) -> tuple[float, Vector]:
    """Estimate the dominant eigenpair of a real symmetric matrix.

    Repeatedly applies *matrix* to a random unit vector, renormalizing each
    step, until the Rayleigh-quotient eigenvalue estimate stops changing. The
    iterate converges to the eigenvector whose eigenvalue is largest in
    magnitude; the matching eigenvalue (returned with its sign) comes from the
    Rayleigh quotient ``v.T @ matrix @ v``.

    Each iteration costs a single matrix-vector product, so this is far cheaper
    than a full :func:`~cvx.linalg.eigh` when only the leading eigenpair is
    needed and convergence is fast (i.e. a clear spectral gap). Convergence
    slows as the ratio of the two largest eigenvalue magnitudes approaches 1.

    The matrix is assumed symmetric; only the result's interpretation as an
    eigenpair relies on it. Like :func:`~cvx.linalg.svd`, this is a raw
    primitive and is **not** NaN-aware.

    Args:
        matrix: Square (symmetric) input matrix of shape ``(n, n)``.
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
        NotAMatrixError: If *matrix* is not 2-D.
        NonSquareMatrixError: If *matrix* is not square.

    Example:
        >>> import numpy as np
        >>> from cvx.linalg import power_iteration
        >>> matrix = np.diag([3.0, 2.0, 1.0])
        >>> eigenvalue, eigenvector = power_iteration(matrix, seed=0)
        >>> bool(np.isclose(eigenvalue, 3.0))
        True
        >>> bool(np.isclose(abs(eigenvector[0]), 1.0))
        True

        The estimate agrees with the largest-magnitude eigenvalue from a full
        symmetric eigendecomposition:

        >>> from cvx.linalg import rand_cov
        >>> cov = rand_cov(8, seed=42)
        >>> eigenvalue, _ = power_iteration(cov, seed=0)
        >>> dominant = np.linalg.eigvalsh(cov)[-1]
        >>> bool(np.isclose(eigenvalue, dominant))
        True
    """
    if matrix.ndim != 2:
        raise NotAMatrixError(matrix.ndim, func="power_iteration")
    rows, cols = matrix.shape
    if rows != cols:
        raise NonSquareMatrixError(rows, cols)

    rng = np.random.default_rng(seed)
    v = rng.standard_normal((cols,))
    v = v / np.linalg.norm(v)

    eigenvalue = float(v @ (matrix @ v))
    for _ in range(n_iter):
        w = matrix @ v
        norm = float(np.linalg.norm(w))
        if norm < 1e-15:
            # matrix annihilates the iterate: the dominant eigenvalue is zero.
            return 0.0, v
        v = w / norm
        new_eigenvalue = float(v @ (matrix @ v))
        if abs(new_eigenvalue - eigenvalue) <= tol * max(1.0, abs(new_eigenvalue)):
            return new_eigenvalue, v
        eigenvalue = new_eigenvalue

    return eigenvalue, v
