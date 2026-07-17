"""Shared reference helpers for the operator-backend tests.

Every backend must reproduce the linear-algebra products of the dense matrix it
represents; these helpers provide the dense reference each per-backend module
(``test_dense``/``test_gram``/``test_factor``) checks itself against.
"""

from __future__ import annotations

import numpy as np

from cvx.linalg import SymmetricOperator


def rcond_reference(block: np.ndarray) -> float:
    """Reference reciprocal 2-norm condition number from the symmetric eigenvalues."""
    if block.shape[0] == 0:
        return 1.0
    eig = np.linalg.eigvalsh(block)
    lam_max = float(eig[-1])
    if lam_max <= 0.0:
        return 0.0
    return max(float(eig[0]), 0.0) / lam_max


def check_against_dense(op: SymmetricOperator, a: np.ndarray, rng: np.random.Generator) -> None:
    """Assert *op* reproduces every product of the dense matrix ``a``."""
    n = a.shape[0]
    x = rng.standard_normal(n)
    assert np.allclose(op.matvec(x), a @ x)

    assert np.allclose(op.diag, np.diag(a))

    perm = rng.permutation(n)
    free = np.sort(perm[: n // 2])
    bound = np.setdiff1d(np.arange(n), free)

    # Free-block forward product (drives a matrix-free CG solve).
    vf = rng.standard_normal(len(free))
    assert np.allclose(op.apply_free(free, vf), a[np.ix_(free, free)] @ vf)

    # Cross product on disjoint sets (the reduced gradient at bound indices).
    assert np.allclose(op.block_matvec(bound, free, vf), a[np.ix_(bound, free)] @ vf)

    # Direct free-block solve, vector and matrix right-hand sides.
    rhs = rng.standard_normal(len(free))
    x_free = op.solve_free(free, rhs)
    assert np.allclose(a[np.ix_(free, free)] @ x_free, rhs)

    rhs_mat = rng.standard_normal((len(free), 3))
    x_mat = op.solve_free(free, rhs_mat)
    assert np.allclose(a[np.ix_(free, free)] @ x_mat, rhs_mat)
