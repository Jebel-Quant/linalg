"""Bordered KKT (saddle-point) solve over a symmetric operator's free block."""

from __future__ import annotations

from typing import cast

import numpy as np

from ..core.types import Matrix, Vector
from ..operators import SymmetricOperator


def bordered_solve(
    operator: SymmetricOperator,
    free: object,
    c_free: Matrix,
    rhs: Vector | Matrix,
    d: Vector | Matrix,
) -> tuple[Vector | Matrix, Vector | Matrix]:
    """Solve the bordered KKT system ``[[H_FF, C.T], [C, 0]] @ [x; nu] = [rhs; d]``.

    ``H_FF`` is the principal ``free`` block of *operator*, reached only through
    :meth:`~cvx.linalg.SymmetricOperator.solve_free` (so structured backends never
    materialise an ``n x n`` matrix). ``C = c_free`` is the ``(mc, len(free))``
    constraint block on the free coordinates. This is the range-space (Schur
    complement) solution of an equality-constrained quadratic: the constraint
    columns ``C.T`` and the right-hand side are solved against ``H_FF`` together,
    so ``H_FF`` is factorised once, then the ``mc x mc`` Schur complement
    ``C H_FF^{-1} C.T`` gives the multipliers.

    Multiple right-hand sides are supported in one call: pass ``rhs`` and ``d`` as
    matrices whose columns are independent systems (they share the ``H_FF`` and
    Schur factorisations). ``mc == 0`` (no constraint rows) is the plain free-block
    solve, with an empty multiplier block.

    Args:
        operator: The symmetric operator whose free block is ``H_FF``.
        free: Index set of the free coordinates (as accepted by *operator*).
        c_free: Constraint block ``C`` of shape ``(mc, len(free))``.
        rhs: Right-hand side of shape ``(len(free),)`` or ``(len(free), k)``.
        d: Constraint target of shape ``(mc,)`` or ``(mc, k)``.

    Returns:
        A pair ``(x, nu)``: the free solution (same trailing shape as *rhs*) and the
        constraint multipliers (shape ``(mc,)`` or ``(mc, k)``; empty when ``mc == 0``).

    Example:
        >>> import numpy as np
        >>> from cvx.linalg import DenseOperator, bordered_solve
        >>> A = np.array([[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]])
        >>> op = DenseOperator(A)
        >>> free = np.array([0, 1, 2])
        >>> C = np.array([[1.0, 1.0, 1.0]])
        >>> x, nu = bordered_solve(op, free, C, np.zeros(3), np.array([1.0]))
        >>> bool(np.isclose(C @ x, 1.0))          # constraint satisfied
        True
        >>> np.allclose(A @ x + C.T @ nu, 0.0)    # stationarity: A x + C.T nu = rhs (= 0)
        True
    """
    c_free = np.asarray(c_free, dtype=np.float64)
    rhs = np.asarray(rhs, dtype=np.float64)
    d = np.asarray(d, dtype=np.float64)
    vector_rhs = rhs.ndim == 1
    rhs_2d = rhs.reshape(rhs.shape[0], -1)
    mc = c_free.shape[0]

    if mc == 0:
        x_only = operator.solve_free(free, rhs)
        empty = np.zeros((0,) if vector_rhs else (0, rhs_2d.shape[1]))
        return x_only, empty

    # One multi-RHS solve against H_FF covers the constraint columns C.T and the
    # right-hand side, so H_FF is factorised a single time.
    solved = cast("Matrix", operator.solve_free(free, np.column_stack([c_free.T, rhs_2d])))
    y = solved[:, :mc]  # H_FF^{-1} C.T
    z = solved[:, mc:]  # H_FF^{-1} rhs

    # Schur complement C H_FF^{-1} C.T and the multipliers, then back-substitute.
    schur = c_free @ y
    d_2d = d.reshape(mc, -1)
    nu = cast("Matrix", np.linalg.solve(schur, c_free @ z - d_2d))
    x = z - y @ nu

    if vector_rhs:
        return x[:, 0], nu[:, 0]
    return x, nu
