"""Type aliases and structural protocols for linear algebra utilities.

Defines the NumPy ndarray aliases used across the linalg subpackage and the
:class:`SupportsMatvec` protocol -- the minimal matrix-free contract (a dimension
``n`` and a ``matvec``) that lets lower layers accept an operator structurally,
without importing the :mod:`~cvx.linalg.operators` backends.
"""

from typing import Protocol, TypeAlias, runtime_checkable

import numpy as np
import numpy.typing as npt

Matrix: TypeAlias = npt.NDArray[np.float64]
"""A 2-D float64 NumPy array."""

Vector: TypeAlias = npt.NDArray[np.float64]
"""A 1-D float64 NumPy array."""


@runtime_checkable
class SupportsMatvec(Protocol):
    """Structural protocol for an operator applied matrix-free.

    Anything exposing a dimension ``n`` and a ``matvec(x) -> A @ x`` satisfies it,
    so a routine can accept an operator by shape rather than by importing a concrete
    class. Every :class:`~cvx.linalg.SymmetricOperator` conforms; plain arrays and
    bare callables deliberately do not. It lives in :mod:`~cvx.linalg.core.types`
    (the lowest layer) so consumers such as
    :func:`~cvx.linalg.power_iteration` can ``isinstance``-check against it without
    pulling in the operator backends.
    """

    @property
    def n(self) -> int:
        """Dimension of the operator (it acts on vectors of length ``n``)."""
        ...

    def matvec(self, x: Vector | Matrix) -> Vector | Matrix:
        """Return ``A @ x``."""
        ...
