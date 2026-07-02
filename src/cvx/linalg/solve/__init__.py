"""Linear-system solvers and related quantities: solve, lstsq, inv, det."""

from .det import det as det
from .inv import inv as inv
from .lstsq import lstsq as lstsq
from .solve import solve as solve

__all__ = [
    "det",
    "inv",
    "lstsq",
    "solve",
]
