"""Symmetric linear operators and matrix-free backends."""

from .operators import DenseOperator as DenseOperator
from .operators import FactorOperator as FactorOperator
from .operators import GramOperator as GramOperator
from .operators import SymmetricOperator as SymmetricOperator

__all__ = [
    "DenseOperator",
    "FactorOperator",
    "GramOperator",
    "SymmetricOperator",
]
