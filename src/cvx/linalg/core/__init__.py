"""Core building blocks: type aliases, exceptions, and matrix validation."""

from .exceptions import DEFAULT_COND_THRESHOLD as DEFAULT_COND_THRESHOLD
from .exceptions import DimensionMismatchError as DimensionMismatchError
from .exceptions import IllConditionedMatrixWarning as IllConditionedMatrixWarning
from .exceptions import InvalidComponentsError as InvalidComponentsError
from .exceptions import NegativeWarmupError as NegativeWarmupError
from .exceptions import NonIntegerWarmupError as NonIntegerWarmupError
from .exceptions import NonSquareMatrixError as NonSquareMatrixError
from .exceptions import NotAMatrixError as NotAMatrixError
from .exceptions import SingularMatrixError as SingularMatrixError
from .exceptions import check_and_warn_condition as check_and_warn_condition
from .exceptions import cond as cond
from .exceptions import warn_ill_conditioned as warn_ill_conditioned
from .types import Matrix as Matrix
from .types import Vector as Vector
from .valid import valid as valid

__all__ = [
    "DEFAULT_COND_THRESHOLD",
    "DimensionMismatchError",
    "IllConditionedMatrixWarning",
    "InvalidComponentsError",
    "Matrix",
    "NegativeWarmupError",
    "NonIntegerWarmupError",
    "NonSquareMatrixError",
    "NotAMatrixError",
    "SingularMatrixError",
    "Vector",
    "check_and_warn_condition",
    "cond",
    "valid",
    "warn_ill_conditioned",
]
