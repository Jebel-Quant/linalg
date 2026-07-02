"""Type aliases for linear algebra utilities.

Defines NumPy ndarray aliases used across the linalg subpackage.
"""

from typing import TypeAlias

import numpy as np
import numpy.typing as npt

Matrix: TypeAlias = npt.NDArray[np.float64]
"""A 2-D float64 NumPy array."""

Vector: TypeAlias = npt.NDArray[np.float64]
"""A 1-D float64 NumPy array."""
