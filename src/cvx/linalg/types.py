"""Type aliases for linear algebra utilities.

Defines a NumPy ndarray alias used across the linalg subpackage.
"""

from typing import TypeAlias

import numpy as np
import numpy.typing as npt

Matrix: TypeAlias = npt.NDArray[np.float64]
