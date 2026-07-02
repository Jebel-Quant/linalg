"""Matrix validation utilities for handling non-finite values.

This module provides functions for validating and cleaning matrices that may
contain non-finite values (NaN or infinity). This is particularly useful when
working with financial data where missing values are common.

Example:
    Extract the valid submatrix from a covariance matrix with missing data:

    >>> import numpy as np
    >>> from cvx.linalg import valid
    >>> # Create a covariance matrix with some NaN values on diagonal
    >>> cov = np.array([[np.nan, 0.5, 0.2],
    ...                 [0.5, 2.0, 0.3],
    ...                 [0.2, 0.3, np.nan]])
    >>> # Get valid indicator and submatrix
    >>> v, submatrix = valid(cov)
    >>> v  # Second row/column is valid
    array([False,  True, False])
    >>> submatrix
    array([[2.]])

"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from .exceptions import NonSquareMatrixError
from .types import Matrix


def valid(matrix: Matrix) -> tuple[npt.NDArray[np.bool_], Matrix]:
    """Extract the valid subset of a matrix by removing rows/columns with non-finite values.

    This function identifies rows and columns in a square matrix that contain
    non-finite values (NaN or infinity) on the diagonal and removes them,
    returning both the indicator vector and the resulting valid submatrix.

    This is useful when working with covariance matrices where some assets
    may have missing or invalid data.

    Args:
        matrix: A square n x n matrix to be validated. Typically a covariance
            or correlation matrix.

    Returns:
        A tuple containing:
            - v: Boolean vector of shape (n,) indicating which rows/columns are
              valid (True for valid, False for invalid).
            - submatrix: The valid submatrix with invalid rows/columns removed.
              Shape is (k, k) where k is the number of True values in v.

    Raises:
        NonSquareMatrixError: If the input matrix is not square (n x n).

    Example:
        Basic usage with a covariance matrix:

        >>> import numpy as np
        >>> from cvx.linalg import valid
        >>> # Create a 3x3 matrix with one invalid entry
        >>> cov = np.array([[1.0, 0.5, 0.2],
        ...                 [0.5, np.nan, 0.3],
        ...                 [0.2, 0.3, 1.0]])
        >>> v, submatrix = valid(cov)
        >>> v
        array([ True, False,  True])
        >>> submatrix
        array([[1. , 0.2],
               [0.2, 1. ]])

        Handling a fully valid matrix:

        >>> cov = np.array([[1.0, 0.5], [0.5, 1.0]])
        >>> v, submatrix = valid(cov)
        >>> v
        array([ True,  True])
        >>> np.allclose(submatrix, cov)
        True

        Handling infinity values:

        >>> cov = np.array([[1.0, 0.5, 0.2],
        ...                 [0.5, np.inf, 0.3],
        ...                 [0.2, 0.3, 1.0]])
        >>> v, submatrix = valid(cov)
        >>> v
        array([ True, False,  True])
        >>> submatrix
        array([[1. , 0.2],
               [0.2, 1. ]])

        Multiple invalid entries:

        >>> cov = np.array([[np.nan, 0.1, 0.2, 0.3],
        ...                 [0.1, 2.0, 0.4, 0.5],
        ...                 [0.2, 0.4, np.nan, 0.6],
        ...                 [0.3, 0.5, 0.6, 3.0]])
        >>> v, submatrix = valid(cov)
        >>> v
        array([False,  True, False,  True])
        >>> submatrix.shape
        (2, 2)
        >>> submatrix
        array([[2. , 0.5],
               [0.5, 3. ]])

        Non-square matrix raises NonSquareMatrixError:

        >>> try:
        ...     valid(np.array([[1, 2, 3], [4, 5, 6]]))
        ... except NonSquareMatrixError:
        ...     print("Caught NonSquareMatrixError for non-square matrix")
        Caught NonSquareMatrixError for non-square matrix

    Note:
        The function checks only the diagonal elements for validity. It assumes
        that if the diagonal is finite, the entire row/column is valid. This is
        a common assumption for covariance matrices.

    """
    if matrix.shape[0] != matrix.shape[1]:
        raise NonSquareMatrixError(matrix.shape[0], matrix.shape[1])

    v = np.isfinite(np.diag(matrix))
    return v, matrix[:, v][v]
