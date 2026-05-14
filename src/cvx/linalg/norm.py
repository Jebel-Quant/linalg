"""Norm utilities for vectors with optional NaN-aware matrix weighting."""

from __future__ import annotations

import numpy as np

from .valid import valid


def a_norm(vector: np.ndarray, matrix: np.ndarray | None = None) -> float:
    """Calculate the generalized norm of a vector with respect to a matrix.

    Args:
        vector: The input vector.
        matrix: Optional square matrix defining the quadratic form.

    Returns:
        The Euclidean norm of the finite vector entries, or ``sqrt(v.T @ A @ v)``
        after dropping rows and columns whose diagonal entries are not finite.

    Raises:
        AssertionError: If the matrix is not square or is incompatible with the vector.

    """
    if matrix is None:
        return float(np.linalg.norm(vector[np.isfinite(vector)], 2))

    if matrix.shape[0] != matrix.shape[1]:
        raise AssertionError

    if vector.size != matrix.shape[0]:
        raise AssertionError

    mask, submatrix = valid(matrix)
    if mask.any():
        return float(np.sqrt(np.dot(vector[mask], np.dot(submatrix, vector[mask]))))

    return float("nan")


def inv_a_norm(vector: np.ndarray, matrix: np.ndarray | None = None) -> float:
    """Calculate the inverse A-norm of a vector using an optional matrix.

    Args:
        vector: The input vector.
        matrix: Optional square matrix defining the quadratic form.

    Returns:
        The Euclidean norm of the finite vector entries, or ``sqrt(v.T @ A^-1 @ v)``
        after dropping rows and columns whose diagonal entries are not finite.

    Raises:
        AssertionError: If the matrix is not square or is incompatible with the vector.

    """
    if matrix is None:
        return float(np.linalg.norm(vector[np.isfinite(vector)], 2))

    if matrix.shape[0] != matrix.shape[1]:
        raise AssertionError

    if vector.size != matrix.shape[0]:
        raise AssertionError

    mask, submatrix = valid(matrix)
    if mask.any():
        return float(np.sqrt(np.dot(vector[mask], np.linalg.solve(submatrix, vector[mask]))))

    return float("nan")
