# cvx-linalg

[![PyPI version](https://badge.fury.io/py/cvx-linalg.svg)](https://badge.fury.io/py/cvx-linalg)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://pypi.org/project/cvx-linalg/)
[![Downloads](https://static.pepy.tech/personalized-badge/cvx-linalg?period=month&units=international_system&left_color=black&right_color=orange&left_text=PyPI%20downloads%20per%20month)](https://pepy.tech/project/cvx-linalg)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/Jebel-Quant/linalg/blob/main/LICENSE)
[![CodeFactor](https://www.codefactor.io/repository/github/jebel-quant/linalg/badge)](https://www.codefactor.io/repository/github/jebel-quant/linalg)
[![Rhiza](https://img.shields.io/badge/dynamic/yaml?url=https%3A%2F%2Fraw.githubusercontent.com%2FJebel-Quant%2Flinalg%2Fmain%2F.rhiza%2Ftemplate.yml&query=%24.ref&label=rhiza)](https://github.com/jebel-quant/rhiza)
[![Coverage](https://jebel-quant.github.io/linalg/coverage-badge.svg)](https://jebel-quant.github.io/linalg/reports/html-coverage/)

Linear algebra utilities for portfolio optimization, part of the [jebel-quant](https://github.com/jebel-quant) ecosystem.

## Installation

```bash
pip install cvx-linalg
```

## Usage

```python
from cvx.linalg import (
    a_norm, cholesky, ewm_covariance,
    inv, inv_a_norm, is_positive_definite, pca, qr, rand_cov, solve, valid,
)
```

## Functions

- **[`a_norm(vector, matrix=None)`](https://github.com/Jebel-Quant/linalg/blob/main/src/cvx/linalg/norm.py#L20)** — Euclidean norm or NaN-aware matrix norm
- **[`cholesky(cov, rhs=None)`](https://github.com/Jebel-Quant/linalg/blob/main/src/cvx/linalg/cholesky.py#L9)** — Upper triangular Cholesky factor R such that R.T @ R = cov; when *rhs* is given, solves cov @ x = rhs (falls back to LU for non-positive-definite matrices)
- **[`det(matrix, cond_threshold=1e12)`](https://github.com/Jebel-Quant/linalg/blob/main/src/cvx/linalg/det.py#L18)** — Determinant of a square matrix with NaN-aware submatrix handling; emits `IllConditionedMatrixWarning` when near-singular
- **[`ewm_covariance(data, assets, index_col, window, is_halflife, warmup)`](https://github.com/Jebel-Quant/linalg/blob/main/src/cvx/linalg/ewm_cov.py#L15)** — Exponentially weighted covariance matrices from a Polars DataFrame
- **[`inv(matrix, cond_threshold=1e12)`](https://github.com/Jebel-Quant/linalg/blob/main/src/cvx/linalg/inv.py#L20)** — Invert a matrix restricted to valid rows/columns; NaN rows/columns are returned for invalid positions
- **[`inv_a_norm(vector, matrix=None)`](https://github.com/Jebel-Quant/linalg/blob/main/src/cvx/linalg/norm.py#L58)** — Euclidean norm or inverse NaN-aware matrix norm
- **[`is_positive_definite(matrix)`](https://github.com/Jebel-Quant/linalg/blob/main/src/cvx/linalg/cholesky.py#L57)** — Return True if the matrix is symmetric positive-definite
- **[`pca(returns, n_components)`](https://github.com/Jebel-Quant/linalg/blob/main/src/cvx/linalg/pca.py#L61)** — Principal Component Analysis via SVD
- **[`qr(matrix)`](https://github.com/Jebel-Quant/linalg/blob/main/src/cvx/linalg/qr.py#L10)** — Reduced QR decomposition, matching `np.linalg.qr(mode='reduced')`
- **[`rand_cov(n, seed)`](https://github.com/Jebel-Quant/linalg/blob/main/src/cvx/linalg/rand_cov.py#L29)** — Random positive semi-definite covariance matrix
- **[`solve(matrix, rhs, cond_threshold=1e12)`](https://github.com/Jebel-Quant/linalg/blob/main/src/cvx/linalg/solve.py#L22)** — Solve a linear system restricted to valid rows/columns; NaN entries are returned for invalid positions
- **[`valid(matrix)`](https://github.com/Jebel-Quant/linalg/blob/main/src/cvx/linalg/valid.py#L32)** — Return a boolean mask and valid submatrix by removing rows/columns with non-finite diagonal entries

## Exceptions & Warnings

- **[`DimensionMismatchError`](https://github.com/Jebel-Quant/linalg/blob/main/src/cvx/linalg/exceptions.py#L31)** — Raised when vector length does not match matrix dimension
- **[`IllConditionedMatrixWarning`](https://github.com/Jebel-Quant/linalg/blob/main/src/cvx/linalg/exceptions.py#L73)** — Emitted when the condition number exceeds a configurable threshold
- **[`NegativeWarmupError`](https://github.com/Jebel-Quant/linalg/blob/main/src/cvx/linalg/ewm_cov.py#L11)** — Raised when a negative warmup period is passed to `ewm_covariance`
- **[`NonSquareMatrixError`](https://github.com/Jebel-Quant/linalg/blob/main/src/cvx/linalg/exceptions.py#L10)** — Raised when a square matrix is required but the input is not square
- **[`SingularMatrixError`](https://github.com/Jebel-Quant/linalg/blob/main/src/cvx/linalg/exceptions.py#L52)** — Raised when a matrix is numerically singular
- **[`check_and_warn_condition(matrix, threshold)`](https://github.com/Jebel-Quant/linalg/blob/main/src/cvx/linalg/exceptions.py#L82)** — Emit `IllConditionedMatrixWarning` when the condition number exceeds the threshold

## Types

- **[`Matrix`](https://github.com/Jebel-Quant/linalg/blob/main/src/cvx/linalg/types.py#L11)** — Type alias for `numpy.ndarray` with `float64` dtype
