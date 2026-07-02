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

The `ewm_covariance` function requires the optional [Polars](https://pola.rs) dependency:

```bash
pip install 'cvx-linalg[ewm]'
```

## Usage

```python
from cvx.linalg import (
    a_norm, cholesky, cholesky_solve,
    inv, inv_a_norm, is_positive_definite, lstsq, pca, rand_cov, solve, valid,
)
from cvx.linalg.covariance.ewm_cov import ewm_covariance  # requires the 'ewm' extra (polars)
```

## Functions

- **[`a_norm(vector, matrix=None)`](https://github.com/Jebel-Quant/linalg/blob/main/src/cvx/linalg/norm.py)** — Euclidean norm or NaN-aware matrix norm
- **[`cholesky(cov)`](https://github.com/Jebel-Quant/linalg/blob/main/src/cvx/linalg/cholesky.py)** — Upper triangular Cholesky factor R such that R.T @ R = cov
- **[`cholesky_solve(cov, rhs)`](https://github.com/Jebel-Quant/linalg/blob/main/src/cvx/linalg/cholesky.py)** — Solve cov @ x = rhs via Cholesky decomposition (falls back to LU for non-positive-definite matrices)
- **[`cond(matrix, p=None)`](https://github.com/Jebel-Quant/linalg/blob/main/src/cvx/linalg/exceptions.py)** — Condition number of a matrix (NaN-aware); accepts the same `p` norm values as `numpy.linalg.cond`
- **[`det(matrix, cond_threshold=1e12)`](https://github.com/Jebel-Quant/linalg/blob/main/src/cvx/linalg/det.py)** — Determinant of a square matrix with NaN-aware submatrix handling; emits `IllConditionedMatrixWarning` when near-singular
- **[`eigvals(matrix)`](https://github.com/Jebel-Quant/linalg/blob/main/src/cvx/linalg/eigvals.py)** — Eigenvalues of a general square matrix (supports complex output for non-symmetric matrices)
- **[`eigh(matrix)`](https://github.com/Jebel-Quant/linalg/blob/main/src/cvx/linalg/eigh.py)** — Eigenvalues/eigenvectors of the valid symmetric/Hermitian submatrix in ascending eigenvalue order
- **[`eigvalsh(matrix)`](https://github.com/Jebel-Quant/linalg/blob/main/src/cvx/linalg/eigh.py)** — Eigenvalues-only convenience wrapper around `eigh`
- **[`ewm_covariance(data, assets, index_col, window, is_halflife, warmup)`](https://github.com/Jebel-Quant/linalg/blob/main/src/cvx/linalg/ewm_cov.py)** — Exponentially weighted covariance matrices from a Polars DataFrame
- **[`inv(matrix, cond_threshold=1e12)`](https://github.com/Jebel-Quant/linalg/blob/main/src/cvx/linalg/inv.py)** — Invert a matrix restricted to valid rows/columns; NaN rows/columns are returned for invalid positions
- **[`inv_a_norm(vector, matrix=None)`](https://github.com/Jebel-Quant/linalg/blob/main/src/cvx/linalg/norm.py)** — Euclidean norm or inverse NaN-aware matrix norm
- **[`lstsq(matrix, rhs, cond_threshold=1e12)`](https://github.com/Jebel-Quant/linalg/blob/main/src/cvx/linalg/lstsq.py)** — Solve a least-squares system with NaN-aware row filtering; returns `(x, residuals, rank, sv)` consistent with `numpy.linalg.lstsq`
- **[`is_positive_definite(matrix)`](https://github.com/Jebel-Quant/linalg/blob/main/src/cvx/linalg/cholesky.py)** — Return True if the matrix is symmetric positive-definite
- **[`pca(returns, n_components)`](https://github.com/Jebel-Quant/linalg/blob/main/src/cvx/linalg/pca.py)** — Principal Component Analysis via SVD
- **[`power_iteration(matrix, n_iter=1000, tol=1e-9, seed=None)`](https://github.com/Jebel-Quant/linalg/blob/main/src/cvx/linalg/power_iteration.py)** — Estimate the dominant (largest-magnitude) eigenpair of a symmetric matrix via power iteration
- **[`qr(matrix)`](https://github.com/Jebel-Quant/linalg/blob/main/src/cvx/linalg/qr.py)** — Reduced QR decomposition, matching `np.linalg.qr(mode='reduced')`
- **[`rand_cov(n, seed)`](https://github.com/Jebel-Quant/linalg/blob/main/src/cvx/linalg/rand_cov.py)** — Random positive semi-definite covariance matrix
- **[`solve(matrix, rhs, cond_threshold=1e12)`](https://github.com/Jebel-Quant/linalg/blob/main/src/cvx/linalg/solve.py)** — Solve a linear system (vector or matrix rhs) restricted to valid rows/columns; NaN entries are returned for invalid positions
- **[`svd(matrix)`](https://github.com/Jebel-Quant/linalg/blob/main/src/cvx/linalg/svd.py)** — Raw compact singular value decomposition via `np.linalg.svd(full_matrices=False)`
- **[`svd_k(matrix, k)`](https://github.com/Jebel-Quant/linalg/blob/main/src/cvx/linalg/svd.py)** — Exact truncated rank-`k` SVD (the best rank-`k` approximation; leading triplets of the compact SVD)
- **[`valid(matrix)`](https://github.com/Jebel-Quant/linalg/blob/main/src/cvx/linalg/valid.py)** — Return a boolean mask and valid submatrix by removing rows/columns with non-finite diagonal entries

## Exceptions & Warnings

All exceptions and warnings live in [`exceptions.py`](https://github.com/Jebel-Quant/linalg/blob/main/src/cvx/linalg/exceptions.py):

- **`DimensionMismatchError`** — Raised when vector length does not match matrix dimension
- **`IllConditionedMatrixWarning`** — Emitted when the condition number exceeds a configurable threshold
- **`InvalidComponentsError`** — Raised when `pca` is asked for fewer than 1 or more components than the data supports
- **`NegativeWarmupError`** — Raised when a negative warmup period is passed to `ewm_covariance`
- **`NonIntegerWarmupError`** — Raised when a non-integer warmup is passed to `ewm_covariance`
- **`NonSquareMatrixError`** — Raised when a square matrix is required but the input is not square
- **`NotAMatrixError`** — Raised when a 2-D matrix is required but the input has different dimensionality
- **`SingularMatrixError`** — Raised when a matrix is numerically singular
- **`check_and_warn_condition(matrix, threshold)`** — Emit `IllConditionedMatrixWarning` when the condition number exceeds the threshold

## Types

- **[`Matrix`](https://github.com/Jebel-Quant/linalg/blob/main/src/cvx/linalg/types.py)** — Type alias for a 2-D `numpy.ndarray` with `float64` dtype
- **[`Vector`](https://github.com/Jebel-Quant/linalg/blob/main/src/cvx/linalg/types.py)** — Type alias for a 1-D `numpy.ndarray` with `float64` dtype

The package ships a `py.typed` marker; all public signatures are precisely annotated and verified with [ty](https://github.com/astral-sh/ty) in CI.

## Stability policy

This package follows [semantic versioning](https://semver.org). The public API
is everything importable from `cvx.linalg` (plus `cvx.linalg.covariance.ewm_cov`):

- **Breaking changes** only occur in major releases.
- **Deprecations** are announced at least one minor release before removal and
  emit a `DeprecationWarning` in the meantime (currently: the two-argument
  `cholesky(cov, rhs)` form — use `cholesky_solve` — slated for removal in 1.0).
- **Supported environments:** Python 3.11–3.14, NumPy ≥ 2.0. The optional
  `ewm` extra requires Polars ≥ 1.40.

Numerical conventions (NaN handling, condition-number warnings, dtype
contract) are documented in
[Numerical Behavior](https://jebel-quant.github.io/linalg/numerical-behavior/).
