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

The entire public API is re-exported at the top level, so a flat import is all
you ever need:

```python
from cvx.linalg import (
    a_norm, cholesky, cholesky_solve, cov_to_corr,
    inv, inv_a_norm, is_positive_definite, lstsq, pca, rand_cov, solve, valid,
    GramOperator, bordered_solve,
)
from cvx.linalg.covariance.ewm_cov import ewm_covariance  # requires the 'ewm' extra (polars)
```

Internally the package is organized into subpackages (which can also be
imported directly, e.g. `from cvx.linalg.decomposition import qr`):

| Subpackage | Contents |
|---|---|
| [`cvx.linalg.core`](https://github.com/Jebel-Quant/linalg/tree/main/src/cvx/linalg/core) | Type aliases, exceptions/warnings, condition-number helpers, matrix validation |
| [`cvx.linalg.covariance`](https://github.com/Jebel-Quant/linalg/tree/main/src/cvx/linalg/covariance) | Covariance utilities: PCA, random covariance, correlation conversion, EWM covariance |
| [`cvx.linalg.decomposition`](https://github.com/Jebel-Quant/linalg/tree/main/src/cvx/linalg/decomposition) | Matrix decompositions: Cholesky, QR, SVD, eigenvalue routines, power iteration |
| [`cvx.linalg.kkt`](https://github.com/Jebel-Quant/linalg/tree/main/src/cvx/linalg/kkt) | Equality-constrained solves: bordered KKT systems, affine projections |
| [`cvx.linalg.norm`](https://github.com/Jebel-Quant/linalg/tree/main/src/cvx/linalg/norm) | NaN-aware vector and matrix norms, A-norms and their inverses |
| [`cvx.linalg.operators`](https://github.com/Jebel-Quant/linalg/tree/main/src/cvx/linalg/operators) | Symmetric linear operators (dense, Gram, factor, incremental) for active-set and Krylov solvers |
| [`cvx.linalg.solve`](https://github.com/Jebel-Quant/linalg/tree/main/src/cvx/linalg/solve) | NaN-aware linear-system solvers: `solve`, `lstsq`, `inv`, `det` |

The one exception to the flat-import rule is `ewm_covariance`: it lives in
`cvx.linalg.covariance.ewm_cov` and is deliberately not re-exported because it
requires the optional `polars` dependency.

## Core (`cvx.linalg.core`)

- **[`valid(matrix)`](https://github.com/Jebel-Quant/linalg/blob/main/src/cvx/linalg/core/valid.py)** — Return a boolean mask and valid submatrix by removing rows/columns with non-finite diagonal entries
- **[`cond(matrix, p=None)`](https://github.com/Jebel-Quant/linalg/blob/main/src/cvx/linalg/core/exceptions.py)** — Condition number of a matrix (NaN-aware); accepts the same `p` norm values as `numpy.linalg.cond`
- **[`check_and_warn_condition(matrix, threshold)`](https://github.com/Jebel-Quant/linalg/blob/main/src/cvx/linalg/core/exceptions.py)** — Emit `IllConditionedMatrixWarning` when the condition number exceeds the threshold
- **[`warn_ill_conditioned(cond_value, threshold)`](https://github.com/Jebel-Quant/linalg/blob/main/src/cvx/linalg/core/exceptions.py)** — Emit `IllConditionedMatrixWarning` for an already-computed condition number
- **`DEFAULT_COND_THRESHOLD`** — Default condition-number threshold (`1e12`) used by the ill-conditioning checks

### Types

- **[`Matrix`](https://github.com/Jebel-Quant/linalg/blob/main/src/cvx/linalg/core/types.py)** — Type alias for a 2-D `numpy.ndarray` with `float64` dtype
- **[`Vector`](https://github.com/Jebel-Quant/linalg/blob/main/src/cvx/linalg/core/types.py)** — Type alias for a 1-D `numpy.ndarray` with `float64` dtype

The package ships a `py.typed` marker; all public signatures are precisely annotated and verified with [ty](https://github.com/astral-sh/ty) in CI.

### Exceptions & Warnings

All exceptions and warnings live in [`core/exceptions.py`](https://github.com/Jebel-Quant/linalg/blob/main/src/cvx/linalg/core/exceptions.py):

- **`DimensionMismatchError`** — Raised when vector length does not match matrix dimension
- **`IllConditionedMatrixWarning`** — Emitted when the condition number exceeds a configurable threshold
- **`InvalidComponentsError`** — Raised when `pca` is asked for fewer than 1 or more components than the data supports
- **`NegativeWarmupError`** — Raised when a negative warmup period is passed to `ewm_covariance`
- **`NonIntegerWarmupError`** — Raised when a non-integer warmup is passed to `ewm_covariance`
- **`NonSquareMatrixError`** — Raised when a square matrix is required but the input is not square
- **`NotAMatrixError`** — Raised when a 2-D matrix is required but the input has different dimensionality
- **`SingularMatrixError`** — Raised when a matrix is numerically singular

## Covariance (`cvx.linalg.covariance`)

- **[`cov_to_corr(cov, min_var=1e-14)`](https://github.com/Jebel-Quant/linalg/blob/main/src/cvx/linalg/covariance/cov_to_corr.py)** — Convert a covariance matrix to a correlation matrix
- **[`ewm_covariance(data, assets, index_col, window, is_halflife, warmup)`](https://github.com/Jebel-Quant/linalg/blob/main/src/cvx/linalg/covariance/ewm_cov.py)** — Exponentially weighted covariance matrices from a Polars DataFrame (requires the `ewm` extra)
- **[`pca(returns, n_components=10)`](https://github.com/Jebel-Quant/linalg/blob/main/src/cvx/linalg/covariance/pca.py)** — Principal Component Analysis via SVD
- **[`rand_cov(n, seed=None)`](https://github.com/Jebel-Quant/linalg/blob/main/src/cvx/linalg/covariance/rand_cov.py)** — Random positive semi-definite covariance matrix

## Decompositions (`cvx.linalg.decomposition`)

- **[`cholesky(cov)`](https://github.com/Jebel-Quant/linalg/blob/main/src/cvx/linalg/decomposition/cholesky.py)** — Upper triangular Cholesky factor R such that R.T @ R = cov
- **[`cholesky_solve(cov, rhs)`](https://github.com/Jebel-Quant/linalg/blob/main/src/cvx/linalg/decomposition/cholesky.py)** — Solve cov @ x = rhs via Cholesky decomposition (falls back to LU for non-positive-definite matrices)
- **[`is_positive_definite(matrix)`](https://github.com/Jebel-Quant/linalg/blob/main/src/cvx/linalg/decomposition/cholesky.py)** — Return True if the matrix is symmetric positive-definite
- **[`eigh(matrix)`](https://github.com/Jebel-Quant/linalg/blob/main/src/cvx/linalg/decomposition/eigh.py)** — Eigenvalues/eigenvectors of the valid symmetric/Hermitian submatrix in ascending eigenvalue order
- **[`eigvalsh(matrix)`](https://github.com/Jebel-Quant/linalg/blob/main/src/cvx/linalg/decomposition/eigh.py)** — Eigenvalues-only convenience wrapper around `eigh`
- **[`eigvals(matrix)`](https://github.com/Jebel-Quant/linalg/blob/main/src/cvx/linalg/decomposition/eigvals.py)** — Eigenvalues of a general square matrix (supports complex output for non-symmetric matrices)
- **[`power_iteration(matrix, n_iter=1000, tol=1e-9, seed=None)`](https://github.com/Jebel-Quant/linalg/blob/main/src/cvx/linalg/decomposition/power_iteration.py)** — Estimate the dominant (largest-magnitude) eigenpair of a symmetric matrix via power iteration
- **[`qr(matrix)`](https://github.com/Jebel-Quant/linalg/blob/main/src/cvx/linalg/decomposition/qr.py)** — Reduced QR decomposition, matching `np.linalg.qr(mode='reduced')`
- **[`svd(matrix)`](https://github.com/Jebel-Quant/linalg/blob/main/src/cvx/linalg/decomposition/svd.py)** — Raw compact singular value decomposition via `np.linalg.svd(full_matrices=False)`
- **[`svd_k(matrix, k)`](https://github.com/Jebel-Quant/linalg/blob/main/src/cvx/linalg/decomposition/svd.py)** — Exact truncated rank-`k` SVD (the best rank-`k` approximation; leading triplets of the compact SVD)

## Norms (`cvx.linalg.norm`)

- **[`norm(x, ord=None)`](https://github.com/Jebel-Quant/linalg/blob/main/src/cvx/linalg/norm/norm.py)** — Norm of a vector or matrix, ignoring non-finite entries; supports all `ord` values of `np.linalg.norm`
- **[`a_norm(vector, matrix=None)`](https://github.com/Jebel-Quant/linalg/blob/main/src/cvx/linalg/norm/norm.py)** — Euclidean norm or NaN-aware matrix norm
- **[`inv_a_norm(vector, matrix=None)`](https://github.com/Jebel-Quant/linalg/blob/main/src/cvx/linalg/norm/norm.py)** — Euclidean norm or inverse NaN-aware matrix norm

## Solvers (`cvx.linalg.solve`)

- **[`solve(matrix, rhs, cond_threshold=1e12)`](https://github.com/Jebel-Quant/linalg/blob/main/src/cvx/linalg/solve/solve.py)** — Solve a linear system (vector or matrix rhs) restricted to valid rows/columns; NaN entries are returned for invalid positions
- **[`lstsq(matrix, rhs, cond_threshold=1e12)`](https://github.com/Jebel-Quant/linalg/blob/main/src/cvx/linalg/solve/lstsq.py)** — Solve a least-squares system with NaN-aware row filtering; returns `(x, residuals, rank, sv)` consistent with `numpy.linalg.lstsq`
- **[`inv(matrix, cond_threshold=1e12)`](https://github.com/Jebel-Quant/linalg/blob/main/src/cvx/linalg/solve/inv.py)** — Invert a matrix restricted to valid rows/columns; NaN rows/columns are returned for invalid positions
- **[`det(matrix, cond_threshold=1e12)`](https://github.com/Jebel-Quant/linalg/blob/main/src/cvx/linalg/solve/det.py)** — Determinant of a square matrix with NaN-aware submatrix handling; emits `IllConditionedMatrixWarning` when near-singular

## Operators (`cvx.linalg.operators`)

Active-set and Krylov solvers never need a symmetric matrix `A` as an explicit
array — they touch it through a handful of products: a full matrix-vector
product, sub-block products, and a direct solve against a principal ("free")
sub-block. The [`SymmetricOperator`](https://github.com/Jebel-Quant/linalg/blob/main/src/cvx/linalg/operators/base.py)
protocol captures exactly that contract (including `rcond_free` for detecting
rank-deficient free blocks and `restricted(free)` for a pre-sliced free-block
view), and the backends implement it at very different cost:

- **[`SymmetricOperator`](https://github.com/Jebel-Quant/linalg/blob/main/src/cvx/linalg/operators/base.py)** — Protocol exposing a symmetric matrix through `matvec`, `block_matvec`, `solve_free`, `apply_free`, `rcond_free`, `restricted`, and `diag`
- **[`DenseOperator`](https://github.com/Jebel-Quant/linalg/blob/main/src/cvx/linalg/operators/dense.py)** — Backend wrapping an explicit dense `n x n` matrix
- **[`IncrementalDenseOperator`](https://github.com/Jebel-Quant/linalg/blob/main/src/cvx/linalg/operators/dense.py)** — `DenseOperator` maintaining the free-block inverse across single-index flips for active-set sweeps
- **[`GramOperator`](https://github.com/Jebel-Quant/linalg/blob/main/src/cvx/linalg/operators/gram.py)** — Matrix-free backend for `A = M.T @ M`, represented by its factor `M`; never forms the Gram matrix
- **[`FactorOperator`](https://github.com/Jebel-Quant/linalg/blob/main/src/cvx/linalg/operators/factor.py)** — Diagonal-plus-low-rank backend `A = diag(d) + U @ Delta @ U.T` with Woodbury free-block solves
- **[`SumOperator`](https://github.com/Jebel-Quant/linalg/blob/main/src/cvx/linalg/operators/composite.py)** — Weighted sum of symmetric operators (forward-only; feed `apply_free` to a Krylov solver)

## Constrained solves (`cvx.linalg.kkt`)

Building blocks for active-set and path-tracing solvers, built on top of the
operator protocol:

- **[`bordered_solve(operator, free, c_free, rhs, d)`](https://github.com/Jebel-Quant/linalg/blob/main/src/cvx/linalg/kkt/bordered.py)** — Range-space (Schur complement) solve of the bordered KKT system `[[H_FF, C.T], [C, 0]] @ [x; nu] = [rhs; d]`, reaching the Hessian block only through a `SymmetricOperator`
- **[`AffineProjection(c, d)`](https://github.com/Jebel-Quant/linalg/blob/main/src/cvx/linalg/kkt/projection.py)** — Euclidean projection onto the affine set `{x : C x = d}`, caching the Gram matrix `C C.T` for repeated use

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
