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
from cvx.linalg import a_norm, cholesky, ewm_covariance, inv_a_norm, pca, rand_cov, solve, valid
```

## Functions

- **`a_norm(vector, matrix=None)`** — Euclidean norm or NaN-aware matrix norm
- **`cholesky(cov)`** — Upper triangular Cholesky factor R such that R.T @ R = cov
- **`ewm_covariance(data, assets, index_col, window, is_halflife, warmup)`** — Exponentially weighted covariance matrices from a Polars DataFrame
- **`inv_a_norm(vector, matrix=None)`** — Euclidean norm or inverse NaN-aware matrix norm
- **`pca(returns, n_components)`** — Principal Component Analysis via SVD
- **`rand_cov(n, seed)`** — Random positive semi-definite covariance matrix
- **`solve(matrix, rhs)`** — Solve a linear system after removing invalid rows/columns
- **`valid(matrix)`** — Extract valid submatrix by removing rows/columns with non-finite diagonal entries
