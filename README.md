# cvx-linalg

Linear algebra utilities for portfolio optimization, part of the [jebel-quant](https://github.com/jebel-quant) ecosystem.

## Installation

```bash
pip install cvx-linalg
```

## Usage

```python
from cvx.linalg import a_norm, cholesky, inv_a_norm, pca, rand_cov, solve, valid
```

## Functions

- **`a_norm(vector, matrix=None)`** — Euclidean norm or NaN-aware matrix norm
- **`cholesky(cov)`** — Upper triangular Cholesky factor R such that R.T @ R = cov
- **`inv_a_norm(vector, matrix=None)`** — Euclidean norm or inverse NaN-aware matrix norm
- **`pca(returns, n_components)`** — Principal Component Analysis via SVD
- **`rand_cov(n, seed)`** — Random positive semi-definite covariance matrix
- **`solve(matrix, rhs)`** — Solve a linear system after removing invalid rows/columns
- **`valid(matrix)`** — Extract valid submatrix by removing rows/columns with non-finite diagonal entries
