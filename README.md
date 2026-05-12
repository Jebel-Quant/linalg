# cvx-linalg

Linear algebra utilities for portfolio optimization, part of the [jebel-quant](https://github.com/jebel-quant) ecosystem.

## Installation

```bash
pip install cvx-linalg
```

## Usage

```python
from cvx.linalg import cholesky, pca, rand_cov, valid
```

## Functions

- **`cholesky(cov)`** — Upper triangular Cholesky factor R such that R.T @ R = cov
- **`pca(returns, n_components)`** — Principal Component Analysis via SVD
- **`rand_cov(n, seed)`** — Random positive semi-definite covariance matrix
- **`valid(matrix)`** — Extract valid submatrix by removing rows/columns with non-finite diagonal entries
