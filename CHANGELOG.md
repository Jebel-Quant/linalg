## [0.6.2] - 2026-06-08

### 🐛 Bug Fixes

- Bump pandas minimum to 3.0.0 to fix lowest-direct CI
- Bump rhiza_benchmark.yml reference to v0.18.4 (#44)

### 💼 Other

- Bump version 0.6.1 → 0.6.2

### ⚙️ Miscellaneous Tasks

- Bump rhiza to v0.14.1 with github-project profile (#36)
- Bump rhiza template to v0.11.0 (#37)
- Update rhiza to v0.15.2 (#39)
- Bump rhiza to v0.17.0 (#40)
- Update rhiza to v0.18.4 (#41)
- Add pip dependabot entry for .rhiza/requirements
- Bump rhiza to v0.18.8 (#47)
## [0.6.1] - 2026-05-20

### 🚀 Features

- Add cov_to_corr — convert covariance matrix to correlation matrix (#34)

### 💼 Other

- Remove ewm_covariance from top-level API to avoid eager polars import
- Bump version 0.6.0 → 0.6.1
## [0.6.0] - 2026-05-18

### 🚀 Features

- Add det() — matrix determinant with NaN-aware handling
- Add inv() — guarded matrix inversion
- Add norm() — NaN-aware general matrix/vector norm (closes #12)
- Add qr decomposition utility
- Expose cond() — public NaN-aware condition number (#23)
- *(linalg)* Add standalone `svd()` and route `pca()` through it (#26)
- Add lstsq() — least-squares solver with NaN-aware row filtering (#22)

### 🐛 Bug Fixes

- Update inv() doctests for numpy scalar repr compatibility
- Tighten type annotations to satisfy ty type checker

### 💼 Other

- Bump version 0.5.1 → 0.6.0
## [0.5.1] - 2026-05-18

### 💼 Other

- Bump version 0.5.0 → 0.5.1

### 🚜 Refactor

- Merge cholesky and cholesky_solve into one function

### 📚 Documentation

- Update README with v0.5.0 API and source links

### 🧪 Testing

- Integrate basanos linalg tests into cvx-linalg
## [0.5.0] - 2026-05-18

### 🚀 Features

- Add domain errors, Cholesky-first solving, and condition-number checking

### 🐛 Bug Fixes

- Correct doctest examples in exceptions and solve modules

### 💼 Other

- Bump version 0.4.1 → 0.5.0
## [0.4.1] - 2026-05-14

### 🚀 Features

- Add ewm_covariance with pure-numpy implementation
- Restore polars-based ewm_covariance implementation
- Export ewm_covariance from package and update README

### 💼 Other

- Bump version 0.4.0 → 0.4.1

### 📚 Documentation

- Add badges to README

### 🧪 Testing

- Bring coverage to 100%
## [0.4.0] - 2026-05-14

### 🚀 Features

- Add tinycta linalg helpers

### 💼 Other

- Bump version 0.3.0 → 0.4.0

### 🚜 Refactor

- Streamline norm helper calculations

### 📚 Documentation

- List inv_a_norm in package api docs
## [0.3.0] - 2026-05-13

### 💼 Other

- Bump version 0.2.0 → 0.3.0
## [0.2.0] - 2026-05-13

### 🚀 Features

- Add rhiza template seed files
- Integrate rhiza framework

### 🐛 Bug Fixes

- Remove stubs extra-path from ty config (no stubs dir)
- Add module docstrings to test packages
- Make all passes cleanly

### 💼 Other

- Bump version 0.1.0 → 0.2.0

### 🚜 Refactor

- Migrate from pandas to polars

### ⚙️ Miscellaneous Tasks

- Add uv.lock
- Remove dev dependency group and bumpversion config
