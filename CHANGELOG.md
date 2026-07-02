# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com),
and entries are generated from [Conventional Commits](https://www.conventionalcommits.org).

## [0.9.1] - 2026-07-02

### New Features
- Add rcond_free and IncrementalDenseOperator to the operator layer (#80)

## [0.9.0] - 2026-07-02

### New Features
- Add ClusterFuzzLite fuzzing scaffold for cvx-linalg (#61)
- SymmetricOperator abstraction and backends (#76)
- Add ridge (Tikhonov) support to GramOperator (#78)

### Documentation
- *(api)* Document the 8 missing public exports in API reference (#69)
- *(api)* Declare explicit __all__ for the public API (#70) (#72)
- Complete the package-level export overview in __init__ (#71) (#73)

### Maintenance
- Chore(deps-dev)(deps-dev): bump the python-dependencies group with 4 updates (#60)
- Chore(deps)(deps): bump the github-actions group with 3 updates (#64)
- Chore(deps-dev)(deps-dev): bump the python-dependencies group with 3 updates (#63)
- Update rhiza to v1.0.0 (#74)
- Chore(deps-dev)(deps-dev): bump polars in the python-dependencies group (#75)
- Update rhiza to v1.0.1 (#77)
- Group linalg modules into subpackages (#79)

### Other Changes
- Sync Rhiza template v0.19.3 → v0.19.4 (#59)
- Sync Rhiza template v0.19.4 → v0.19.6 (#62)
- Sync Rhiza template v0.19.6 → v0.19.9 (#67)
- Bump version 0.8.0 → 0.9.0

## [0.8.0] - 2026-06-16

### New Features
- Add power_iteration and svd_k primitives (#58)

### Maintenance
- Chore(deps)(deps): bump the github-actions group with 8 updates (#55)
- Add Rhiza Claude commands (/rhiza_quality, /rhiza_update) (#54)
- Chore(deps-dev)(deps-dev): bump the python-dependencies group with 3 updates (#56)

### Other Changes
- Quality hardening: typing, coverage gates, property tests, mutation baseline (#52)
- Sync Rhiza template v0.18.8 → v0.19.3 (#57)
- Bump version 0.7.0 → 0.8.0

## [0.7.0] - 2026-06-11

### Maintenance
- Chore(deps)(deps): bump actions/checkout in the github-actions group (#48)
- Chore(deps-dev)(deps-dev): bump the python-dependencies group with 3 updates (#49)
- Address code-quality, packaging, and API consistency issues (#50)

### Other Changes
- Bump version 0.6.2 → 0.7.0

## [0.6.2] - 2026-06-08

### Bug Fixes
- Bump pandas minimum to 3.0.0 to fix lowest-direct CI
- Bump rhiza_benchmark.yml reference to v0.18.4 (#44)

### Maintenance
- Update rhiza to v0.15.2 (#39)
- Update rhiza to v0.18.4 (#41)
- Add pip dependabot entry for .rhiza/requirements
- Chore(deps)(deps): bump the python-dependencies group with 3 updates (#43)
- Chore(deps)(deps): bump the github-actions group with 8 updates (#42)
- Chore(deps)(deps): bump the python-dependencies group with 2 updates (#46)
- Chore(deps)(deps): bump the github-actions group with 9 updates (#45)

### Other Changes
- Sync rhiza templates to v0.10.9 (#35)
- Rhiza-version
- Remove commented and unused GitHub templates
- Bump version 0.6.1 → 0.6.2

## [0.6.1] - 2026-05-20

### New Features
- Add cov_to_corr — convert covariance matrix to correlation matrix (#34)

### Maintenance
- Chore(deps)(deps): bump the python-dependencies group with 2 updates (#33)

### Other Changes
- Remove ewm_covariance from top-level API to avoid eager polars import
- Sync rhiza templates to v0.10.7 (#32)
- Bump version 0.6.0 → 0.6.1

## [0.6.0] - 2026-05-18

### New Features
- Add det() — matrix determinant with NaN-aware handling
- Add inv() — guarded matrix inversion
- Add norm() — NaN-aware general matrix/vector norm (closes #12)
- Add qr decomposition utility
- Expose cond() — public NaN-aware condition number (#23)
- *(linalg)* Add standalone `svd()` and route `pca()` through it (#26)
- Add lstsq() — least-squares solver with NaN-aware row filtering (#22)

### Bug Fixes
- Update inv() doctests for numpy scalar repr compatibility
- Tighten type annotations to satisfy ty type checker

### Other Changes
- Initial plan
- Merge pull request #21 from Jebel-Quant/copilot/add-det-function-matrix-determinant
- Initial plan
- Merge pull request #24 from Jebel-Quant/copilot/add-inv-guarded-matrix-inversion
- Merge pull request #25 from Jebel-Quant/norm
- Initial plan
- Merge pull request #27 from Jebel-Quant/copilot/add-qr-decomposition
- Add NaN-aware `eigh()` and `eigvalsh()` for symmetric/Hermitian eigendecomposition (#28)
- Marimo as dev dependency
- Add `eigvals()` API for general square matrices (#29)
- Add Marimo notebook sources under `book/marimo/notebooks` (#31)
- Improve mkdocs.yml: add notebooks/reports nav, matrix logo, src path
- Improve API reference page with icon, intro, and import example
- Rewrite api.md with per-symbol directives grouped by category
- Bump version 0.5.1 → 0.6.0

## [0.5.1] - 2026-05-18

### Documentation
- Update README with v0.5.0 API and source links

### Maintenance
- Integrate basanos linalg tests into cvx-linalg
- Merge cholesky and cholesky_solve into one function

### Other Changes
- Bump version 0.5.0 → 0.5.1

## [0.5.0] - 2026-05-18

### New Features
- Add domain errors, Cholesky-first solving, and condition-number checking

### Bug Fixes
- Correct doctest examples in exceptions and solve modules

### Other Changes
- Merge pull request #11 from Jebel-Quant/domain-errors
- Bump version 0.4.1 → 0.5.0

## [0.4.1] - 2026-05-14

### New Features
- Add ewm_covariance with pure-numpy implementation
- Restore polars-based ewm_covariance implementation
- Export ewm_covariance from package and update README

### Documentation
- Add badges to README

### Maintenance
- Bring coverage to 100%

### Other Changes
- Merge pull request #10 from Jebel-Quant/ewm_cov
- Bump version 0.4.0 → 0.4.1

## [0.4.0] - 2026-05-14

### New Features
- Add tinycta linalg helpers

### Documentation
- List inv_a_norm in package api docs

### Maintenance
- Streamline norm helper calculations

### Other Changes
- Initial plan
- Potential fix for pull request finding
- Potential fix for pull request finding
- Merge pull request #9 from Jebel-Quant/copilot/bring-over-linalg
- Bump version 0.3.0 → 0.4.0

## [0.3.0] - 2026-05-13

### Other Changes
- Initial plan
- Remove license blurb from all source files
- Merge pull request #4 from Jebel-Quant/copilot/remove-license-blurb
- Initial plan
- Add types.py with Matrix type alias and export from __init__
- Add test for Matrix type alias in types.py
- Merge pull request #6 from Jebel-Quant/copilot/add-types-py-for-linear-algebra
- Remove polars dependency, make pca fully numpy-based
- Fix incorrect False in pca doctest for reconstruction check
- Potential fix for pull request finding
- Merge pull request #7 from Jebel-Quant/polars
- Bump version 0.2.0 → 0.3.0

## [0.2.0] - 2026-05-13

### New Features
- Add rhiza template seed files
- Integrate rhiza framework

### Bug Fixes
- Remove stubs extra-path from ty config (no stubs dir)
- Add module docstrings to test packages
- Make all passes cleanly

### Maintenance
- Add uv.lock
- Chore(deps)(deps): bump github/codeql-action
- Remove dev dependency group and bumpversion config
- Migrate from pandas to polars

### Other Changes
- Initial commit: extract cvx.linalg from cvxrisk
- Add README.md
- Merge pull request #1 from Jebel-Quant/dependabot/github_actions/github-actions-8abaa2cbc6
- Update pyproject.toml
- Disable coverage report exclusions
- Remove unused coverage report settings
- Merge pull request #2 from Jebel-Quant/tschm-patch-1
- Bump version 0.1.0 → 0.2.0

<!-- generated by git-cliff -->
