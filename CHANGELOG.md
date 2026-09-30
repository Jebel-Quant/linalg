# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com),
and entries are generated from [Conventional Commits](https://www.conventionalcommits.org).

## [1.1.0] - 2026-09-30

### New Features
- Allow cond_threshold=None to skip the condition-number check (#181)

### Bug Fixes
- Mark package Production/Stable to match 1.0.0 (#113) (#115)
- Declare a discoverable bumpversion config in pyproject.toml (#125)
- Point the cholesky rhs deprecation at 2.0 instead of the shipped 1.0 (#130)
- Enforce the documented 100% coverage gate (#171) (#174)
- *(cholesky)* Solve with the factor, not two general LU solves (#178)

### Documentation
- Add root CLAUDE.md documenting owner split (#106)
- Replace stale `make validate` with `make rhiza-test` in CLAUDE.md (#141)
- Add a verified NaN-aware quickstart to the README (#146)
- Write README examples as pycon doctests (#177)
- Document cond_threshold=None in the README (#182)

### Performance
- Reuse factorisations; fix NaN rhs regression from #178 (#180)

### Dependencies
- *(deps)* Prune [dependency-groups] to what the environment actually needs (#133)
- *(deps)* Prune [dependency-groups] to what the environment needs, and drop `lint` (#134)

### Maintenance
- Chore(deps)(deps): bump docker/login-action in the github-actions group (#94)
- Chore(deps-dev)(deps-dev): bump the python-dependencies group with 2 updates (#95)
- Update rhiza to v1.1.1 (#101)
- Add SupportsMatvec protocol and extract validation helpers (#102)
- Update rhiza to v1.1.2 (#103)
- Update rhiza to v1.1.3 (#104)
- Chore(deps-dev)(deps-dev): bump the python-dependencies group with 5 updates (#107)
- Exclude fuzzing, scorecard, and weekly workflows from sync (#108)
- Update rhiza to v1.2.1 (#109)
- Mirror tests one-file-per-source-module (#110, #111) (#112)
- Chore(deps-dev)(deps-dev): bump the python-dependencies group with 4 updates (#117)
- Chore(deps)(deps): bump the github-actions group with 10 updates (#116)
- *(pyproject)* Modernize Python version and license metadata (#118)
- Chore(deps-dev)(deps-dev): bump the python-dependencies group with 6 updates (#120)
- Chore(deps)(deps): bump docker/login-action in the github-actions group (#119)
- Update rhiza to v1.2.5 (#121)
- Update rhiza to v1.3.0 (#124)
- Make the shared operator helpers public within base.py (#129)
- Chore(deps)(deps): bump the github-actions group with 3 updates (#131)
- Chore(deps-dev)(deps-dev): bump the python-dependencies group with 4 updates (#132)
- Update rhiza to v1.3.2 (#135)
- *(ci)* Bump the rhiza pin to v1.3.3 (#138)
- Chore(deps-dev)(deps-dev): bump the python-dependencies group with 2 updates (#137)
- Chore(deps)(deps): bump the github-actions group with 2 updates (#136)
- Update rhiza to v1.3.3 (#139)
- Chore(deps-dev)(deps-dev): bump hypothesis (#142)
- Update rhiza to v1.3.4 (#143)
- Remove .rhiza/.env and .rhiza/.gitignore (#147)
- Update rhiza to v1.4.2 (#148)
- Update rhiza to v1.4.2 (#150)
- Update rhiza to v1.5.0 (#151)
- Remove stale template-owned files that v1.5.0 dropped (#152)
- Prune exclude entries the template no longer ships (#153)
- Remove mutation testing (#154)
- Drop the exclude entries for the retired mutation/fuzzing workflows (#155)
- Update rhiza to v1.6.0 (#156)
- Chore(deps-dev)(deps-dev): bump the python-dependencies group with 2 updates (#157)
- Update rhiza to v1.7.1 (#158)
- Chore(deps-dev)(deps-dev): bump polars in the python-dependencies group (#159)
- Update rhiza to v1.7.2 (#160)
- Chore(deps-dev)(deps-dev): bump hypothesis (#161)
- Update rhiza to v1.8.0 (#163)
- Derive the version from the git tag (#164)
- Chore(deps-dev)(deps-dev): bump the python-dependencies group with 3 updates (#165)
- Chore(deps)(deps): bump starlette from 1.2.0 to 1.3.1 (#167)
- Chore(deps)(deps): bump anyio from 4.13.0 to 4.14.2 (#169)
- Chore(deps)(deps): bump python-multipart from 0.0.29 to 0.0.31 (#168)
- Chore(deps)(deps): bump pymdown-extensions from 10.21.3 to 11.0.1 (#166)
- Chore(deps-dev)(deps-dev): bump pandas in the python-dependencies group (#170)
- Assert exact solution in test_lstsq_nan_rhs_filtered (#172) (#173)
- Chore(deps-dev)(deps-dev): bump the python-dependencies group with 2 updates (#175)
- Update rhiza to v1.9.0 (#176)

### Other Changes
- Declare the deliberate test-layout deviations in pyproject.toml (#123)
- Modify excluded paths in template.yml (#149)

## [1.0.0] - 2026-07-07

### Documentation
- Restructure README around the nested subpackage layout (#90)

### Maintenance
- Update rhiza to v1.0.2 (#93)

## [0.9.6] - 2026-07-03

### Other Changes
- *(free)* Pre-sliced free-block operators (closes #88) (#89)
- Bump version 0.9.5 → 0.9.6

## [0.9.5] - 2026-07-03

### New Features
- Add diag attribute to SymmetricOperator and its backends (#87)

### Other Changes
- Bump version 0.9.4 → 0.9.5

## [0.9.4] - 2026-07-02

### New Features
- Operator-aware power_iteration + SumOperator composite (#86)

### Maintenance
- *(operators)* Split module and cover degenerate paths (#85)

### Other Changes
- Bump version 0.9.3 → 0.9.4

## [0.9.3] - 2026-07-02

### New Features
- Add k attribute (number of factors) to FactorOperator (#82)

### Other Changes
- Bump version 0.9.2 → 0.9.3

## [0.9.2] - 2026-07-02

### New Features
- Kkt subpackage — bordered_solve + AffineProjection (#81)

### Other Changes
- Bump version 0.9.1 → 0.9.2

## [0.9.1] - 2026-07-02

### New Features
- Add rcond_free and IncrementalDenseOperator to the operator layer (#80)

### Other Changes
- Bump version 0.9.0 → 0.9.1

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
