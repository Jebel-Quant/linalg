#!/bin/bash -eu
# ClusterFuzzLite build script — installs cvx-linalg and compiles each Python
# harness in tests/fuzz/ via OSS-Fuzz's compile_python_fuzzer helper.

cd "$SRC"

# Pin pip so the build environment is reproducible and only changes through a
# reviewed bump (the same rationale as the SHA-pinned base image).
pip3 install --upgrade "pip==24.3.1"

# Install the package and its runtime dependencies (numpy, pandas) into the
# build environment so PyInstaller can discover and bundle cvx.linalg into each
# frozen fuzzer binary. Without this, the harness would fail to import
# cvx.linalg at runtime inside the ClusterFuzzLite runner.
pip3 install .

for fuzzer in tests/fuzz/fuzz_*.py; do
  compile_python_fuzzer "$fuzzer"
done
