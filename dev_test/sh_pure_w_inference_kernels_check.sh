#!/usr/bin/env bash
# Standalone arithmetic checks; no model, MPI, BLAS or training data needed.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
BUILD=${SUS2_TEST_BUILD_DIR:-$ROOT/.local_tmp/pure-w-inference-kernels}
mkdir -p "$BUILD"
cd "$ROOT"
CXX=${CXX:-c++}
FLAGS=(-std=c++11 -O3 -I.)
if [[ ${SUS2_TEST_SANITIZE:-0} == 1 ]]; then
  FLAGS=(-std=c++11 -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer -I.)
fi
"$CXX" "${FLAGS[@]}" native_tests/radial_coordinate_only_check.cpp \
  src/radial_basis.cpp src/common/utils.cpp -o "$BUILD/radial"
"$BUILD/radial"
for kind in basis_kernels edge_force; do
  "$CXX" "${FLAGS[@]}" "native_tests/pure_w_${kind}_check.cpp" \
    dev_src/pure_w_basis_kernels.cpp -o "$BUILD/$kind"
  "$BUILD/$kind"
done
