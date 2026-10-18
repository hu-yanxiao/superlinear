#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
BUILD=${SUS2_TEST_BUILD_DIR:-$ROOT/.local_tmp/radial-hermite-table}
mkdir -p "$BUILD"
cd "$ROOT"
FLAGS=(-std=c++11 -O3 -I.)
if [[ ${SUS2_TEST_SANITIZE:-0} == 1 ]]; then
  FLAGS=(-std=c++11 -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer -I.)
elif [[ ${SUS2_TEST_SANITIZE:-0} == undefined ]]; then
  FLAGS=(-std=c++11 -O1 -g -fsanitize=undefined -fno-omit-frame-pointer -I.)
fi
"${CXX:-c++}" "${FLAGS[@]}" native_tests/radial_hermite_table_check.cpp \
  src/radial_basis.cpp src/common/utils.cpp -o "$BUILD/radial"
"$BUILD/radial"
