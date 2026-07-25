#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

tmp_dir=".codex_tmp/bfgs_restart_state"
mkdir -p "$tmp_dir"

cxx="${CXX:-c++}"
read -r -a extra_cppflags <<< "${BFGS_TEST_CPPFLAGS:-}"
read -r -a extra_ldflags <<< "${BFGS_TEST_LDFLAGS:--lblas}"

"$cxx" \
  -std=c++11 \
  -O2 \
  -Wall \
  -Wextra \
  -pedantic \
  -I. \
  "${extra_cppflags[@]}" \
  codex_tests/bfgs_restart_state_check.cpp \
  src/common/bfgs.cpp \
  "${extra_ldflags[@]}" \
  -o "$tmp_dir/bfgs_restart_state_check"

"$tmp_dir/bfgs_restart_state_check"
