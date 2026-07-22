#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

tmp_dir=".codex_tmp/sh_optimizer_accepted_state"
mkdir -p "$tmp_dir"

cxx="${CXX:-c++}"
"$cxx" \
  -std=c++11 \
  -O2 \
  -Wall \
  -Wextra \
  -pedantic \
  -I. \
  codex_tests/optimizer_accepted_state_check.cpp \
  -o "$tmp_dir/optimizer_accepted_state_check"

"$tmp_dir/optimizer_accepted_state_check"
