#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

tmp_dir=".codex_tmp/sh_release3_radial_weight_norm_rechart"
mkdir -p "$tmp_dir"

cxx="${CXX:-c++}"
"$cxx" \
  -std=c++11 \
  -O2 \
  -Wall \
  -Wextra \
  -pedantic \
  -I. \
  codex_tests/radial_weight_norm_rechart_check.cpp \
  -o "$tmp_dir/radial_weight_norm_rechart_check"

"$tmp_dir/radial_weight_norm_rechart_check"
