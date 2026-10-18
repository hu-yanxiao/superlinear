#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${TMPDIR:-/tmp}/sus2_zbl_pair_check.$$"
mkdir -p "$build_dir"
trap 'rm -rf "$build_dir"' EXIT

cxx="${CXX:-c++}"
"$cxx" -std=c++11 -I"$repo_dir" \
	"$repo_dir/native_tests/zbl_pair_check.cpp" \
	"$repo_dir/src/zbl_pair.cpp" \
	-o "$build_dir/zbl_pair_check"
"$build_dir/zbl_pair_check"
