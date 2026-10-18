#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
build_dir=$(mktemp -d "${TMPDIR:-/tmp}/sus2-kk-gate-cache-layout.XXXXXX")
trap 'rm -rf "$build_dir"' EXIT

cxx=${CXX:-c++}
"$cxx" -std=c++17 -O2 -Wall -Wextra -pedantic \
  -I"$repo_dir/interfaces/lammps/ML-SUS2-KK" \
  "$repo_dir/native_tests/lammps_kk_gate_cache_layout_check.cpp" \
  -o "$build_dir/lammps_kk_gate_cache_layout_check"

"$build_dir/lammps_kk_gate_cache_layout_check" \
  "$repo_dir/interfaces/lammps/ML-SUS2-KK/pair_sus2_mtp_kokkos.cpp"
