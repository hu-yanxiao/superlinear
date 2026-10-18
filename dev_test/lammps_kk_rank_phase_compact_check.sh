#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
build_dir=$(mktemp -d "${TMPDIR:-/tmp}/sus2-kk-rank-phase-compact.XXXXXX")
trap 'rm -rf "$build_dir"' EXIT

cxx=${CXX:-c++}
"$cxx" -std=c++17 -O2 -Wall -Wextra -pedantic \
  -I"$repo_dir/interfaces/lammps/ML-SUS2-KK" \
  "$repo_dir/native_tests/lammps_kk_rank_phase_compact_check.cpp" \
  -o "$build_dir/lammps_kk_rank_phase_compact_check"

"$build_dir/lammps_kk_rank_phase_compact_check" \
  "$repo_dir/interfaces/lammps/ML-SUS2-KK/pair_sus2_mtp_kokkos.cpp" \
  "$repo_dir/interfaces/lammps/ML-SUS2-KK/pair_sus2_mtp_kokkos.h"

install_root="$build_dir/lammps-install-smoke"
mkdir -p "$install_root/src"
"$repo_dir/interfaces/lammps/scripts/install_into_lammps.sh" \
  "$install_root" >/dev/null
test -f "$install_root/src/KOKKOS/sus2_mtp_rank_phase_layout.h"
