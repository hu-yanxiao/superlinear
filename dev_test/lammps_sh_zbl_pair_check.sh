#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${TMPDIR:-/tmp}/lammps_sh_zbl_pair_check"

g++ -std=c++17 -O2 -I"$ROOT" \
  "$ROOT/native_tests/lammps_sh_zbl_pair_check.cpp" \
  -o "$OUT"
"$OUT"
