#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
TMP_ROOT="$ROOT/.codex_tmp"
mkdir -p "$TMP_ROOT"
TMP=$(mktemp -d "$TMP_ROOT/full-head-coordinate-scaling.XXXXXX")
trap 'rm -rf "$TMP"' EXIT

${CXX:-c++} -std=c++11 -O2 -I"$ROOT" \
  "$ROOT/codex_tests/full_head_coordinate_scaling_check.cpp" \
  -o "$TMP/check"
"$TMP/check"
