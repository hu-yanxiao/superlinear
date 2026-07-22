#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

tmp_dir=".codex_tmp/sh_optimizer_checkpoint"
mkdir -p "$tmp_dir"

cxx="${CXX:-c++}"
"$cxx" \
  -std=c++11 \
  -O2 \
  -Wall \
  -Wextra \
  -pedantic \
  -I. \
  codex_tests/optimizer_checkpoint_check.cpp \
  -o "$tmp_dir/optimizer_checkpoint_check"

"$tmp_dir/optimizer_checkpoint_check" "$tmp_dir/current.mtp"

if compgen -G "$tmp_dir/current.mtp.tmp.*" >/dev/null; then
  echo "optimizer checkpoint test left a temporary file" >&2
  exit 1
fi
