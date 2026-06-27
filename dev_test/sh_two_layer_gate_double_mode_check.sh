#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

tmp_dir=".codex_tmp/sh_two_layer_gate_double_mode"
mkdir -p "$tmp_dir"

model="$tmp_dir/double_rejected.mtp"
log="$tmp_dir/double_rejected.log"
rm -f "$model" "$log"

if ./bin/mlp-sus2 init-sh "$model" \
  --species-count=2 \
  --l-max=2 \
  --k-max=2 \
  --body-order=4 \
  --body-l-max=2,2,2,2 \
  --radial-basis-size=4 \
  --cutoff=5.0 \
  --write-sh-scalar-info \
  --two-layer-gate \
  --two-layer-gate-site-mode=double >"$log" 2>&1; then
  echo "double gate mode should be rejected" >&2
  exit 1
fi

grep -q "double is obsolete" "$log"

echo "release double-gate rejection check: PASS"
