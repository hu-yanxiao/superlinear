#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

tmp_dir=".local_tmp/sh_two_layer_residual_init"
mkdir -p "$tmp_dir"

model="$tmp_dir/residual.sus2"
log="$tmp_dir/init.log"
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
  --two-layer-gate-shared-radial \
  --two-layer-residual >"$log" 2>&1; then
  echo "residual two-layer gate should be rejected by mu-body-order gate mode" >&2
  exit 1
fi

grep -q "not supported by mu-body-order gate" "$log"
