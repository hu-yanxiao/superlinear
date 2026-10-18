#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

tmp_dir=".local_tmp/sh_two_layer_gate_shared_radial_default"
mkdir -p "$tmp_dir"

gate_init="$tmp_dir/gate_default.sus2"
plain_init="$tmp_dir/plain_init.sus2"
train="$tmp_dir/one.cfg"
upgraded="$tmp_dir/upgraded_default_gate.sus2"
upgrade_log="$tmp_dir/upgrade_train.log"
rm -f "$gate_init" "$plain_init" "$train" "$upgraded" "$upgrade_log"
awk '{print} /^END_CFG/{exit}' example/train.cfg > "$train"

./bin/mlp-sus2 init-sh "$gate_init" \
  --species-count=2 \
  --l-max=2 \
  --k-max=2 \
  --body-order=4 \
  --body-l-max=2,2,2,2 \
  --radial-basis-size=4 \
  --cutoff=5.0 \
  --write-sh-scalar-info \
  --two-layer-gate >/dev/null

grep -q "two_layer_gate_radial_mode = shared-radial" "$gate_init"
grep -q "two_layer_gate_radial_coeff_count = 24" "$gate_init"
grep -q "two_layer_gate_radial_coeffs" "$gate_init"

./bin/mlp-sus2 init-sh "$plain_init" \
  --species-count=2 \
  --l-max=2 \
  --k-max=2 \
  --body-order=4 \
  --body-l-max=2,2,2,2 \
  --radial-basis-size=4 \
  --cutoff=5.0 >/dev/null

./bin/mlp-sus2 train "$plain_init" "$train" \
  --trained-pot-name="$upgraded" \
  --max-iter=1 \
  --skip-preinit \
  --two-layer-gate \
  --energy-weight=1 \
  --force-weight=0 \
  --stress-weight=0 >"$upgrade_log"

grep -q "SUS2-SH plain-to-gate upgrade enabled" "$upgrade_log"
grep -q "two_layer_gate_radial_mode = shared-radial" "$upgraded"
grep -q "two_layer_gate_radial_coeff_count = 24" "$upgraded"
grep -q "two_layer_gate_radial_coeffs" "$upgraded"
