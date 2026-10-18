#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

tmp_dir=".local_tmp/sh_two_layer_gate_init"
mkdir -p "$tmp_dir"

model="$tmp_dir/gated.sus2"
saved="$tmp_dir/gated.saved.sus2"
pred="$tmp_dir/pred.cfg"
train="$tmp_dir/one.cfg"
rm -f "$model" "$saved" "$pred" "$train"
awk '{print} /^END_CFG/{exit}' example/train.cfg > "$train"

./bin/mlp-sus2 init-sh "$model" \
  --species-count=2 \
  --l-max=2 \
  --k-max=2 \
  --body-order=4 \
  --body-l-max=2,2,2,2 \
  --radial-basis-size=4 \
  --cutoff=5.0 \
  --write-sh-scalar-info \
  --two-layer-gate

grep -q "two_layer_gate_enabled = true" "$model"
grep -q "two_layer_gate_include_one_body = false" "$model"
grep -q "two_layer_gate_body_order_max = 3" "$model"
grep -q "two_layer_gate_scalar_indices" "$model"
grep -q "two_layer_gate_weights" "$model"

./bin/mlp-sus2 calc-efs "$model" "$train" "$pred" >/dev/null
./bin/mlp-sus2 train "$model" "$train" \
  --max-iter=1 \
  --skip-preinit \
  --trained-pot-name="$saved" \
  --energy-weight=1 \
  --force-weight=0 \
  --stress-weight=0 >/dev/null

grep -q "two_layer_gate_enabled = true" "$saved"
grep -q "two_layer_gate_weights" "$saved"
