#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

tmp_dir=".codex_tmp/sh_two_layer_gate_edge_l1_init"
mkdir -p "$tmp_dir"

edge_model="$tmp_dir/edge_l1.mtp"
legacy_model="$tmp_dir/legacy_gate.mtp"
legacy_pred="$tmp_dir/legacy_gate.pred.cfg"
reject_log="$tmp_dir/reject_double.log"

./bin/mlp-sus2 init-sh "$edge_model" \
  --species-count=2 \
  --l-max=2 \
  --k-max=3 \
  --body-order=4 \
  --body-l-max=2,2,2,2 \
  --radial-basis-size=4 \
  --cutoff=5.0 \
  --write-sh-scalar-info \
  --two-layer-gate \
  --two-layer-gate-edge-l1 >/dev/null

grep -q "two_layer_gate_enabled = true" "$edge_model"
grep -q "two_layer_gate_edge_l1_enabled = true" "$edge_model"
grep -q "two_layer_gate_edge_l1_source_moment_count" "$edge_model"
grep -q "two_layer_gate_edge_l1_source_moment_indices" "$edge_model"
if grep -q "two_layer_gate_edge_l1_source_mu_count" "$edge_model"; then
  echo "edge L1 init should write full L=1 moment sources, not raw mu sources" >&2
  exit 1
fi
source_count=$(awk '/two_layer_gate_edge_l1_source_moment_count/ {print $3}' "$edge_model")
if [[ "$source_count" -le 3 ]]; then
  echo "full edge L1 gate should include multi-body L=1 source moments; got $source_count" >&2
  exit 1
fi
weight_count=$(awk '/two_layer_gate_edge_l1_weight_count/ {print $3}' "$edge_model")
if [[ "$weight_count" -ne $((9 * source_count)) ]]; then
  echo "edge L1 weight count should be radial_func_count * source_count; got $weight_count for $source_count sources" >&2
  exit 1
fi
grep -q "two_layer_gate_edge_l1_weights" "$edge_model"

if ./bin/mlp-sus2 init-sh "$tmp_dir/reject_double.mtp" \
  --species-count=1 \
  --l-max=1 \
  --k-max=2 \
  --body-order=3 \
  --body-l-max=1,1,1,1 \
  --radial-basis-size=3 \
  --cutoff=5.0 \
  --two-layer-gate \
  --two-layer-gate-site-mode=double \
  --two-layer-gate-edge-l1 >"$reject_log" 2>&1; then
  echo "edge L1 gate should reject double site mode in the first version" >&2
  exit 1
fi
grep -q "double is obsolete" "$reject_log"

./bin/mlp-sus2 init-sh "$legacy_model" \
  --species-count=2 \
  --l-max=2 \
  --k-max=3 \
  --body-order=4 \
  --body-l-max=2,2,2,2 \
  --radial-basis-size=4 \
  --cutoff=5.0 \
  --write-sh-scalar-info \
  --two-layer-gate >/dev/null

./bin/mlp-sus2 calc-efs "$legacy_model" example/train.cfg "$legacy_pred" >/dev/null

grep -q "Energy" "$legacy_pred"
if grep -q "two_layer_gate_edge_l1_enabled = true" "$legacy_model"; then
  echo "legacy gate init unexpectedly enabled edge L1 metadata" >&2
  exit 1
fi
