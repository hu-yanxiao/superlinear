#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

bash dev_test/diagonal_coordinate_scaling_check.sh

tmp_dir=".codex_tmp/sh_release3_bfgs_coordinate_normalization"
mkdir -p "$tmp_dir"

initial_model="$tmp_dir/initial.mtp"
seed_model="$tmp_dir/seed.mtp"
train_cfg="$tmp_dir/one.cfg"
physical_trace="$tmp_dir/physical.csv"
normalized_trace="$tmp_dir/normalized.csv"
normalized_repeat_trace="$tmp_dir/normalized_repeat.csv"
normalized_model="$tmp_dir/normalized.mtp"
normalized_repeat_model="$tmp_dir/normalized_repeat.mtp"
combined_model="$tmp_dir/combined_weight_norm.mtp"
rm -f "$tmp_dir"/*.mtp "$tmp_dir"/*.cfg "$tmp_dir"/*.csv "$tmp_dir"/*.log

awk '{print} /^END_CFG/{exit}' example/train.cfg > "$train_cfg"

./bin/mlp-sus2 init-sh "$initial_model" \
  --species-count=2 \
  --l-max=2 \
  --k-max=2 \
  --body-order=3 \
  --body-l-max=2,2,2,2 \
  --radial-basis-size=4 \
  --init-params=same \
  --cutoff=5.0 \
  --write-sh-scalar-info \
  --two-layer-gate

./bin/mlp-sus2 train "$initial_model" "$train_cfg" \
  --init-params=same \
  --trained-pot-name="$seed_model" \
  --max-iter=1 \
  --skip-preinit \
  --energy-weight=1 \
  --force-weight=0.1 \
  --stress-weight=0.01 \
  --radial-smooth=0 >"$tmp_dir/seed.log" 2>&1

common_train_args=(
  --max-iter=1
  --skip-preinit
  --energy-weight=1
  --force-weight=0.1
  --stress-weight=0.01
  --radial-smooth=0
  --scalar-head-l2=1e-5
  --gate-scalar-l2=1e-4
  --gate-mix-l2=1e-5
  --gate-full-l2=1e-4
  --gate-x-l2=1e-4
)

./bin/mlp-sus2 train "$seed_model" "$train_cfg" \
  "${common_train_args[@]}" \
  --bfgs-coordinate-normalization=none \
  --optimizer-trace-file="$physical_trace" \
  --trained-pot-name="$tmp_dir/physical.mtp" \
  >"$tmp_dir/physical.log" 2>&1

./bin/mlp-sus2 train "$seed_model" "$train_cfg" \
  "${common_train_args[@]}" \
  --bfgs-coordinate-normalization=efs-gn \
  --bfgs-coordinate-probes=2 \
  --bfgs-coordinate-floor=1e-6 \
  --optimizer-trace-file="$normalized_trace" \
  --curr-pot-name="$normalized_model" \
  --trained-pot-name="$tmp_dir/normalized_trained.mtp" \
  >"$tmp_dir/normalized.log" 2>&1

./bin/mlp-sus2 train "$seed_model" "$train_cfg" \
  "${common_train_args[@]}" \
  --bfgs-coordinate-normalization=efs-gn \
  --bfgs-coordinate-probes=2 \
  --bfgs-coordinate-floor=1e-6 \
  --optimizer-trace-file="$normalized_repeat_trace" \
  --curr-pot-name="$normalized_repeat_model" \
  --trained-pot-name="$tmp_dir/normalized_repeat_trained.mtp" \
  >"$tmp_dir/normalized_repeat.log" 2>&1

grep -q "BFGS coordinate normalization: efs-gn probes=2 floor=1e-06" \
  "$tmp_dir/normalized.log"
grep -q "BFGS coordinate normalization block name=species-prefactor" \
  "$tmp_dir/normalized.log"
grep -q "BFGS coordinate normalization block name=scalar-head" \
  "$tmp_dir/normalized.log"
grep -q "BFGS coordinate normalization block name=gate1-additive" \
  "$tmp_dir/normalized.log"
grep -q "BFGS coordinate normalization block name=gate1-weight" \
  "$tmp_dir/normalized.log"
grep -q "BFGS coordinate normalization initialized mode=efs-gn" \
  "$tmp_dir/normalized.log"
grep -q "physical_reference=exact" "$tmp_dir/normalized.log"

if grep -Eq \
  "bfgs_coordinate|coordinate_normalization|coordinate_scale|hutchinson" \
  "$normalized_model"; then
  echo "BFGS coordinate-normalization state leaked into the model file" >&2
  exit 1
fi

grep "BFGS coordinate normalization block" "$tmp_dir/normalized.log" \
  | sed -E 's/^\[[^]]+\] //' >"$tmp_dir/normalized.blocks"
grep "BFGS coordinate normalization block" "$tmp_dir/normalized_repeat.log" \
  | sed -E 's/^\[[^]]+\] //' >"$tmp_dir/normalized_repeat.blocks"
cmp "$tmp_dir/normalized.blocks" "$tmp_dir/normalized_repeat.blocks"
cmp "$normalized_trace" "$normalized_repeat_trace"

./bin/mlp-sus2 train "$seed_model" "$train_cfg" \
  "${common_train_args[@]}" \
  --radial-parameterization=weight-norm \
  --bfgs-coordinate-normalization=efs-gn \
  --bfgs-coordinate-probes=2 \
  --trained-pot-name="$combined_model" \
  >"$tmp_dir/combined.log" 2>&1
grep -q "outer radial weight-normalization coordinates enabled" \
  "$tmp_dir/combined.log"
grep -q "BFGS coordinate normalization initialized mode=efs-gn" \
  "$tmp_dir/combined.log"

python3 - "$physical_trace" "$normalized_trace" <<'PY'
import csv
import math
import pathlib
import sys

physical_path, normalized_path = map(pathlib.Path, sys.argv[1:])


def read_losses(path):
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise SystemExit(f"missing optimizer trace row in {path}")
    losses = [float(row["total_loss"]) for row in rows]
    if not all(math.isfinite(value) for value in losses):
        raise SystemExit(f"non-finite optimizer trace loss in {path}")
    return losses


read_losses(physical_path)
read_losses(normalized_path)
PY

./bin/mlp-sus2 calc-efs \
  "$normalized_model" "$train_cfg" "$tmp_dir/predicted.cfg"

if [[ -n "${MLIP_OLD_RELEASE3_BIN:-}" ]]; then
  "$MLIP_OLD_RELEASE3_BIN" calc-efs \
    "$normalized_model" "$train_cfg" "$tmp_dir/old_release3.cfg"
fi

echo "release3 BFGS coordinate-normalization check: PASS"
