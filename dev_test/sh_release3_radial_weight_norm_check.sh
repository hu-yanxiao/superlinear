#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

tmp_dir=".codex_tmp/sh_release3_radial_weight_norm"
mkdir -p "$tmp_dir"

initial_model="$tmp_dir/initial.mtp"
seed_model="$tmp_dir/seed.mtp"
negative_gate_model="$tmp_dir/negative_gate_seed.mtp"
untrained_weight_norm_model="$tmp_dir/untrained_weight_norm.mtp"
train_cfg="$tmp_dir/one.cfg"
physical_model="$tmp_dir/physical_step0.mtp"
weight_norm_model="$tmp_dir/weight_norm_step0.mtp"
do_lin_model="$tmp_dir/weight_norm_do_lin_step0.mtp"
physical_trace="$tmp_dir/physical.csv"
weight_norm_trace="$tmp_dir/weight_norm.csv"
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

# Cover the initialization path where TrainLinear runs before the nonlinear
# optimizer has created its weight-normalization coordinates.
./bin/mlp-sus2 train "$initial_model" "$train_cfg" \
  --radial-parameterization=weight-norm \
  --init-params=same \
  --trained-pot-name="$untrained_weight_norm_model" \
  --max-iter=1 \
  --skip-preinit \
  --energy-weight=1 \
  --force-weight=0.1 \
  --stress-weight=0 \
  --radial-smooth=0 >"$tmp_dir/untrained_weight_norm.log" 2>&1
grep -q "outer radial weight-normalization coordinates enabled" \
  "$tmp_dir/untrained_weight_norm.log"
grep -Eq "gate1_rows=[1-9][0-9]*" "$tmp_dir/untrained_weight_norm.log"

./bin/mlp-sus2 check-radial-weight-norm-roundtrip-dev \
  "$untrained_weight_norm_model" \
  --abs-tolerance=2e-12 \
  --rel-tolerance=2e-12 >"$tmp_dir/roundtrip.log"

# Produce one complete model so both coordinate systems start from identical
# physical coefficients without an implicit untrained-model warmup.
./bin/mlp-sus2 train "$initial_model" "$train_cfg" \
  --init-params=same \
  --trained-pot-name="$seed_model" \
  --max-iter=1 \
  --skip-preinit \
  --energy-weight=1 \
  --force-weight=0.1 \
  --stress-weight=0 \
  --radial-smooth=0 >"$tmp_dir/seed.log" 2>&1

for mu in 0 3; do
  ./bin/mlp-sus2 check-radial-weight-norm-gradient-dev \
    "$seed_model" "$train_cfg" \
    --radial-block=outer \
    --mu="$mu" \
    --max-configs=1 \
    --energy-weight=1 \
    --force-weight=0.1 \
    --stress-weight=0.01 \
    --radial-smooth=1e-6 \
    --scalar-head-l2=1e-5 \
    --gate-scalar-l2=1e-4 \
    --gate-mix-l2=1e-5 \
    --gate-full-l2=1e-4 \
    --gate-x-l2=1e-4 \
    --displacement=1e-6 \
    --abs-tolerance=2e-5 \
    --rel-tolerance=2e-4 >"$tmp_dir/gradient_mu${mu}.log"

  ./bin/mlp-sus2 check-radial-weight-norm-gradient-dev \
    "$seed_model" "$train_cfg" \
    --radial-block=gate1 \
    --mu="$mu" \
    --max-configs=1 \
    --energy-weight=1 \
    --force-weight=0.1 \
    --stress-weight=0.01 \
    --radial-smooth=1e-6 \
    --scalar-head-l2=1e-5 \
    --gate-scalar-l2=1e-4 \
    --gate-mix-l2=1e-5 \
    --gate-full-l2=1e-4 \
    --gate-x-l2=1e-4 \
    --displacement=1e-3 \
    --abs-tolerance=2e-4 \
    --rel-tolerance=2e-4 >"$tmp_dir/gradient_gate1_mu${mu}.log"
done

# A continued gate model may have either sign in every radial row. Verify that
# the trainer selects a signed chart without changing the serialized values.
python3 - "$seed_model" "$negative_gate_model" <<'PY'
import pathlib
import re
import sys

source, target = map(pathlib.Path, sys.argv[1:])
text = source.read_text()
pattern = re.compile(r"(two_layer_gate_radial_coeffs\s*=\s*\{)([^,}]+)")
match = pattern.search(text)
if not match:
    raise SystemExit("missing gate1 radial coefficient block")
first = float(match.group(2))
replacement = match.group(1) + f"{-abs(first):.15e}"
target.write_text(text[: match.start()] + replacement + text[match.end() :])
PY

./bin/mlp-sus2 check-radial-weight-norm-roundtrip-dev \
  "$negative_gate_model" \
  --abs-tolerance=2e-12 \
  --rel-tolerance=2e-12 >"$tmp_dir/negative_gate_roundtrip.log"

./bin/mlp-sus2 train "$negative_gate_model" "$train_cfg" \
  --radial-parameterization=weight-norm \
  --trained-pot-name="$tmp_dir/negative_gate_trained.mtp" \
  --max-iter=1 \
  --skip-preinit \
  --energy-weight=1 \
  --force-weight=0.1 \
  --stress-weight=0 \
  --radial-smooth=0 >"$tmp_dir/negative_gate_train.log" 2>&1
grep -Eq "gate1_rows=[1-9][0-9]*" "$tmp_dir/negative_gate_train.log"

./bin/mlp-sus2 train "$seed_model" "$train_cfg" \
  --radial-parameterization=physical \
  --optimizer-trace-file="$physical_trace" \
  --curr-pot-name="$physical_model" \
  --trained-pot-name="$tmp_dir/physical_trained.mtp" \
  --max-iter=1 \
  --skip-preinit \
  --energy-weight=1 \
  --force-weight=0.1 \
  --stress-weight=0 \
  --radial-smooth=0 >"$tmp_dir/physical.log" 2>&1

./bin/mlp-sus2 train "$seed_model" "$train_cfg" \
  --radial-parameterization=weight-norm \
  --optimizer-trace-file="$weight_norm_trace" \
  --curr-pot-name="$weight_norm_model" \
  --trained-pot-name="$tmp_dir/weight_norm_trained.mtp" \
  --max-iter=1 \
  --skip-preinit \
  --energy-weight=1 \
  --force-weight=0.1 \
  --stress-weight=0 \
  --radial-smooth=0 >"$tmp_dir/weight_norm.log" 2>&1

grep -q "Radial parameterization: weight-norm" "$tmp_dir/weight_norm.log"
grep -q "outer radial weight-normalization coordinates enabled" "$tmp_dir/weight_norm.log"
grep -Eq "gate1_rows=[1-9][0-9]*" "$tmp_dir/weight_norm.log"
if grep -q "radial_parameterization\|weight_norm" "$weight_norm_model"; then
  echo "weight-normalization training state leaked into the model file" >&2
  exit 1
fi

# Exercise the TrainLinear refresh. In weight-norm mode it must not project the
# radial rows through the legacy Orthogonalize() call.
./bin/mlp-sus2 train "$seed_model" "$train_cfg" \
  --radial-parameterization=weight-norm \
  --do-lin \
  --do-lin-steps=1 \
  --do-lin-freq=1 \
  --curr-pot-name="$do_lin_model" \
  --trained-pot-name="$tmp_dir/do_lin_trained.mtp" \
  --max-iter=1 \
  --skip-preinit \
  --energy-weight=1 \
  --force-weight=0.1 \
  --stress-weight=0 \
  --radial-smooth=0 >"$tmp_dir/do_lin.log" 2>&1

python3 - "$physical_trace" "$weight_norm_trace" \
  "$seed_model" "$do_lin_model" <<'PY'
import csv
import pathlib
import re
import sys

physical_trace, weight_norm_trace, seed_model, do_lin_model = map(
    pathlib.Path, sys.argv[1:]
)


def first_loss(path):
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise SystemExit(f"missing optimizer trace row in {path}")
    return float(rows[0]["total_loss"])


physical_loss = first_loss(physical_trace)
weight_norm_loss = first_loss(weight_norm_trace)
loss_scale = max(1.0, abs(physical_loss), abs(weight_norm_loss))
if abs(physical_loss - weight_norm_loss) > 1.0e-10 * loss_scale:
    raise SystemExit(
        f"weight-norm changed the initial physical objective: "
        f"{weight_norm_loss} vs {physical_loss}"
    )


def outer_radial(path):
    text = path.read_text()
    match = re.search(
        r"\bradial_coeffs\s*(.*?)\balpha_moments_count\s*=",
        text,
        re.S,
    )
    if not match:
        raise SystemExit(f"cannot find outer radial block in {path}")
    rows = re.findall(r"\{([^{}]+)\}", match.group(1))
    return [
        [float(value.strip()) for value in row.split(",")]
        for row in rows
    ]


def named_vector(path, name):
    text = path.read_text()
    match = re.search(rf"\b{name}\s*=\s*\{{([^}}]+)\}}", text)
    if not match:
        raise SystemExit(f"cannot find {name} in {path}")
    return [float(value.strip()) for value in match.group(1).split(",")]


seed_rows = outer_radial(seed_model)
do_lin_rows = outer_radial(do_lin_model)
if len(seed_rows) != len(do_lin_rows):
    raise SystemExit("TrainLinear changed outer radial row count")
max_radial_error = 0.0
for seed_row, do_lin_row in zip(seed_rows, do_lin_rows):
    # Type coefficients follow the primitive radial coefficients in the same
    # serialized row. TrainLinear does not change either part.
    if len(seed_row) != len(do_lin_row):
        raise SystemExit("TrainLinear changed outer radial row width")
    for lhs, rhs in zip(seed_row, do_lin_row):
        max_radial_error = max(max_radial_error, abs(lhs - rhs))
if max_radial_error > 2.0e-13:
    raise SystemExit(
        f"weight-norm TrainLinear refresh changed outer radial coefficients: "
        f"{max_radial_error}"
    )

seed_gate_radial = named_vector(seed_model, "two_layer_gate_radial_coeffs")
do_lin_gate_radial = named_vector(do_lin_model, "two_layer_gate_radial_coeffs")
if len(seed_gate_radial) != len(do_lin_gate_radial):
    raise SystemExit("TrainLinear changed gate1 radial coefficient count")
max_gate_radial_error = max(
    abs(lhs - rhs)
    for lhs, rhs in zip(seed_gate_radial, do_lin_gate_radial)
)
if max_gate_radial_error > 2.0e-13:
    raise SystemExit(
        f"weight-norm TrainLinear refresh changed gate1 radial coefficients: "
        f"{max_gate_radial_error}"
    )
PY

if [[ -n "${MLIP_OLD_RELEASE3_BIN:-}" ]]; then
  "$MLIP_OLD_RELEASE3_BIN" calc-efs \
    "$weight_norm_model" "$train_cfg" "$tmp_dir/old_release3.cfg"
fi

echo "release3 radial weight-normalization check: PASS"
