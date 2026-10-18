#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

tmp_dir=".local_tmp/sh_release3_radial_weight_norm"
mkdir -p "$tmp_dir"

initial_model="$tmp_dir/initial.sus2"
seed_model="$tmp_dir/seed.sus2"
signed_resume_model="$tmp_dir/signed_resume_seed.sus2"
signed_resume_trace="$tmp_dir/signed_resume.csv"
signed_resume_none_trace="$tmp_dir/signed_resume_none.csv"
signed_resume_refusal_model="$tmp_dir/signed_resume_refusal.sus2"
signed_resume_opt_in_model="$tmp_dir/signed_resume_opt_in.sus2"
untrained_weight_norm_model="$tmp_dir/untrained_weight_norm.sus2"
train_cfg="$tmp_dir/one.cfg"
physical_model="$tmp_dir/physical_step0.sus2"
weight_norm_model="$tmp_dir/weight_norm_step0.sus2"
physical_trace="$tmp_dir/physical.csv"
weight_norm_trace="$tmp_dir/weight_norm.csv"
rm -f "$tmp_dir"/*.sus2 "$tmp_dir"/*.cfg "$tmp_dir"/*.csv "$tmp_dir"/*.log

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

# A continued physical model may have either sign in every radial row. Verify
# that enabling both normalization layers converts outer and gate1 rows to
# optimizer coordinates without changing the loaded physical coefficients.
python3 - "$seed_model" "$signed_resume_model" <<'PY'
import pathlib
import re
import sys

source, target = map(pathlib.Path, sys.argv[1:])
text = source.read_text()


def negate_first(pattern, label, value):
    match = pattern.search(value)
    if not match:
        raise SystemExit(f"missing {label} radial coefficient block")
    first = float(match.group(2))
    replacement = match.group(1) + f"{-abs(first):.15e}"
    return value[: match.start()] + replacement + value[match.end() :]


text = negate_first(
    re.compile(
        r"(\bradial_coeffs\b\s*\d+\s*-\s*\d+\s*\{\s*)"
        r"([-+0-9.eE]+)",
        re.S,
    ),
    "outer",
    text,
)
text = negate_first(
    re.compile(r"(two_layer_gate_radial_coeffs\s*=\s*\{)([^,}]+)"),
    "gate1",
    text,
)
target.write_text(text)
PY

# A complete checkpoint must never be silently rewritten merely because the
# caller selected physical optimizer coordinates.  The old conversion remains
# available only through an explicit compatibility opt-in.
signed_resume_sha_before="$(sha256_file "$signed_resume_model")"
if ./bin/mlp-sus2 train "$signed_resume_model" "$train_cfg" \
  --radial-parameterization=physical \
  --trained-pot-name="$signed_resume_refusal_model" \
  --max-iter=1 \
  --skip-preinit \
  --energy-weight=1 \
  --force-weight=0.1 \
  --stress-weight=0 \
  --radial-smooth=0 >"$tmp_dir/signed_resume_refusal.log" 2>&1; then
  echo "physical resume unexpectedly modified a complete checkpoint without opt-in" >&2
  exit 1
fi
signed_resume_sha_after="$(sha256_file "$signed_resume_model")"
if [[ "$signed_resume_sha_before" != "$signed_resume_sha_after" ]]; then
  echo "failed physical resume modified its input checkpoint" >&2
  exit 1
fi
test ! -e "$signed_resume_refusal_model"
grep -q "Refusing to modify a complete trained checkpoint" \
  "$tmp_dir/signed_resume_refusal.log"
grep -q -- "--radial-parameterization=weight-norm" \
  "$tmp_dir/signed_resume_refusal.log"
grep -q -- "--allow-physical-radial-sign-repair" \
  "$tmp_dir/signed_resume_refusal.log"

./bin/mlp-sus2 train "$signed_resume_model" "$train_cfg" \
  --radial-parameterization=physical \
  --allow-physical-radial-sign-repair \
  --trained-pot-name="$signed_resume_opt_in_model" \
  --max-iter=1 \
  --skip-preinit \
  --energy-weight=1 \
  --force-weight=0.1 \
  --stress-weight=0 \
  --radial-smooth=0 >"$tmp_dir/signed_resume_opt_in.log" 2>&1
test -s "$signed_resume_opt_in_model"
grep -q "Legacy physical radial sign repair explicitly enabled" \
  "$tmp_dir/signed_resume_opt_in.log"
grep -Eq "Repaired .*radial first coefficients; running one linear solve" \
  "$tmp_dir/signed_resume_opt_in.log"

./bin/mlp-sus2 check-radial-weight-norm-roundtrip-dev \
  "$signed_resume_model" \
  --abs-tolerance=2e-12 \
  --rel-tolerance=2e-12 >"$tmp_dir/signed_resume_roundtrip.log"

./bin/mlp-sus2 train "$signed_resume_model" "$train_cfg" \
  --radial-parameterization=weight-norm \
  --bfgs-coordinate-normalization=none \
  --optimizer-trace-file="$signed_resume_none_trace" \
  --trained-pot-name="$tmp_dir/signed_resume_none_trained.sus2" \
  --max-iter=1 \
  --skip-preinit \
  --energy-weight=1 \
  --force-weight=0.1 \
  --stress-weight=0 \
  --radial-smooth=0 >"$tmp_dir/signed_resume_none.log" 2>&1

./bin/mlp-sus2 train "$signed_resume_model" "$train_cfg" \
  --radial-parameterization=weight-norm \
  --bfgs-coordinate-normalization=efs-gn \
  --bfgs-coordinate-probes=2 \
  --optimizer-trace-file="$signed_resume_trace" \
  --trained-pot-name="$tmp_dir/signed_resume_trained.sus2" \
  --max-iter=1 \
  --skip-preinit \
  --energy-weight=1 \
  --force-weight=0.1 \
  --stress-weight=0 \
  --radial-smooth=0 >"$tmp_dir/signed_resume_train.log" 2>&1
grep -q "physical_load=roundtrip-verified" "$tmp_dir/signed_resume_train.log"
grep -q "physical_reference=exact" "$tmp_dir/signed_resume_train.log"
test -s "$signed_resume_trace"
if grep -Eq "Repaired .*radial first|radial first-coeff sign repair" \
  "$tmp_dir/signed_resume_train.log"; then
  echo "weight-normalized resume unexpectedly repaired physical radial coefficients" >&2
  exit 1
fi

python3 - "$tmp_dir/signed_resume_none.log" "$tmp_dir/signed_resume_train.log" <<'PY'
import math
import pathlib
import re
import sys

pattern = re.compile(
    r"optimizer physical startup objective total_loss=([0-9eE+.-]+)"
)


def startup_loss(path):
    match = pattern.search(path.read_text())
    if match is None:
        raise SystemExit(f"missing startup physical objective in {path}")
    value = float(match.group(1))
    if not math.isfinite(value):
        raise SystemExit(f"non-finite startup physical objective in {path}")
    return value


none_path, normalized_path = map(pathlib.Path, sys.argv[1:])
none_loss = startup_loss(none_path)
normalized_loss = startup_loss(normalized_path)
scale = max(1.0, abs(none_loss), abs(normalized_loss))
if abs(none_loss - normalized_loss) > 2.0e-12 * scale:
    raise SystemExit(
        "combined weight normalization and EFS-GN changed the loaded physical "
        f"objective: none={none_loss:.17e} normalized={normalized_loss:.17e}"
    )
PY

./bin/mlp-sus2 train "$seed_model" "$train_cfg" \
  --radial-parameterization=physical \
  --optimizer-trace-file="$physical_trace" \
  --curr-pot-name="$physical_model" \
  --trained-pot-name="$tmp_dir/physical_trained.sus2" \
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
  --trained-pot-name="$tmp_dir/weight_norm_trained.sus2" \
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
  --trained-pot-name="$tmp_dir/do_lin_trained.sus2" \
  --max-iter=1 \
  --skip-preinit \
  --energy-weight=1 \
  --force-weight=0.1 \
  --stress-weight=0 \
  --radial-smooth=0 >"$tmp_dir/do_lin.log" 2>&1
# MPI builds report the completed solve explicitly.  The serial implementation
# executes the same initial refresh but historically has no per-solve log line,
# so retain a portable assertion for both build modes.
if grep -q "Mode: Serial" "$tmp_dir/do_lin.log"; then
  grep -q "do-lin enabled for first 1 optimizer steps, frequency 1" \
    "$tmp_dir/do_lin.log"
else
  grep -q "TrainLinear solve done" "$tmp_dir/do_lin.log"
fi

python3 - "$tmp_dir/physical.log" "$tmp_dir/weight_norm.log" <<'PY'
import pathlib
import re
import sys

physical_log, weight_norm_log = map(pathlib.Path, sys.argv[1:])
pattern = re.compile(
    r"optimizer physical startup objective total_loss=([0-9eE+.-]+)"
)


def startup_loss(path):
    match = pattern.search(path.read_text())
    if match is None:
        raise SystemExit(f"missing optimizer startup objective in {path}")
    return float(match.group(1))


physical_loss = startup_loss(physical_log)
weight_norm_loss = startup_loss(weight_norm_log)
loss_scale = max(1.0, abs(physical_loss), abs(weight_norm_loss))
if abs(physical_loss - weight_norm_loss) > 1.0e-10 * loss_scale:
    raise SystemExit(
        f"weight-norm changed the initial physical objective: "
        f"{weight_norm_loss} vs {physical_loss}"
    )
PY

if [[ -n "${MLIP_OLD_RELEASE3_BIN:-}" ]]; then
  "$MLIP_OLD_RELEASE3_BIN" calc-efs \
    "$weight_norm_model" "$train_cfg" "$tmp_dir/old_release3.cfg"
fi

echo "release3 radial weight-normalization check: PASS"
