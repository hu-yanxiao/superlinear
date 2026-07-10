#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

tmp_dir=".codex_tmp/sh_two_layer_gate_decoupled_scaling"
mkdir -p "$tmp_dir"

base_model="$tmp_dir/gated_decoupled.mtp"
lmp_model="$tmp_dir/gated_decoupled_lmp.mtp"
bad_model="$tmp_dir/bad_multi_decoupled.mtp"
shared_init="$tmp_dir/shared_init.mtp"
shared_model="$tmp_dir/shared_complete.mtp"
train="$tmp_dir/one.cfg"
base_pred="$tmp_dir/base_pred.cfg"
lmp_pred="$tmp_dir/lmp_pred.cfg"
bad_log="$tmp_dir/bad.log"
shared_train_log="$tmp_dir/shared_train.log"
rm -f "$base_model" "$lmp_model" "$bad_model" "$shared_init" "$shared_model" \
  "$train" "$base_pred" "$lmp_pred" "$bad_log" "$shared_train_log"
awk '{print} /^END_CFG/{exit}' example/train.cfg > "$train"

./bin/mlp-sus2 init-sh "$base_model" \
  --species-count=2 \
  --l-max=2 \
  --k-max=2 \
  --body-order=4 \
  --body-l-max=2,2,2,2 \
  --radial-basis-size=4 \
  --cutoff=5.0 \
  --write-sh-scalar-info \
  --two-layer-gate \
  --two-layer-gate-decoupled-scaling

grep -q "two_layer_gate_scaling_mode = decoupled" "$base_model"
grep -q "two_layer_gate_scaling_coeff_count = 48" "$base_model"
grep -q "two_layer_gate_scaling_coeffs" "$base_model"

if ./bin/mlp-sus2 init-sh "$bad_model" \
  --species-count=2 \
  --l-max=2 \
  --k-max=2 \
  --body-order=4 \
  --body-l-max=2,2,2,2 \
  --radial-basis-size=4 \
  --cutoff=5.0 \
  --two-layer-gate \
  --multi-layer-gate \
  --two-layer-gate-decoupled-scaling >"$bad_log" 2>&1; then
  echo "expected multi-layer decoupled scaling init to fail" >&2
  exit 1
fi
grep -q "one-layer gate" "$bad_log"

python3 - "$base_model" "$lmp_model" <<'PY'
import re
import sys

base, lmp = sys.argv[1:]
text = open(base).read()

def rewrite_vector(src, name, values_fn):
    pattern = rf"{name} = \{{([^}}]*)\}}"
    match = re.search(pattern, src, re.S)
    if not match:
        raise SystemExit(f"missing {name}")
    values = [
        float(x)
        for x in re.findall(
            r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?",
            match.group(1),
        )
    ]
    values = values_fn(values)
    replacement = name + " = {" + ", ".join(f"{x:.15e}" for x in values) + "}"
    return src[:match.start()] + replacement + src[match.end():]

def set_prefix(seed):
    def apply(values):
        for i, value in enumerate(seed[: len(values)]):
            values[i] = value
        return values
    return apply

def perturb_gate_scaling(values):
    for i in range(len(values)):
        values[i] += (0.07 if i % 2 == 0 else -0.04) * (1.0 + 0.03 * (i % 5))
    return values

text = rewrite_vector(text, "two_layer_gate_weights", set_prefix([0.31, -0.17, 0.12, -0.06]))
text = rewrite_vector(text, "two_layer_gate_scaling_coeffs", perturb_gate_scaling)
open(base, "w").write(text)
open(lmp, "w").write(
    text.replace("radial_basis_type = RBChebyshev_sss",
                 "radial_basis_type = RBChebyshev_sss_lmp",
                 1)
)
PY

./bin/mlp-sus2 calc-efs "$base_model" "$train" "$base_pred" >/dev/null
./bin/mlp-sus2 calc-efs "$lmp_model" "$train" "$lmp_pred" >/dev/null

python3 - "$base_pred" "$lmp_pred" <<'PY'
import math
import sys

def read_values(path):
    values = []
    force_columns = None
    expect_energy = False
    with open(path) as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if expect_energy:
                values.append(float(line.split()[0]))
                expect_energy = False
            elif line == "Energy":
                expect_energy = True
            elif line.startswith("AtomData:"):
                header = line.split()[1:]
                force_columns = [header.index(name) for name in ("fx", "fy", "fz")]
            elif force_columns is not None and line[0].isdigit():
                parts = line.split()
                if len(parts) > max(force_columns):
                    values.extend(float(parts[idx]) for idx in force_columns)
    return values

base = read_values(sys.argv[1])
lmp = read_values(sys.argv[2])
if not base or not lmp:
    raise SystemExit("failed to parse prediction output")
if len(base) != len(lmp):
    raise SystemExit(f"value count mismatch: {len(base)} vs {len(lmp)}")
max_err = max((abs(a - b) for a, b in zip(base, lmp)), default=0.0)
if not math.isfinite(max_err) or max_err > 2.0e-5:
    raise SystemExit(f"decoupled _lmp gate table prediction mismatch: {max_err}")
PY

./bin/mlp-sus2 check-loss-gradient-dev "$base_model" "$train" \
  --max-configs=1 \
  --energy-weight=1 \
  --force-weight=0.2 \
  --stress-weight=0 \
  --gate-x-l2=0 \
  --displacement=1.0e-3 \
  --abs-tolerance=3.0e-5 \
  --rel-tolerance=3.0e-4

./bin/mlp-sus2 init-sh "$shared_init" \
  --species-count=2 \
  --l-max=2 \
  --k-max=2 \
  --body-order=4 \
  --body-l-max=2,2,2,2 \
  --radial-basis-size=4 \
  --cutoff=5.0 \
  --write-sh-scalar-info \
  --two-layer-gate \
  --two-layer-gate-mode=mu-scalar-full >/dev/null

./bin/mlp-sus2 train "$shared_init" "$train" \
  --trained-pot-name="$shared_model" \
  --max-iter=1 \
  --skip-preinit \
  --shift \
  --energy-weight=0 \
  --force-weight=0 \
  --stress-weight=0 \
  --radial-smooth=0 >"$shared_train_log"

./bin/mlp-sus2 check-two-layer-gate-decoupled-upgrade-dev \
  "$shared_model" "$train" \
  --max-configs=1 \
  --abs-tolerance=1.0e-10
