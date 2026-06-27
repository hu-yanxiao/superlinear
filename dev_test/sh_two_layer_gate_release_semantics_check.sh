#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

tmp_dir=".codex_tmp/sh_two_layer_gate_release_semantics"
mkdir -p "$tmp_dir"

model="$tmp_dir/release_gate.mtp"
reject_log="$tmp_dir/double_reject.log"

./bin/mlp-sus2 init-sh "$model" \
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

python3 - "$model" <<'PY'
import re
import sys

text = open(sys.argv[1], encoding="utf-8").read()

def scalar_int(name):
    match = re.search(rf"^\s*{name}\s*=\s*(\d+)\s*$", text, re.M)
    if not match:
        raise SystemExit(f"missing {name}")
    return int(match.group(1))

species_count = scalar_int("species_count")
mu_count = scalar_int("radial_funcs_count")
additive_count = scalar_int("two_layer_gate_additive_coeff_count")
gate_type_count = scalar_int("two_layer_gate_type_coeff_count")

expected_additive = species_count * mu_count
if additive_count != expected_additive:
    raise SystemExit(
        f"expected additive coeff count {expected_additive}, got {additive_count}"
    )
if gate_type_count != species_count:
    raise SystemExit(
        f"expected gate type coeff count {species_count}, got {gate_type_count}"
    )
if not re.search(r"^two_layer_gate_type_coeffs\s*=\s*\{", text, re.M):
    raise SystemExit("missing independent gate type coefficients")
if re.search(r"^two_layer_gate_tanh_amplitude\s*=", text, re.M):
    raise SystemExit("release gate should not write tanh amplitude metadata")
if not re.search(r"^two_layer_gate_scale_mode\s*=\s*additive-node\s*$", text, re.M):
    raise SystemExit("missing additive-node gate scale mode")
PY

if ./bin/mlp-sus2 init-sh "$tmp_dir/double_gate.mtp" \
  --species-count=2 \
  --l-max=2 \
  --k-max=2 \
  --body-order=4 \
  --body-l-max=2,2,2,2 \
  --radial-basis-size=4 \
  --cutoff=5.0 \
  --write-sh-scalar-info \
  --two-layer-gate \
  --two-layer-gate-site-mode=double >"$reject_log" 2>&1; then
  echo "expected --two-layer-gate-site-mode=double to be rejected" >&2
  exit 1
fi

grep -Eiq "double|neighbor|obsolete|unsupported" "$reject_log"

echo "release two-layer gate semantics check: PASS"
