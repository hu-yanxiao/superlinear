#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

tmp_dir=".codex_tmp/sh_outer_type_mu_active_gradient"
mkdir -p "$tmp_dir"

model="$tmp_dir/gated_init.mtp"
train="$tmp_dir/one.cfg"
rm -f "$model" "$train"
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

read -r coeff_start coeff_end < <(python3 - "$model" <<'PY'
import re
import sys

text = open(sys.argv[1]).read()
def get_int(name):
    match = re.search(rf"^\s*{name}\s*=\s*(\d+)", text, re.M)
    if not match:
        raise SystemExit(f"missing {name}")
    return int(match.group(1))

C = get_int("species_count")
R = get_int("radial_basis_size")
K = get_int("radial_funcs_count")
if K < 2:
    raise SystemExit("test requires at least two mu channels")

radial_begin = C + 2 * C * C * K
block = R + C
mu = 1
start = radial_begin + mu * block + R
print(start, start + C)
PY
)

output=$(./bin/mlp-sus2 check-loss-gradient-dev "$model" "$train" \
  --max-configs=1 \
  --energy-weight=1 \
  --force-weight=1 \
  --stress-weight=0 \
  --gate-x-l2=0 \
  --displacement=1.0e-3 \
  --abs-tolerance=1.0e-5 \
  --rel-tolerance=1.0e-4 \
  --coeff-start="$coeff_start" \
  --coeff-end="$coeff_end" \
  --stage-active-only)

echo "$output"
echo "$output" | grep -q "checked_coeffs=2"
echo "outer type mu active gradient check: PASS"
