#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

tmp_dir=".local_tmp/sh_two_layer_gate_shared_radial"
mkdir -p "$tmp_dir"

plain="$tmp_dir/plain.sus2"
shared="$tmp_dir/shared.sus2"
saved="$tmp_dir/shared.saved.sus2"
train="$tmp_dir/one.cfg"
plain_pred="$tmp_dir/plain_pred.cfg"
shared_pred="$tmp_dir/shared_pred.cfg"
rm -f "$plain" "$shared" "$saved" "$train" "$plain_pred" "$shared_pred"
awk '{print} /^END_CFG/{exit}' example/train.cfg > "$train"

common_opts=(
  --species-count=2
  --l-max=2
  --k-max=2
  --body-order=4
  --body-l-max=2,2,2,2
  --radial-basis-size=4
  --cutoff=5.0
  --write-sh-scalar-info
  --two-layer-gate
)

./bin/mlp-sus2 init-sh "$plain" "${common_opts[@]}"
./bin/mlp-sus2 init-sh "$shared" "${common_opts[@]}" \
  --two-layer-gate-shared-radial

python3 - "$plain" "$shared" <<'PY'
import re
import sys

for path in sys.argv[1:]:
    text = open(path).read()
    match = re.search(r"two_layer_gate_weights = \{([^}]*)\}", text)
    if not match:
        raise SystemExit(f"missing two_layer_gate_weights in {path}")
    weights = [float(x.strip()) for x in match.group(1).split(",") if x.strip()]
    replacement = "two_layer_gate_weights = {" + ", ".join("0.000000000000000e+00" for _ in weights) + "}"
    open(path, "w").write(text[:match.start()] + replacement + text[match.end():])
PY

grep -q "two_layer_gate_radial_mode = shared-radial" "$shared"
grep -q "two_layer_gate_radial_coeff_count = 24" "$shared"
grep -q "two_layer_gate_radial_coeffs" "$shared"

./bin/mlp-sus2 calc-efs "$plain" "$train" "$plain_pred" >/dev/null
./bin/mlp-sus2 calc-efs "$shared" "$train" "$shared_pred" >/dev/null

python3 - "$plain_pred" "$shared_pred" <<'PY'
import math
import sys

def read_numbers(path):
    vals = []
    with open(path) as handle:
        for raw in handle:
            for token in raw.split():
                try:
                    vals.append(float(token))
                except ValueError:
                    pass
    return vals

plain = read_numbers(sys.argv[1])
shared = read_numbers(sys.argv[2])
if len(plain) != len(shared):
    raise SystemExit(f"value count mismatch: {len(plain)} vs {len(shared)}")
err = max((abs(a - b) for a, b in zip(plain, shared)), default=0.0)
if not math.isfinite(err) or err > 1.0e-10:
    raise SystemExit(f"zero-gate shared-radial predictions differ: {err}")
PY

./bin/mlp-sus2 train "$shared" "$train" \
  --max-iter=1 \
  --skip-preinit \
  --trained-pot-name="$saved" \
  --energy-weight=1 \
  --force-weight=0 \
  --stress-weight=0 >/dev/null

grep -q "two_layer_gate_radial_mode = shared-radial" "$saved"
grep -q "two_layer_gate_radial_coeff_count = 24" "$saved"
grep -q "two_layer_gate_radial_coeffs" "$saved"
