#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

tmp_dir=".local_tmp/sh_two_layer_gate_lmp_table"
mkdir -p "$tmp_dir"

base_model="$tmp_dir/gated_base.sus2"
nonzero_model="$tmp_dir/gated_nonzero.sus2"
lmp_model="$tmp_dir/gated_lmp.sus2"
train="$tmp_dir/one.cfg"
base_pred="$tmp_dir/base_pred.cfg"
lmp_pred="$tmp_dir/lmp_pred.cfg"
rm -f "$base_model" "$nonzero_model" "$lmp_model" "$train" "$base_pred" "$lmp_pred"
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
  --two-layer-gate-shared-radial

python3 - "$base_model" "$nonzero_model" "$lmp_model" <<'PY'
import re
import sys

base, nonzero, lmp = sys.argv[1:]
text = open(base).read()

match = re.search(r"two_layer_gate_weights\s*=\s*\{([^}]*)\}", text, re.S)
if not match:
    raise SystemExit("missing two_layer_gate_weights")
weights = [float(x) for x in re.findall(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?", match.group(1))]
if not weights:
    raise SystemExit("empty two_layer_gate_weights")
weights[0] = 0.25
if len(weights) > 1:
    weights[1] = -0.10
replacement = "two_layer_gate_weights = {" + ", ".join(f"{x:.15e}" for x in weights) + "}"
nonzero_text = text[:match.start()] + replacement + text[match.end():]
open(nonzero, "w").write(nonzero_text)
open(lmp, "w").write(nonzero_text.replace(
    "radial_basis_type = RBChebyshev_sss",
    "radial_basis_type = RBChebyshev_sss_lmp",
    1,
))
PY

./bin/mlp-sus2 calc-efs "$nonzero_model" "$train" "$base_pred" >/dev/null
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
if not base:
    raise SystemExit(f"failed to parse base prediction {sys.argv[1]}")
if not lmp:
    raise SystemExit(f"failed to parse _lmp prediction {sys.argv[2]}")
if len(base) != len(lmp):
    raise SystemExit(f"value count mismatch: {len(base)} vs {len(lmp)}")
max_err = max((abs(a - b) for a, b in zip(base, lmp)), default=0.0)
if not math.isfinite(max_err) or max_err > 2.0e-5:
    raise SystemExit(f"_lmp gate table prediction mismatch: {max_err}")
PY
