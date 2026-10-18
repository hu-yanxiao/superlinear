#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

tmp_dir=".local_tmp/sh_two_layer_gate_zero"
mkdir -p "$tmp_dir"

plain="$tmp_dir/plain.sus2"
gated="$tmp_dir/gated.sus2"
plain_pred="$tmp_dir/plain_pred.cfg"
gated_pred="$tmp_dir/gated_pred.cfg"
train="$tmp_dir/one.cfg"
rm -f "$plain" "$gated" "$plain_pred" "$gated_pred" "$train"
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
)

./bin/mlp-sus2 init-sh "$plain" "${common_opts[@]}"
./bin/mlp-sus2 init-sh "$gated" "${common_opts[@]}" \
  --two-layer-gate

python3 - "$gated" <<'PY'
import re
import sys

path = sys.argv[1]
text = open(path).read()
match = re.search(r"two_layer_gate_weights = \{([^}]*)\}", text)
if not match:
    raise SystemExit("missing two_layer_gate_weights")
weights = [float(x.strip()) for x in match.group(1).split(",") if x.strip()]
replacement = "two_layer_gate_weights = {" + ", ".join("0.000000000000000e+00" for _ in weights) + "}"
open(path, "w").write(text[:match.start()] + replacement + text[match.end():])
PY

./bin/mlp-sus2 calc-efs "$plain" "$train" "$plain_pred" >/dev/null
./bin/mlp-sus2 calc-efs "$gated" "$train" "$gated_pred" >/dev/null

python3 - "$plain_pred" "$gated_pred" <<'PY'
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

plain = read_values(sys.argv[1])
gated = read_values(sys.argv[2])
if len(plain) != len(gated):
    raise SystemExit(f"value count mismatch: {len(plain)} vs {len(gated)}")
err = max((abs(a - b) for a, b in zip(plain, gated)), default=0.0)
if err >= 1.0e-10:
    raise SystemExit(f"zero-gate predictions differ: {err}")
PY
