#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

tmp_dir=".codex_tmp/sh_two_layer_gate_readout_zero_gate_efs_fd"
mkdir -p "$tmp_dir"

init_model="$tmp_dir/readout_init.mtp"
model="$tmp_dir/readout_zero_gate.mtp"
cfg="$tmp_dir/one.cfg"
rm -f "$init_model" "$model" "$cfg"
awk '{print} /^END_CFG/{exit}' example/train.cfg > "$cfg"

./bin/mlp-sus2 init-sh "$init_model" \
  --species-count=2 \
  --l-max=2 \
  --k-max=2 \
  --body-order=4 \
  --body-l-max=2,2,2,2 \
  --radial-basis-size=4 \
  --cutoff=5.0 \
  --write-sh-scalar-info \
  --two-layer-gate \
  --two-layer-gate-readout >/dev/null

./bin/mlp-sus2 train "$init_model" "$cfg" \
  --trained-pot-name="$model" \
  --max-iter=1 \
  --skip-preinit \
  --energy-weight=1 \
  --force-weight=1 \
  --stress-weight=0 >/dev/null

python3 - "$model" "$cfg" <<'PY'
import re
import sys

path = sys.argv[1]
cfg_path = sys.argv[2]
text = open(path).read()

def replace_values(name, value_fn):
    global text
    match = re.search(rf"{name} = \{{([^}}]*)\}}", text)
    if not match:
        raise SystemExit(f"missing {name}")
    values = [float(x.strip()) for x in match.group(1).split(",") if x.strip()]
    replacement = name + " = {" + ", ".join(
        f"{value_fn(i):.15e}" for i in range(len(values))
    ) + "}"
    text = text[:match.start()] + replacement + text[match.end():]

replace_values("two_layer_gate_weights", lambda _: 0.0)
replace_values(
    "two_layer_gate_readout_coeffs",
    lambda i: (1.0 if i % 2 == 0 else -1.0) * (0.05 + 0.002 * i),
)
replace_values("species_coeffs", lambda _: 1.0)
open(path, "w").write(text)

lines = open(cfg_path).read().splitlines(True)
for line_index, line in enumerate(lines):
    if line.strip().startswith("AtomData:"):
        header = line.split()[1:]
        atom_line = line_index + 1
        columns = {name: header.index(name) for name in header}
        break
else:
    raise SystemExit("missing AtomData")
parts = lines[atom_line].split()
parts[columns["cartes_x"]] = f"{float(parts[columns['cartes_x']]) + 0.031:.16e}"
parts[columns["cartes_y"]] = f"{float(parts[columns['cartes_y']]) - 0.017:.16e}"
lines[atom_line] = "\t".join(parts) + "\n"
open(cfg_path, "w").writelines(lines)
PY

./bin/mlp-sus2 check-efs-fd-dev "$model" "$cfg" \
  --max-configs=1 \
  --max-atoms=2 \
  --displacement=1e-4 \
  --abs-tolerance=3e-5 \
  --rel-tolerance=2e-3

./bin/mlp-sus2 check-linear-components-fd-dev "$model" "$cfg" \
  --max-configs=1 \
  --max-atoms=2 \
  --displacement=1e-5 \
  --abs-tolerance=2e-5 \
  --rel-tolerance=2e-3

./bin/mlp-sus2 check-linear-readout-dev "$model" "$cfg" \
  --max-configs=1 \
  --max-atoms=2 \
  --abs-tolerance=1e-8 \
  --rel-tolerance=1e-8
