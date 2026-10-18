#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

tmp_dir=".local_tmp/sh_two_layer_gate_force_fd"
mkdir -p "$tmp_dir"

init_model="$tmp_dir/gated_init.sus2"
trained_model="$tmp_dir/gated_trained.sus2"
nonzero_model="$tmp_dir/gated_nonzero.sus2"
train="$tmp_dir/one.cfg"
base="$tmp_dir/base.cfg"
plus="$tmp_dir/plus.cfg"
minus="$tmp_dir/minus.cfg"
base_pred="$tmp_dir/base_pred.cfg"
plus_pred="$tmp_dir/plus_pred.cfg"
minus_pred="$tmp_dir/minus_pred.cfg"
rm -f "$init_model" "$trained_model" "$nonzero_model" \
  "$train" "$base" "$plus" "$minus" \
  "$base_pred" "$plus_pred" "$minus_pred"
awk '{print} /^END_CFG/{exit}' example/train.cfg > "$train"

./bin/mlp-sus2 init-sh "$init_model" \
  --species-count=2 \
  --l-max=2 \
  --k-max=2 \
  --body-order=4 \
  --body-l-max=2,2,2,2 \
  --radial-basis-size=4 \
  --cutoff=5.0 \
  --write-sh-scalar-info \
  --two-layer-gate

./bin/mlp-sus2 train "$init_model" "$train" \
  --trained-pot-name="$trained_model" \
  --max-iter=1 \
  --skip-preinit \
  --energy-weight=1 \
  --force-weight=1 \
  --stress-weight=0 >/dev/null

python3 - "$trained_model" "$nonzero_model" "$train" "$base" "$plus" "$minus" <<'PY'
import re
import sys

model, out_model, src, base, plus, minus = sys.argv[1:]
text = open(model).read()
match = re.search(r"two_layer_gate_weights = \{([^}]*)\}", text)
if not match:
    raise SystemExit("missing two_layer_gate_weights")
weights = [float(x.strip()) for x in match.group(1).split(",") if x.strip()]
if len(weights) < 2:
    raise SystemExit("expected at least two gate weights")
weights[0] = 5.0
weights[1] = 2.0
replacement = "two_layer_gate_weights = {" + ", ".join(f"{x:.15e}" for x in weights) + "}"
open(out_model, "w").write(text[:match.start()] + replacement + text[match.end():])

lines = open(src).read().splitlines(True)
for i, line in enumerate(lines):
    if line.strip().startswith("AtomData:"):
        header = line.split()[1:]
        atom_line = i + 1
        cols = {name: header.index(name) for name in header}
        break
else:
    raise SystemExit("missing AtomData")

def write_cfg(path, delta):
    out = lines.copy()
    parts = out[atom_line].split()
    parts[cols["cartes_x"]] = f"{float(parts[cols['cartes_x']]) + 0.031 + delta:.16e}"
    parts[cols["cartes_y"]] = f"{float(parts[cols['cartes_y']]) - 0.017:.16e}"
    out[atom_line] = "\t".join(parts) + "\n"
    open(path, "w").writelines(out)

eps = 1.0e-3
write_cfg(base, 0.0)
write_cfg(plus, eps)
write_cfg(minus, -eps)
PY

./bin/mlp-sus2 calc-efs "$nonzero_model" "$base" "$base_pred" >/dev/null
./bin/mlp-sus2 calc-efs "$nonzero_model" "$plus" "$plus_pred" >/dev/null
./bin/mlp-sus2 calc-efs "$nonzero_model" "$minus" "$minus_pred" >/dev/null

python3 - "$base_pred" "$plus_pred" "$minus_pred" <<'PY'
import sys

def read(path):
    energy = None
    forces = []
    expect_energy = False
    force_cols = None
    with open(path) as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if expect_energy:
                energy = float(line.split()[0])
                expect_energy = False
            elif line == "Energy":
                expect_energy = True
            elif line.startswith("AtomData:"):
                header = line.split()[1:]
                force_cols = [header.index(name) for name in ("fx", "fy", "fz")]
            elif force_cols is not None and line[0].isdigit():
                parts = line.split()
                if len(parts) > max(force_cols):
                    forces.append([float(parts[idx]) for idx in force_cols])
    if energy is None or not forces:
        raise SystemExit(f"failed to parse prediction {path}")
    return energy, forces

base_energy, base_forces = read(sys.argv[1])
plus_energy, _ = read(sys.argv[2])
minus_energy, _ = read(sys.argv[3])
fd_dedx = (plus_energy - minus_energy) / (2.0e-3)
analytic_force = base_forces[0][0]
err = abs(analytic_force + fd_dedx)
scale = max(1.0e-8, abs(analytic_force), abs(fd_dedx))
if err > max(2.0e-6, 5.0e-2 * scale):
    raise SystemExit(
        f"two-layer gate force FD mismatch: force={analytic_force} dE/dx={fd_dedx} err={err}"
    )
PY
