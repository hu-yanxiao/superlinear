#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

tmp_dir=".codex_tmp/sh_two_layer_gate_edge_l1_efs_fd"
mkdir -p "$tmp_dir"

model="$tmp_dir/edge_l1_efs_fd.mtp"
cfg="$tmp_dir/random_non_symmetric.cfg"

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
  --two-layer-gate-edge-l1 >/dev/null

python3 - "$model" "$cfg" <<'PY'
import re
import sys

model_path, cfg_path = sys.argv[1:3]
text = open(model_path).read()

def scalar_int(name):
    return int(re.search(rf"{name} = (\d+)", text).group(1))

def replace_block(name, values, body):
    line = f"{name} = " + "{" + ", ".join(f"{v:.15e}" for v in values) + "}"
    return re.sub(rf"{name} = \{{[^}}]*\}}", line, body, flags=re.S)

species = scalar_int("species_count")
alpha = scalar_int("alpha_scalar_moments")
edge_l1_count = scalar_int("two_layer_gate_edge_l1_weight_count")
text = replace_block("species_coeffs", [1.0] * species, text)
text = replace_block("moment_coeffs", [1.0] * alpha, text)
text = replace_block(
    "two_layer_gate_edge_l1_weights",
    [1000.0 * ((i % 5) - 2) for i in range(edge_l1_count)],
    text,
)
open(model_path, "w").write(text)

atoms = [
    (1, 0, 1.13, 1.72, 1.41, 0.03, -0.02, 0.01),
    (2, 1, 2.47, 1.19, 3.06, -0.01, 0.04, -0.03),
    (3, 0, 3.81, 2.68, 1.97, 0.02, 0.01, -0.04),
    (4, 1, 5.08, 3.33, 4.44, -0.03, -0.01, 0.02),
    (5, 0, 1.86, 5.21, 5.64, 0.01, -0.05, 0.03),
    (6, 1, 4.62, 5.87, 2.58, -0.02, 0.03, 0.04),
    (7, 0, 6.19, 1.93, 5.16, 0.05, -0.01, -0.02),
    (8, 1, 2.94, 6.36, 1.24, -0.04, 0.02, 0.01),
]
with open(cfg_path, "w") as out:
    out.write("BEGIN_CFG\n")
    out.write(" Size\n")
    out.write(f"      {len(atoms)}\n")
    out.write(" Supercell\n")
    out.write("   8.5000000000    0.0000000000    0.0000000000\n")
    out.write("   0.0000000000    8.5000000000    0.0000000000\n")
    out.write("   0.0000000000    0.0000000000    8.5000000000\n")
    out.write(" AtomData:  id type       cartes_x      cartes_y      cartes_z     fx          fy          fz\n")
    for row in atoms:
        out.write("%7d %6d %12.6f %12.6f %12.6f %11.6f %11.6f %11.6f\n" % row)
    out.write(" Energy\n")
    out.write("    -12.345678\n")
    out.write(" PlusStress:  xx          yy          zz          yz          xz          xy\n")
    out.write("    0.030000   -0.020000    0.010000    0.004000   -0.003000    0.002000\n")
    out.write("END_CFG\n")
PY

./bin/mlp-sus2 check-efs-fd-dev "$model" "$cfg" \
  --max-configs=1 \
  --max-atoms=2 \
  --displacement=1e-4 \
  --abs-tolerance=1e-6 \
  --rel-tolerance=1e-4
