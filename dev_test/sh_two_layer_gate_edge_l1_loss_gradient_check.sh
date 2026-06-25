#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

tmp_dir=".codex_tmp/sh_two_layer_gate_edge_l1_loss_gradient"
mkdir -p "$tmp_dir"

model="$tmp_dir/edge_l1_grad.mtp"
cfg="$tmp_dir/random_non_symmetric.cfg"
range_file="$tmp_dir/edge_l1_range.txt"

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

python3 - "$model" "$range_file" "$cfg" <<'PY'
import re
import sys

model_path, range_path, cfg_path = sys.argv[1:4]
text = open(model_path).read()

def scalar_int(name):
    return int(re.search(rf"{name} = (\d+)", text).group(1))

def replace_block(name, values, body):
    line = f"{name} = " + "{" + ", ".join(f"{v:.15e}" for v in values) + "}"
    return re.sub(rf"{name} = \{{[^}}]*\}}", line, body, flags=re.S)

species = scalar_int("species_count")
radial_count = scalar_int("radial_funcs_count")
rb_size = scalar_int("radial_basis_size")
alpha = scalar_int("alpha_scalar_moments")
gate_radial_count = scalar_int("two_layer_gate_radial_coeff_count")
gate_additive_count = scalar_int("two_layer_gate_additive_coeff_count")
gate_weight_count = scalar_int("two_layer_gate_weight_count")
gate_body_mix_count = scalar_int("two_layer_gate_body_mix_weight_count")
edge_l1_count = scalar_int("two_layer_gate_edge_l1_weight_count")

species_values = [1.0 for _ in range(species)]
moment_values = [0.25 + 0.01 * ((i % 7) - 3) for i in range(alpha)]
edge_values = [0.2 * ((i % 5) - 2) for i in range(edge_l1_count)]
text = replace_block("species_coeffs", species_values, text)
text = replace_block("moment_coeffs", moment_values, text)
text = replace_block("two_layer_gate_edge_l1_weights", edge_values, text)
open(model_path, "w").write(text)

base_nonlinear = (
    species
    + 2 * species * species * radial_count
    + radial_count * (rb_size + species)
)
edge_begin = (
    base_nonlinear
    + gate_radial_count
    + gate_additive_count
    + gate_weight_count
    + gate_body_mix_count
)
edge_end = edge_begin + edge_l1_count
open(range_path, "w").write(f"{edge_begin} {edge_end}\n")

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

read -r edge_begin edge_end < "$range_file"

for weights in E F S EFS; do
  case "$weights" in
    E) ew=1; fw=0; sw=0 ;;
    F) ew=0; fw=1; sw=0 ;;
    S) ew=0; fw=0; sw=1 ;;
    EFS) ew=1; fw=1; sw=1 ;;
  esac
  ./bin/mlp-sus2 check-loss-gradient-direction-dev "$model" "$cfg" \
    --max-configs=1 \
    --energy-weight="$ew" \
    --force-weight="$fw" \
    --stress-weight="$sw" \
    --radial-smooth=0 \
    --scalar-head-l2=0 \
    --gate-scalar-l2=0 \
    --gate-mix-l2=0 \
    --gate-full-l2=0 \
    --displacement=1e-2 \
    --abs-tolerance=1e-12 \
    --rel-tolerance=1e-6 \
    --coeff-start="$edge_begin" \
    --coeff-end="$edge_end"
done
