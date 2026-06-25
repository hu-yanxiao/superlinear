#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

tmp_dir=".codex_tmp/sh_two_layer_gate_edge_l1_rotation"
mkdir -p "$tmp_dir"

model="$tmp_dir/edge_l1_rotation.mtp"
cfg="$tmp_dir/original.cfg"
rot_cfg="$tmp_dir/rotated.cfg"
pred="$tmp_dir/original_pred.cfg"
rot_pred="$tmp_dir/rotated_pred.cfg"

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

python3 - "$model" "$cfg" "$rot_cfg" <<'PY'
import re
import sys

model_path, cfg_path, rot_cfg_path = sys.argv[1:4]
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
text = replace_block(
    "moment_coeffs",
    [0.35 + 0.02 * ((i % 5) - 2) for i in range(alpha)],
    text,
)
text = replace_block(
    "two_layer_gate_edge_l1_weights",
    [0.3 * ((i % 5) - 2) for i in range(edge_l1_count)],
    text,
)
open(model_path, "w").write(text)

L = 8.5
center = 0.5 * L
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

def rot_pos(x, y, z):
    dx, dy = x - center, y - center
    return center - dy, center + dx, z

def write_cfg(path, rows):
    with open(path, "w") as out:
        out.write("BEGIN_CFG\n")
        out.write(" Size\n")
        out.write(f"      {len(rows)}\n")
        out.write(" Supercell\n")
        out.write("   8.5000000000    0.0000000000    0.0000000000\n")
        out.write("   0.0000000000    8.5000000000    0.0000000000\n")
        out.write("   0.0000000000    0.0000000000    8.5000000000\n")
        out.write(" AtomData:  id type       cartes_x      cartes_y      cartes_z     fx          fy          fz\n")
        for row in rows:
            out.write("%7d %6d %12.6f %12.6f %12.6f %11.6f %11.6f %11.6f\n" % row)
        out.write(" Energy\n")
        out.write("    -12.345678\n")
        out.write(" PlusStress:  xx          yy          zz          yz          xz          xy\n")
        out.write("    0.030000   -0.020000    0.010000    0.004000   -0.003000    0.002000\n")
        out.write("END_CFG\n")

write_cfg(cfg_path, atoms)
rot_atoms = []
for atom_id, atom_type, x, y, z, fx, fy, fz in atoms:
    xr, yr, zr = rot_pos(x, y, z)
    rot_atoms.append((atom_id, atom_type, xr, yr, zr, -fy, fx, fz))
write_cfg(rot_cfg_path, rot_atoms)
PY

./bin/mlp-sus2 calc-efs "$model" "$cfg" "$pred" >/dev/null
./bin/mlp-sus2 calc-efs "$model" "$rot_cfg" "$rot_pred" >/dev/null

python3 - "$pred" "$rot_pred" <<'PY'
import math
import sys

def parse_cfg(path):
    forces = []
    sites = []
    stress = None
    atom_cols = None
    need_stress = False
    with open(path) as handle:
        for raw in handle:
            line = raw.strip()
            parts = line.split()
            if not parts:
                continue
            if need_stress:
                vals = [float(x) for x in parts[:6]]
                xx, yy, zz, yz, xz, xy = vals
                stress = [
                    [xx, xy, xz],
                    [xy, yy, yz],
                    [xz, yz, zz],
                ]
                need_stress = False
                continue
            if line.startswith("AtomData:"):
                header = line.split()[1:]
                atom_cols = {
                    "fx": header.index("fx"),
                    "fy": header.index("fy"),
                    "fz": header.index("fz"),
                    "site_en": header.index("site_en"),
                }
            elif atom_cols is not None and line[:1].isdigit():
                forces.append(tuple(float(parts[atom_cols[key]]) for key in ("fx", "fy", "fz")))
                sites.append(float(parts[atom_cols["site_en"]]))
            elif parts[0] == "PlusStress:":
                need_stress = True
    if stress is None:
        raise SystemExit(f"missing stress in {path}")
    return sum(sites), forces, stress

def matmul(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)] for i in range(3)]

def transpose(a):
    return [[a[j][i] for j in range(3)] for i in range(3)]

e0, f0, s0 = parse_cfg(sys.argv[1])
er, fr, sr = parse_cfg(sys.argv[2])
if len(f0) != len(fr):
    raise SystemExit("force count mismatch")

force_err = 0.0
for a, b in zip(f0, fr):
    b_back = (b[1], -b[0], b[2])
    force_err = max(force_err, max(abs(x - y) for x, y in zip(a, b_back)))

R = [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
s_back = matmul(matmul(transpose(R), sr), R)
stress_err = max(abs(s0[i][j] - s_back[i][j]) for i in range(3) for j in range(3))
energy_err = abs(e0 - er)

if not math.isfinite(energy_err) or energy_err > 1.0e-10:
    raise SystemExit(f"rotated site-energy sum mismatch: {energy_err}")
if not math.isfinite(force_err) or force_err > 1.0e-8:
    raise SystemExit(f"rotated force mismatch: {force_err}")
if not math.isfinite(stress_err) or stress_err > 1.0e-8:
    raise SystemExit(f"rotated stress mismatch: {stress_err}")
print(
    "edge-L1 rotation invariance OK: "
    f"site_energy_sum_err={energy_err:.3e} "
    f"force_err={force_err:.3e} stress_err={stress_err:.3e}"
)
PY
