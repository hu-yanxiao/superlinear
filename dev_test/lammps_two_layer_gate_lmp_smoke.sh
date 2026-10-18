#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

MLP_BIN="${MLP_BIN:-./bin/mlp-sus2}"
LAMMPS_BIN="${LAMMPS_BIN:-/work/phy-weigw/apps/lammps-10Dec2025/src/lmp_ml_sus2_avx2}"
MPI_RUN="${MPI_RUN:-mpirun}"
GATE_MODE="${GATE_MODE:-mu-body-linear-combo}"

tmp_dir=".local_tmp/lammps_two_layer_gate_lmp_smoke_${GATE_MODE}"
rm -rf "$tmp_dir"
mkdir -p "$tmp_dir"

base_model="$tmp_dir/gated_base.sus2"
direct_model="$tmp_dir/gated_nonzero.sus2"
lmp_model="$tmp_dir/gated_nonzero_lmp.sus2"

"$MLP_BIN" init-sh "$base_model" \
  --species-count=2 \
  --l-max=2 \
  --k-max=2 \
  --body-order=4 \
  --body-l-max=2,2,2,2 \
  --radial-basis-size=4 \
  --cutoff=5.0 \
  --write-sh-scalar-info \
  --two-layer-gate \
  --two-layer-gate-mode="$GATE_MODE" \
  --two-layer-gate-shared-radial >/dev/null

python3 - "$base_model" "$direct_model" "$lmp_model" <<'PY'
import re
import sys

base, direct, lmp = sys.argv[1:]
text = open(base).read()

species_match = re.search(r"species_count\s*=\s*(\d+)", text)
scalar_match = re.search(r"alpha_scalar_moments\s*=\s*(\d+)", text)
if not species_match or not scalar_match:
    raise SystemExit("missing species_count or alpha_scalar_moments")
species_count = int(species_match.group(1))
scalar_count = int(scalar_match.group(1))

def with_linear_coeffs(model_text):
    if "species_coeffs" in model_text and "moment_coeffs" in model_text:
        return model_text
    species = [0.0] * species_count
    moments = [0.0] * scalar_count
    if scalar_count:
        moments[0] = 1.0
    if scalar_count > 1:
        moments[1] = -0.25
    if scalar_count > 2:
        moments[2] = 0.10
    return (
        model_text.rstrip()
        + "\n"
        + "species_coeffs = {" + ", ".join(f"{x:.15e}" for x in species) + "}\n"
        + "moment_coeffs = {" + ", ".join(f"{x:.15e}" for x in moments) + "}\n"
    )

text = with_linear_coeffs(text)
open(base, "w").write(text)
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
direct_text = text[:match.start()] + replacement + text[match.end():]
open(direct, "w").write(direct_text)
open(lmp, "w").write(direct_text.replace(
    "radial_basis_type = RBChebyshev_sss",
    "radial_basis_type = RBChebyshev_sss_lmp",
    1,
))
PY

cat > "$tmp_dir/data.small" <<'DATA'
LAMMPS data

8 atoms
2 atom types

0.0 12.0 xlo xhi
0.0 12.0 ylo yhi
0.0 12.0 zlo zhi

Masses

1 1.0
2 1.0

Atoms # atomic

1 1 1.0 1.0 1.0
2 1 2.3 1.2 1.1
3 2 4.8 1.4 1.2
4 2 5.7 1.6 1.3
5 1 6.4 1.8 1.1
6 2 7.3 2.0 1.2
7 1 9.8 1.5 1.4
8 2 10.6 1.3 1.0
DATA

make_input() {
  local model="$1"
  local label="$2"
  cat > "$tmp_dir/in.$label" <<LAMMPS
units metal
atom_style atomic
boundary p p p
newton on
read_data $tmp_dir/data.small
pair_style sus2mtp $model tabstep 0.001
pair_coeff * *
neighbor 1.0 bin
neigh_modify every 1 delay 0 check yes
thermo_style custom step pe
thermo 1
run 0 post no
write_dump all custom $tmp_dir/dump.$label id type x y z fx fy fz modify sort id
LAMMPS
}

run_lammps() {
  local ranks="$1"
  local model="$2"
  local label="$3"
  make_input "$model" "$label"
  "$MPI_RUN" -n "$ranks" "$LAMMPS_BIN" \
    -in "$tmp_dir/in.$label" \
    -log "$tmp_dir/log.$label" \
    > "$tmp_dir/stdout.$label" 2> "$tmp_dir/stderr.$label"
}

run_lammps 1 "$base_model" gate_zero_1
run_lammps 1 "$direct_model" direct_1
run_lammps 1 "$lmp_model" lmp_1
run_lammps 2 "$direct_model" direct_2
run_lammps 2 "$lmp_model" lmp_2

python3 - "$tmp_dir" <<'PY'
import math
import pathlib
import re
import sys

root = pathlib.Path(sys.argv[1])

def read_energy(label):
    lines = (root / f"log.{label}").read_text().splitlines()
    for idx, line in enumerate(lines):
        if re.match(r"^\s*Step\s+PotEng\s*$", line):
            if idx + 1 >= len(lines):
                break
            parts = lines[idx + 1].split()
            if len(parts) >= 2 and parts[0] == "0":
                return float(parts[1])
    raise SystemExit(f"failed to parse PotEng from log.{label}")

def read_dump_forces(label):
    lines = (root / f"dump.{label}").read_text().splitlines()
    for idx, line in enumerate(lines):
        if line.startswith("ITEM: ATOMS"):
            header = line.split()[2:]
            id_col = header.index("id")
            f_cols = [header.index(name) for name in ("fx", "fy", "fz")]
            rows = []
            for raw in lines[idx + 1:]:
                if raw.startswith("ITEM:"):
                    break
                parts = raw.split()
                if not parts:
                    continue
                rows.append((int(parts[id_col]), [float(parts[col]) for col in f_cols]))
            rows.sort()
            return [value for _, force in rows for value in force]
    raise SystemExit(f"failed to parse forces from dump.{label}")

def values(label):
    return [read_energy(label)] + read_dump_forces(label)

def compare(a_label, b_label, tol):
    a = values(a_label)
    b = values(b_label)
    if len(a) != len(b):
        raise SystemExit(f"{a_label}/{b_label} length mismatch: {len(a)} vs {len(b)}")
    max_err = max((abs(x - y) for x, y in zip(a, b)), default=0.0)
    if not math.isfinite(max_err) or max_err > tol:
        raise SystemExit(f"{a_label} vs {b_label} max abs error {max_err:.6e} > {tol:.6e}")
    print(f"{a_label} vs {b_label}: max_abs={max_err:.6e}")

compare("direct_1", "lmp_1", 2.0e-4)
compare("direct_2", "lmp_2", 2.0e-4)
compare("direct_1", "direct_2", 2.0e-8)
compare("lmp_1", "lmp_2", 2.0e-8)
print("LAMMPS_TWO_LAYER_GATE_LMP_SMOKE: ok")
PY
