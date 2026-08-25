#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

bin=${1:-./bin/mlp-sus2}
bin=$(cd "$(dirname "$bin")" && pwd)/$(basename "$bin")
tmp_dir=$(mktemp -d "${TMPDIR:-/tmp}/sus2sh-rank-edge-l1-XXXXXX")
if [[ ${SUS2_TEST_KEEP_TMP:-0} == 1 ]]; then
  echo "keeping test directory: $tmp_dir"
else
  trap 'rm -rf "$tmp_dir"' EXIT
fi

cfg="$tmp_dir/non-symmetric.cfg"
rot_cfg="$tmp_dir/non-symmetric-rotated.cfg"
linear_cfg="$tmp_dir/linear-train.cfg"
python3 - "$cfg" "$rot_cfg" "$linear_cfg" <<'PY'
import math
import sys

path, rotated_path, linear_path = sys.argv[1:]
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

def write_cfg(target, rows, mode="w", cell=8.5, energy=-12.345678, stress=None):
  if stress is None:
    stress = (0.03, -0.02, 0.01, 0.004, -0.003, 0.002)
  with open(target, mode, encoding="utf-8") as out:
    out.write("BEGIN_CFG\n Size\n      8\n Supercell\n")
    out.write(f"   {cell:.10f}    0.0000000000    0.0000000000\n")
    out.write(f"   0.0000000000    {cell:.10f}    0.0000000000\n")
    out.write(f"   0.0000000000    0.0000000000    {cell:.10f}\n")
    out.write(" AtomData:  id type       cartes_x      cartes_y      cartes_z     fx          fy          fz\n")
    for row in rows:
        out.write("%7d %6d %12.6f %12.6f %12.6f %11.6f %11.6f %11.6f\n" % row)
    out.write(" Energy\n")
    out.write(f"    {energy:.12f}\n")
    out.write(" PlusStress:  xx          yy          zz          yz          xz          xy\n")
    out.write("    " + "   ".join(f"{value:.8f}" for value in stress) + "\n")
    out.write("END_CFG\n")

write_cfg(path, atoms)
center = 4.25
rotated = []
for atom_id, atom_type, x, y, z, fx, fy, fz in atoms:
    dx, dy = x - center, y - center
    rotated.append((
        atom_id, atom_type, center - dy, center + dx, z,
        -fy, fx, fz))
write_cfg(rotated_path, rotated)

for sample in range(24):
    scale = 1.0 + 0.002 * (sample - 11.5)
    rows = []
    for atom_id, atom_type, x, y, z, _fx, _fy, _fz in atoms:
        phase = 0.37 * (sample + 1) * (atom_id + 1)
        rows.append((
            atom_id,
            atom_type,
            x * scale + 0.035 * math.sin(phase),
            y * scale + 0.030 * math.cos(0.71 * phase),
            z * scale + 0.025 * math.sin(1.13 * phase),
            0.02 * math.cos(phase),
            -0.018 * math.sin(0.83 * phase),
            0.016 * math.cos(1.21 * phase),
        ))
    write_cfg(
        linear_path,
        rows,
        "w" if sample == 0 else "a",
        8.5 * scale,
        -12.5 + 0.075 * sample + 0.03 * math.sin(sample),
        (0.001 * sample, -0.0007 * sample, 0.0004 * sample,
         0.0003, -0.0002, 0.0001),
    )
PY

common=(
  --species-count=2
  --l-max=2
  --k-max=3
  --body-order=4
  --body-l-max=2,2,2,2
  --radial-basis-size=3
  --radial-basis-type=RBChebyshev_sss
  --cutoff=5.0
  --sh-factor-pruning=q-total
  --two-layer-gate
  --two-layer-gate-mode=mu-scalar-full
  --two-layer-gate-body-order=3
  --two-layer-gate-factor-pruning=q-total
  --two-layer-gate-shared-radial
  --two-layer-gate-edge-l1
  --init-params=same
)

outer_only="$tmp_dir/outer2-edge.mtp"
gate_ranked="$tmp_dir/outer2-gate1-edge.mtp"
plain_identity="$tmp_dir/plain-edge.mtp"
rank_identity="$tmp_dir/outer3-gate3-edge.mtp"
outer_no_edge="$tmp_dir/outer2-no-edge.mtp"
gate_ranked_no_edge="$tmp_dir/outer2-gate1-no-edge.mtp"
"$bin" init-sh "$outer_only" "${common[@]}" --outer-k-rank=2 >/dev/null
"$bin" init-sh "$gate_ranked" "${common[@]}" \
  --outer-k-rank=2 --two-layer-gate-k-rank=1 >/dev/null
"$bin" init-sh "$plain_identity" "${common[@]}" >/dev/null
"$bin" init-sh "$rank_identity" "${common[@]}" \
  --outer-k-rank=3 --two-layer-gate-k-rank=3 >/dev/null
common_no_edge=()
for arg in "${common[@]}"; do
  [[ "$arg" == --two-layer-gate-edge-l1 ]] || common_no_edge+=("$arg")
done
"$bin" init-sh "$outer_no_edge" "${common_no_edge[@]}" \
  --outer-k-rank=2 >/dev/null
"$bin" init-sh "$gate_ranked_no_edge" "${common_no_edge[@]}" \
  --outer-k-rank=2 --two-layer-gate-k-rank=1 >/dev/null

python3 - "$outer_only" "$gate_ranked" <<'PY'
import re
import sys

number = r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?"
def text(path):
    return open(path, encoding="utf-8").read()
def integer(data, name):
    match = re.search(rf"(?m)^\s*{re.escape(name)}\s*=\s*(\d+)\s*$", data)
    if not match:
        raise SystemExit(f"missing {name}")
    return int(match.group(1))
def vector(data, name):
    match = re.search(rf"(?m)^\s*{re.escape(name)}\s*=\s*\{{([^}}]*)\}}", data, re.S)
    if not match:
        raise SystemExit(f"missing {name}")
    return [int(float(x)) for x in re.findall(number, match.group(1))]

outer = text(sys.argv[1])
ranked = text(sys.argv[2])
outer_begin = integer(outer, "outer_k_rank1_basic_begin")
outer_sources = vector(outer, "two_layer_gate_edge_l1_source_moment_indices")
if not outer_sources or any(base < 0 or base + 2 >= outer_begin for base in outer_sources):
    raise SystemExit("outer-only ranked edge-L1 sources escaped the preserved raw graph")
if integer(outer, "two_layer_gate_edge_l1_source_moment_count") != len(outer_sources):
    raise SystemExit("outer-only ranked edge-L1 source count is inconsistent")
if len(outer_sources) <= 3:
    raise SystemExit("outer-only ranked edge-L1 omitted multi-body L=1 sources")
if integer(outer, "two_layer_gate_edge_l1_weight_count") != (
        integer(outer, "radial_funcs_count") * len(outer_sources)):
    raise SystemExit("outer-only ranked edge-L1 weight count is inconsistent")
gate_begin = integer(ranked, "two_layer_gate_k_rank_basic_begin")
ranked_outer_begin = integer(ranked, "outer_k_rank1_basic_begin")
gate_sources = vector(ranked, "two_layer_gate_edge_l1_source_moment_indices")
if not gate_sources or any(
        base < gate_begin or base + 2 >= ranked_outer_begin for base in gate_sources):
    raise SystemExit("gate-ranked edge-L1 sources escaped the contracted gate graph")
if integer(ranked, "two_layer_gate_edge_l1_source_moment_count") != len(gate_sources):
    raise SystemExit("gate-ranked edge-L1 source count is inconsistent")
if len(gate_sources) <= 1:
    raise SystemExit("gate-ranked edge-L1 omitted multi-body L=1 sources")
if integer(ranked, "two_layer_gate_edge_l1_weight_count") != (
        integer(ranked, "radial_funcs_count") * len(gate_sources)):
    raise SystemExit("gate-ranked edge-L1 weight count is inconsistent")
if integer(ranked, "two_layer_gate_k_rank_product_end") != integer(
        ranked, "outer_k_rank1_product_begin"):
    raise SystemExit("gate-ranked and outer-ranked product stages overlap")
if vector(ranked, "two_layer_gate_scalar_indices") != list(range(
        len(vector(ranked, "two_layer_gate_scalar_indices")))):
    raise SystemExit("gate-ranked scalar ordinals are not canonical")
print(
    f"rank_edge_sources_ok outer_only={len(outer_sources)} "
    f"gate_ranked={len(gate_sources)}")
PY

activate_model() {
  local model=$1
  local change_rank_weights=$2
  python3 - "$model" "$change_rank_weights" <<'PY'
import re
import sys

path, change_rank_weights = sys.argv[1], sys.argv[2] == "yes"
data = open(path, encoding="utf-8").read()
number = r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?"

def replace(name, fn, required=True):
    global data
    match = re.search(rf"(?m)^\s*{re.escape(name)}\s*=\s*\{{([^}}]*)\}}", data, re.S)
    if not match:
        if required:
            raise SystemExit(f"missing {name}")
        return
    old = [float(x) for x in re.findall(number, match.group(1))]
    values = [fn(i, value, len(old)) for i, value in enumerate(old)]
    block = name + " = {" + ", ".join(f"{value:.15e}" for value in values) + "}"
    data = data[:match.start()] + block + data[match.end():]

replace("species_coeffs", lambda i, _v, _n: 1.0, required=False)
replace("moment_coeffs", lambda i, _v, _n: (-1.0 if i % 3 == 1 else 1.0) * (0.035 + 0.001 * (i % 11)), required=False)
replace("two_layer_gate_radial_coeffs", lambda i, _v, _n: (-1.0 if i % 7 == 4 else 1.0) * (0.11 + 0.003 * (i % 9)))
replace("two_layer_gate_additive_coeffs", lambda i, _v, _n: (-1.0 if i % 4 == 1 else 1.0) * (0.72 + 0.02 * (i % 5)))
replace("two_layer_gate_type_coeffs", lambda i, _v, _n: 0.83 + 0.025 * (i % 7))
replace("two_layer_gate_weights", lambda i, _v, _n: (-1.0 if i % 5 == 2 else 1.0) * (0.12 + 0.002 * (i % 13)))
replace("two_layer_gate_edge_l1_weights", lambda i, _v, _n: (-1.0 if i % 2 else 1.0) * (0.8 + 0.03 * (i % 9)))
if change_rank_weights:
    replace("two_layer_gate_k_rank_weights", lambda i, _v, _n: (-1.0 if i % 3 == 1 else 1.0) * (0.61 + 0.04 * (i % 5)), required=False)
    replace("outer_k_rank1_weights", lambda i, _v, _n: (-1.0 if i % 4 == 2 else 1.0) * (0.73 + 0.03 * (i % 7)), required=False)
open(path, "w", encoding="utf-8").write(data)
PY
}

activate_model "$outer_only" yes
activate_model "$gate_ranked" yes
activate_model "$plain_identity" no
activate_model "$rank_identity" no

compare_numeric_files() {
  python3 - "$1" "$2" "${3:-3e-10}" <<'PY'
import re
import sys

number = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")
left = [float(x) for x in number.findall(open(sys.argv[1], encoding="utf-8").read())]
right = [float(x) for x in number.findall(open(sys.argv[2], encoding="utf-8").read())]
if not left or len(left) != len(right):
    raise SystemExit("numeric output shape differs")
worst = max(abs(a-b) for a, b in zip(left, right))
scale = max([1.0] + [abs(x) for x in left] + [abs(x) for x in right])
if worst > float(sys.argv[3]) * scale:
    raise SystemExit(f"numeric parity failed: worst={worst:.3e} scale={scale:.3e}")
print(f"numeric_parity worst={worst:.3e} scale={scale:.3e}")
PY
}

"$bin" calc-efs "$plain_identity" "$cfg" "$tmp_dir/plain-identity.cfg" >/dev/null
"$bin" calc-efs "$rank_identity" "$cfg" "$tmp_dir/rank-identity.cfg" >/dev/null
compare_numeric_files "$tmp_dir/plain-identity.cfg" "$tmp_dir/rank-identity.cfg" 3e-10

for model in "$outer_only" "$gate_ranked"; do
  stem=$(basename "$model" .mtp)
  roundtrip="$tmp_dir/$stem-roundtrip.mtp"
  "$bin" roundtrip-model-dev "$model" "$roundtrip" >/dev/null
  "$bin" calc-efs "$model" "$cfg" "$tmp_dir/$stem.cfg" >/dev/null
  "$bin" calc-efs "$roundtrip" "$cfg" "$tmp_dir/$stem-roundtrip.cfg" >/dev/null
  compare_numeric_files "$tmp_dir/$stem.cfg" "$tmp_dir/$stem-roundtrip.cfg" 3e-12
done

for model in "$outer_only" "$gate_ranked"; do
  echo "ranked edge-L1 E/F/S finite difference: $(basename "$model")"
  "$bin" check-efs-fd-dev "$model" "$cfg" \
    --max-configs=1 --max-atoms=2 \
    --displacement=1e-5 --abs-tolerance=4e-5 --rel-tolerance=4e-3
done

layout=$("$bin" check-loss-gradient-dev "$gate_ranked" "$cfg" \
  --max-configs=1 --energy-weight=1 --force-weight=0 --stress-weight=0 \
  --radial-smooth=0 --scalar-head-l2=0 --gate-scalar-l2=0 \
  --gate-mix-l2=0 --gate-full-l2=0 --gate-x-l2=0 \
  --coeff-start=0 --coeff-end=1 \
  --displacement=1e-5 --abs-tolerance=1 --rel-tolerance=1)

field() {
  local name=$1
  sed -n "s/.* ${name}=\([^ ]*\).*/\1/p" <<<"$layout"
}

run_gradient_block() {
  local model=$1
  local name=$2
  local begin=$3
  local end=$4
  local energy_weight=$5
  local force_weight=$6
  local stress_weight=$7
  if [[ -z "$begin" || -z "$end" || "$end" -le "$begin" ]]; then
    echo "invalid $name coefficient range [$begin,$end)" >&2
    exit 1
  fi
  echo "ranked edge-L1 gradient block=$name range=[$begin,$end) weights=$energy_weight/$force_weight/$stress_weight"
  output=$("$bin" check-loss-gradient-direction-dev "$model" "$cfg" \
    --max-configs=1 \
    --energy-weight="$energy_weight" --force-weight="$force_weight" \
    --stress-weight="$stress_weight" \
    --radial-smooth=0 --scalar-head-l2=0 --gate-scalar-l2=0 \
    --gate-mix-l2=0 --gate-full-l2=0 --gate-x-l2=0 \
    --coeff-start="$begin" --coeff-end="$end" \
    --displacement=1e-4 --abs-tolerance=5e-5 --rel-tolerance=5e-3)
  printf '%s\n' "$output"
  analytic=$(sed -n 's/.*analytic_directional=\([^ ]*\).*/\1/p' <<<"$output")
  python3 - "$analytic" <<'PY'
import math
import sys
value = float(sys.argv[1])
if not math.isfinite(value) or abs(value) <= 1e-14:
    raise SystemExit("ranked edge-L1 directional gradient is vacuous or non-finite")
PY
}

edge_begin=$(field gate_edge_l1_begin)
edge_end=$(field gate_edge_l1_end)
gate_rank_begin=$(field gate_k_rank_begin)
gate_rank_end=$(field gate_k_rank_end)
outer_rank_begin=$(field outer_k_rank_begin)
outer_rank_end=$(field outer_k_rank_end)
gate_radial_begin=$(field gate_radial_begin)
gate_radial_end=$(field gate_radial_end)
gate_type_begin=$(field gate_type_begin)
gate_type_end=$(field gate_type_end)

run_gradient_block "$gate_ranked" edge_l1_W "$edge_begin" "$edge_end" 1 0.1 0.01
run_gradient_block "$gate_ranked" gate_rank_W_energy "$gate_rank_begin" "$gate_rank_end" 1 0 0
run_gradient_block "$gate_ranked" gate_rank_W_force "$gate_rank_begin" "$gate_rank_end" 0 1 0
run_gradient_block "$gate_ranked" outer_rank_W "$outer_rank_begin" "$outer_rank_end" 1 0.1 0.01
run_gradient_block "$gate_ranked" gate_radial "$gate_radial_begin" "$gate_radial_end" 1 0.1 0.01
run_gradient_block "$gate_ranked" gate_type "$gate_type_begin" "$gate_type_end" 1 0.1 0.01

layout=$("$bin" check-loss-gradient-dev "$outer_only" "$cfg" \
  --max-configs=1 --energy-weight=1 --force-weight=0 --stress-weight=0 \
  --radial-smooth=0 --scalar-head-l2=0 --gate-scalar-l2=0 \
  --gate-mix-l2=0 --gate-full-l2=0 --gate-x-l2=0 \
  --coeff-start=0 --coeff-end=1 \
  --displacement=1e-5 --abs-tolerance=1 --rel-tolerance=1)
edge_begin=$(field gate_edge_l1_begin)
edge_end=$(field gate_edge_l1_end)
outer_rank_begin=$(field outer_k_rank_begin)
outer_rank_end=$(field outer_k_rank_end)
gate_radial_begin=$(field gate_radial_begin)
gate_radial_end=$(field gate_radial_end)
gate_type_begin=$(field gate_type_begin)
gate_type_end=$(field gate_type_end)
run_gradient_block "$outer_only" outer_only_edge_l1_W "$edge_begin" "$edge_end" 1 0.1 0.01
run_gradient_block "$outer_only" outer_only_outer_rank_W "$outer_rank_begin" "$outer_rank_end" 1 0.1 0.01
run_gradient_block "$outer_only" outer_only_gate_radial "$gate_radial_begin" "$gate_radial_end" 1 0.1 0.01
run_gradient_block "$outer_only" outer_only_gate_type "$gate_type_begin" "$gate_type_end" 1 0.1 0.01

for model in "$outer_only" "$gate_ranked"; do
  stem=$(basename "$model" .mtp)
  "$bin" calc-efs "$model" "$cfg" "$tmp_dir/$stem-rotation-original.cfg" >/dev/null
  "$bin" calc-efs "$model" "$rot_cfg" "$tmp_dir/$stem-rotation-rotated.cfg" >/dev/null
  python3 - "$tmp_dir/$stem-rotation-original.cfg" \
    "$tmp_dir/$stem-rotation-rotated.cfg" <<'PY'
import math
import sys

def parse_cfg(path):
    forces, sites = [], []
    stress = None
    atom_cols = None
    need_stress = False
    with open(path, encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            parts = line.split()
            if not parts:
                continue
            if need_stress:
                xx, yy, zz, yz, xz, xy = [float(x) for x in parts[:6]]
                stress = [[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]]
                need_stress = False
                continue
            if line.startswith("AtomData:"):
                header = line.split()[1:]
                atom_cols = {key: header.index(key) for key in ("fx", "fy", "fz", "site_en")}
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

e0, f0, s0 = parse_cfg(sys.argv[1])
er, fr, sr = parse_cfg(sys.argv[2])
force_err = max(
    abs(a - b)
    for original, rotated in zip(f0, fr)
    for a, b in zip(original, (rotated[1], -rotated[0], rotated[2])))
rotation = [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
transpose = [[rotation[j][i] for j in range(3)] for i in range(3)]
stress_back = matmul(matmul(transpose, sr), rotation)
stress_err = max(abs(s0[i][j] - stress_back[i][j]) for i in range(3) for j in range(3))
energy_err = abs(e0 - er)
if not all(math.isfinite(x) for x in (energy_err, force_err, stress_err)):
    raise SystemExit("ranked edge-L1 rotation check produced a non-finite error")
if energy_err > 1e-10 or force_err > 1e-8 or stress_err > 1e-8:
    raise SystemExit(
        f"ranked edge-L1 rotation mismatch: E={energy_err:.3e} "
        f"F={force_err:.3e} S={stress_err:.3e}")
print(f"ranked_edge_rotation E={energy_err:.3e} F={force_err:.3e} S={stress_err:.3e}")
PY
done

"$bin" check-linear-components-fd-dev "$gate_ranked" "$cfg" \
  --max-configs=1 --max-atoms=2 \
  --displacement=1e-5 --abs-tolerance=5e-5 --rel-tolerance=5e-3

trained="$tmp_dir/gate-ranked-trained.mtp"
"$bin" train "$gate_ranked" "$linear_cfg" \
  --trained-pot-name="$trained" \
  --max-iter=1 --skip-preinit --do-lin --do-lin-steps=1 \
  --do-lin-freq=1 --optimizer=lbfgs --lbfgs-initial-step=1e-8 \
  --init-params=same \
  --energy-weight=1 --force-weight=0 --stress-weight=0 \
  --radial-parameterization=weight-norm \
  --radial-smooth=0 --scalar-head-l2=1e-5 --gate-scalar-l2=0 \
  --gate-mix-l2=0 --gate-full-l2=0 --gate-x-l2=0 >/dev/null
test -s "$trained"
"$bin" check-linear-readout-dev "$trained" "$cfg" \
  --max-configs=1 --max-atoms=2 --abs-tolerance=2e-7 --rel-tolerance=2e-6

if "$bin" help train 2>&1 | grep -q "Mode: MPI" \
    && command -v mpirun >/dev/null 2>&1; then
  mpi_cfg="$tmp_dir/four-train.cfg"
  awk 'BEGIN {n=0} {print} /^END_CFG/ {n++; if (n==4) exit}' \
    "$linear_cfg" >"$mpi_cfg"
  if [[ $(grep -c '^END_CFG' "$mpi_cfg") -ne 4 ]]; then
    echo "failed to prepare ranked edge-L1 MPI training set" >&2
    exit 1
  fi
  mpi_materialized="$tmp_dir/gate-ranked-mpi-materialized.mtp"
  "$bin" roundtrip-model-dev "$gate_ranked" "$mpi_materialized" >/dev/null
  mpi_serial_model="$tmp_dir/gate-ranked-serial-trained.mtp"
  mpi_two_model="$tmp_dir/gate-ranked-mpi2-trained.mtp"
  mpi_train_common=(
    --max-iter=1
    --skip-preinit
    --do-samp
    --energy-weight=1
    --force-weight=0.01
    --stress-weight=0.001
    --radial-smooth=0
    --scalar-head-l2=0
    --gate-scalar-l2=0
    --gate-mix-l2=0
    --gate-full-l2=0
    --gate-x-l2=0
  )
  "$bin" train "$mpi_materialized" "$mpi_cfg" \
    "${mpi_train_common[@]}" --trained-pot-name="$mpi_serial_model" \
    >"$tmp_dir/gate-ranked-serial-train.log" 2>&1
  timeout 300 mpirun -n 2 "$bin" train "$mpi_materialized" "$mpi_cfg" \
    "${mpi_train_common[@]}" --trained-pot-name="$mpi_two_model" \
    >"$tmp_dir/gate-ranked-mpi2-train.log" 2>&1
  test -s "$mpi_serial_model"
  test -s "$mpi_two_model"
  "$bin" calc-efs "$mpi_serial_model" "$mpi_cfg" \
    "$tmp_dir/gate-ranked-serial-trained.cfg" >/dev/null
  "$bin" calc-efs "$mpi_two_model" "$mpi_cfg" \
    "$tmp_dir/gate-ranked-mpi2-trained.cfg" >/dev/null
  compare_numeric_files \
    "$tmp_dir/gate-ranked-serial-trained.cfg" \
    "$tmp_dir/gate-ranked-mpi2-trained.cfg" 2e-7
fi

for model in "$outer_no_edge" "$gate_ranked_no_edge"; do
  stem=$(basename "$model" .mtp)
  if "$bin" train "$model" "$cfg" --two-layer-gate-edge-l1 \
       --max-iter=0 --skip-preinit \
       --trained-pot-name="$tmp_dir/$stem-invalid-upgrade.mtp" \
       >"$tmp_dir/$stem-upgrade.err" 2>&1; then
    echo "an existing ranked model was incorrectly upgraded to edge-L1" >&2
    exit 1
  fi
  grep -q 'initialize a new model with --two-layer-gate-edge-l1' \
    "$tmp_dir/$stem-upgrade.err"
done

python3 - "$outer_only" "$gate_ranked" "$tmp_dir" <<'PY'
import re
import sys

outer_path, gate_path, target_dir = sys.argv[1:]

def integer(text, name):
    match = re.search(rf"(?m)^\s*{re.escape(name)}\s*=\s*(\d+)\s*$", text)
    if not match:
        raise SystemExit(f"missing {name}")
    return int(match.group(1))

def corrupt(source, target, replacement):
    text = open(source, encoding="utf-8").read()
    pattern = r"(?m)^(\s*two_layer_gate_edge_l1_source_moment_indices\s*=\s*\{\s*)(\d+)"
    text, count = re.subn(pattern, lambda match: match.group(1) + str(replacement), text, count=1)
    if count != 1:
        raise SystemExit("failed to corrupt edge-L1 source metadata")
    open(target, "w", encoding="utf-8").write(text)

outer = open(outer_path, encoding="utf-8").read()
gate = open(gate_path, encoding="utf-8").read()
corrupt(outer_path, target_dir + "/invalid-outer-source.mtp",
        integer(outer, "outer_k_rank1_basic_begin"))
corrupt(gate_path, target_dir + "/invalid-gate-source.mtp",
        integer(gate, "two_layer_gate_k_rank_basic_begin") - 1)
PY

if "$bin" calc-efs "$tmp_dir/invalid-outer-source.mtp" "$cfg" \
     "$tmp_dir/invalid-outer-source.cfg" >"$tmp_dir/invalid-outer-source.err" 2>&1; then
  echo "outer-ranked edge-L1 accepted a source outside the raw gate graph" >&2
  exit 1
fi
grep -q 'preserved raw gate graph' "$tmp_dir/invalid-outer-source.err"

if "$bin" calc-efs "$tmp_dir/invalid-gate-source.mtp" "$cfg" \
     "$tmp_dir/invalid-gate-source.cfg" >"$tmp_dir/invalid-gate-source.err" 2>&1; then
  echo "gate-ranked edge-L1 accepted a source outside the contracted gate graph" >&2
  exit 1
fi
grep -q 'must belong to the gate K-rank graph' "$tmp_dir/invalid-gate-source.err"

legacy="$tmp_dir/invalid-legacy-gate-rank-edge.mtp"
python3 - "$gate_ranked" "$legacy" <<'PY'
import sys
source, target = sys.argv[1:]
text = open(source, encoding="utf-8").read()
text = text.replace(
    "two_layer_gate_edge_l1_source_moment_count",
    "two_layer_gate_edge_l1_source_mu_count")
text = text.replace(
    "two_layer_gate_edge_l1_source_moment_indices",
    "two_layer_gate_edge_l1_source_mu_indices")
open(target, "w", encoding="utf-8").write(text)
PY
if "$bin" calc-efs "$legacy" "$cfg" "$tmp_dir/invalid.cfg" \
     >"$tmp_dir/legacy.err" 2>&1; then
  echo "gate-ranked edge-L1 legacy raw source metadata was accepted" >&2
  exit 1
fi
grep -q 'require full source-moment metadata' "$tmp_dir/legacy.err"

echo "outer/gate K-rank edge-L1 checks passed"
