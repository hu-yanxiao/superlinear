#!/usr/bin/env bash
# Verify that an SH scaling_map=K model is mathematically identical in LAMMPS
# to the exact scaling_map=LK expansion that repeats every k block for all l.
# The test covers the CPU and/or Kokkos pair style, Chebyshev and Laguerre
# radial bases, direct and _lmp table paths, and plain-full, gate-full, and
# gate+rank model layouts.
set -Eeuo pipefail
trap 'echo "LAMMPS scaling_map=K parity failed at line ${LINENO}: ${BASH_COMMAND}" >&2' ERR

repo_root=$(cd "$(dirname "$0")/.." && pwd)
cd "$repo_root"

MLP_BIN="${MLP_BIN:-./bin/mlp-sus2}"
CPU_LAMMPS_BIN="${CPU_LAMMPS_BIN:-}"
KOKKOS_LAMMPS_BIN="${KOKKOS_LAMMPS_BIN:-}"
CPU_MPI_RANKS="${CPU_MPI_RANKS:-1}"
KOKKOS_MPI_RANKS="${KOKKOS_MPI_RANKS:-1}"
MPI_RUN="${MPI_RUN:-mpirun}"
KOKKOS_ARGS="${KOKKOS_ARGS:--k on g 1 -sf kk -pk kokkos neigh half newton on}"
KOKKOS_CHUNKSIZE="${KOKKOS_CHUNKSIZE:-128}"
KEEP="${SUS2_SH_TEST_KEEP_TMP:-0}"
SH_L_MAX="${SH_L_MAX:-2}"
SH_K_MAX="${SH_K_MAX:-2}"
SH_RADIAL_BASIS_SIZE="${SH_RADIAL_BASIS_SIZE:-4}"
PREPARE_ONLY=0

usage() {
  cat <<'EOF'
Usage: dev_test/lammps_sh_scaling_map_k_parity.sh [options]

  --mlp PATH       standalone mlp-sus2 binary (default: ./bin/mlp-sus2)
  --cpu PATH       CPU LAMMPS binary using pair_style sus2mtp
  --kokkos PATH    Kokkos LAMMPS binary using pair_style sus2mtp/kk
  --keep           retain temporary files
  --prepare-only   generate exact K/LK fixtures, then exit before LAMMPS

Unless --prepare-only is used, at least one of --cpu or --kokkos is required.  CPU_MPI_RANKS,
KOKKOS_MPI_RANKS, MPI_RUN, KOKKOS_ARGS, and KOKKOS_CHUNKSIZE may be
overridden for the local installation.  The script runs only tiny run-0
calculations; it does not submit a scheduler job.
EOF
}

while (($#)); do
  case "$1" in
    --mlp) MLP_BIN="$2"; shift 2 ;;
    --cpu) CPU_LAMMPS_BIN="$2"; shift 2 ;;
    --kokkos) KOKKOS_LAMMPS_BIN="$2"; shift 2 ;;
    --keep) KEEP=1; shift ;;
    --prepare-only) PREPARE_ONLY=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if (( ! PREPARE_ONLY )) && [[ -z "$CPU_LAMMPS_BIN" && -z "$KOKKOS_LAMMPS_BIN" ]]; then
  echo "provide --cpu and/or --kokkos" >&2
  usage >&2
  exit 2
fi
[[ -x "$MLP_BIN" ]] || {
  echo "MLP binary is not executable: $MLP_BIN" >&2
  exit 2
}
[[ -z "$CPU_LAMMPS_BIN" || -x "$CPU_LAMMPS_BIN" ]] || {
  echo "CPU LAMMPS binary is not executable: $CPU_LAMMPS_BIN" >&2
  exit 2
}
[[ -z "$KOKKOS_LAMMPS_BIN" || -x "$KOKKOS_LAMMPS_BIN" ]] || {
  echo "Kokkos LAMMPS binary is not executable: $KOKKOS_LAMMPS_BIN" >&2
  exit 2
}
if ! [[ "$KOKKOS_CHUNKSIZE" =~ ^[1-9][0-9]*$ ]]; then
  echo "KOKKOS_CHUNKSIZE must be a positive integer" >&2
  exit 2
fi
if ! [[ "$SH_L_MAX" =~ ^[0-4]$ ]] || \
   ! [[ "$SH_K_MAX" =~ ^[1-9][0-9]*$ ]] || \
   ! [[ "$SH_RADIAL_BASIS_SIZE" =~ ^[1-9][0-9]*$ ]]; then
  echo "SH_L_MAX must be 0..4; SH_K_MAX and SH_RADIAL_BASIS_SIZE must be positive" >&2
  exit 2
fi

tmp_dir=$(mktemp -d "${TMPDIR:-/tmp}/sus2sh-lammps-scaling-map-k-XXXXXX")
if [[ "$KEEP" == 1 ]]; then
  echo "keeping parity artifacts: $tmp_dir"
else
  trap 'rm -rf "$tmp_dir"' EXIT
fi

data_file="$tmp_dir/data.small"
cat >"$data_file" <<'DATA'
LAMMPS scaling_map=K parity

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

common_init=(
  --species-count=2
  --l-max="$SH_L_MAX"
  --k-max="$SH_K_MAX"
  --body-order=4
  --body-l-max="$SH_L_MAX,$SH_L_MAX,$SH_L_MAX,$SH_L_MAX"
  --radial-basis-size="$SH_RADIAL_BASIS_SIZE"
  --radial-basis-type=RBChebyshev_sss
  --cutoff=5.0
  --sh-factor-pruning=q-total
  --scaling-map=K
  --init-params=same
)

prepare_pair() {
  local case_name=$1
  shift
  local init_model="$tmp_dir/$case_name.init.mtp"
  local materialized="$tmp_dir/$case_name.materialized.mtp"

  "$MLP_BIN" init-sh "$init_model" "${common_init[@]}" "$@" >/dev/null
  "$MLP_BIN" roundtrip-model-dev "$init_model" "$materialized" >/dev/null

  python3 - "$materialized" "$tmp_dir" "$case_name" \
    "$SH_K_MAX" "$((SH_L_MAX + 1))" <<'PY'
import pathlib
import re
import sys

source = pathlib.Path(sys.argv[1])
root = pathlib.Path(sys.argv[2])
case = sys.argv[3]
kmax = int(sys.argv[4])
lchannels = int(sys.argv[5])
text = source.read_text(encoding="utf-8")
number = r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?"

species = 2
pairs = species * species
block_width = 2 * pairs

def vector(name, content=None):
    if content is None:
        content = text
    match = re.search(
        rf"^{re.escape(name)}\s*=\s*\{{([^}}]*)\}}",
        content,
        re.M | re.S,
    )
    if not match:
        raise SystemExit(f"{case}: missing {name}")
    return [float(value) for value in re.findall(number, match.group(1))]

def replace_vector(content, name, values, required=True):
    match = re.search(
        rf"^{re.escape(name)}\s*=\s*\{{([^}}]*)\}}",
        content,
        re.M | re.S,
    )
    if not match:
        if required:
            raise SystemExit(f"{case}: missing {name}")
        return content
    replacement = (
        name + " = {" +
        ", ".join(f"{value:.17e}" for value in values) +
        "}"
    )
    return content[:match.start()] + replacement + content[match.end():]

def radial_section(content):
    match = re.search(
        r"(?ms)^\s*radial_coeffs\s*$.*?(?=^alpha_moments_count\s*=)",
        content,
    )
    if not match:
        raise SystemExit(f"{case}: missing radial_coeffs section")
    return match.group(0)

if not re.search(r"^scaling_map\s*=\s*K\s*$", text, re.M):
    raise SystemExit(f"{case}: init-sh did not write scaling_map=K")
has_gate = re.search(
    r"^two_layer_gate_enabled\s*=\s*(?:1|true)\s*$", text, re.M
)
if case == "full" and has_gate:
    raise SystemExit("full: unexpectedly serialized a gate")
if case in ("gate", "rank") and not has_gate:
    raise SystemExit(f"{case}: gate metadata is missing")
if case == "rank":
    for rank_vector in (
        "outer_k_rank1_weights",
        "two_layer_gate_k_rank_weights",
    ):
        if not re.search(rf"^{rank_vector}\s*=", text, re.M):
            raise SystemExit(f"rank: missing {rank_vector}")

# Deliberately distinct k blocks make a wrong l/k indexing immediately visible.
k_blocks = []
for k in range(kmax):
    slopes = [2.71 + 0.31 * k + 0.041 * pair for pair in range(pairs)]
    shifts = [1.23 + 0.19 * k + 0.029 * pair for pair in range(pairs)]
    k_blocks.append(slopes + shifts)
text = replace_vector(text, "scal_coeffs", [
    value for block in k_blocks for value in block
])

# Keep all physical paths nonzero.  The initialized radial coefficient matrix
# is already nonzero and is serialized as a labelled multiline section rather
# than a single ``name = {...}`` vector.  Leave that full mu=(k,l) matrix
# byte-identical in the K/LK pair.
moments = vector("moment_coeffs", text)
text = replace_vector(text, "moment_coeffs", [
    ((0.018 + 0.0013 * (index % 19)) *
     (-1.0 if index % 5 == 2 else 1.0))
    if index < min(32, len(moments)) else 0.0
    for index in range(len(moments))
])
species_coeffs = vector("species_coeffs", text)
text = replace_vector(
    text, "species_coeffs",
    [0.37 + 0.11 * index for index in range(len(species_coeffs))]
)

gate_specs = {
    "two_layer_gate_radial_coeffs": (0.031, 0.006, 7),
    "two_layer_gate_additive_coeffs": (0.17, 0.013, 5),
    "two_layer_gate_type_coeffs": (0.73, 0.037, 11),
    "two_layer_gate_body_mix_weights": (0.12, 0.009, 13),
    "two_layer_gate_weights": (0.021, 0.0007, 17),
}
for name, (base, step, period) in gate_specs.items():
    if not re.search(rf"^{re.escape(name)}\s*=", text, re.M):
        continue
    values = vector(name, text)
    text = replace_vector(text, name, [
        (base + step * (index % period)) *
        (-1.0 if index % 7 == 3 else 1.0)
        for index in range(len(values))
    ])

for name, base, step, period in (
    ("outer_k_rank1_weights", 0.41, 0.019, 9),
    ("two_layer_gate_k_rank_weights", 0.67, -0.027, 7),
):
    if not re.search(rf"^{re.escape(name)}\s*=", text, re.M):
        continue
    values = vector(name, text)
    text = replace_vector(text, name, [
        (base + step * (index % period)) *
        (-1.0 if index % 6 == 1 else 1.0)
        for index in range(len(values))
    ])

k_text = text
lk_text, replacements = re.subn(
    r"^scaling_map\s*=\s*K\s*$",
    "scaling_map = LK",
    k_text,
    count=1,
    flags=re.M,
)
if replacements != 1:
    raise SystemExit(f"{case}: could not create LK expansion")

# Exact expansion: mu=k*(lmax+1)+l receives the original k block.
expanded = []
for mu in range(kmax * lchannels):
    expanded.extend(k_blocks[mu // lchannels])
lk_text = replace_vector(lk_text, "scal_coeffs", expanded)

if len(vector("scal_coeffs", k_text)) != kmax * block_width:
    raise SystemExit(f"{case}: wrong K scal_coeffs count")
if len(vector("scal_coeffs", lk_text)) != kmax * lchannels * block_width:
    raise SystemExit(f"{case}: wrong expanded LK scal_coeffs count")
if radial_section(k_text) != radial_section(lk_text):
    raise SystemExit(f"{case}: LK expansion changed mu-local radial coefficients")

for basis_label, basis_type in (
    ("chebyshev", "RBChebyshev_sss"),
    ("laguerre", "RBLaguerre_log1p"),
):
    for mode, basis_suffix in (("direct", ""), ("table", "_lmp")):
        for mapping, content in (("K", k_text), ("LK", lk_text)):
            content, count = re.subn(
                r"^radial_basis_type\s*=\s*RBChebyshev_sss\s*$",
                f"radial_basis_type = {basis_type}{basis_suffix}",
                content,
                count=1,
                flags=re.M,
            )
            if count != 1:
                raise SystemExit(f"{case}: missing direct radial marker")
            target = root / f"{case}.{basis_label}.{mode}.{mapping}.mtp"
            target.write_text(content, encoding="utf-8")
PY
}

gate_args=(
  --two-layer-gate
  --two-layer-gate-factor-pruning=q-total
  --two-layer-gate-mode=mu-scalar-full
  --two-layer-gate-site-mode=neighbor
  --two-layer-gate-shared-radial
)
outer_rank=$((SH_K_MAX < 2 ? SH_K_MAX : 2))

prepare_pair full
prepare_pair gate "${gate_args[@]}"
prepare_pair rank "${gate_args[@]}" --outer-k-rank="$outer_rank" --two-layer-gate-k-rank=1

if ((PREPARE_ONLY)); then
  echo "LAMMPS_SH_SCALING_MAP_K_FIXTURES: $tmp_dir"
  exit 0
fi

make_input() {
  local model=$1
  local label=$2
  local style=$3
  local radial_mode=$4
  local style_args=""
  if [[ "$style" == "sus2mtp/kk" ]]; then
    style_args="chunksize $KOKKOS_CHUNKSIZE"
  fi
  cat >"$tmp_dir/in.$label" <<LAMMPS
units metal
atom_style atomic
boundary p p p
newton on
read_data $data_file
pair_style $style $model $style_args radial_mode $radial_mode tabstep 0.0005
pair_coeff * *
neighbor 1.0 bin
neigh_modify every 1 delay 0 check yes
compute atom_energy all pe/atom
compute sum_atom_energy all reduce sum c_atom_energy
thermo_style custom step pe c_sum_atom_energy pxx pyy pzz pyz pxz pxy vol
thermo_modify format float %.17g
thermo 1
run 0 post no
write_dump all custom $tmp_dir/dump.$label id type fx fy fz modify sort id format float %.17g
LAMMPS
}

run_lammps() {
  local executable=$1
  local ranks=$2
  local style=$3
  local model=$4
  local label=$5
  local radial_mode=$6
  make_input "$model" "$label" "$style" "$radial_mode"
  local -a extra=()
  if [[ "$style" == "sus2mtp/kk" ]]; then
    read -r -a extra <<<"$KOKKOS_ARGS"
  fi
  "$MPI_RUN" -n "$ranks" "$executable" "${extra[@]}" \
    -in "$tmp_dir/in.$label" -log "$tmp_dir/log.$label" \
    >"$tmp_dir/stdout.$label" 2>"$tmp_dir/stderr.$label"
  grep -Fq "resolved_mode=$radial_mode, requested_mode=$radial_mode" \
    "$tmp_dir/log.$label"
}

for case_name in full gate rank; do
  for basis_name in chebyshev laguerre; do
    for radial_mode in direct table; do
      for mapping in K LK; do
        model="$tmp_dir/$case_name.$basis_name.$radial_mode.$mapping.mtp"
        if [[ -n "$CPU_LAMMPS_BIN" ]]; then
          run_lammps "$CPU_LAMMPS_BIN" "$CPU_MPI_RANKS" sus2mtp \
            "$model" "cpu.$case_name.$basis_name.$radial_mode.$mapping" "$radial_mode"
        fi
        if [[ -n "$KOKKOS_LAMMPS_BIN" ]]; then
          run_lammps "$KOKKOS_LAMMPS_BIN" "$KOKKOS_MPI_RANKS" sus2mtp/kk \
            "$model" "kk.$case_name.$basis_name.$radial_mode.$mapping" "$radial_mode"
        fi
      done
    done
  done
done

python3 - "$tmp_dir" <<'PY'
import math
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
pressure_to_ev_a3 = 1602176.6208

def parse(label):
    lines = (root / f"log.{label}").read_text(encoding="utf-8").splitlines()
    energy = atom_energy = virial = None
    for pos, raw in enumerate(lines):
        header = raw.split()
        required = {
            "Step", "PotEng", "c_sum_atom_energy",
            "Pxx", "Pyy", "Pzz", "Pyz", "Pxz", "Pxy", "Volume",
        }
        if not required.issubset(header) or pos + 1 >= len(lines):
            continue
        row = lines[pos + 1].split()
        if len(row) != len(header) or row[header.index("Step")] != "0":
            continue
        energy = float(row[header.index("PotEng")])
        atom_energy = float(row[header.index("c_sum_atom_energy")])
        volume = float(row[header.index("Volume")])
        virial = [
            float(row[header.index(key)]) * volume / pressure_to_ev_a3
            for key in ("Pxx", "Pyy", "Pzz", "Pyz", "Pxz", "Pxy")
        ]

    dump = (root / f"dump.{label}").read_text(encoding="utf-8").splitlines()
    forces = None
    for pos, raw in enumerate(dump):
        if not raw.startswith("ITEM: ATOMS"):
            continue
        header = raw.split()[2:]
        id_col = header.index("id")
        force_cols = [header.index(key) for key in ("fx", "fy", "fz")]
        rows = []
        for line in dump[pos + 1:]:
            if line.startswith("ITEM:"):
                break
            fields = line.split()
            if fields:
                rows.append((
                    int(fields[id_col]),
                    [float(fields[col]) for col in force_cols],
                ))
        rows.sort()
        forces = [value for _, force in rows for value in force]
        break

    if energy is None or atom_energy is None or virial is None or \
            forces is None or len(forces) != 24:
        raise SystemExit(f"{label}: cannot parse E/F/virial")
    values = [energy, atom_energy, *forces, *virial]
    if not all(math.isfinite(value) for value in values):
        raise SystemExit(f"{label}: non-finite E/F/virial")
    if abs(energy - atom_energy) > 2.0e-10:
        raise SystemExit(
            f"{label}: global/per-atom energy mismatch "
            f"{abs(energy - atom_energy):.3e}"
        )
    return energy, forces, virial

def max_error(lhs, rhs):
    return max(abs(x - y) for x, y in zip(lhs, rhs))

seen = 0
for engine in ("cpu", "kk"):
    for case in ("full", "gate", "rank"):
        for basis in ("chebyshev", "laguerre"):
            for mode in ("direct", "table"):
                k_label = f"{engine}.{case}.{basis}.{mode}.K"
                lk_label = f"{engine}.{case}.{basis}.{mode}.LK"
                if not (root / f"log.{k_label}").exists():
                    continue
                k_energy, k_forces, k_virial = parse(k_label)
                l_energy, l_forces, l_virial = parse(lk_label)
                errors = (
                    abs(k_energy - l_energy),
                    max_error(k_forces, l_forces),
                    max_error(k_virial, l_virial),
                )
                tolerance = 2.0e-10 if engine == "cpu" else 5.0e-6
                scale = max(
                    1.0,
                    abs(k_energy),
                    *(abs(value) for value in k_forces),
                    *(abs(value) for value in k_virial),
                )
                if max(errors) > tolerance * scale:
                    raise SystemExit(
                        f"{engine}/{case}/{basis}/{mode}: "
                        f"K vs expanded-LK mismatch dE={errors[0]:.3e} "
                        f"dF={errors[1]:.3e} dV={errors[2]:.3e}, "
                        f"limit={tolerance * scale:.3e}"
                    )
                if max(abs(value) for value in k_forces) < 1.0e-8:
                    raise SystemExit(
                        f"{engine}/{case}/{basis}/{mode}: trivial zero forces"
                    )
                if max(abs(value) for value in k_virial) < 1.0e-8:
                    raise SystemExit(
                        f"{engine}/{case}/{basis}/{mode}: trivial zero virial"
                    )
                print(
                    f"{engine}/{case}/{basis}/{mode}: "
                    f"dE={errors[0]:.3e} dF={errors[1]:.3e} "
                    f"dV={errors[2]:.3e}"
                )
                seen += 1

if seen == 0:
    raise SystemExit("no LAMMPS result pair was checked")
print(f"LAMMPS_SH_SCALING_MAP_K_PARITY: ok ({seen} result pairs)")
PY
