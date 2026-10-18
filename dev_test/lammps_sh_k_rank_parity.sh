#!/usr/bin/env bash
# Compare release3 rank models between standalone calc-efs and the CPU and/or
# Kokkos LAMMPS pair styles.  It is intended to be run on the server after the
# corresponding LAMMPS binaries have been built; it does not submit jobs.
set -euo pipefail

cd "$(dirname "$0")/.."

MLP_BIN="${MLP_BIN:-./bin/mlp-sus2}"
CPU_LAMMPS_BIN="${CPU_LAMMPS_BIN:-}"
KOKKOS_LAMMPS_BIN="${KOKKOS_LAMMPS_BIN:-}"
CPU_MPI_RANKS="${CPU_MPI_RANKS:-1}"
KOKKOS_MPI_RANKS="${KOKKOS_MPI_RANKS:-1}"
MPI_RUN="${MPI_RUN:-mpirun}"
KOKKOS_ARGS="${KOKKOS_ARGS:--k on g 1 -sf kk -pk kokkos neigh half newton on}"
KOKKOS_CHUNKSIZE="${KOKKOS_CHUNKSIZE:-2048}"
PARITY_K_MAX="${SUS2_SH_PARITY_K_MAX:-4}"
RADIAL_MODE="${SUS2_SH_PARITY_RADIAL_MODE:-auto}"
RADIAL_BASIS_TYPE="${SUS2_SH_PARITY_RADIAL_BASIS_TYPE:-RBChebyshev_sss}"
INIT_RADIAL_BASIS_TYPE="${SUS2_SH_PARITY_INIT_RADIAL_BASIS_TYPE:-$RADIAL_BASIS_TYPE}"
KEEP="${SUS2_SH_TEST_KEEP_TMP:-0}"

usage() {
  cat <<'EOF'
Usage: dev_test/lammps_sh_k_rank_parity.sh [options]

  --mlp PATH       standalone mlp-sus2 binary (default: ./bin/mlp-sus2)
  --cpu PATH       CPU LAMMPS binary using pair_style sus2mtp
  --kokkos PATH    Kokkos LAMMPS binary using pair_style sus2mtp/kk
  --keep           retain temporary test files

At least one of --cpu or --kokkos is required.  KOKKOS_ARGS may be overridden
for a site-specific GPU/host Kokkos launch configuration.  KOKKOS_CHUNKSIZE
may be set to a positive integer to exercise multi-chunk execution.
SUS2_SH_PARITY_K_MAX may be set to an integer >= 2 (default: 4); using 3
exercises the Kokkos no-moment-cache recomputation path.
SUS2_SH_PARITY_RADIAL_MODE selects auto, table, or direct (default: auto).
SUS2_SH_PARITY_RADIAL_BASIS_TYPE selects the serialized radial basis; when a
desired basis cannot be emitted directly by init-sh,
SUS2_SH_PARITY_INIT_RADIAL_BASIS_TYPE may select its initialization basis.
EOF
}

while (($#)); do
  case "$1" in
    --mlp) MLP_BIN="$2"; shift 2 ;;
    --cpu) CPU_LAMMPS_BIN="$2"; shift 2 ;;
    --kokkos) KOKKOS_LAMMPS_BIN="$2"; shift 2 ;;
    --keep) KEEP=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$CPU_LAMMPS_BIN" && -z "$KOKKOS_LAMMPS_BIN" ]]; then
  echo "provide --cpu and/or --kokkos" >&2
  usage >&2
  exit 2
fi
if ! [[ "$KOKKOS_CHUNKSIZE" =~ ^[1-9][0-9]*$ ]]; then
  echo "KOKKOS_CHUNKSIZE must be a positive integer" >&2
  exit 2
fi
if ! [[ "$PARITY_K_MAX" =~ ^[0-9]+$ ]] || ((PARITY_K_MAX < 2)); then
  echo "SUS2_SH_PARITY_K_MAX must be an integer >= 2" >&2
  exit 2
fi
case "$RADIAL_MODE" in
  auto|table|direct) ;;
  *) echo "SUS2_SH_PARITY_RADIAL_MODE must be auto, table, or direct" >&2; exit 2 ;;
esac
[[ -x "$MLP_BIN" ]] || { echo "MLP binary is not executable: $MLP_BIN" >&2; exit 2; }
[[ -z "$CPU_LAMMPS_BIN" || -x "$CPU_LAMMPS_BIN" ]] || {
  echo "CPU LAMMPS binary is not executable: $CPU_LAMMPS_BIN" >&2; exit 2; }
[[ -z "$KOKKOS_LAMMPS_BIN" || -x "$KOKKOS_LAMMPS_BIN" ]] || {
  echo "Kokkos LAMMPS binary is not executable: $KOKKOS_LAMMPS_BIN" >&2; exit 2; }

tmp_dir=$(mktemp -d "${TMPDIR:-/tmp}/sus2sh-lammps-k-rank-parity-XXXXXX")
if [[ "$KEEP" == 1 ]]; then
  echo "keeping parity artifacts: $tmp_dir"
else
  trap 'rm -rf "$tmp_dir"' EXIT
fi

cfg="$tmp_dir/one.cfg"
data="$tmp_dir/data.small"

cat > "$cfg" <<'CFG'
BEGIN_CFG
 Size
    8
 Supercell
    1.2000000000000000e+01 0.0000000000000000e+00 0.0000000000000000e+00
    0.0000000000000000e+00 1.2000000000000000e+01 0.0000000000000000e+00
    0.0000000000000000e+00 0.0000000000000000e+00 1.2000000000000000e+01
 AtomData:  id type       cartes_x      cartes_y      cartes_z           fx          fy          fz
    1 0  1.0  1.0  1.0  0 0 0
    2 0  2.3  1.2  1.1  0 0 0
    3 1  4.8  1.4  1.2  0 0 0
    4 1  5.7  1.6  1.3  0 0 0
    5 0  6.4  1.8  1.1  0 0 0
    6 1  7.3  2.0  1.2  0 0 0
    7 0  9.8  1.5  1.4  0 0 0
    8 1 10.6  1.3  1.0  0 0 0
 Energy
    0.0
 PlusStress:  xx          yy          zz          yz          xz          xy
     0 0 0 0 0 0
END_CFG
CFG

cat > "$data" <<'DATA'
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

common_init=(
  --species-count=2 --l-max=4 --k-max="$PARITY_K_MAX" --body-order=6
  --body-l-max=4,4,4,2,1 --radial-basis-size=4
  --radial-basis-type="$INIT_RADIAL_BASIS_TYPE" --cutoff=5.0
  --sh-factor-pruning=q-total --two-layer-gate
  --two-layer-gate-factor-pruning=q-total
  --two-layer-gate-mode=mu-scalar-full --two-layer-gate-body-order=5
  --two-layer-gate-site-mode=neighbor --two-layer-gate-shared-radial
)

prepare_model() {
  local name="$1"; shift
  local init="$tmp_dir/$name.init.sus2"
  local materialized="$tmp_dir/$name.materialized.sus2"
  local model="$tmp_dir/$name.lmp.sus2"
  "$MLP_BIN" init-sh "$init" "${common_init[@]}" "$@" >/dev/null || return
  "$MLP_BIN" roundtrip-model-dev "$init" "$materialized" >/dev/null || return
  python3 - "$materialized" "$model" "$name" "$PARITY_K_MAX" \
    "$RADIAL_BASIS_TYPE" "$INIT_RADIAL_BASIS_TYPE" <<'PY' || return
import re
import sys

source, target, name, kmax_text, radial_basis_type, init_radial_basis_type = sys.argv[1:]
kmax = int(kmax_text)
lmax = 4
text = open(source, encoding="utf-8").read()
number = r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?"

def replace_vector(key, values, required=True):
    global text
    match = re.search(rf"{re.escape(key)}\s*=\s*\{{([^}}]*)\}}", text, re.S)
    if not match:
        if required:
            raise SystemExit(f"{name}: missing {key}")
        return
    replacement = key + " = {" + ", ".join(f"{x:.16e}" for x in values) + "}"
    text = text[:match.start()] + replacement + text[match.end():]

def vector(key):
    match = re.search(rf"{re.escape(key)}\s*=\s*\{{([^}}]*)\}}", text, re.S)
    if not match:
        raise SystemExit(f"{name}: missing {key}")
    return [float(x) for x in re.findall(number, match.group(1))]

moments = vector("moment_coeffs")
for q in range(len(moments)):
    moments[q] = 0.0
for q in range(min(12, len(moments))):
    moments[q] = (0.011 + 0.002 * q) * (-1.0 if q % 3 == 1 else 1.0)
replace_vector("moment_coeffs", moments)

species = vector("species_coeffs")
replace_vector("species_coeffs", [0.9 + 0.1 * i for i in range(len(species))])

gate = vector("two_layer_gate_weights")
for q in range(len(gate)):
    gate[q] = 0.0
for q in range(min(20, len(gate))):
    gate[q] = (0.017 + 0.001 * (q % 7)) * (-1.0 if q % 5 == 2 else 1.0)
replace_vector("two_layer_gate_weights", gate)

# Rank-2 weights deliberately differ by l/channel/k, exercising the exact
# serialized l-major layout.  Rank-1 outer uses direct radial and stays one.
if "outer-rank2" in name or "gate2-outer2" in name or "outer2-gate1" in name:
    weights = vector("outer_k_rank1_weights")
    replace_vector("outer_k_rank1_weights", [
        0.31 + 0.017 * (q % 9) * (-1.0 if q % 4 == 1 else 1.0)
        for q in range(len(weights))])
if "outer-rankK-identity" in name:
    weights = vector("outer_k_rank1_weights")
    rank = kmax
    expected = (lmax + 1) * rank * kmax
    if len(weights) != expected:
        raise SystemExit(f"{name}: unexpected rank-K weight count {len(weights)}")
    identity = []
    for l in range(lmax + 1):
        for channel in range(rank):
            for k in range(kmax):
                identity.append(1.0 if channel == k else 0.0)
    replace_vector("outer_k_rank1_weights", identity)
if "gate1-outer1" in name or "outer2-gate1" in name:
    weights = vector("two_layer_gate_k_rank_weights")
    replace_vector("two_layer_gate_k_rank_weights", [
        0.71 - 0.09 * (q % 4) for q in range(len(weights))])
if "gate2-outer2" in name:
    weights = vector("two_layer_gate_k_rank_weights")
    replace_vector("two_layer_gate_k_rank_weights", [
        -0.23 + 0.041 * (q % 11) * (-1.0 if q % 3 == 0 else 1.0)
        for q in range(len(weights))])
if "outer1-gate2" in name:
    weights = vector("outer_k_rank1_weights")
    replace_vector("outer_k_rank1_weights", [
        0.43 - 0.029 * (q % 7) * (-1.0 if q % 5 == 0 else 1.0)
        for q in range(len(weights))])
    weights = vector("two_layer_gate_k_rank_weights")
    replace_vector("two_layer_gate_k_rank_weights", [
        -0.19 + 0.037 * (q % 13) * (-1.0 if q % 4 == 2 else 1.0)
        for q in range(len(weights))])
if "default-mode-outer2-gate1" in name:
    # Deliberately keep every default-mode signal path visibly active: shared
    # gate radial, additive node signal, type signal, body mix and scalar
    # gate weights.  This exercises the distinct Kokkos default-mode finalize.
    radial = vector("two_layer_gate_radial_coeffs")
    replace_vector("two_layer_gate_radial_coeffs", [
        0.031 + 0.006 * (q % 5) for q in range(len(radial))])
    additive = vector("two_layer_gate_additive_coeffs")
    replace_vector("two_layer_gate_additive_coeffs", [
        0.19 + 0.013 * (q % 7) for q in range(len(additive))])
    type_coeffs = vector("two_layer_gate_type_coeffs")
    replace_vector("two_layer_gate_type_coeffs", [
        0.73 + 0.09 * (q % 4) for q in range(len(type_coeffs))])
    mix = vector("two_layer_gate_body_mix_weights")
    replace_vector("two_layer_gate_body_mix_weights", [
        0.17 - 0.023 * (q % 9) * (-1.0 if q % 4 == 1 else 1.0)
        for q in range(len(mix))])
if name.startswith("readout-") and name != "readout-zero":
    readout = vector("two_layer_gate_readout_coeffs")
    for q in range(len(readout)):
        readout[q] = (0.013 + 0.0007 * (q % 17)) * (
            -1.0 if q % 4 == 1 else 1.0)
    replace_vector("two_layer_gate_readout_coeffs", readout)
    if "zero-gate" in name:
        # The readout is an independent linear head on the first gate scalars.
        # Zeroing the ordinary gate contraction makes accidental routing
        # through two_layer_gate_weights immediately visible in E/F/virial.
        replace_vector("two_layer_gate_weights", [0.0] * len(gate))

direct_marker = f"radial_basis_type = {init_radial_basis_type}"
table_marker = f"radial_basis_type = {radial_basis_type}_lmp"
if direct_marker not in text:
    raise SystemExit(f"{name}: missing radial basis marker {direct_marker}")
text = text.replace(direct_marker, table_marker, 1)
open(target, "w", encoding="utf-8").write(text)
PY
  [[ -s "$model" ]] || {
    echo "failed to produce model: $model" >&2
    return 1
  }
  echo "$model"
}

make_input() {
  local model="$1" label="$2" style="$3"
  local style_args=""
  if [[ "$style" == sus2mtp/kk ]]; then
    style_args="chunksize $KOKKOS_CHUNKSIZE"
  fi
  cat > "$tmp_dir/in.$label" <<LAMMPS
units metal
atom_style atomic
boundary p p p
newton on
read_data $data
pair_style $style $model $style_args radial_mode $RADIAL_MODE tabstep 0.0005
pair_coeff * *
neighbor 1.0 bin
neigh_modify every 1 delay 0 check yes
compute atom_energy all pe/atom
compute sum_atom_energy all reduce sum c_atom_energy
# The pair style accumulates only global virial[].  The stress/atom compute
# requests per-atom vatom, which this pair style does not fill, so it is not
# a valid virial source here.  In metal units P*V/1602176.6208 is eV.
thermo_style custom step pe c_sum_atom_energy pxx pyy pzz pyz pxz pxy vol
thermo_modify format float %.17g
thermo 1
run 0 post no
write_dump all custom $tmp_dir/dump.$label id type x y z fx fy fz modify sort id format float %.17g
LAMMPS
}

make_force_only_input() {
  local model="$1" label="$2" style="$3"
  local style_args=""
  if [[ "$style" == sus2mtp/kk ]]; then
    style_args="chunksize $KOKKOS_CHUNKSIZE"
  fi
  cat > "$tmp_dir/in.$label" <<LAMMPS
units metal
atom_style atomic
boundary p p p
newton on
read_data $data
pair_style $style $model $style_args radial_mode $RADIAL_MODE tabstep 0.0005
pair_coeff * *
neighbor 1.0 bin
neigh_modify every 1 delay 0 check yes
thermo_style custom step
thermo 1
run 0 post no
write_dump all custom $tmp_dir/dump.$label id type x y z fx fy fz modify sort id format float %.17g
LAMMPS
}

run_lammps() {
  local executable="$1" ranks="$2" style="$3" model="$4" label="$5"
  make_input "$model" "$label" "$style"
  local -a extra=()
  if [[ "$style" == sus2mtp/kk ]]; then
    read -r -a extra <<<"$KOKKOS_ARGS"
  fi
  "$MPI_RUN" -n "$ranks" "$executable" "${extra[@]}" \
    -in "$tmp_dir/in.$label" -log "$tmp_dir/log.$label" \
    >"$tmp_dir/stdout.$label" 2>"$tmp_dir/stderr.$label"
}

run_force_only_lammps() {
  local executable="$1" ranks="$2" style="$3" model="$4" label="$5"
  make_force_only_input "$model" "$label" "$style"
  local -a extra=()
  if [[ "$style" == sus2mtp/kk ]]; then
    read -r -a extra <<<"$KOKKOS_ARGS"
  fi
  "$MPI_RUN" -n "$ranks" "$executable" "${extra[@]}" \
    -in "$tmp_dir/in.$label" -log "$tmp_dir/log.$label" \
    >"$tmp_dir/stdout.$label" 2>"$tmp_dir/stderr.$label"
}

assert_model_rejected() {
  local executable="$1" ranks="$2" style="$3" model="$4" label="$5"
  local expected_diagnostic="$6"
  make_input "$model" "$label" "$style"
  local -a extra=()
  if [[ "$style" == sus2mtp/kk ]]; then
    read -r -a extra <<<"$KOKKOS_ARGS"
  fi
  if "$MPI_RUN" -n "$ranks" "$executable" "${extra[@]}" \
      -in "$tmp_dir/in.$label" -log "$tmp_dir/log.$label" \
      >"$tmp_dir/stdout.$label" 2>"$tmp_dir/stderr.$label"; then
    echo "$label unexpectedly accepted an unsupported model" >&2
    return 1
  fi
  if ! grep -Fq "$expected_diagnostic" \
      "$tmp_dir/stdout.$label" "$tmp_dir/stderr.$label" "$tmp_dir/log.$label"; then
    echo "$label rejected the model without the required diagnostic" >&2
    return 1
  fi
  echo "$label: unsupported-model rejection ok"
}

cases=(rank0 outer-rank1-direct outer-rank2 outer-rankK-identity gate1-outer1 gate2-outer2 outer2-gate1 outer1-gate2 default-mode-outer2-gate1 readout-full-zero-gate readout-body-zero-gate readout-full-mixed readout-zero)
models=()
model="$(prepare_model rank0)" || exit 1; models+=("$model")
model="$(prepare_model outer-rank1-direct --outer-k-rank=1)" || exit 1; models+=("$model")
model="$(prepare_model outer-rank2 --outer-k-rank=2)" || exit 1; models+=("$model")
model="$(prepare_model outer-rankK-identity --outer-k-rank="$PARITY_K_MAX")" || exit 1; models+=("$model")
model="$(prepare_model gate1-outer1 --outer-k-rank=1 --two-layer-gate-k-rank=1)" || exit 1; models+=("$model")
model="$(prepare_model gate2-outer2 --outer-k-rank=2 --two-layer-gate-k-rank=2)" || exit 1; models+=("$model")
model="$(prepare_model outer2-gate1 --outer-k-rank=2 --two-layer-gate-k-rank=1)" || exit 1; models+=("$model")
model="$(prepare_model outer1-gate2 --outer-k-rank=1 --two-layer-gate-k-rank=2)" || exit 1; models+=("$model")
model="$(prepare_model default-mode-outer2-gate1 --outer-k-rank=2 --two-layer-gate-k-rank=1 --two-layer-gate-mode=mu-body-linear-combo)" || exit 1; models+=("$model")
model="$(prepare_model readout-full-zero-gate --two-layer-gate-readout)" || exit 1; models+=("$model")
model="$(prepare_model readout-body-zero-gate --two-layer-gate-readout --two-layer-gate-mode=mu-body-linear-combo)" || exit 1; models+=("$model")
model="$(prepare_model readout-full-mixed --two-layer-gate-readout)" || exit 1; models+=("$model")
model="$(prepare_model readout-zero --two-layer-gate-readout)" || exit 1; models+=("$model")
decoupled_model="$(prepare_model decoupled-scaling-rejected \
  --outer-k-rank=2 --two-layer-gate-k-rank=1 \
  --two-layer-gate-decoupled-scaling)" || exit 1
multi_layer_model="$(prepare_model multi-layer-rejected \
  --multi-layer-gate)" || exit 1

for index in "${!cases[@]}"; do
  case_name="${cases[$index]}"
  model="${models[$index]}"
  "$MLP_BIN" calc-efs "$model" "$cfg" "$tmp_dir/$case_name.standalone.cfg" >/dev/null
  if [[ -n "$CPU_LAMMPS_BIN" ]]; then
    run_lammps "$CPU_LAMMPS_BIN" "$CPU_MPI_RANKS" sus2mtp "$model" "cpu.$case_name"
  fi
  if [[ -n "$KOKKOS_LAMMPS_BIN" ]]; then
    run_lammps "$KOKKOS_LAMMPS_BIN" "$KOKKOS_MPI_RANKS" sus2mtp/kk "$model" "kk.$case_name"
  fi
done

if [[ "$RADIAL_MODE" == direct ]]; then
  for engine in cpu kk; do
    if [[ "$engine" == cpu && -z "$CPU_LAMMPS_BIN" ]] ||
       [[ "$engine" == kk && -z "$KOKKOS_LAMMPS_BIN" ]]; then
      continue
    fi
    for case_name in "${cases[@]}"; do
      grep -Fq \
        'SUS2 radial evaluation: resolved_mode=direct, requested_mode=direct' \
        "$tmp_dir/log.$engine.$case_name"
      if grep -Fq 'Preinterpolation table built successfully' \
          "$tmp_dir/log.$engine.$case_name"; then
        echo "$engine.$case_name unexpectedly built a radial table" >&2
        exit 1
      fi
    done
  done
fi

for force_only_case in readout-full-zero-gate readout-body-zero-gate \
    readout-full-mixed; do
  for index in "${!cases[@]}"; do
    if [[ "${cases[$index]}" != "$force_only_case" ]]; then
      continue
    fi
    model="${models[$index]}"
    if [[ -n "$CPU_LAMMPS_BIN" ]]; then
      run_force_only_lammps "$CPU_LAMMPS_BIN" "$CPU_MPI_RANKS" sus2mtp \
        "$model" "cpu.force-only.$force_only_case"
    fi
    if [[ -n "$KOKKOS_LAMMPS_BIN" ]]; then
      run_force_only_lammps "$KOKKOS_LAMMPS_BIN" "$KOKKOS_MPI_RANKS" \
        sus2mtp/kk "$model" "kk.force-only.$force_only_case"
    fi
    break
  done
done

if [[ -n "$CPU_LAMMPS_BIN" ]]; then
  assert_model_rejected "$CPU_LAMMPS_BIN" "$CPU_MPI_RANKS" sus2mtp \
    "$decoupled_model" cpu.decoupled-scaling-rejected \
    "do not support two_layer_gate_scaling_mode=decoupled"
  assert_model_rejected "$CPU_LAMMPS_BIN" "$CPU_MPI_RANKS" sus2mtp \
    "$multi_layer_model" cpu.multi-layer-rejected \
    "do not support multi_layer_gate_enabled models"
fi
if [[ -n "$KOKKOS_LAMMPS_BIN" ]]; then
  assert_model_rejected "$KOKKOS_LAMMPS_BIN" "$KOKKOS_MPI_RANKS" sus2mtp/kk \
    "$decoupled_model" kk.decoupled-scaling-rejected \
    "do not support two_layer_gate_scaling_mode=decoupled"
  assert_model_rejected "$KOKKOS_LAMMPS_BIN" "$KOKKOS_MPI_RANKS" sus2mtp/kk \
    "$multi_layer_model" kk.multi-layer-rejected \
    "do not support multi_layer_gate_enabled models"
fi

python3 - "$tmp_dir" "${cases[@]}" <<'PY'
import math
import pathlib
import re
import sys

root = pathlib.Path(sys.argv[1])
cases = sys.argv[2:]

def standalone(path):
    energy = None
    forces, stress = [], None
    columns = None
    next_energy = False
    next_stress = False
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if line == "Energy":
            next_energy = True
            continue
        if line.startswith("PlusStress:"):
            next_stress = True
            continue
        if next_energy and line:
            energy = float(line.split()[0]); next_energy = False; continue
        if next_stress and line:
            stress = [float(x) for x in line.split()[:6]]; next_stress = False; continue
        if line.startswith("AtomData:"):
            header = line.split()[1:]
            columns = [header.index(x) for x in ("fx", "fy", "fz")]
        elif columns is not None and line and line[0].isdigit():
            fields = line.split()
            forces.extend(float(fields[i]) for i in columns)
    if energy is None or len(forces) != 24 or stress is None:
        raise SystemExit(f"cannot parse standalone E/F/virial: {path}")
    return energy, forces, stress

def lammps(label):
    log = (root / f"log.{label}").read_text().splitlines()
    energy = atom_energy = stress = None
    for pos, raw in enumerate(log):
        header = raw.split()
        if "Step" not in header or "PotEng" not in header or \
                "c_sum_atom_energy" not in header:
            continue
        row = log[pos + 1].split()
        if len(row) == len(header) and row[header.index("Step")] == "0":
            volume = float(row[header.index("Volume")])
            # LAMMPS metal pressure is bar and its global virial has the same
            # sign convention as PlusStress.  1 eV/A^3 = 1602176.6208 bar.
            energy = float(row[header.index("PotEng")])
            atom_energy = float(row[header.index("c_sum_atom_energy")])
            stress = [
                float(row[header.index(key)]) * volume / 1602176.6208
                for key in ("Pxx", "Pyy", "Pzz", "Pyz", "Pxz", "Pxy")
            ]
    dump = (root / f"dump.{label}").read_text().splitlines()
    forces = None
    for pos, raw in enumerate(dump):
        if raw.startswith("ITEM: ATOMS"):
            header = raw.split()[2:]
            ids, cols = header.index("id"), [header.index(x) for x in ("fx", "fy", "fz")]
            rows = []
            for line in dump[pos + 1:]:
                if line.startswith("ITEM:"): break
                fields = line.split()
                if fields: rows.append((int(fields[ids]), [float(fields[c]) for c in cols]))
            rows.sort(); forces = [x for _, f in rows for x in f]; break
    if energy is None or atom_energy is None or stress is None or \
            forces is None or len(forces) != 24:
        raise SystemExit(f"cannot parse LAMMPS E/F/virial: {label}")
    if abs(energy - atom_energy) > 5e-12:
        raise SystemExit(
            f"{label}: sum(per-atom energy) differs from global energy by "
            f"{abs(energy - atom_energy):.3e}")
    return energy, forces, stress

def dump_forces(label):
    dump = (root / f"dump.{label}").read_text().splitlines()
    for pos, raw in enumerate(dump):
        if raw.startswith("ITEM: ATOMS"):
            header = raw.split()[2:]
            ids = header.index("id")
            cols = [header.index(x) for x in ("fx", "fy", "fz")]
            rows = []
            for line in dump[pos + 1:]:
                if line.startswith("ITEM:"):
                    break
                fields = line.split()
                if fields:
                    rows.append(
                        (int(fields[ids]), [float(fields[c]) for c in cols]))
            rows.sort()
            forces = [x for _, force in rows for x in force]
            if len(forces) == 24:
                return forces
    raise SystemExit(f"cannot parse LAMMPS forces: {label}")

def check(case, engine, reference, candidate):
    e0, f0, s0 = reference; e1, f1, s1 = candidate
    force_error = max(abs(x-y) for x, y in zip(f0, f1))
    stress_error = max(abs(x-y) for x, y in zip(s0, s1))
    energy_error = abs(e0-e1)
    if not all(map(math.isfinite, [energy_error, force_error, stress_error])):
        raise SystemExit(f"{case}/{engine}: non-finite parity error")
    # calc-efs serializes PlusStress with five digits after the decimal,
    # hence an irreducible <= 5e-6 eV text roundoff; this is not a physics
    # tolerance or a tabulation allowance.
    if energy_error > 5e-6 or force_error > 5e-6 or stress_error > 6e-6:
        raise SystemExit(
            f"{case}/{engine}: E={energy_error:.3e} F={force_error:.3e} V={stress_error:.3e}")
    print(f"{case}/{engine}: dE={energy_error:.3e} dF={force_error:.3e} dV={stress_error:.3e}")

for case in cases:
    ref = standalone(root / f"{case}.standalone.cfg")
    for engine in ("cpu", "kk"):
        if (root / f"log.{engine}.{case}").exists():
            check(case, engine, ref, lammps(f"{engine}.{case}"))
        force_only = root / f"dump.{engine}.force-only.{case}"
        if force_only.exists():
            regular_forces = lammps(f"{engine}.{case}")[1]
            force_only_values = dump_forces(f"{engine}.force-only.{case}")
            error = max(abs(x-y) for x, y in zip(
                regular_forces, force_only_values))
            if error > 5e-12:
                raise SystemExit(
                    f"{case}/{engine}/force-only: dF={error:.3e}")
            print(f"{case}/{engine}/force-only: dF={error:.3e}")

zero_ref = standalone(root / "rank0.standalone.cfg")
zero_readout_ref = standalone(root / "readout-zero.standalone.cfg")
zero_error = max(
    abs(zero_ref[0] - zero_readout_ref[0]),
    max(abs(x-y) for x, y in zip(zero_ref[1], zero_readout_ref[1])),
    max(abs(x-y) for x, y in zip(zero_ref[2], zero_readout_ref[2])))
if zero_error > 5e-12:
    raise SystemExit(
        f"zero readout metadata changed standalone output by {zero_error:.3e}")
print("LAMMPS_SH_K_RANK_PARITY: ok")
PY
