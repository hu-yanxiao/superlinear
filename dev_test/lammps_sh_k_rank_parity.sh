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
  --species-count=2 --l-max=4 --k-max=4 --body-order=6
  --body-l-max=4,4,4,2,1 --radial-basis-size=4
  --radial-basis-type=RBChebyshev_sss --cutoff=5.0
  --sh-factor-pruning=q-total --two-layer-gate
  --two-layer-gate-factor-pruning=q-total
  --two-layer-gate-mode=mu-scalar-full --two-layer-gate-body-order=5
  --two-layer-gate-site-mode=neighbor --two-layer-gate-shared-radial
)

prepare_model() {
  local name="$1"; shift
  local init="$tmp_dir/$name.init.mtp"
  local materialized="$tmp_dir/$name.materialized.mtp"
  local model="$tmp_dir/$name.lmp.mtp"
  "$MLP_BIN" init-sh "$init" "${common_init[@]}" "$@" >/dev/null || return
  "$MLP_BIN" roundtrip-model-dev "$init" "$materialized" >/dev/null || return
  python3 - "$materialized" "$model" "$name" <<'PY' || return
import re
import sys

source, target, name = sys.argv[1:]
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
    kmax = 4
    rank = 4
    expected = (kmax + 1) * rank * kmax
    if len(weights) != expected:
        raise SystemExit(f"{name}: unexpected rank-K weight count {len(weights)}")
    identity = []
    for l in range(kmax + 1):
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
if "readout-rejected" in name:
    readout = vector("two_layer_gate_readout_coeffs")
    readout[0] = 0.037
    replace_vector("two_layer_gate_readout_coeffs", readout)

text = text.replace("radial_basis_type = RBChebyshev_sss",
                    "radial_basis_type = RBChebyshev_sss_lmp", 1)
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
pair_style $style $model $style_args tabstep 0.0005
pair_coeff * *
neighbor 1.0 bin
neigh_modify every 1 delay 0 check yes
# The pair style accumulates only global virial[].  The stress/atom compute
# requests per-atom vatom, which this pair style does not fill, so it is not
# a valid virial source here.  In metal units P*V/1602176.6208 is eV.
thermo_style custom step pe pxx pyy pzz pyz pxz pxy vol
thermo_modify format float %.17g
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

cases=(rank0 outer-rank1-direct outer-rank2 outer-rankK-identity gate1-outer1 gate2-outer2 outer2-gate1 outer1-gate2 default-mode-outer2-gate1)
models=()
model="$(prepare_model rank0)" || exit 1; models+=("$model")
model="$(prepare_model outer-rank1-direct --outer-k-rank=1)" || exit 1; models+=("$model")
model="$(prepare_model outer-rank2 --outer-k-rank=2)" || exit 1; models+=("$model")
model="$(prepare_model outer-rankK-identity --outer-k-rank=4)" || exit 1; models+=("$model")
model="$(prepare_model gate1-outer1 --outer-k-rank=1 --two-layer-gate-k-rank=1)" || exit 1; models+=("$model")
model="$(prepare_model gate2-outer2 --outer-k-rank=2 --two-layer-gate-k-rank=2)" || exit 1; models+=("$model")
model="$(prepare_model outer2-gate1 --outer-k-rank=2 --two-layer-gate-k-rank=1)" || exit 1; models+=("$model")
model="$(prepare_model outer1-gate2 --outer-k-rank=1 --two-layer-gate-k-rank=2)" || exit 1; models+=("$model")
model="$(prepare_model default-mode-outer2-gate1 --outer-k-rank=2 --two-layer-gate-k-rank=1 --two-layer-gate-mode=mu-body-linear-combo)" || exit 1; models+=("$model")
readout_model="$(prepare_model readout-rejected --two-layer-gate-readout)" || exit 1
decoupled_model="$(prepare_model decoupled-scaling-rejected \
  --outer-k-rank=2 --two-layer-gate-k-rank=1 \
  --two-layer-gate-decoupled-scaling)" || exit 1

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

if [[ -n "$CPU_LAMMPS_BIN" ]]; then
  assert_model_rejected "$CPU_LAMMPS_BIN" "$CPU_MPI_RANKS" sus2mtp \
    "$readout_model" cpu.readout-rejected \
    "do not support two_layer_gate_readout_enabled"
  assert_model_rejected "$CPU_LAMMPS_BIN" "$CPU_MPI_RANKS" sus2mtp \
    "$decoupled_model" cpu.decoupled-scaling-rejected \
    "do not support two_layer_gate_scaling_mode=decoupled"
fi
if [[ -n "$KOKKOS_LAMMPS_BIN" ]]; then
  assert_model_rejected "$KOKKOS_LAMMPS_BIN" "$KOKKOS_MPI_RANKS" sus2mtp/kk \
    "$readout_model" kk.readout-rejected \
    "do not support two_layer_gate_readout_enabled"
  assert_model_rejected "$KOKKOS_LAMMPS_BIN" "$KOKKOS_MPI_RANKS" sus2mtp/kk \
    "$decoupled_model" kk.decoupled-scaling-rejected \
    "do not support two_layer_gate_scaling_mode=decoupled"
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
    energy = stress = None
    for pos, raw in enumerate(log):
        header = raw.split()
        if header[:2] != ["Step", "PotEng"] or len(header) != 9:
            continue
        row = log[pos + 1].split()
        if len(row) == 9 and row[0] == "0":
            volume = float(row[8])
            # LAMMPS metal pressure is bar and its global virial has the same
            # sign convention as PlusStress.  1 eV/A^3 = 1602176.6208 bar.
            energy = float(row[1])
            stress = [float(x) * volume / 1602176.6208 for x in row[2:8]]
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
    if energy is None or stress is None or forces is None or len(forces) != 24:
        raise SystemExit(f"cannot parse LAMMPS E/F/virial: {label}")
    return energy, forces, stress

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
print("LAMMPS_SH_K_RANK_PARITY: ok")
PY
