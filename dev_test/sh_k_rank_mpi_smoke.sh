#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

bin=${1:-./bin/mlp-sus2}
bin=$(cd "$(dirname "$bin")" && pwd)/$(basename "$bin")
tmp_dir=$(mktemp -d "${TMPDIR:-/tmp}/sus2sh-k-rank-mpi-XXXXXX")
if [[ ${SUS2_TEST_KEEP_TMP:-0} == 1 ]]; then
  echo "keeping test directory: $tmp_dir"
else
  trap 'rm -rf "$tmp_dir"' EXIT
fi

if ! "$bin" help train 2>&1 | grep -q "Mode: MPI"; then
  echo "K-rank MPI smoke requires an MPI binary" >&2
  exit 1
fi
if ! command -v mpirun >/dev/null 2>&1; then
  echo "K-rank MPI smoke requires mpirun" >&2
  exit 1
fi

# Four configurations ensure that both ranks own real work.  Unlike calc-efs,
# the train command shards the input on rank 0 and reduces the objective and
# gradient, so this exercises the actual MPI mathematics instead of racing two
# ranks on one output file.
cfg="$tmp_dir/four.cfg"
awk 'BEGIN {n=0} {print} /^END_CFG/ {n++; if (n==4) exit}' example/train.cfg >"$cfg"
if [[ $(grep -c '^END_CFG' "$cfg") -ne 4 ]]; then
  echo "failed to prepare four-configuration MPI test set" >&2
  exit 1
fi

common=(
  --species-count=2
  --l-max=2
  --k-max=2
  --body-order=3
  --body-l-max=2,2,2,2
  --radial-basis-size=3
  --radial-basis-type=RBChebyshev_sss
  --cutoff=5.0
  --sh-factor-pruning=q-total
  --two-layer-gate
  --two-layer-gate-factor-pruning=q-total
  --two-layer-gate-mode=mu-scalar-full
  --two-layer-gate-body-order=3
  --two-layer-gate-shared-radial
  --init-params=same
)

train_common=(
  # Start from a materialized checkpoint and take one real BFGS update.  The
  # structural init path also runs Rescale+TrainLinear; on this four-CFG smoke
  # set its zero-ridge normal equation is deliberately underdetermined and can
  # amplify harmless MPI reduction-order roundoff by many orders of magnitude.
  --max-iter=2
  --skip-preinit
  --do-samp
  --init-params=same
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

for mode in rank0 rank1 rank2 gate1_outer1 gate2_outer1 \
            gate1_outer2_body_default; do
  init="$tmp_dir/$mode-init.mtp"
  extra=()
  init_common=("${common[@]}")
  if [[ $mode == rank1 ]]; then
    extra=(--outer-k-rank=1)
  elif [[ $mode == rank2 ]]; then
    extra=(--outer-k-rank=2)
  elif [[ $mode == gate1_outer1 ]]; then
    extra=(--outer-k-rank=1 --two-layer-gate-k-rank=1)
  elif [[ $mode == gate2_outer1 ]]; then
    extra=(--outer-k-rank=1 --two-layer-gate-k-rank=2)
  elif [[ $mode == gate1_outer2_body_default ]]; then
    extra=(--outer-k-rank=2 --two-layer-gate-k-rank=1)
    init_common=()
    for option in "${common[@]}"; do
      if [[ $option != --two-layer-gate-mode=* ]]; then
        init_common+=("$option")
      fi
    done
  fi
  "$bin" init-sh "$init" "${init_common[@]}" "${extra[@]}" >/dev/null
  materialized="$tmp_dir/$mode-materialized.mtp"
  "$bin" roundtrip-model-dev "$init" "$materialized" >/dev/null

  serial_model="$tmp_dir/$mode-serial.mtp"
  mpi_model="$tmp_dir/$mode-mpi2.mtp"
  "$bin" train "$materialized" "$cfg" "${train_common[@]}" \
    --trained-pot-name="$serial_model" >"$tmp_dir/$mode-serial.log" 2>&1
  timeout 300 mpirun -n 2 "$bin" train "$materialized" "$cfg" \
    "${train_common[@]}" --trained-pot-name="$mpi_model" \
    >"$tmp_dir/$mode-mpi2.log" 2>&1
  test -s "$serial_model"
  test -s "$mpi_model"

  serial_pred="$tmp_dir/$mode-serial.cfg"
  mpi_pred="$tmp_dir/$mode-mpi2.cfg"
  "$bin" calc-efs "$serial_model" "$cfg" "$serial_pred" >/dev/null
  "$bin" calc-efs "$mpi_model" "$cfg" "$mpi_pred" >/dev/null

  python3 - "$mode" "$serial_pred" "$mpi_pred" <<'PY'
import re
import sys

mode, serial_path, mpi_path = sys.argv[1:]
number = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")

def predicted_values(path):
    values = {"energy": [], "force": [], "stress": []}
    in_atoms = False
    take_energy = False
    take_stress = False
    for raw in open(path, encoding="utf-8"):
        line = raw.strip()
        if line.startswith("AtomData:"):
            in_atoms = True
            continue
        if line == "Energy":
            in_atoms = False
            take_energy = True
            continue
        if line.startswith("PlusStress:"):
            take_stress = True
            continue
        if in_atoms and line:
            fields = line.split()
            if fields[0].lstrip("+-").isdigit() and len(fields) >= 8:
                values["force"].extend(float(x) for x in fields[5:8])
        elif take_energy and line:
            values["energy"].append(float(number.search(line).group()))
            take_energy = False
        elif take_stress and line:
            row = [float(x) for x in number.findall(line)]
            values["stress"].extend(row[:6])
            take_stress = False
    return values

a = predicted_values(serial_path)
b = predicted_values(mpi_path)
tolerances = {"energy": 1.0e-7, "force": 1.0e-8, "stress": 1.0e-5}
parts = []
for group in ("energy", "force", "stress"):
    left, right = a[group], b[group]
    if not left or len(left) != len(right):
        raise SystemExit(f"{mode}: serial/MPI {group} shape mismatch")
    worst = max(abs(x-y) for x, y in zip(left, right))
    if worst > tolerances[group]:
        raise SystemExit(
            f"{mode}: serial/MPI trained-model {group} mismatch "
            f"worst={worst:.3e} tolerance={tolerances[group]:.3e}")
    parts.append(f"{group}={worst:.3e}")
print(f"k_rank_mpi_training_parity mode={mode} " + " ".join(parts))
PY
done

echo "K-rank MPI training parity passed for rank0, direct rank1, rank2, equal rank, both mixed-rank orders, and default body gate"
