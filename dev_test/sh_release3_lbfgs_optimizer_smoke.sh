#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

tmp_dir=".codex_tmp/sh_release3_lbfgs_optimizer"
mkdir -p "$tmp_dir"

initial_model="$tmp_dir/initial.mtp"
trained_model="$tmp_dir/trained.mtp"
current_model="$tmp_dir/current.mtp"
train_cfg="$tmp_dir/two.cfg"
trace_file="$tmp_dir/lbfgs.csv"
log_file="$tmp_dir/train.log"
mpi2_model="$tmp_dir/trained_mpi2.mtp"
mpi2_trace="$tmp_dir/lbfgs_mpi2.csv"
mpi2_log="$tmp_dir/train_mpi2.log"
mpi3_model="$tmp_dir/trained_mpi3.mtp"
mpi3_trace="$tmp_dir/lbfgs_mpi3.csv"
mpi3_log="$tmp_dir/train_mpi3.log"
compare_serial_model="$tmp_dir/trained_compare_serial.mtp"
compare_serial_trace="$tmp_dir/lbfgs_compare_serial.csv"
compare_serial_log="$tmp_dir/train_compare_serial.log"
rm -f "$initial_model" "$trained_model" "$current_model" \
  "$train_cfg" "$trace_file" "$log_file" \
  "$mpi2_model" "$mpi2_trace" "$mpi2_log" \
  "$mpi3_model" "$mpi3_trace" "$mpi3_log" \
  "$compare_serial_model" "$compare_serial_trace" "$compare_serial_log"
awk 'BEGIN {count=0} {print} /^END_CFG/ {count++; if (count == 2) exit}' \
  example/train.cfg > "$train_cfg"

./bin/mlp-sus2 init-sh "$initial_model" \
  --species-count=2 \
  --l-max=2 \
  --k-max=2 \
  --body-order=4 \
  --body-l-max=2,2,2,2 \
  --radial-basis-size=4 \
  --cutoff=5.0 \
  --write-sh-scalar-info \
  --two-layer-gate

./bin/mlp-sus2 train "$initial_model" "$train_cfg" \
  --optimizer=lbfgs \
  --optimizer-line-search=armijo-value \
  --optimizer-trace-file="$trace_file" \
  --lbfgs-history=4 \
  --trained-pot-name="$trained_model" \
  --curr-pot-name="$current_model" \
  --max-iter=3 \
  --skip-preinit \
  --energy-weight=1 \
  --force-weight=0.1 \
  --stress-weight=0 \
  --radial-smooth=0 >"$log_file" 2>&1

python3 - "$trace_file" "$trained_model" "$log_file" <<'PY'
import csv
import math
import pathlib
import sys

trace_path, model_path, log_path = map(pathlib.Path, sys.argv[1:])
if not model_path.is_file() or model_path.stat().st_size == 0:
    raise SystemExit("missing trained L-BFGS model")

with trace_path.open(newline="") as stream:
    rows = list(csv.DictReader(stream))
if not rows:
    raise SystemExit("L-BFGS trace has no accepted steps")

required = {
    "total_loss",
    "lbfgs_history",
    "lbfgs_step",
    "lbfgs_gamma",
    "lbfgs_curvature",
    "lbfgs_relative_curvature",
}
if not required.issubset(rows[0]):
    raise SystemExit("L-BFGS trace is missing diagnostics")

losses = [float(row["total_loss"]) for row in rows]
if not all(math.isfinite(value) for value in losses):
    raise SystemExit("L-BFGS trace contains a non-finite loss")
if any(curr > prev + 1.0e-10 * max(1.0, abs(prev))
       for prev, curr in zip(losses, losses[1:])):
    raise SystemExit("accepted L-BFGS losses are not monotone")
if any(int(row["lbfgs_history"]) > 4 for row in rows):
    raise SystemExit("L-BFGS history exceeded the requested limit")

log_text = log_path.read_text()
if "Optimizer: lbfgs" not in log_text or "L-BFGS summary" not in log_text:
    raise SystemExit("training log does not identify the L-BFGS path")
PY

binary_banner="$(./bin/mlp-sus2 help train 2>&1 || true)"
binary_is_mpi=0
if grep -q "Mode: MPI" <<<"$binary_banner"; then
  binary_is_mpi=1
fi
if [[ "${MLIP_REQUIRE_MPI_SMOKE:-0}" == "1" && "$binary_is_mpi" != "1" ]]; then
  echo "MLIP_REQUIRE_MPI_SMOKE=1 but ./bin/mlp-sus2 is not an MPI build" >&2
  exit 1
fi

if command -v mpirun >/dev/null 2>&1 \
  && [[ "$binary_is_mpi" == "1" ]] \
  && [[ "${MLIP_SKIP_MPI_SMOKE:-0}" != "1" ]]; then
  # The first serial run turns the untrained initializer into one common,
  # fully initialized start model. All consistency runs below read that exact
  # file, so MPI-specific warmup/rescaling cannot change their starting point.
  ./bin/mlp-sus2 train "$trained_model" "$train_cfg" \
    --optimizer=lbfgs \
    --optimizer-trace-file="$compare_serial_trace" \
    --lbfgs-history=4 \
    --trained-pot-name="$compare_serial_model" \
    --max-iter=2 \
    --skip-preinit \
    --energy-weight=1 \
    --force-weight=0.1 \
    --stress-weight=0 \
    --radial-smooth=0 >"$compare_serial_log" 2>&1

  mpirun -n 2 ./bin/mlp-sus2 train "$trained_model" "$train_cfg" \
    --optimizer=lbfgs \
    --optimizer-trace-file="$mpi2_trace" \
    --lbfgs-history=4 \
    --trained-pot-name="$mpi2_model" \
    --max-iter=2 \
    --skip-preinit \
    --energy-weight=1 \
    --force-weight=0.1 \
    --stress-weight=0 \
    --radial-smooth=0 >"$mpi2_log" 2>&1

  # Three ranks for two configurations exercises the inactive-rank train
  # communicator path that must not enter active-rank collectives.
  timeout 180 mpirun -n 3 ./bin/mlp-sus2 train "$trained_model" "$train_cfg" \
    --optimizer=lbfgs \
    --optimizer-trace-file="$mpi3_trace" \
    --lbfgs-history=4 \
    --trained-pot-name="$mpi3_model" \
    --max-iter=2 \
    --skip-preinit \
    --energy-weight=1 \
    --force-weight=0.1 \
    --stress-weight=0 \
    --radial-smooth=0 >"$mpi3_log" 2>&1

  python3 - "$compare_serial_trace" "$mpi2_trace" "$mpi3_trace" <<'PY'
import csv
import math
import pathlib
import sys

def losses(path):
    with pathlib.Path(path).open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise SystemExit(f"no accepted L-BFGS steps in {path}")
    values = [float(row["total_loss"]) for row in rows]
    if not all(math.isfinite(value) for value in values):
        raise SystemExit(f"non-finite L-BFGS loss in {path}")
    return values

serial = losses(sys.argv[1])
mpi2 = losses(sys.argv[2])
mpi3 = losses(sys.argv[3])
for label, candidate in (("mpi2", mpi2), ("mpi3", mpi3)):
    common = min(len(serial), len(candidate))
    for index in range(common):
        scale = max(1.0, abs(serial[index]), abs(candidate[index]))
        if abs(serial[index] - candidate[index]) > 1.0e-10 * scale:
            raise SystemExit(
                f"{label} loss differs from serial at accepted step {index}: "
                f"{candidate[index]} vs {serial[index]}"
            )
PY
fi
