#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

bin=${1:-./bin/mlp-sus2}
mpi_run=${MPI_RUN:-mpirun}
tmp_dir=$(mktemp -d "${TMPDIR:-/tmp}/sus2sh-full-linear-l2-XXXXXX")
if [[ ${SUS2_TEST_KEEP_TMP:-0} == 1 ]]; then
  echo "keeping test directory: $tmp_dir"
else
  trap 'rm -rf "$tmp_dir"' EXIT
fi

train_cfg="$tmp_dir/four.cfg"
awk 'BEGIN {n=0} {print} /^END_CFG/ {n++; if (n==4) exit}' \
  example/train.cfg >"$train_cfg"
if [[ $(grep -c '^END_CFG' "$train_cfg") -ne 4 ]]; then
  echo "failed to prepare four-configuration linear-regression set" >&2
  exit 1
fi

initial="$tmp_dir/initial.mtp"
"$bin" init-sh "$initial" \
  --species-count=2 \
  --l-max=2 \
  --k-max=2 \
  --body-order=3 \
  --body-l-max=2,2,2,2 \
  --radial-basis-size=3 \
  --radial-basis-type=RBChebyshev_sss \
  --cutoff=5.0 \
  --sh-factor-pruning=q-total \
  --init-params=same >/dev/null

common_train=(
  --max-iter=1
  --skip-preinit
  --init-params=same
  --energy-weight=1
  --force-weight=0.1
  --stress-weight=0.01
  --radial-smooth=0
)

unregularized="$tmp_dir/unregularized.mtp"
regularized_serial="$tmp_dir/regularized-serial.mtp"
regularized_mpi="$tmp_dir/regularized-mpi.mtp"

"$bin" train "$initial" "$train_cfg" \
  --trained-pot-name="$unregularized" \
  --scalar-head-l2=0 "${common_train[@]}" \
  >"$tmp_dir/unregularized.log" 2>&1
"$bin" train "$initial" "$train_cfg" \
  --trained-pot-name="$regularized_serial" \
  --scalar-head-l2=10 "${common_train[@]}" \
  >"$tmp_dir/regularized-serial.log" 2>&1
"$mpi_run" -n 2 "$bin" train "$initial" "$train_cfg" \
  --trained-pot-name="$regularized_mpi" \
  --scalar-head-l2=10 "${common_train[@]}" \
  >"$tmp_dir/regularized-mpi.log" 2>&1

for log in "$tmp_dir"/*.log; do
  grep -q 'TrainLinear solve done' "$log"
done

# Rescale and initialization are deterministic and identical in these three
# runs.  A large head penalty must therefore visibly shrink the serialized
# scalar-head block; this catches the historical full-TrainLinear path that
# omitted scalar-head L2 from its normal matrix entirely.
python3 - "$unregularized" "$regularized_serial" <<'PY'
import math
import re
import sys

number = r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?"

def head_norm(path):
    text = open(path, encoding="utf-8").read()
    match = re.search(r"moment_coeffs\s*=\s*\{([^}]*)\}", text, re.S)
    if not match:
        raise SystemExit(f"missing moment_coeffs in {path}")
    values = [float(x) for x in re.findall(number, match.group(1))]
    if not values or not all(math.isfinite(x) for x in values):
        raise SystemExit(f"invalid moment_coeffs in {path}")
    return math.sqrt(sum(x*x for x in values))

plain = head_norm(sys.argv[1])
regularized = head_norm(sys.argv[2])
if not regularized < 0.99 * plain:
    raise SystemExit(
        f"full TrainLinear scalar L2 did not shrink the head: "
        f"unregularized={plain:.16e} regularized={regularized:.16e}")
print(f"full_linear_head_norm unregularized={plain:.6e} regularized={regularized:.6e}")
PY

serial_pred="$tmp_dir/regularized-serial.cfg"
mpi_pred="$tmp_dir/regularized-mpi.cfg"
"$bin" calc-efs "$regularized_serial" "$train_cfg" "$serial_pred" >/dev/null
"$bin" calc-efs "$regularized_mpi" "$train_cfg" "$mpi_pred" >/dev/null
python3 - "$serial_pred" "$mpi_pred" <<'PY'
import re
import sys

number = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")
a = [float(x) for x in number.findall(open(sys.argv[1], encoding="utf-8").read())]
b = [float(x) for x in number.findall(open(sys.argv[2], encoding="utf-8").read())]
if not a or len(a) != len(b):
    raise SystemExit("serial/MPI regularized prediction shape differs")
worst = max(abs(x-y) for x, y in zip(a, b))
scale = max([1.0] + [abs(x) for x in a] + [abs(x) for x in b])
if worst > 2.0e-10 * scale:
    raise SystemExit(
        f"serial/MPI full TrainLinear scalar-L2 mismatch: "
        f"worst={worst:.3e} scale={scale:.3e}")
print(f"full_linear_l2_serial_mpi_worst={worst:.3e} scale={scale:.3e}")
PY

echo "full TrainLinear scalar-head L2 check passed"
