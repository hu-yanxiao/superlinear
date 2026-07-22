#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

binary="${MLIP_TEST_BINARY:-./bin/mlp-sus2}"
tmp_dir=".codex_tmp/sh_optimizer_checkpoint_training"
rm -rf "$tmp_dir"
mkdir -p "$tmp_dir"

train_cfg="$tmp_dir/two.cfg"
initial_model="$tmp_dir/initial.mtp"
seed_model="$tmp_dir/seed.mtp"
current_model="$tmp_dir/current.mtp"
best_model="$tmp_dir/current.best.mtp"
trained_model="$tmp_dir/trained.mtp"
trace_file="$tmp_dir/trace.csv"
log_file="$tmp_dir/train.log"

awk 'BEGIN {count=0} {print} /^END_CFG/ {count++; if (count == 2) exit}' \
  example/train.cfg > "$train_cfg"

"$binary" init-sh "$initial_model" \
  --species-count=2 \
  --l-max=2 \
  --k-max=2 \
  --body-order=3 \
  --body-l-max=2,2,2,2 \
  --radial-basis-size=4 \
  --init-params=same \
  --cutoff=5.0 \
  --write-sh-scalar-info \
  --two-layer-gate

"$binary" train "$initial_model" "$train_cfg" \
  --init-params=same \
  --trained-pot-name="$seed_model" \
  --max-iter=1 \
  --skip-preinit \
  --energy-weight=1 \
  --force-weight=0.1 \
  --stress-weight=0 \
  --radial-smooth=0 >"$tmp_dir/seed.log" 2>&1

"$binary" train "$seed_model" "$train_cfg" \
  --trained-pot-name="$trained_model" \
  --curr-pot-name="$current_model" \
  --optimizer-trace-file="$trace_file" \
  --max-iter=3 \
  --skip-preinit \
  --energy-weight=1 \
  --force-weight=0.1 \
  --stress-weight=0 \
  --radial-smooth=0 >"$log_file" 2>&1

for path in "$current_model" "$best_model" "$trained_model" "$trace_file"; do
  if [[ ! -s "$path" ]]; then
    echo "missing non-empty checkpoint test artifact: $path" >&2
    exit 1
  fi
done

grep -q "checkpoint safety enabled" "$log_file"
grep -q "best checkpoint updated kind=evaluated-boundary" "$log_file"
cmp "$current_model" "$best_model"
cmp "$current_model" "$trained_model"

python3 - "$trace_file" "$log_file" <<'PY'
import csv
import math
import pathlib
import re
import sys

trace_path = pathlib.Path(sys.argv[1])
log_path = pathlib.Path(sys.argv[2])
with trace_path.open(newline="") as stream:
    rows = list(csv.DictReader(stream))
if not rows:
    raise SystemExit("checkpoint training trace contains no accepted trial")
losses = [float(row["total_loss"]) for row in rows]
if not all(math.isfinite(value) for value in losses):
    raise SystemExit("checkpoint training trace contains non-finite loss")
for previous, current in zip(losses, losses[1:]):
    tolerance = 1.0e-8 * max(1.0, abs(previous), abs(current))
    if current > previous + tolerance:
        raise SystemExit("accepted checkpoint losses are not monotone")

updates = [
    float(value)
    for value in re.findall(r"best checkpoint updated.*?total_loss=([0-9eE+.-]+)",
                            log_path.read_text())
]
if not updates or not all(math.isfinite(value) for value in updates):
    raise SystemExit("missing finite best-checkpoint update loss")
for previous, current in zip(updates, updates[1:]):
    if current >= previous:
        raise SystemExit("best-checkpoint update loss did not strictly improve")
if updates[-1] > min(losses) + 1.0e-10 * max(1.0, abs(min(losses))):
    raise SystemExit("best checkpoint loss is above the accepted trace minimum")
PY

if command -v sha256sum >/dev/null 2>&1; then
  old_best_sha="$(sha256sum "$best_model" | awk '{print $1}')"
else
  old_best_sha="$(shasum -a 256 "$best_model" | awk '{print $1}')"
fi

"$binary" train "$trained_model" "$train_cfg" \
  --trained-pot-name="$tmp_dir/restarted-trained.mtp" \
  --curr-pot-name="$current_model" \
  --max-iter=1 \
  --skip-preinit \
  --energy-weight=1 \
  --force-weight=0.1 \
  --stress-weight=0 \
  --radial-smooth=0 >"$tmp_dir/restarted.log" 2>&1

archive_matched=0
while IFS= read -r archive; do
  if command -v sha256sum >/dev/null 2>&1; then
    archive_sha="$(sha256sum "$archive" | awk '{print $1}')"
  else
    archive_sha="$(shasum -a 256 "$archive" | awk '{print $1}')"
  fi
  if [[ "$archive_sha" == "$old_best_sha" ]]; then
    archive_matched=1
  fi
done < <(find "$tmp_dir" -maxdepth 1 -type f \
  -name 'current.best.mtp.previous-*' -print)
if [[ "$archive_matched" != "1" ]]; then
  echo "restart did not preserve the previous best checkpoint" >&2
  exit 1
fi

alias_log="$tmp_dir/alias.log"
if "$binary" train "$seed_model" "$train_cfg" \
  --trained-pot-name="$tmp_dir/alias-trained.mtp" \
  --curr-pot-name="$tmp_dir/alias-current.mtp" \
  --best-pot-name="$tmp_dir/./alias-current.mtp" \
  --max-iter=1 \
  --skip-preinit >"$alias_log" 2>&1; then
  echo "lexically aliased current and best paths were accepted" >&2
  exit 1
fi
grep -q -- "--best-pot-name must differ from --curr-pot-name" "$alias_log"

binary_banner="$("$binary" help train 2>&1 || true)"
if command -v mpirun >/dev/null 2>&1 \
  && grep -q "Mode: MPI" <<<"$binary_banner" \
  && [[ "${MLIP_SKIP_MPI_SMOKE:-0}" != "1" ]]; then
  missing_parent="$tmp_dir/missing-parent"
  mpi_failure_log="$tmp_dir/mpi-failure.log"
  set +e
  timeout 120 mpirun -n 2 "$binary" train "$seed_model" "$train_cfg" \
    --trained-pot-name="$tmp_dir/mpi-trained.mtp" \
    --curr-pot-name="$tmp_dir/mpi-current.mtp" \
    --best-pot-name="$missing_parent/best.mtp" \
    --max-iter=1 \
    --skip-preinit \
    --energy-weight=1 \
    --force-weight=0.1 \
    --stress-weight=0 \
    --radial-smooth=0 >"$mpi_failure_log" 2>&1
  mpi_status=$?
  set -e
  if [[ "$mpi_status" == "0" ]]; then
    echo "MPI checkpoint failure injection unexpectedly succeeded" >&2
    exit 1
  fi
  if [[ "$mpi_status" == "124" ]]; then
    echo "MPI checkpoint failure injection hung" >&2
    exit 1
  fi
  grep -q "Checkpoint I/O failed during best accepted model" "$mpi_failure_log"
fi

echo "optimizer checkpoint training check passed"
