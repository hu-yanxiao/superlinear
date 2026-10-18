#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

tmp_dir=".local_tmp/sh_initial_linear_warmup"
mkdir -p "$tmp_dir"

init_model="$tmp_dir/init.sus2"
trained_model="$tmp_dir/trained.sus2"
continued_model="$tmp_dir/continued.sus2"
train="$tmp_dir/one.cfg"
init_log="$tmp_dir/init_train.log"
continued_log="$tmp_dir/continued_train.log"
rm -f "$init_model" "$trained_model" "$continued_model" \
  "$train" "$init_log" "$continued_log"
awk '{print} /^END_CFG/{exit}' example/train.cfg > "$train"

./bin/mlp-sus2 init-sh "$init_model" \
  --species-count=2 \
  --l-max=2 \
  --k-max=2 \
  --body-order=4 \
  --body-l-max=2,2,2,2 \
  --radial-basis-size=4 \
  --cutoff=5.0 \
  --write-sh-scalar-info >/dev/null

./bin/mlp-sus2 train "$init_model" "$train" \
  --trained-pot-name="$trained_model" \
  --max-iter=1 \
  --skip-preinit \
  --energy-weight=1 \
  --force-weight=0 \
  --stress-weight=0 >"$init_log"

grep -q "Initial untrained model warmup Rescale start" "$init_log"
grep -q "TrainLinear\\[initial untrained model warmup\\]" "$init_log"
grep -q "Initial untrained model warmup rescale+linear solve done" "$init_log"
grep -q "do-lin disabled" "$init_log"

./bin/mlp-sus2 train "$trained_model" "$train" \
  --trained-pot-name="$continued_model" \
  --max-iter=1 \
  --skip-preinit \
  --energy-weight=1 \
  --force-weight=0 \
  --stress-weight=0 >"$continued_log"

if grep -q "Initial untrained model warmup" "$continued_log"; then
  echo "complete trained model unexpectedly ran implicit initial warmup" >&2
  exit 1
fi
if grep -q "TrainLinear\\[initial untrained model warmup\\]" "$continued_log"; then
  echo "complete trained model unexpectedly ran implicit TrainLinear warmup" >&2
  exit 1
fi
grep -q "do-lin disabled" "$continued_log"
