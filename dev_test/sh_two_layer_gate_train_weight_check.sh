#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

tmp_dir=".local_tmp/sh_two_layer_gate_train_weight"
mkdir -p "$tmp_dir"

init_model="$tmp_dir/gated_init.sus2"
trained_model="$tmp_dir/gated_trained.sus2"
train="$tmp_dir/one.cfg"
rm -f "$init_model" "$trained_model" "$train"
awk '{print} /^END_CFG/{exit}' example/train.cfg > "$train"

./bin/mlp-sus2 init-sh "$init_model" \
  --species-count=2 \
  --l-max=2 \
  --k-max=2 \
  --body-order=4 \
  --body-l-max=2,2,2,2 \
  --radial-basis-size=4 \
  --cutoff=5.0 \
  --write-sh-scalar-info \
  --two-layer-gate

./bin/mlp-sus2 train "$init_model" "$train" \
  --trained-pot-name="$trained_model" \
  --max-iter=3 \
  --skip-preinit \
  --energy-weight=1 \
  --force-weight=1 \
  --stress-weight=0 >/dev/null

python3 - "$trained_model" <<'PY'
import re
import sys

text = open(sys.argv[1]).read()
match = re.search(r"two_layer_gate_weights = \{([^}]*)\}", text)
if not match:
    raise SystemExit("missing two_layer_gate_weights")
weights = [float(x.strip()) for x in match.group(1).split(",") if x.strip()]
if not weights:
    raise SystemExit("empty two_layer_gate_weights")
max_move = max(abs(x - 1.0) for x in weights)
if max_move <= 1.0e-12:
    raise SystemExit("two-layer gate weights did not move from one")
PY
