#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

tmp_dir=".codex_tmp/sh_two_layer_gate_loss_gradient"
mkdir -p "$tmp_dir"

init_model="$tmp_dir/gated_init.mtp"
nonzero_model="$tmp_dir/gated_nonzero.mtp"
train="$tmp_dir/one.cfg"
rm -f "$init_model" "$nonzero_model" "$train"
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

python3 - "$init_model" "$nonzero_model" <<'PY'
import re
import sys

source, target = sys.argv[1], sys.argv[2]
text = open(source).read()
match = re.search(r"two_layer_gate_weights = \{([^}]*)\}", text)
if not match:
    raise SystemExit("missing two_layer_gate_weights")
weights = [float(x.strip()) for x in match.group(1).split(",") if x.strip()]
if len(weights) < 3:
    raise SystemExit("expected at least three gate weights")
weights[0] = 0.35
weights[1] = -0.20
weights[2] = 0.15
replacement = "two_layer_gate_weights = {" + ", ".join(f"{x:.15e}" for x in weights) + "}"
open(target, "w").write(text[:match.start()] + replacement + text[match.end():])
PY

./bin/mlp-sus2 check-loss-gradient-dev "$nonzero_model" "$train" \
  --max-configs=1 \
  --energy-weight=1 \
  --force-weight=1 \
  --stress-weight=0 \
  --gate-x-l2=0 \
  --displacement=1.0e-3 \
  --abs-tolerance=1.0e-5 \
  --rel-tolerance=1.0e-4
