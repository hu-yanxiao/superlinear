#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

tmp_dir=".codex_tmp/sh_two_layer_gate_mu_scalar_full_loss_gradient"
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
  --two-layer-gate \
  --two-layer-gate-mode=mu-scalar-full

python3 - "$init_model" "$nonzero_model" <<'PY'
import re
import sys

source, target = sys.argv[1], sys.argv[2]
text = open(source).read()
mode = re.search(r"two_layer_gate_mode\s*=\s*(\S+)", text)
if not mode or mode.group(1) != "mu-scalar-full":
    raise SystemExit("expected two_layer_gate_mode = mu-scalar-full")
match = re.search(r"two_layer_gate_weights = \{([^}]*)\}", text)
if not match:
    raise SystemExit("missing two_layer_gate_weights")
weights = [float(x.strip()) for x in match.group(1).split(",") if x.strip()]
if len(weights) < 6:
    raise SystemExit("expected full mu x scalar gate weights")
weights[0] = 0.35
weights[1] = -0.20
weights[2] = 0.15
weights[5] = -0.10
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
