#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
bin="${1:-$repo_root/bin/mlp-sus2}"
tmp_dir="$(mktemp -d "${TMPDIR:-/tmp}/adaptive-radial-balanced-init-XXXXXX")"
one_cfg="$tmp_dir/one.cfg"
trained="$tmp_dir/balanced.sus2"
log="$tmp_dir/train.log"

awk 'BEGIN {n=0} {print} /^END_CFG/ {n++; if (n==1) exit}' \
  "$repo_root/example/train.cfg" >"$one_cfg"

"$bin" train "$one_cfg" \
  --init-sh \
  --species-count=2 \
  --l-max=2 \
  --k-max=2 \
  --body-order=4 \
  --body-l-max=2,2,2,2 \
  --radial-basis-size=4 \
  --cutoff=5.2 \
  --radial-basis-type=RBChebyshev_sss \
  --scaling-map=K \
  --superlinear-mode=radial-coefficient-gate \
  --outer-k-rank=2 \
  --two-layer-gate \
  --two-layer-gate-mode=mu-scalar-full \
  --two-layer-gate-body-order=4 \
  --two-layer-gate-k-rank=2 \
  --write-sh-scalar-info \
  --trained-pot-name="$trained" \
  --max-iter=0 \
  --init-params=same \
  --energy-weight=1 \
  --force-weight=0.1 \
  --stress-weight=0.01 \
  --adaptive-radial-initialization=balanced \
  --adaptive-radial-init-relative-rms=0.03 \
  >"$log" 2>&1

python3 - "$trained" "$log" <<'PY'
import math
import re
import sys

model_path, log_path = sys.argv[1], sys.argv[2]
model = open(model_path, encoding="utf-8").read()
log = open(log_path, encoding="utf-8").read()
num = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"


def vector(name):
    match = re.search(rf"(?ms)^\s*{re.escape(name)}\s*=\s*\{{([^}}]*)\}}", model)
    if not match:
        raise SystemExit(f"missing {name}")
    return [float(x) for x in re.findall(num, match.group(1))]


for key in (
    "adaptive_radial_weights",
    "adaptive_radial_amplitudes",
    "adaptive_radial_weight_optimizer_scales",
    "adaptive_radial_amplitude_optimizer_scales",
):
    values = vector(key)
    if not values:
        raise SystemExit(f"empty {key}")
    if not all(math.isfinite(x) for x in values):
        raise SystemExit(f"non-finite {key}")
    if max(abs(x) for x in values) <= 0.0:
        raise SystemExit(f"zero {key}")

if "[adaptive-radial-init]" not in log:
    raise SystemExit("missing adaptive radial initialization statistics log")
if "no Rescale or 75-step pretraining" not in log:
    raise SystemExit("balanced initialization did not skip legacy pretraining")
if "gate_x_l2_regularization=0" not in log:
    raise SystemExit("adaptive radial default gate_x_l2 was not zero")
PY

if "$bin" train "$one_cfg" \
  --init-sh \
  --species-count=2 \
  --l-max=2 \
  --k-max=2 \
  --body-order=4 \
  --body-l-max=2,2,2,2 \
  --radial-basis-size=4 \
  --cutoff=5.2 \
  --radial-basis-type=RBChebyshev_sss \
  --scaling-map=K \
  --superlinear-mode=radial-coefficient-gate \
  --outer-k-rank=2 \
  --two-layer-gate \
  --two-layer-gate-mode=mu-scalar-full \
  --two-layer-gate-body-order=4 \
  --two-layer-gate-k-rank=2 \
  --trained-pot-name="$tmp_dir/invalid-gate-x.sus2" \
  --max-iter=0 \
  --init-params=same \
  --adaptive-radial-initialization=balanced \
  --gate-x-l2=1e-4 \
  >"$tmp_dir/invalid-gate-x.log" 2>&1; then
  echo "adaptive radial accepted nonzero --gate-x-l2" >&2
  exit 1
fi

grep -q "gate-x-l2 is not valid for adaptive radial coefficient models" \
  "$tmp_dir/invalid-gate-x.log"

echo "PASS adaptive radial balanced initialization smoke: $tmp_dir"
