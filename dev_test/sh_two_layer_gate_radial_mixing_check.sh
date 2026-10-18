#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

tmp_dir=".local_tmp/sh_two_layer_gate_radial_mixing"
mkdir -p "$tmp_dir"

init_model="$tmp_dir/init.sus2"
roundtrip_model="$tmp_dir/roundtrip.sus2"
legacy_layout_model="$tmp_dir/legacy-layout.sus2"
nonzero_model="$tmp_dir/nonzero.sus2"
train="$tmp_dir/one.cfg"

awk '{print} /^END_CFG/{exit}' example/train.cfg > "$train"

./bin/mlp-sus2 init-sh "$init_model" \
  --species-count=2 \
  --l-max=2 \
  --k-max=3 \
  --body-order=4 \
  --body-l-max=2,2,2,2 \
  --radial-basis-size=4 \
  --cutoff=5.0 \
  --write-sh-scalar-info \
  --two-layer-gate \
  --two-layer-gate-mode=mu-scalar-full \
  --two-layer-gate-body-order=3 \
  --two-layer-gate-k-rank=2 \
  --outer-k-rank=2 \
  --two-layer-gate-radial-mixing=full >/dev/null

python3 - "$init_model" <<'PY'
import re
import sys

text = open(sys.argv[1]).read()
if "two_layer_gate_radial_mixing = full" not in text:
    raise SystemExit("missing radial mixing mode marker")
if "two_layer_gate_radial_mixing_layout = direct-k" not in text:
    raise SystemExit("missing direct-K radial mixing layout marker")
match = re.search(r"two_layer_gate_radial_mixing_coeff_count\s*=\s*(\d+)", text)
if not match:
    raise SystemExit("missing radial mixing coefficient count")
count = int(match.group(1))
expected = (2 + 1) * 2 * 3 * 4
if count != expected:
    raise SystemExit(f"unexpected radial mixing coefficient count: {count} != {expected}")
PY

./bin/mlp-sus2 roundtrip-model-dev "$init_model" "$roundtrip_model" >/dev/null
grep -q "two_layer_gate_radial_mixing = full" "$roundtrip_model"
grep -q "two_layer_gate_radial_mixing_layout = direct-k" "$roundtrip_model"
grep -q "two_layer_gate_radial_mixing_coeff_count = 72" "$roundtrip_model"

python3 - "$init_model" "$legacy_layout_model" <<'PY'
import sys

source, target = sys.argv[1:]
lines = open(source, encoding="utf-8").readlines()
lines = [
    line for line in lines
    if not line.startswith("two_layer_gate_radial_mixing_layout =")
]
open(target, "w", encoding="utf-8").writelines(lines)
PY
if ./bin/mlp-sus2 roundtrip-model-dev \
  "$legacy_layout_model" "$tmp_dir/legacy-accepted.sus2" \
  >"$tmp_dir/legacy.out" 2>"$tmp_dir/legacy.err"; then
  echo "unmarked legacy radial mixer was accepted" >&2
  exit 1
fi
grep -q "legacy gate-rank radial mixer models are not compatible" \
  "$tmp_dir/legacy.err"

python3 - "$init_model" "$nonzero_model" <<'PY'
import re
import sys

source, target = sys.argv[1], sys.argv[2]
text = open(source).read()

def rewrite_vector(name, callback):
    global text
    match = re.search(rf"{name} = \{{([^}}]*)\}}", text)
    if not match:
        raise SystemExit(f"missing {name}")
    values = [float(x.strip()) for x in match.group(1).split(",") if x.strip()]
    for i, value in enumerate(values):
        values[i] = callback(i, value)
    replacement = f"{name} = {{" + ", ".join(f"{x:.15e}" for x in values) + "}"
    text = text[:match.start()] + replacement + text[match.end():]

rewrite_vector(
    "two_layer_gate_radial_mixing_coeffs",
    lambda i, _: 0.03 * ((i % 7) - 3) / 3.0,
)
rewrite_vector(
    "two_layer_gate_k_rank_weights",
    lambda i, _: 0.2 * ((i % 5) - 2) / 2.0,
)
open(target, "w").write(text)
PY

./bin/mlp-sus2 check-efs-fd-dev "$nonzero_model" "$train" \
  --max-configs=1 \
  --max-atoms=4 \
  --displacement=1.0e-4 \
  --abs-tolerance=1.0e-4 \
  --rel-tolerance=2.0e-3 >/dev/null

./bin/mlp-sus2 check-loss-gradient-dev "$nonzero_model" "$train" \
  --max-configs=1 \
  --energy-weight=1 \
  --force-weight=1 \
  --stress-weight=0 \
  --gate-x-l2=0 \
  --displacement=1.0e-4 \
  --abs-tolerance=1.0e-4 \
  --rel-tolerance=1.0e-2 >/dev/null

layout_line=$(./bin/mlp-sus2 check-loss-gradient-dev "$init_model" "$train" \
  --max-configs=1 \
  --energy-weight=1 \
  --force-weight=1 \
  --stress-weight=0 \
  --gate-x-l2=0 \
  --displacement=1.0e-3 \
  --abs-tolerance=1.0e99 \
  --rel-tolerance=1.0e99 2>/dev/null | tail -1)
mix_begin=$(python3 - "$layout_line" <<'PY'
import re
import sys

line = sys.argv[1]
match = re.search(r"gate_body_mix_end=(\d+)", line)
if not match:
    raise SystemExit("missing gate_body_mix_end in gradient-check layout")
print(match.group(1))
PY
)
mix_end=$(python3 - "$layout_line" <<'PY'
import re
import sys

line = sys.argv[1]
match = re.search(r"gate_edge_l1_begin=(\d+)", line)
if not match:
    raise SystemExit("missing gate_edge_l1_begin in gradient-check layout")
print(match.group(1))
PY
)
./bin/mlp-sus2 check-loss-gradient-dev "$init_model" "$train" \
  --max-configs=1 \
  --energy-weight=1 \
  --force-weight=1 \
  --stress-weight=0 \
  --gate-x-l2=0 \
  --displacement=1.0e-3 \
  --abs-tolerance=1.0e-6 \
  --rel-tolerance=2.0e-3 \
  --coeff-start="$mix_begin" \
  --coeff-end="$mix_end" >/dev/null

echo "two-layer gate radial mixing checks passed"
