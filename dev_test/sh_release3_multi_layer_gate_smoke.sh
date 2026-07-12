#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

tmp_dir=".codex_tmp/sh_release3_multi_layer_gate"
mkdir -p "$tmp_dir"

model="$tmp_dir/release3_multi_gate.mtp"
probe_model="$tmp_dir/release3_multi_gate_probe.mtp"
train="$tmp_dir/one.cfg"
pred="$tmp_dir/one_out.cfg"
rm -f "$model" "$probe_model" "$train" "$pred"
awk '{print} /^END_CFG/{exit}' example/train.cfg > "$train"

./bin/mlp-sus2 init-sh "$model" \
  --species-count=2 \
  --l-max=2 \
  --k-max=2 \
  --body-order=4 \
  --body-l-max=2,2,2,2 \
  --radial-basis-size=4 \
  --cutoff=5.0 \
  --write-sh-scalar-info \
  --two-layer-gate \
  --multi-layer-gate \
  --second-layer-gate-body-order=3

grep -q "two_layer_gate_body_order_max = 4" "$model"
grep -q "multi_layer_gate_enabled = true" "$model"
grep -q "second_layer_gate_body_order_max = 3" "$model"
grep -q "second_layer_gate_mode = mu-body-linear-combo" "$model"

python3 - "$model" <<'PY'
import re
import sys

path = sys.argv[1]
text = open(path).read()

info_match = re.search(r"sh_scalar_info = \{(.*?)\}\n", text, re.S)
if not info_match:
    raise SystemExit("missing sh_scalar_info")
entries = re.findall(r"\{([^{}]+)\}", info_match.group(1))
body_orders = [int(entry.split(",", 1)[0].strip()) for entry in entries]

idx_match = re.search(r"second_layer_gate_scalar_indices = \{([^}]*)\}", text)
if not idx_match:
    raise SystemExit("missing second_layer_gate_scalar_indices")
indices = [int(x.strip()) for x in idx_match.group(1).split(",") if x.strip()]
if not indices:
    raise SystemExit("empty second-layer gate scalar index list")

selected = [body_orders[i] for i in indices]
selected_set = set(selected)
if selected_set != {2, 3}:
    raise SystemExit(f"expected selected body orders {{2,3}}, got {sorted(selected_set)}")
if any(order > 3 for order in selected):
    raise SystemExit("second-layer gate selected a scalar above its body-order cutoff")

count_match = re.search(r"second_layer_gate_weight_count = (\d+)", text)
if not count_match:
    raise SystemExit("missing second_layer_gate_weight_count")
if int(count_match.group(1)) != len(indices):
    raise SystemExit("second-layer weight count does not match scalar index count")

radial_match = re.search(r"radial_funcs_count = (\d+)", text)
mix_match = re.search(r"second_layer_gate_body_mix_weight_count = (\d+)", text)
body_match = re.search(r"second_layer_gate_body_order_max = (\d+)", text)
if not radial_match or not mix_match or not body_match:
    raise SystemExit("missing second-layer body-mix metadata")
expected_mix = int(radial_match.group(1)) * (int(body_match.group(1)) - 1)
if int(mix_match.group(1)) != expected_mix:
    raise SystemExit("second-layer body-mix count is inconsistent")
PY

./bin/mlp-sus2 calc-efs "$model" "$train" "$pred"

python3 - "$model" "$probe_model" <<'PY'
import re
import sys

source, target = sys.argv[1], sys.argv[2]
text = open(source).read()

def rewrite_vector(src, name, values_fn):
    pattern = rf"{name} = \{{([^}}]*)\}}"
    match = re.search(pattern, src)
    if not match:
        raise SystemExit(f"missing {name}")
    values = [float(x.strip()) for x in match.group(1).split(",") if x.strip()]
    values = values_fn(values)
    replacement = name + " = {" + ", ".join(f"{x:.15e}" for x in values) + "}"
    return src[:match.start()] + replacement + src[match.end():]

def set_prefix(seed):
    def apply(values):
        for i in range(min(len(values), len(seed))):
            values[i] = seed[i]
        return values
    return apply

def set_alternating(scale):
    def apply(values):
        for i in range(len(values)):
            sign = -1.0 if i % 2 else 1.0
            values[i] = sign * scale * (1.0 + 0.05 * (i % 5))
        return values
    return apply

text = rewrite_vector(text, "two_layer_gate_weights", set_prefix([0.25, -0.18, 0.11, -0.07]))
text = rewrite_vector(text, "two_layer_gate_additive_coeffs", set_alternating(0.08))
text = rewrite_vector(text, "second_layer_gate_weights", set_prefix([0.19, -0.13, 0.09, -0.05]))
text = rewrite_vector(text, "second_layer_gate_additive_coeffs", set_alternating(0.06))
open(target, "w").write(text)
PY

./bin/mlp-sus2 check-linear-components-fd-dev "$probe_model" "$train" \
  --max-configs=1 \
  --max-atoms=1 \
  --displacement=1.0e-5 \
  --abs-tolerance=1.0e-5 \
  --rel-tolerance=1.0e-4

./bin/mlp-sus2 check-loss-gradient-direction-dev "$probe_model" "$train" \
  --max-configs=1 \
  --energy-weight=1 \
  --force-weight=0.1 \
  --stress-weight=0 \
  --radial-smooth=0 \
  --scalar-head-l2=0 \
  --gate-scalar-l2=0 \
  --gate-mix-l2=0 \
  --gate-full-l2=0 \
  --gate-x-l2=0 \
  --displacement=1.0e-4 \
  --abs-tolerance=1.0e-6 \
  --rel-tolerance=2.0e-2
