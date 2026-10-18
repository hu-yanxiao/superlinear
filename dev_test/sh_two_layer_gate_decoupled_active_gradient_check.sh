#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

tmp_dir=".local_tmp/sh_two_layer_gate_decoupled_active_gradient"
mkdir -p "$tmp_dir"

init_model="$tmp_dir/init.sus2"
active_model="$tmp_dir/active.sus2"
train_cfg="$tmp_dir/train.cfg"
probe_cfg="$tmp_dir/probe.cfg"
train_log="$tmp_dir/train.log"
rm -f "$init_model" "$active_model" "$train_cfg" "$probe_cfg" "$train_log"

awk 'BEGIN {n=0} /^BEGIN_CFG/ {n++} n==1 {print} n==1 && /^END_CFG/ {exit}' \
  example/train.cfg > "$train_cfg"
awk 'BEGIN {n=0} /^BEGIN_CFG/ {n++} n==2 {print} n==2 && /^END_CFG/ {exit}' \
  example/train.cfg > "$probe_cfg"

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
  --two-layer-gate-mode=mu-scalar-full \
  --two-layer-gate-full-decoupled >/dev/null

./bin/mlp-sus2 train "$init_model" "$train_cfg" \
  --trained-pot-name="$active_model" \
  --max-iter=1 \
  --skip-preinit \
  --shift \
  --energy-weight=1 \
  --force-weight=0.2 \
  --stress-weight=0 \
  --radial-smooth=0 >"$train_log"

python3 - "$active_model" <<'PY'
import re
import sys

path = sys.argv[1]
text = open(path).read()

def rewrite(name, update):
    global text
    match = re.search(rf"{name} = \{{([^}}]*)\}}", text, re.S)
    if not match:
        raise SystemExit(f"missing {name}")
    values = [
        float(value)
        for value in re.findall(
            r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?",
            match.group(1),
        )
    ]
    update(values)
    replacement = name + " = {" + ", ".join(f"{value:.15e}" for value in values) + "}"
    text = text[:match.start()] + replacement + text[match.end():]

def separate_scaling(values):
    for index in range(len(values)):
        values[index] += (0.08 if index % 2 == 0 else -0.05) * (1.0 + 0.02 * (index % 7))

def activate_weights(values):
    for index in range(len(values)):
        values[index] += 0.03 * ((index % 5) - 2)

rewrite("two_layer_gate_scaling_coeffs", separate_scaling)
rewrite("two_layer_gate_weights", activate_weights)
open(path, "w").write(text)
PY

layout_output=$(./bin/mlp-sus2 check-loss-gradient-dev "$active_model" "$probe_cfg" \
  --max-configs=1 \
  --energy-weight=1 \
  --force-weight=0.2 \
  --stress-weight=0 \
  --radial-smooth=0 \
  --gate-x-l2=0 \
  --coeff-start=0 \
  --coeff-end=1 \
  --displacement=1e-5 \
  --abs-tolerance=1e99 \
  --rel-tolerance=1e99 | grep 'checked_coeffs=')

gate_begin=$(awk '{for (i=1; i<=NF; ++i) if ($i ~ /^gate_radial_begin=/) {sub(/^[^=]*=/, "", $i); print $i}}' <<<"$layout_output")
gate_end=$(awk '{for (i=1; i<=NF; ++i) if ($i ~ /^gate_radial_end=/) {sub(/^[^=]*=/, "", $i); print $i}}' <<<"$layout_output")
if [[ ! "$gate_begin" =~ ^[0-9]+$ || ! "$gate_end" =~ ^[0-9]+$ \
      || "$gate_begin" -ge "$gate_end" ]]; then
  echo "failed to parse gate radial coefficient range" >&2
  exit 1
fi

run_gradient_check() {
  local cache_mode=$1
  shift
  local output
  output=$("$@" ./bin/mlp-sus2 check-loss-gradient-direction-dev \
    "$active_model" "$probe_cfg" \
    --max-configs=1 \
    --energy-weight=1 \
    --force-weight=0.2 \
    --stress-weight=0 \
    --radial-smooth=0 \
    --gate-x-l2=0 \
    --coeff-start="$gate_begin" \
    --coeff-end="$gate_end" \
    --displacement=1e-5 \
    --abs-tolerance=3e-6 \
    --rel-tolerance=3e-4 | grep 'checked_coeffs=')
  echo "$cache_mode $output"
  python3 - "$output" <<'PY'
import re
import sys

line = sys.argv[1]
analytic = float(re.search(r"analytic(?:_directional)?=([^ ]+)", line).group(1))
finite_difference = float(re.search(r"finite_difference=([^ ]+)", line).group(1))
if max(abs(analytic), abs(finite_difference)) <= 1.0e-8:
    raise SystemExit("decoupled gate radial gradient probe is inactive")
PY
}

run_gradient_check cache_on env
run_gradient_check cache_off env SUS2_SH_DISABLE_TWO_LAYER_EDGE_CACHE=1
