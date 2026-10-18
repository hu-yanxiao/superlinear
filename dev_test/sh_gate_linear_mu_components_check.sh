#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

tmp_dir=".local_tmp/sh_gate_linear_mu_components"
mkdir -p "$tmp_dir"

model="$tmp_dir/mu_full.sus2"
probe="$tmp_dir/mu_full_probe.sus2"
train="$tmp_dir/one.cfg"
log="$tmp_dir/components_fd.log"
rm -f "$model" "$probe" "$train" "$log"
awk '
  /^BEGIN_CFG/ { ++cfg_index }
  cfg_index == 2 { print }
  cfg_index == 2 && /^END_CFG/ { exit }
' example/train.cfg > "$train"

./bin/mlp-sus2 init-sh "$model" \
  --species-count=2 \
  --l-max=2 \
  --k-max=2 \
  --body-order=4 \
  --body-l-max=2,2,2,2 \
  --radial-basis-size=4 \
  --init-params=same \
  --cutoff=5.0 \
  --write-sh-scalar-info \
  --two-layer-gate \
  --two-layer-gate-mode=mu-scalar-full \
  --two-layer-gate-shared-radial

python3 - "$model" "$probe" <<'PY'
import re
import sys

source, target = sys.argv[1:]
text = open(source).read()

radial_match = re.search(r"radial_funcs_count = (\d+)", text)
weight_match = re.search(r"two_layer_gate_weights = \{([^}]*)\}", text)
if not radial_match or not weight_match:
    raise SystemExit("missing radial count or full gate weights")

mu_count = int(radial_match.group(1))
weights = [float(x.strip()) for x in weight_match.group(1).split(",") if x.strip()]
if not weights or len(weights) % mu_count:
    raise SystemExit("full gate weight matrix has an inconsistent size")

scalar_count = len(weights) // mu_count
for mu in range(mu_count):
    for q in range(scalar_count):
        sign = -1.0 if (mu + q) % 2 else 1.0
        weights[mu * scalar_count + q] = (
            sign * 2.0 * (1.0 + 0.07 * mu + 0.03 * (q % 5))
        )

replacement = "two_layer_gate_weights = {" + ", ".join(
    f"{value:.15e}" for value in weights
) + "}"
text = text[:weight_match.start()] + replacement + text[weight_match.end():]

for name, scale in (
    ("two_layer_gate_additive_coeffs", 0.75),
    ("two_layer_gate_type_coeffs", 0.90),
):
    match = re.search(rf"{name} = \{{([^}}]*)\}}", text)
    if not match:
        raise SystemExit(f"missing {name}")
    values = [float(x.strip()) for x in match.group(1).split(",") if x.strip()]
    for index in range(len(values)):
        sign = -1.0 if name.endswith("additive_coeffs") and index % 2 else 1.0
        values[index] = sign * scale * (1.0 + 0.11 * (index % mu_count))
    replacement = name + " = {" + ", ".join(
        f"{value:.15e}" for value in values
    ) + "}"
    text = text[:match.start()] + replacement + text[match.end():]
open(target, "w").write(text)
PY

./bin/mlp-sus2 check-linear-components-fd-dev "$probe" "$train" \
  --max-configs=1 \
  --max-atoms=1 \
  --displacement=1e-5 \
  --abs-tolerance=1e-7 \
  --rel-tolerance=1e-6 >"$log" 2>&1

grep -q "checked_components=" "$log"
cat "$log"
