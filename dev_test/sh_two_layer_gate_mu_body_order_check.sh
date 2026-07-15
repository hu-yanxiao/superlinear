#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

tmp_dir=".codex_tmp/sh_two_layer_gate_mu_body_order"
mkdir -p "$tmp_dir"

model="$tmp_dir/mu_body_gate.mtp"
old_opt_model="$tmp_dir/old_opt.mtp"
low_body_model="$tmp_dir/low_body.mtp"
train="$tmp_dir/one.cfg"
rm -f "$model" "$old_opt_model" "$low_body_model" "$train"
awk '{print} /^END_CFG/{exit}' example/train.cfg > "$train"

./bin/mlp-sus2 init-sh "$old_opt_model" \
  --species-count=2 \
  --l-max=2 \
  --k-max=4 \
  --body-order=6 \
  --body-l-max=2,2,2,2,2 \
  --radial-basis-size=4 \
  --cutoff=5.0 \
  --two-layer-gate \
  --two-layer-gate-body-order=3 >/dev/null

grep -q "two_layer_gate_body_order_max = 3" "$old_opt_model"

./bin/mlp-sus2 init-sh "$low_body_model" \
  --species-count=2 \
  --l-max=2 \
  --k-max=4 \
  --body-order=4 \
  --body-l-max=2,2,2,2 \
  --radial-basis-size=4 \
  --cutoff=5.0 \
  --two-layer-gate >/dev/null

grep -q "two_layer_gate_body_order_max = 4" "$low_body_model"

./bin/mlp-sus2 init-sh "$model" \
  --species-count=2 \
  --l-max=2 \
  --k-max=4 \
  --body-order=6 \
  --body-l-max=2,2,2,2,2 \
  --radial-basis-size=4 \
  --cutoff=5.0 \
  --write-sh-scalar-info \
  --two-layer-gate

grep -q "two_layer_gate_mode = mu-body-linear-combo" "$model"
grep -q "two_layer_gate_body_order_max = 6" "$model"
grep -q "two_layer_gate_site_mode = neighbor" "$model"
if grep -q "two_layer_gate_bias" "$model"; then
  echo "mu-body-order gate model should not save two_layer_gate_bias" >&2
  exit 1
fi

if ./bin/mlp-sus2 init-sh "$tmp_dir/bad_site_mode.mtp" \
  --species-count=2 \
  --l-max=2 \
  --k-max=2 \
  --body-order=4 \
  --body-l-max=2,2,2,2 \
  --radial-basis-size=4 \
  --cutoff=5.0 \
  --two-layer-gate \
  --two-layer-gate-site-mode=bad >/dev/null 2>&1; then
  echo "--two-layer-gate-site-mode should reject unknown modes" >&2
  exit 1
fi

if ./bin/mlp-sus2 init-sh "$tmp_dir/no_gate_site_mode.mtp" \
  --species-count=2 \
  --l-max=2 \
  --k-max=2 \
  --body-order=4 \
  --body-l-max=2,2,2,2 \
  --radial-basis-size=4 \
  --cutoff=5.0 \
  --two-layer-gate-site-mode=double >/dev/null 2>&1; then
  echo "--two-layer-gate-site-mode should require --two-layer-gate" >&2
  exit 1
fi

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

idx_match = re.search(r"two_layer_gate_scalar_indices = \{([^}]*)\}", text)
if not idx_match:
    raise SystemExit("missing two_layer_gate_scalar_indices")
indices = [int(x.strip()) for x in idx_match.group(1).split(",") if x.strip()]
if not indices:
    raise SystemExit("empty gate scalar index list")

selected = [body_orders[i] for i in indices]
selected_set = set(selected)
if selected_set != {2, 3, 4, 5, 6}:
    raise SystemExit(f"expected selected body orders {{2,3,4,5,6}}, got {sorted(selected_set)}")
for order in range(2, 7):
    if order not in selected_set:
        raise SystemExit(f"missing gate scalar body-order {order}")

count_match = re.search(r"two_layer_gate_weight_count = (\d+)", text)
if not count_match:
    raise SystemExit("missing two_layer_gate_weight_count")
radial_match = re.search(r"radial_funcs_count = (\d+)", text)
if not radial_match:
    raise SystemExit("missing radial_funcs_count")
expected_weight_count = len(indices)
if int(count_match.group(1)) != expected_weight_count:
    raise SystemExit("gate weight count does not match scalar index count")
mix_count_match = re.search(r"two_layer_gate_body_mix_weight_count = (\d+)", text)
if not mix_count_match:
    raise SystemExit("missing two_layer_gate_body_mix_weight_count")
gate_body_order_match = re.search(r"two_layer_gate_body_order_max = (\d+)", text)
if not gate_body_order_match:
    raise SystemExit("missing two_layer_gate_body_order_max")
expected_mix_count = int(radial_match.group(1)) * (int(gate_body_order_match.group(1)) - 1)
if int(mix_count_match.group(1)) != expected_mix_count:
    raise SystemExit("gate body mix weight count does not match radial_funcs_count * gate_body_order_count")
PY

./bin/mlp-sus2 check-two-layer-gate-mu-body-order-dev "$model" "$train" \
  --max-configs=1 \
  --max-atoms=2 \
  --probe-weight=0.5 \
  --abs-tolerance=1.0e-12 \
  --rel-tolerance=1.0e-10
