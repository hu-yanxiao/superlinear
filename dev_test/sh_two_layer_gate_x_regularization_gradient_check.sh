#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

tmp_dir=".codex_tmp/sh_two_layer_gate_x_regularization_gradient"
mkdir -p "$tmp_dir"

train="$tmp_dir/one.cfg"
node_model="$tmp_dir/gate_x_node.mtp"
edge_model="$tmp_dir/gate_x_edge_l1.mtp"
rm -f "$train" "$node_model" "$edge_model"
cat > "$train" <<'CFG'
BEGIN_CFG
 Size
      4
 Supercell
   7.0000000000    0.0000000000    0.0000000000
   0.0000000000    7.0000000000    0.0000000000
   0.0000000000    0.0000000000    7.0000000000
 AtomData:  id type       cartes_x      cartes_y      cartes_z     fx          fy          fz
      1      0     0.300000     0.400000     0.500000    0.000000    0.000000    0.000000
      2      1     1.700000     0.900000     2.100000    0.000000    0.000000    0.000000
      3      0     3.100000     2.400000     1.300000    0.000000    0.000000    0.000000
      4      1     4.200000     3.500000     3.800000    0.000000    0.000000    0.000000
 Energy
    -1.234567
END_CFG
CFG

common_init=(
  --species-count=2
  --l-max=1
  --k-max=1
  --body-order=3
  --body-l-max=1,1,1,1
  --radial-basis-size=3
  --cutoff=5.0
  --write-sh-scalar-info
  --two-layer-gate
  --two-layer-gate-mode=mu-scalar-full
)

./bin/mlp-sus2 init-sh "$node_model" "${common_init[@]}" >/dev/null
./bin/mlp-sus2 init-sh "$edge_model" "${common_init[@]}" --two-layer-gate-edge-l1 >/dev/null

python3 - "$node_model" "$edge_model" <<'PY'
import re
import sys

node_path, edge_path = sys.argv[1:3]

def replace_block(text, name, values):
    return re.sub(
        rf"{name}\s*=\s*\{{[^}}]*\}}",
        f"{name} = " + "{" + ", ".join(f"{v:.15e}" for v in values) + "}",
        text,
        flags=re.S,
    )

def list_block(text, name):
    m = re.search(rf"{name}\s*=\s*\{{([^}}]*)\}}", text, re.S)
    if not m:
        raise SystemExit(f"missing {name}")
    return [float(x.strip()) for x in m.group(1).split(",") if x.strip()]

def scalar_int(text, name):
    m = re.search(rf"{name}\s*=\s*(\d+)", text)
    if not m:
        raise SystemExit(f"missing {name}")
    return int(m.group(1))

def tune(path, edge):
    text = open(path).read()
    weights = list_block(text, "two_layer_gate_weights")
    if len(weights) < 2:
        raise SystemExit("expected at least two gate weights")
    weights = [0.0 for _ in weights]
    for i, value in enumerate([0.35, -0.21, 0.17, 0.09]):
        if i >= len(weights):
            break
        weights[i] = value
    text = replace_block(text, "two_layer_gate_weights", weights)

    additive = list_block(text, "two_layer_gate_additive_coeffs")
    additive = [0.8 + 0.03 * ((i % 7) - 3) for i in range(len(additive))]
    text = replace_block(text, "two_layer_gate_additive_coeffs", additive)

    if edge:
        edge_weights = list_block(text, "two_layer_gate_edge_l1_weights")
        if not edge_weights:
            raise SystemExit("empty edge L1 weights")
        edge_weights = [0.12 * ((i % 5) - 2) for i in range(len(edge_weights))]
        text = replace_block(text, "two_layer_gate_edge_l1_weights", edge_weights)

    open(path, "w").write(text)

tune(node_path, False)
tune(edge_path, True)
PY

common_check=(
  --max-configs=1
  --energy-weight=0
  --force-weight=0
  --stress-weight=0
  --radial-smooth=0
  --scalar-head-l2=0
  --gate-scalar-l2=0
  --gate-mix-l2=0
  --gate-full-l2=0
  --gate-x-l2=1
)

./bin/mlp-sus2 check-loss-gradient-direction-dev "$node_model" "$train" \
  "${common_check[@]}" \
  --displacement=1e-5 \
  --abs-tolerance=1e-7 \
  --rel-tolerance=1e-5

./bin/mlp-sus2 check-loss-gradient-direction-dev "$edge_model" "$train" \
  "${common_check[@]}" \
  --displacement=1e-5 \
  --abs-tolerance=1e-7 \
  --rel-tolerance=1e-5

echo "release gate x regularization gradient check: PASS"
