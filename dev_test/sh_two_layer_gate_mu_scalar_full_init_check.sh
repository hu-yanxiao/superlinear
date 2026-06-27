#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mlp="${1:-$root/bin/mlp-sus2}"
tmp_dir="$root/.codex_tmp/sh_two_layer_gate_mu_scalar_full_init"
mkdir -p "$tmp_dir"
model="$tmp_dir/gate_full.mtp"

"$mlp" init-sh "$model" \
  --species-count=2 \
  --l-max=2 \
  --k-max=3 \
  --body-order=5 \
  --body-l-max=2,2,2,1 \
  --cutoff=5.0 \
  --radial-basis-size=4 \
  --radial-basis-type=RBChebyshev_sss \
  --two-layer-gate \
  --two-layer-gate-mode=mu-scalar-full \
  --two-layer-gate-shared-radial >/dev/null

python3 - "$model" <<'PY'
import re
import sys
from pathlib import Path

text = Path(sys.argv[1]).read_text()

def int_value(name):
    match = re.search(rf"^{name}\s*=\s*([0-9]+)\s*$", text, re.M)
    if not match:
        raise SystemExit(f"missing {name}")
    return int(match.group(1))

def float_list(name, required=True):
    match = re.search(rf"^{name}\s*=\s*\{{([^}}]*)\}}", text, re.M | re.S)
    if not match:
        if required:
            raise SystemExit(f"missing {name}")
        return []
    body = match.group(1).strip()
    if not body:
        return []
    return [float(x.strip()) for x in body.split(",") if x.strip()]

mode = re.search(r"^two_layer_gate_mode\s*=\s*(\S+)\s*$", text, re.M)
if not mode:
    raise SystemExit("missing two_layer_gate_mode")
if mode.group(1) != "mu-scalar-full":
    raise SystemExit(f"expected mu-scalar-full mode, got {mode.group(1)}")

species = int_value("species_count")
l_max = int_value("sh_l_max")
k_max = int_value("sh_k_max")
radial_func_count = (l_max + 1) * k_max
idx_match = re.search(r"^two_layer_gate_scalar_indices\s*=\s*\{([^}]*)\}", text, re.M | re.S)
if not idx_match:
    raise SystemExit("missing two_layer_gate_scalar_indices")
scalar_count = len([x for x in idx_match.group(1).split(",") if x.strip()])

additive_count = int_value("two_layer_gate_additive_coeff_count")
type_count = int_value("two_layer_gate_type_coeff_count")
weight_count = int_value("two_layer_gate_weight_count")
weights = float_list("two_layer_gate_weights")
mix_count_match = re.search(r"^two_layer_gate_body_mix_weight_count\s*=", text, re.M)
mix_weights = float_list("two_layer_gate_body_mix_weights", required=False)
additive = float_list("two_layer_gate_additive_coeffs")
type_coeffs = float_list("two_layer_gate_type_coeffs")

expected_additive_count = species * radial_func_count
if additive_count != expected_additive_count:
    raise SystemExit(f"expected type/mu additive count {expected_additive_count}, got {additive_count}")
if len(additive) != additive_count:
    raise SystemExit("additive list length mismatch")
expected_additive = 1.0 / 12.0
if any(abs(x - expected_additive) > 1e-7 for x in additive):
    raise SystemExit("additive coefficients should initialize to 1/12")
if type_count != species:
    raise SystemExit(f"expected gate type coeff count {species}, got {type_count}")
if len(type_coeffs) != type_count:
    raise SystemExit("gate type coeff list length mismatch")

expected_weight_count = radial_func_count * scalar_count
if weight_count != expected_weight_count:
    raise SystemExit(f"expected full mu/scalar weight count {expected_weight_count}, got {weight_count}")
if len(weights) != weight_count:
    raise SystemExit("gate weight list length mismatch")
if any(abs(x - 0.1) > 1e-14 for x in weights):
    raise SystemExit("full mu/scalar weights should initialize to 0.1")
if mix_count_match or mix_weights:
    raise SystemExit("mu-scalar-full mode should not save body mix weights")

print("mu-scalar-full gate init metadata OK")
PY
