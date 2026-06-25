#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

tmp_dir=".codex_tmp/sh_two_layer_gate_edge_l1_weighted_cache_profile"
mkdir -p "$tmp_dir"

model="$tmp_dir/edge_l1_cache_profile.mtp"
profile_log="$tmp_dir/profile.log"
pred="$tmp_dir/pred.cfg"

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
  --two-layer-gate-edge-l1 >/dev/null

python3 - "$model" <<'PY'
import re
import sys

path = sys.argv[1]
text = open(path).read()

def replace_block(name, values, body):
    line = f"{name} = " + "{" + ", ".join(f"{v:.15e}" for v in values) + "}"
    return re.sub(rf"{name} = \{{[^}}]*\}}", line, body, flags=re.S)

def scalar_int(name):
    return int(re.search(rf"{name} = (\d+)", text).group(1))

species = scalar_int("species_count")
alpha = scalar_int("alpha_scalar_moments")
edge_l1_count = scalar_int("two_layer_gate_edge_l1_weight_count")

text = replace_block("species_coeffs", [1.0 for _ in range(species)], text)
text = replace_block(
    "moment_coeffs",
    [0.2 + 0.01 * ((i % 5) - 2) for i in range(alpha)],
    text,
)
text = replace_block(
    "two_layer_gate_edge_l1_weights",
    [0.3 * ((i % 7) - 3) for i in range(edge_l1_count)],
    text,
)
open(path, "w").write(text)
PY

SUS2_SH_EDGE_L1_CACHE_PROFILE=1 \
  ./bin/mlp-sus2 calc-efs "$model" example/train.cfg "$pred" >"$profile_log"

grep -q "\\[SUS2_SH_EDGE_L1_CACHE_PROFILE\\]" "$profile_log"
grep -q "weighted_cache_atoms=" "$profile_log"
grep -q "weighted_cache_edge_uses=" "$profile_log"
