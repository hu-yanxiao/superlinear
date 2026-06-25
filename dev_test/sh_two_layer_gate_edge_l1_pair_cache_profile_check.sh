#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

tmp_dir=".codex_tmp/sh_two_layer_gate_edge_l1_pair_cache_profile"
mkdir -p "$tmp_dir"

model="$tmp_dir/edge_l1_pair_cache_profile.mtp"
profile_log="$tmp_dir/profile.log"
disabled_log="$tmp_dir/profile_disabled.log"
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
    [0.25 * ((i % 7) - 3) for i in range(edge_l1_count)],
    text,
)
open(path, "w").write(text)
PY

SUS2_SH_EDGE_L1_PAIR_CACHE_PROFILE=1 \
SUS2_SH_BASIC_MU_FORWARD_PROFILE=1 \
  ./bin/mlp-sus2 calc-efs "$model" example/train.cfg "$pred" >"$profile_log"

grep -q "\\[SUS2_SH_EDGE_L1_PAIR_CACHE_PROFILE\\]" "$profile_log"
grep -q "pair_cache_edges=" "$profile_log"
grep -q "pair_cache_mu=" "$profile_log"
grep -q "pair_cache_fast_path=1" "$profile_log"
grep -q "\\[SUS2_SH_BASIC_MU_FORWARD_PROFILE\\]" "$profile_log"
grep -q "forward_grouped=1" "$profile_log"

SUS2_SH_EDGE_L1_PAIR_CACHE_PROFILE=1 \
SUS2_SH_DISABLE_EDGE_L1_PAIR_CACHE=1 \
  ./bin/mlp-sus2 calc-efs "$model" example/train.cfg "$pred" >"$disabled_log"

if grep -q "\\[SUS2_SH_EDGE_L1_PAIR_CACHE_PROFILE\\]" "$disabled_log"; then
  echo "edge-L1 pair cache disable env did not suppress pair cache profile" >&2
  exit 1
fi
