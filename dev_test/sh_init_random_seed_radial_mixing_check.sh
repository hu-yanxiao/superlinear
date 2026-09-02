#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

bin=${1:-./bin/mlp-sus2}
if [[ ! -x "$bin" ]]; then
  echo "missing mlp-sus2 binary: $bin" >&2
  exit 2
fi

tmp_dir=".codex_tmp/sh_init_random_seed_radial_mixing"
rm -rf "$tmp_dir"
mkdir -p "$tmp_dir"
awk '{print} /^END_CFG/{exit}' example/train.cfg >"$tmp_dir/one.cfg"

common=(
  --init-sh
  --species-count=2
  --l-max=2
  --k-max=3
  --body-order=4
  --body-l-max=2,2,2,2
  --radial-basis-size=4
  --cutoff=5.0
  --two-layer-gate
  --two-layer-gate-mode=mu-scalar-full
  --two-layer-gate-body-order=3
  --two-layer-gate-k-rank=2
  --outer-k-rank=2
  --init-params=random
  --skip-preinit
  --max-iter=0
)

run_init() {
  local label=$1
  local seed=$2
  local mixing=$3
  local scale=$4
  local model="$tmp_dir/${label}.mtp"
  local inline="$tmp_dir/${label}-topology.mtp"
  local args=("${common[@]}")
  if [[ "$mixing" == 1 ]]; then
    args+=(--two-layer-gate-radial-mixing=full)
    args+=(--two-layer-gate-radial-mixing-init-scale="$scale")
  fi
  "$bin" train "$tmp_dir/one.cfg" \
    "${args[@]}" \
    --init-random-seed="$seed" \
    --inline-sh-model="$inline" \
    --trained-pot-name="$model" \
    >"$tmp_dir/${label}.log" 2>&1
  test -s "$model"
  grep -q "seed=${seed}" "$tmp_dir/${label}.log"
}

seed=2026090201
scale=1e-4
run_init a1 "$seed" 0 0
run_init a2 "$seed" 0 0
run_init b1 "$seed" 1 "$scale"
run_init b2 "$seed" 1 "$scale"
run_init bzero "$seed" 1 0
run_init other 2026090202 0 0

python3 - "$tmp_dir" "$scale" <<'PY'
import math
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
scale = float(sys.argv[2])
number = r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?"

def vectors(name):
    text = (root / f"{name}.mtp").read_text(encoding="utf-8")
    result = {}
    for match in re.finditer(r"(?ms)^\s*([A-Za-z0-9_]+)\s*=\s*\{([^}]*)\}", text):
        result[match.group(1)] = tuple(float(x) for x in re.findall(number, match.group(2)))
    radial = re.search(
        r"(?ms)^\s*radial_coeffs\s*$\s*(.*?)^\s*alpha_moments_count\s*=",
        text,
    )
    if not radial:
        raise SystemExit(f"missing radial_coeffs block in {name}.mtp")
    radial_rows = re.findall(r"\{([^}]*)\}", radial.group(1))
    result["radial_coeffs"] = tuple(
        float(x)
        for row in radial_rows
        for x in re.findall(number, row)
    )
    if not result["radial_coeffs"]:
        raise SystemExit(f"empty radial_coeffs block in {name}.mtp")
    return result

models = {name: vectors(name) for name in ("a1", "a2", "b1", "b2", "bzero", "other")}
common = (
    "scal_coeffs",
    "radial_coeffs",
    "two_layer_gate_radial_coeffs",
    "two_layer_gate_additive_coeffs",
    "two_layer_gate_type_coeffs",
    "two_layer_gate_weights",
    "two_layer_gate_k_rank_weights",
    "outer_k_rank1_weights",
)

for key in common:
    for model_name, model in models.items():
        if key not in model or not model[key]:
            raise SystemExit(f"missing common initialization block {key} in {model_name}")
    if models["a1"].get(key) != models["a2"].get(key):
        raise SystemExit(f"same-seed A mismatch in {key}")
    if models["b1"].get(key) != models["b2"].get(key):
        raise SystemExit(f"same-seed B mismatch in {key}")
    for candidate in ("b1", "bzero"):
        if models["a1"].get(key) != models[candidate].get(key):
            raise SystemExit(f"A/B common initialization mismatch in {key}: {candidate}")

if all(models["a1"].get(key) == models["other"].get(key) for key in common):
    raise SystemExit("different seed did not change any common nonlinear block")

mix = models["b1"].get("two_layer_gate_radial_mixing_coeffs")
if not mix:
    raise SystemExit("missing randomized radial mixing coefficients")
if not all(math.isfinite(x) and -scale <= x <= scale for x in mix):
    raise SystemExit("radial mixing initialization is outside requested bounds")
if sum(x != 0.0 for x in mix) < int(0.99 * len(mix)):
    raise SystemExit("radial mixing initialization unexpectedly contains many zeros")
zero_mix = models["bzero"].get("two_layer_gate_radial_mixing_coeffs")
if not zero_mix or any(x != 0.0 for x in zero_mix):
    raise SystemExit("zero-scale radial mixing initialization is not exactly zero")

print(
    "deterministic fresh initialization passed: "
    f"common_blocks={len(common)} mixing_coeffs={len(mix)} scale={scale:.1e}"
)
PY

if "$bin" train "$tmp_dir/one.cfg" "${common[@]}" \
    --init-random-seed=not-a-number \
    --trained-pot-name="$tmp_dir/invalid-seed.mtp" \
    >"$tmp_dir/invalid-seed.log" 2>&1; then
  echo "invalid random seed was accepted" >&2
  exit 1
fi
grep -q "init-random-seed" "$tmp_dir/invalid-seed.log"

if "$bin" train "$tmp_dir/one.cfg" "${common[@]}" \
    --two-layer-gate-radial-mixing=full \
    --two-layer-gate-radial-mixing-init-scale=-1e-4 \
    --init-random-seed="$seed" \
    --trained-pot-name="$tmp_dir/negative-scale.mtp" \
    >"$tmp_dir/negative-scale.log" 2>&1; then
  echo "negative radial mixing initialization scale was accepted" >&2
  exit 1
fi
grep -q "radial-mixing-init-scale" "$tmp_dir/negative-scale.log"

echo "SUS2-SH deterministic random seed and radial mixing initialization check passed"
