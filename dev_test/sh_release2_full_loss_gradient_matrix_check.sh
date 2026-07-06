#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

MLP=${MLP:-./bin/mlp-sus2}
tmp_dir=".codex_tmp/release2_full_loss_gradient_matrix"
log_dir="$tmp_dir/logs"
rm -rf "$tmp_dir"
mkdir -p "$log_dir"

train="$tmp_dir/two_species.cfg"

cat > "$train" <<'CFG'
BEGIN_CFG
 Size
      4
 Supercell
   7.0000000000    0.0000000000    0.0000000000
   0.0000000000    6.8000000000    0.0000000000
   0.0000000000    0.0000000000    6.6000000000
 AtomData:  id type       cartes_x      cartes_y      cartes_z     fx          fy          fz
      1      0     0.400000     0.500000     0.600000    0.110000   -0.070000    0.050000
      2      1     1.700000     0.950000     2.100000   -0.040000    0.090000   -0.080000
      3      0     3.050000     2.400000     1.350000    0.060000    0.030000   -0.020000
      4      1     4.300000     3.550000     3.800000   -0.130000   -0.050000    0.050000
 Energy
    -1.250000
 PlusStress:  xx          yy          zz          yz          xz          xy
     0.020000    -0.030000     0.015000     0.004000    -0.006000     0.008000
END_CFG
BEGIN_CFG
 Size
      4
 Supercell
   5.8000000000    0.0000000000    0.0000000000
   0.0000000000    5.7000000000    0.0000000000
   0.0000000000    0.0000000000    5.6000000000
 AtomData:  id type       cartes_x      cartes_y      cartes_z     fx          fy          fz
      1      0     0.350000     0.450000     0.550000   -0.080000    0.040000    0.020000
      2      1     1.450000     0.820000     1.850000    0.100000   -0.060000    0.070000
      3      0     2.650000     2.050000     1.180000   -0.030000   -0.020000    0.090000
      4      1     3.750000     3.100000     3.250000    0.010000    0.040000   -0.180000
 Energy
    0.650000
 PlusStress:  xx          yy          zz          yz          xz          xy
     0.110000     0.070000     0.095000    -0.012000     0.015000    -0.010000
END_CFG
CFG

common_init=(
  --species-count=2
  --l-max=2
  --k-max=2
  --body-order=4
  --body-l-max=2,2,2,2
  --radial-basis-size=4
  --cutoff=5.0
  --write-sh-scalar-info
)

"$MLP" init-sh "$tmp_dir/nogate.mtp" "${common_init[@]}" >/dev/null
"$MLP" init-sh "$tmp_dir/gate_body.mtp" "${common_init[@]}" --two-layer-gate >/dev/null
"$MLP" init-sh "$tmp_dir/gate_full.mtp" "${common_init[@]}" \
  --two-layer-gate --two-layer-gate-mode=mu-scalar-full >/dev/null
"$MLP" init-sh "$tmp_dir/gate_edge_l1.mtp" "${common_init[@]}" \
  --two-layer-gate --two-layer-gate-mode=mu-scalar-full \
  --two-layer-gate-edge-l1 >/dev/null

python3 - "$tmp_dir/gate_body.mtp" "$tmp_dir/gate_full.mtp" "$tmp_dir/gate_edge_l1.mtp" <<'PY'
import re
import sys

def list_block(text, name, required=False):
    match = re.search(rf"{name}\s*=\s*\{{([^}}]*)\}}", text, re.S)
    if not match:
        if required:
            raise SystemExit(f"missing {name}")
        return None, []
    values = [float(x.strip()) for x in match.group(1).split(",") if x.strip()]
    return match, values

def replace_block(text, match, name, values):
    replacement = f"{name} = " + "{" + ", ".join(f"{v:.15e}" for v in values) + "}"
    return text[:match.start()] + replacement + text[match.end():]

def tune(path):
    text = open(path).read()
    match, values = list_block(text, "two_layer_gate_weights")
    if match:
        values = [0.0 for _ in values]
        pattern = [0.37, -0.23, 0.19, 0.11, -0.17, 0.13, -0.09, 0.07]
        for i in range(len(values)):
            values[i] = pattern[i % len(pattern)] * (1.0 + 0.01 * (i % 5))
        text = replace_block(text, match, "two_layer_gate_weights", values)

    match, values = list_block(text, "two_layer_gate_body_mix_weights")
    if match:
        pattern = [0.85, -0.35, 0.55, 1.15, -0.25, 0.45]
        values = [pattern[i % len(pattern)] for i in range(len(values))]
        text = replace_block(text, match, "two_layer_gate_body_mix_weights", values)

    match, values = list_block(text, "two_layer_gate_additive_coeffs")
    if match:
        values = [0.75 + 0.04 * ((i % 7) - 3) for i in range(len(values))]
        text = replace_block(text, match, "two_layer_gate_additive_coeffs", values)

    match, values = list_block(text, "two_layer_gate_type_coeffs")
    if match:
        values = [1.0 + 0.06 * ((i % 5) - 2) for i in range(len(values))]
        text = replace_block(text, match, "two_layer_gate_type_coeffs", values)

    match, values = list_block(text, "two_layer_gate_edge_l1_weights")
    if match:
        values = [0.14 * ((i % 5) - 2) for i in range(len(values))]
        text = replace_block(text, match, "two_layer_gate_edge_l1_weights", values)

    open(path, "w").write(text)

for path in sys.argv[1:]:
    tune(path)
PY

base_check=(
  --max-configs=2
  --radial-smooth=0
  --scalar-head-l2=0
  --gate-scalar-l2=0
  --gate-mix-l2=0
  --gate-full-l2=0
  --gate-x-l2=0
  --displacement=1e-4
  --abs-tolerance=1e-4
  --rel-tolerance=1e-3
)

run_coeff_check() {
  local name=$1
  local model=$2
  shift 2
  echo "RUN coeff $name"
  "$MLP" check-loss-gradient-dev "$model" "$train" \
    "${base_check[@]}" "$@" | tee "$log_dir/${name}.log"
}

run_direction_check() {
  local name=$1
  local model=$2
  shift 2
  echo "RUN direction $name"
  "$MLP" check-loss-gradient-direction-dev "$model" "$train" \
    "${base_check[@]}" "$@" | tee "$log_dir/${name}.log"
}

models=(
  "nogate:$tmp_dir/nogate.mtp"
  "gate_body:$tmp_dir/gate_body.mtp"
  "gate_full:$tmp_dir/gate_full.mtp"
  "gate_edge_l1:$tmp_dir/gate_edge_l1.mtp"
)

for entry in "${models[@]}"; do
  name=${entry%%:*}
  model=${entry#*:}
  run_coeff_check "${name}_energy" "$model" \
    --energy-weight=1 --force-weight=0 --stress-weight=0
  run_coeff_check "${name}_force" "$model" \
    --energy-weight=0 --force-weight=1 --stress-weight=0
  run_coeff_check "${name}_stress" "$model" \
    --energy-weight=0 --force-weight=0 --stress-weight=1
  run_coeff_check "${name}_efs" "$model" \
    --energy-weight=1 --force-weight=0.5 --stress-weight=0.25
done

guard_common=(
  --energy-weight=0
  --force-weight=0
  --stress-weight=0
  --guard-compression-anchor=both
  --guard-compression-scales=0.95,0.90,0.85,0.80
  --guard-compression-floor-margin=0
  --guard-compression-monotonic-margin=0.2
  --guard-compression-curvature-margin=0
  --guard-compression-smooth=0.02
)

guard_cases=(
  "floor:1:0:0:0"
  "monotonic:0:1:0:0"
  "curvature:0:0:1:0"
  "curvature_active:0:0:1:0"
  "block:0:0:0:0.1"
  "all:1:1:1:0.1"
)

for entry in "${models[@]}"; do
  name=${entry%%:*}
  model=${entry#*:}
  for guard in "${guard_cases[@]}"; do
    IFS=: read -r guard_name floor_w mono_w curv_w block_w <<<"$guard"
    extra_guard_args=()
    if [[ "$guard_name" == "curvature_active" ]]; then
      extra_guard_args+=(--guard-compression-curvature-margin=10)
    fi
    run_direction_check "${name}_guard_${guard_name}" "$model" \
      "${guard_common[@]}" \
      --guard-compression-floor-weight="$floor_w" \
      --guard-compression-monotonic-weight="$mono_w" \
      --guard-compression-curvature-weight="$curv_w" \
      --guard-compression-block-monotonic-weight="$block_w" \
      "${extra_guard_args[@]}"
  done
done

if [[ -n "${SUS2_SH_ACTIVE_BLOCK_MODEL:-}" && -n "${SUS2_SH_ACTIVE_BLOCK_CFG:-}" ]]; then
  echo "RUN direction external_guard_block_active"
  "$MLP" check-loss-gradient-direction-dev \
    "$SUS2_SH_ACTIVE_BLOCK_MODEL" "$SUS2_SH_ACTIVE_BLOCK_CFG" \
    "${base_check[@]}" \
    --max-configs="${SUS2_SH_ACTIVE_BLOCK_MAX_CONFIGS:-4}" \
    --energy-weight=0 --force-weight=0 --stress-weight=0 \
    --guard-compression-anchor=both \
    --guard-compression-scales=0.95,0.90,0.85,0.80 \
    --guard-compression-floor-weight=0 \
    --guard-compression-monotonic-weight=0 \
    --guard-compression-curvature-weight=0 \
    --guard-compression-block-monotonic-weight=1 \
    --guard-compression-floor-margin=0 \
    --guard-compression-monotonic-margin=0.2 \
    --guard-compression-curvature-margin=0 \
    --guard-compression-smooth=0.02 \
    | tee "$log_dir/external_guard_block_active.log"
fi

run_direction_check "nogate_radial_smooth" "$tmp_dir/nogate.mtp" \
  --energy-weight=0 --force-weight=0 --stress-weight=0 \
  --radial-smooth=1e-6 --radial-smooth-grid=64
run_direction_check "nogate_scalar_head_l2" "$tmp_dir/nogate.mtp" \
  --energy-weight=0 --force-weight=0 --stress-weight=0 \
  --scalar-head-l2=1e-3
run_direction_check "gate_body_gate_scalar_l2" "$tmp_dir/gate_body.mtp" \
  --energy-weight=0 --force-weight=0 --stress-weight=0 \
  --gate-scalar-l2=1e-3
run_direction_check "gate_body_gate_mix_l2" "$tmp_dir/gate_body.mtp" \
  --energy-weight=0 --force-weight=0 --stress-weight=0 \
  --gate-mix-l2=1e-3
run_direction_check "gate_full_gate_full_l2" "$tmp_dir/gate_full.mtp" \
  --energy-weight=0 --force-weight=0 --stress-weight=0 \
  --gate-full-l2=1e-3
run_direction_check "gate_full_gate_x_l2" "$tmp_dir/gate_full.mtp" \
  --energy-weight=0 --force-weight=0 --stress-weight=0 \
  --gate-x-l2=1
run_direction_check "gate_edge_l1_gate_x_l2" "$tmp_dir/gate_edge_l1.mtp" \
  --energy-weight=0 --force-weight=0 --stress-weight=0 \
  --gate-x-l2=1

echo "release2 full loss gradient matrix check: PASS"
