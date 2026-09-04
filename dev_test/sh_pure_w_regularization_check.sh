#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
bin=${PATCHED_BIN:-./bin/mlp-sus2}
test -x "$bin"
mkdir -p .codex_tmp
test_dir=$(mktemp -d .codex_tmp/sh_pure_w_regularization.XXXXXX)
cfg=codex_tests/pure_w_regularization.cfg
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export SUS2_SH_PRODUCT_ROWS=0 SUS2_SH_PRODUCT_HVT_REVERSE=0
export SUS2_SH_ACCUM_SKIP_SITE_DERS=0

# Pure mode's physical objective is lambda * mean(l,o,k,p,q) int |W'(r)|^2 dr.
# Unlike legacy smoothness, it excludes type/gate factors (already in x).
# It neither penalizes unused outer R nor adds a new gate-radial penalty.
# Fixed K=2, Lmax=1, O=3>K, B=4; every W row has a nontrivial signed direction.
make_model() {
  local map=$1 outer=$2 target=$3
  "$bin" init-sh "$target.init" \
    --species-count=2 --l-max=1 --k-max=2 --body-order=3 \
    --body-l-max=1,1,1,1 --radial-basis-size=4 --cutoff=5 --scaling=0.1 \
    --scaling-map="$map" --init-params=same --write-sh-scalar-info \
    --two-layer-gate --two-layer-gate-mode=mu-scalar-full \
    --two-layer-gate-body-order=3 --two-layer-gate-k-rank=2 \
    --outer-k-rank="$outer" --superlinear-mode=pure-w-channelwise \
    >"$target.init.log" 2>&1
  # A complete, deterministic test head avoids an untrained linear warmup in
  # gradient probes. For O=6 the W entries repeat the O=3 entries exactly,
  # testing averaging over O independently of the number of outer scalars.
  awk -v outer="$outer" '
    /^alpha_scalar_moments[[:space:]]*=/ { scalars=$3 }
    /^two_layer_gate_radial_coeffs[[:space:]]*=/ {
      line=$0; sub(/^[^{]*\{/, "", line); sub(/\}.*/, "", line)
      count=split(line, vals, ",")
      printf "two_layer_gate_radial_coeffs = {"
      for(i=0;i<count;++i)
        printf "%s%.17e", i ? ", " : "", (i%4==0 ? -0.12 : 0.03*(1+i%4))
      print "}"; next
    }
    /^two_layer_gate_radial_mixing_coeffs[[:space:]]*=/ {
      line=$0; sub(/^[^{]*\{/, "", line); sub(/\}.*/, "", line)
      count=split(line, vals, ",")
      if (count != 2*outer*2*4) exit 3
      printf "two_layer_gate_radial_mixing_coeffs = {"
      for (i=0; i<count; ++i) {
        b=i%4; k=int(i/4)%2; o=int(i/8)%outer; l=int(i/(outer*8))
        value=0.025*(1+l+0.31*(o%3)+0.17*k)*(b==0 ? -1.3 : 0.3*b)
        printf "%s%.17e", i ? ", " : "", value
      }
      print "}"; next
    }
    { print }
    END {
      print "species_coeffs = {1.0, 0.8}"
      printf "moment_coeffs = {"
      for (i=0; i<scalars; ++i) printf "%s%.17e", i ? ", " : "", 0.03*(1+i%5)
      print "}"
    }
  ' "$target.init" > "$target"
}

field() {
  awk -v key="$1" '{ for(i=1;i<=NF;++i) { split($i,v,"="); if(v[1]==key) answer=v[2] } }
    END { if(answer=="") exit 1; print answer }' "$2"
}

smooth_opts=(--max-configs=1 --energy-weight=0 --force-weight=0 --stress-weight=0
  --radial-smooth=0.02 --radial-smooth-grid=32 --gate-x-l2=0
  --scalar-head-l2=0 --gate-scalar-l2=0 --gate-mix-l2=0 --gate-full-l2=0
  --displacement=1e-6 --abs-tolerance=2e-11 --rel-tolerance=2e-5)

for map in K L LK; do
  model="$test_dir/$map.mtp"
  make_model "$map" 3 "$model"
  "$bin" check-loss-gradient-dev "$model" "$cfg" "${smooth_opts[@]}" \
    --coeff-start=0 --coeff-end=1 > "$test_dir/$map.layout.log" 2>&1
  matrix_begin=$(field gate_radial_mixing_begin "$test_dir/$map.layout.log")
  matrix_end=$(field gate_radial_mixing_end "$test_dir/$map.layout.log")
  radial_begin=$(field radial_begin "$test_dir/$map.layout.log")
  radial_end=$(field radial_end "$test_dir/$map.layout.log")
  test "$((matrix_end-matrix_begin))" -eq 48

  # Individual coefficient differences cover ALL W entries and every shared
  # slope/shift pair. K sums over l,o; L sums over k,o; LK sums over o.
  for block in matrix scaling unused_outer; do
    case "$block" in
      matrix) begin=$matrix_begin; end=$matrix_end ;;
      scaling) begin=2; end=$radial_begin ;;
      unused_outer) begin=$radial_begin; end=$radial_end ;;
    esac
    "$bin" check-loss-gradient-dev "$model" "$cfg" "${smooth_opts[@]}" \
      --coeff-start="$begin" --coeff-end="$end" \
      > "$test_dir/$map.$block.smooth-fd.log" 2>&1
    "$bin" check-loss-gradient-direction-dev "$model" "$cfg" "${smooth_opts[@]}" \
      --coeff-start="$begin" --coeff-end="$end" \
      > "$test_dir/$map.$block.smooth-direction.log" 2>&1
    if [[ "$block" == unused_outer ]]; then
      unused_analytic=$(field analytic_directional "$test_dir/$map.$block.smooth-direction.log")
      unused_fd=$(field finite_difference "$test_dir/$map.$block.smooth-direction.log")
      awk -v a="$unused_analytic" -v b="$unused_fd" 'BEGIN { if(a!=0 || b!=0) exit 1 }'
    fi
  done

  "$bin" check-radial-weight-norm-roundtrip-dev "$model" \
    --abs-tolerance=2e-13 --rel-tolerance=2e-13 \
    > "$test_dir/$map.roundtrip.log" 2>&1
  test "$(field outer_rows "$test_dir/$map.roundtrip.log")" -eq 12
  test "$(field gate1_rows "$test_dir/$map.roundtrip.log")" -eq 4
  for row in {0..11}; do
    "$bin" check-radial-weight-norm-gradient-dev "$model" "$cfg" \
      "${smooth_opts[@]}" --radial-block=outer --mu="$row" \
      > "$test_dir/$map.row$row.chart-fd.log" 2>&1
  done

  # Nonzero E/F/stress loss plus smoothness exercises the same chart pullback
  # used by BFGS, not merely a standalone norm/direction implementation.
  "$bin" check-radial-weight-norm-gradient-dev "$model" "$cfg" \
    --radial-block=outer --mu=11 --max-configs=1 \
    --energy-weight=1 --force-weight=0.3 --stress-weight=0.09 \
    --radial-smooth=0.02 --radial-smooth-grid=32 --gate-x-l2=0 \
    --displacement=1e-6 --abs-tolerance=2e-7 --rel-tolerance=2e-4 \
    > "$test_dir/$map.efs.chart-fd.log" 2>&1
  "$bin" check-radial-weight-norm-gradient-dev "$model" "$cfg" \
    --radial-block=gate1 --mu=3 --max-configs=1 \
    --energy-weight=1 --force-weight=0.3 --stress-weight=0.09 \
    --radial-smooth=0.02 --radial-smooth-grid=32 --gate-x-l2=0 \
    --displacement=1e-6 --abs-tolerance=2e-7 --rel-tolerance=2e-4 \
    > "$test_dir/$map.gate.chart-fd.log" 2>&1

  wide_model="$test_dir/$map.wide.mtp"
  make_model "$map" 6 "$wide_model"
  "$bin" check-loss-gradient-dev "$wide_model" "$cfg" "${smooth_opts[@]}" \
    --coeff-start=0 --coeff-end=1 > "$test_dir/$map.wide.log" 2>&1
  narrow_loss=$(field base_loss "$test_dir/$map.layout.log")
  wide_loss=$(field base_loss "$test_dir/$map.wide.log")
  awk -v a="$narrow_loss" -v b="$wide_loss" 'BEGIN {
    d=a-b; if(d<0)d=-d; if(!(a>0) || d>1e-12+1e-10*a) exit 1
  }'

  # Exact zero remains a valid physical row, with no epsilon injection.
  zero_model="$test_dir/$map.zero-row.mtp"
  awk '/^two_layer_gate_radial_mixing_coeffs[[:space:]]*=/ {
      line=$0; sub(/^[^{]*\{/, "", line); sub(/\}.*/, "", line)
      n=split(line,v,","); printf "two_layer_gate_radial_mixing_coeffs = {"
      for(i=1;i<=n;++i) printf "%s%s", (i>1 ? "," : ""), (i<=4 ? "0" : v[i])
      print "}"; next
    } {print}' "$model" > "$zero_model"
  "$bin" check-radial-weight-norm-roundtrip-dev "$zero_model" \
    > "$test_dir/$map.zero-row.roundtrip.log" 2>&1
  test "$(field zero_fallback_rows "$test_dir/$map.zero-row.roundtrip.log")" -eq 1
  "$bin" check-radial-weight-norm-gradient-dev "$zero_model" "$cfg" \
    "${smooth_opts[@]}" --radial-block=outer --mu=0 \
    > "$test_dir/$map.zero-row.gradient.log" 2>&1

  # An actual optimizer construction must count W+gate rows and exclude
  # obsolete R rows. Its periodic linear solve must preserve physical W.
  "$bin" train "$zero_model" "$cfg" --max-iter=2 --skip-preinit \
    --init-params=same --radial-parameterization=weight-norm \
    --bfgs-coordinate-normalization=none \
    --energy-weight=1 --force-weight=0.3 --stress-weight=0.09 \
    --radial-smooth=0.02 --radial-smooth-grid=32 --gate-x-l2=0 \
    --trained-pot-name="$test_dir/$map.trained.mtp" \
    --curr-pot-name="$test_dir/$map.current.mtp" \
    > "$test_dir/$map.train.log" 2>&1
  test "$(field outer_rows "$test_dir/$map.train.log")" -eq 11
  test "$(field outer_zero_fallback_rows "$test_dir/$map.train.log")" -eq 1
  test "$(field gate1_rows "$test_dir/$map.train.log")" -eq 4
  grep -q 'step=1 ' "$test_dir/$map.train.log"
  awk '/^two_layer_gate_radial_mixing_coeffs[[:space:]]*=/ {
    line=$0; sub(/^[^{]*\{/, "", line); split(line,v,",")
    for(i=1;i<=4;++i) total+=v[i]*v[i]
    found=1
  } END { if(!found || !(total>0)) exit 1 }' "$test_dir/$map.trained.mtp"
  printf 'PASS pure-W regularization/weight-norm scaling_map=%s\n' "$map"
done
printf 'Evidence: %s\n' "$test_dir"
