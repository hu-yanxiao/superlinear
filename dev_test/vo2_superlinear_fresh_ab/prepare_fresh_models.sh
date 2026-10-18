#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
SUPERLINEAR_BIN=${SUPERLINEAR_BIN:-$(cd "$SCRIPT_DIR/../.." && pwd)/bin/mlp-sus2}
RELEASE3_BIN=${RELEASE3_BIN:-/work/phy-weigw/app/release3/mlp-sus2}
TRAIN_CFG=${TRAIN_CFG:-/work/phy-weigw/hyx/vo2/9.1/train.cfg}
SEED=${SEED:-2026090201}
MIX_SCALE=${MIX_SCALE:-1e-4}
SEED_DIR=${SEED_DIR:-"$PWD/seed"}
LIVE_VO2_DIR=/work/phy-weigw/hyx/vo2/9.1

reject_live_path() {
  local candidate=$1
  case "$candidate/" in
    "$LIVE_VO2_DIR/"*)
      echo "refusing to create fresh-A/B artifacts inside the live VO2 directory: $candidate" >&2
      exit 2
      ;;
  esac
}

WORK_DIR=$(pwd -P)
SEED_CANONICAL=$(python3 -c 'import os, sys; print(os.path.realpath(sys.argv[1]))' "$SEED_DIR")
reject_live_path "$WORK_DIR"
reject_live_path "$SEED_CANONICAL"

for required in "$SUPERLINEAR_BIN" "$RELEASE3_BIN" "$TRAIN_CFG"
do
  if [[ ! -e "$required" ]]; then
    echo "missing required input: $required" >&2
    exit 2
  fi
done
if [[ -e "$SEED_CANONICAL" ]]; then
  echo "refusing to reuse existing seed directory: $SEED_DIR" >&2
  exit 2
fi
SEED_DIR=$SEED_CANONICAL
mkdir -p "$SEED_DIR"

COMMON=(
  --init-sh
  --species-count=2
  --scaling-map=K
  --l-max=3
  --k-max=8
  --body-order=6
  --body-l-max=3,3,3,2,1
  --cutoff=6.5
  --radial-basis-size=10
  --radial-basis-type=RBChebyshev_sss
  --sh-factor-pruning=q-total
  --two-layer-gate
  --two-layer-gate-mode=mu-scalar-full
  --two-layer-gate-body-order=4
  --two-layer-gate-k-rank=4
  --two-layer-gate-shared-radial
  --outer-k-rank=2
  --init-params=random
  --skip-preinit
  --max-iter=0
  --scal-range=3.5,1.5
)

run_init() {
  local label=$1
  local mixing=$2
  local scale=$3
  local args=("${COMMON[@]}")
  if [[ "$mixing" == 1 ]]; then
    args+=(--two-layer-gate-radial-mixing=full)
    args+=(--two-layer-gate-radial-mixing-init-scale="$scale")
  fi
  "$SUPERLINEAR_BIN" train "$TRAIN_CFG" \
    "${args[@]}" \
    --init-random-seed="$SEED" \
    --inline-sh-model="$SEED_DIR/${label}.topology.sus2" \
    --trained-pot-name="$SEED_DIR/${label}.raw.sus2" \
    >"$SEED_DIR/${label}.init.log" 2>&1
  test -s "$SEED_DIR/${label}.raw.sus2"
  grep -q "seed=${SEED}" "$SEED_DIR/${label}.init.log"
  grep -q "Deterministic fresh initialized model saved" "$SEED_DIR/${label}.init.log"
  if grep -q "Initial untrained model warmup" "$SEED_DIR/${label}.init.log"; then
    echo "raw model unexpectedly ran a training warmup: $label" >&2
    exit 1
  fi
}

run_init A 0 0
run_init B0 1 0
run_init B 1 "$MIX_SCALE"

python3 "$SCRIPT_DIR/audit_fresh_models.py" \
  "$SEED_DIR/A.raw.sus2" "$SEED_DIR/B0.raw.sus2" "$SEED_DIR/B.raw.sus2" \
  --mix-scale "$MIX_SCALE" | tee "$SEED_DIR/model_audit.txt"

awk '{print} /^END_CFG/{exit}' "$TRAIN_CFG" >"$SEED_DIR/smoke.cfg"
"$RELEASE3_BIN" calc-efs "$SEED_DIR/A.raw.sus2" "$SEED_DIR/smoke.cfg" \
  "$SEED_DIR/release3_A.efs.cfg" >"$SEED_DIR/release3_A.calc.log" 2>&1
"$SUPERLINEAR_BIN" calc-efs "$SEED_DIR/A.raw.sus2" "$SEED_DIR/smoke.cfg" \
  "$SEED_DIR/superlinear_A.efs.cfg" >"$SEED_DIR/superlinear_A.calc.log" 2>&1
"$SUPERLINEAR_BIN" calc-efs "$SEED_DIR/B0.raw.sus2" "$SEED_DIR/smoke.cfg" \
  "$SEED_DIR/superlinear_B0.efs.cfg" >"$SEED_DIR/superlinear_B0.calc.log" 2>&1
"$SUPERLINEAR_BIN" calc-efs "$SEED_DIR/B.raw.sus2" "$SEED_DIR/smoke.cfg" \
  "$SEED_DIR/superlinear_B.efs.cfg" >"$SEED_DIR/superlinear_B.calc.log" 2>&1

python3 "$SCRIPT_DIR/compare_cfg_numeric.py" \
  "$SEED_DIR/release3_A.efs.cfg" "$SEED_DIR/superlinear_A.efs.cfg" \
  --tolerance 1e-12 | tee "$SEED_DIR/release3_superlinear_A_parity.txt"
python3 "$SCRIPT_DIR/compare_cfg_numeric.py" \
  "$SEED_DIR/superlinear_A.efs.cfg" "$SEED_DIR/superlinear_B0.efs.cfg" \
  --tolerance 1e-12 | tee "$SEED_DIR/A_B0_parity.txt"
python3 "$SCRIPT_DIR/compare_cfg_numeric.py" \
  "$SEED_DIR/superlinear_B0.efs.cfg" "$SEED_DIR/superlinear_B.efs.cfg" \
  --require-nonzero --min-efs-effect=1e-15 | tee "$SEED_DIR/B0_B_effect.txt"

{
  date -u '+%Y-%m-%dT%H:%M:%SZ'
  echo "seed=$SEED"
  echo "mix_scale=$MIX_SCALE"
  sha256sum "$SUPERLINEAR_BIN" "$RELEASE3_BIN" "$TRAIN_CFG"
  sha256sum "$SEED_DIR/A.raw.sus2" "$SEED_DIR/B0.raw.sus2" "$SEED_DIR/B.raw.sus2"
} >"$SEED_DIR/metadata.txt"

echo "fresh VO2 A/B0/B seed preparation passed: $SEED_DIR"
