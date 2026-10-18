#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

bin="${MLP_BIN:-./bin/mlp-sus2}"
tmp_dir=".local_tmp/sh_zbl_continue_train_roundtrip"
mkdir -p "$tmp_dir"

base_model="$tmp_dir/base_two_layer.sus2"
zbl_init_model="$tmp_dir/zbl_two_layer_init.sus2"
train_cfg="$tmp_dir/train.cfg"
short_cfg="$tmp_dir/short_ho.cfg"
base_pred="$tmp_dir/base_pred.cfg"
zbl_pred="$tmp_dir/zbl_pred.cfg"
continued_model="$tmp_dir/continued_zbl.sus2"
current_model="$tmp_dir/current_zbl.sus2"
train_log="$tmp_dir/train.log"

rm -f "$base_model" "$zbl_init_model" "$train_cfg" "$short_cfg" \
  "$base_pred" "$zbl_pred" "$continued_model" "$current_model" "$train_log"

awk '{print} /^END_CFG/{exit}' dev_test/NaCl_small.cfgs > "$train_cfg"

"$bin" init-sh "$base_model" \
  --species-count=2 \
  --l-max=1 \
  --k-max=1 \
  --body-order=2 \
  --two-layer-gate \
  --two-layer-gate-shared-radial \
  --radial-basis-size=3 \
  --cutoff=3.0

"$bin" init-sh "$zbl_init_model" \
  --species-count=2 \
  --l-max=1 \
  --k-max=1 \
  --body-order=2 \
  --two-layer-gate \
  --two-layer-gate-shared-radial \
  --radial-basis-size=3 \
  --cutoff=3.0 \
  --zbl-inner=0.4 \
  --zbl-outer=0.8 \
  --zbl-elements=H,O \
  --zbl-typewise-cutoff-factor=0.7

grep -q "zbl_enabled = true" "$zbl_init_model"
grep -q "zbl_inner = 0" "$zbl_init_model"
grep -q "zbl_outer = 8.000000000000000e-01" "$zbl_init_model"
grep -q "zbl_typewise_cutoff_enabled = true" "$zbl_init_model"
grep -q "zbl_atomic_numbers = {1, 8}" "$zbl_init_model"

cat > "$short_cfg" <<'CFG'
BEGIN_CFG
 Size
    2
 Supercell
    10.0 0.0 0.0
    0.0 10.0 0.0
    0.0 0.0 10.0
 AtomData:  id type       cartes_x      cartes_y      cartes_z           fx          fy          fz
    1 0  0.0 0.0 0.0  0.0 0.0 0.0
    2 1  0.5 0.0 0.0  0.0 0.0 0.0
 Energy
    0.0
 PlusStress:  xx          yy          zz          yz          xz          xy
    0.0 0.0 0.0 0.0 0.0 0.0
END_CFG
CFG

"$bin" calc-efs "$base_model" "$short_cfg" "$base_pred"
"$bin" calc-efs "$zbl_init_model" "$short_cfg" "$zbl_pred"

python3 - "$base_pred" "$zbl_pred" <<'PY'
import sys

def read_energy(path):
    with open(path) as f:
        lines = iter(f)
        for line in lines:
            if line.strip() == "Energy":
                return float(next(lines).strip())
    raise SystemExit(f"missing Energy in {path}")

base = read_energy(sys.argv[1])
zbl = read_energy(sys.argv[2])
if not (zbl - base > 1.0):
    raise SystemExit(f"expected ZBL to increase short H-O energy, base={base}, zbl={zbl}")
PY

"$bin" train "$base_model" "$train_cfg" \
  --trained-pot-name="$continued_model" \
  --curr-pot-name="$current_model" \
  --max-iter=1 \
  --skip-preinit \
  --energy-weight=1 \
  --force-weight=0 \
  --stress-weight=0 \
  --zbl-inner=0.4 \
  --zbl-outer=0.8 \
  --zbl-elements=H,O \
  --zbl-typewise-cutoff-factor=0.7 > "$train_log"

grep -q "zbl_enabled = true" "$continued_model"
grep -q "zbl_inner = 0.000000000000000e+00" "$continued_model"
grep -q "zbl_outer = 8.000000000000000e-01" "$continued_model"
grep -q "zbl_typewise_cutoff_enabled = true" "$continued_model"
grep -q "zbl_atomic_numbers = {1, 8}" "$continued_model"

"$bin" calc-efs "$continued_model" "$short_cfg" "$zbl_pred"
