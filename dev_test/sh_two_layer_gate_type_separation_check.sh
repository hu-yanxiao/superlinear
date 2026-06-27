#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

tmp_dir=".codex_tmp/sh_two_layer_gate_type_separation"
mkdir -p "$tmp_dir"

model="$tmp_dir/gate_type_separation.mtp"
train="$tmp_dir/one.cfg"
log="$tmp_dir/check.log"
rm -f "$model" "$train" "$log"

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

./bin/mlp-sus2 init-sh "$model" \
  --species-count=2 \
  --l-max=1 \
  --k-max=1 \
  --body-order=3 \
  --body-l-max=1,1,1,1 \
  --radial-basis-size=3 \
  --cutoff=5.0 \
  --write-sh-scalar-info \
  --two-layer-gate \
  --two-layer-gate-mode=mu-scalar-full >/dev/null

./bin/mlp-sus2 check-two-layer-gate-type-separation-dev "$model" "$train" \
  --max-configs=1 \
  --max-atoms=2 \
  --gate-type-scale=2.0 \
  --outer-type-probe=3.0 \
  --abs-tolerance=1e-12 \
  --rel-tolerance=1e-10 >"$log" 2>&1

grep -q "checked_outer_pairs=" "$log"
grep -q "checked_gate_scalars=" "$log"

echo "release two-layer gate type separation check: PASS"
