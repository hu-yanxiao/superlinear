#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 /path/to/mlp-sus2" >&2
  exit 2
fi

BIN=$1
REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)
WORKDIR=$(mktemp -d "${TMPDIR:-/tmp}/sus2sh-product-rows-XXXXXX")
trap 'rm -rf "$WORKDIR"' EXIT

MODEL="$WORKDIR/qtotal_l3k2.sus2"
FLAT_OUT="$WORKDIR/flat.cfg"
ROWS_OUT="$WORKDIR/rows.cfg"
ROWS_LOG="$WORKDIR/rows.log"

"$BIN" init-sh "$MODEL" \
  --species-count=2 \
  --l-max=3 \
  --k-max=2 \
  --body-order=5 \
  --body-l-max=3,3,2,2 \
  --sh-factor-pruning=q-total \
  --cutoff=7.5 \
  --radial-basis-size=10 \
  --radial-basis-type=RBChebyshev_sss \
  --init-params=same >/dev/null

SUS2_SH_PRODUCT_ROWS=0 "$BIN" calc-efs "$MODEL" "$REPO_ROOT/dev_test/NaCl_small.cfgs" "$FLAT_OUT" >/dev/null
SUS2_SH_PRODUCT_ROWS=1 SUS2_SH_PRODUCT_ROWS_TRACE=1 \
  "$BIN" calc-efs "$MODEL" "$REPO_ROOT/dev_test/NaCl_small.cfgs" "$ROWS_OUT" >"$ROWS_LOG"

grep -q "SUS2-SH product rows enabled" "$ROWS_LOG"

python3 - "$FLAT_OUT" "$ROWS_OUT" <<'PY'
import math
import re
import sys

float_re = re.compile(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?")

def values(path):
    with open(path, "r", encoding="utf-8") as fh:
        return [float(x) for x in float_re.findall(fh.read())]

flat = values(sys.argv[1])
rows = values(sys.argv[2])
if len(flat) != len(rows):
    raise SystemExit(f"numeric token count mismatch: {len(flat)} vs {len(rows)}")

max_abs = 0.0
for i, (a, b) in enumerate(zip(flat, rows)):
    diff = abs(a - b)
    max_abs = max(max_abs, diff)
    if diff > 1.0e-10:
        raise SystemExit(f"token {i} differs: {a} vs {b}, diff={diff}")

print(f"SH product row equivalence passed, max_abs={max_abs:.3e}")
PY
