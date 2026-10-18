#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 /path/to/mlp-sus2" >&2
  exit 2
fi

BIN=$1
REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)
WORKDIR=$(mktemp -d "${TMPDIR:-/tmp}/sus2sh-site-der-cache-XXXXXX")
trap 'rm -rf "$WORKDIR"' EXIT

MODEL="$WORKDIR/qtotal_l3k2.sus2"
BASE_OUT="$WORKDIR/base.cfg"
CACHE_OUT="$WORKDIR/cache.cfg"
CACHE_LOG="$WORKDIR/cache.log"

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

SUS2_SH_SITE_DER_CACHE=0 "$BIN" calc-efs "$MODEL" "$REPO_ROOT/dev_test/NaCl_small.cfgs" "$BASE_OUT" >/dev/null
SUS2_SH_SITE_DER_CACHE=1 SUS2_SH_SITE_DER_CACHE_TRACE=1 \
  "$BIN" calc-efs "$MODEL" "$REPO_ROOT/dev_test/NaCl_small.cfgs" "$CACHE_OUT" >"$CACHE_LOG"

grep -q "SUS2-SH site derivative cache enabled" "$CACHE_LOG"

python3 - "$BASE_OUT" "$CACHE_OUT" <<'PY'
import re
import sys

float_re = re.compile(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?")

def values(path):
    with open(path, "r", encoding="utf-8") as fh:
        return [float(x) for x in float_re.findall(fh.read())]

base = values(sys.argv[1])
cache = values(sys.argv[2])
if len(base) != len(cache):
    raise SystemExit(f"numeric token count mismatch: {len(base)} vs {len(cache)}")

max_abs = 0.0
for i, (a, b) in enumerate(zip(base, cache)):
    diff = abs(a - b)
    max_abs = max(max_abs, diff)
    if diff > 1.0e-10:
        raise SystemExit(f"token {i} differs: {a} vs {b}, diff={diff}")

print(f"SH site derivative cache equivalence passed, max_abs={max_abs:.3e}")
PY
