#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 /path/to/mlp-sus2" >&2
  exit 2
fi

BIN=$(cd "$(dirname "$1")" && pwd)/$(basename "$1")
REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)
WORKDIR=$(mktemp -d "${TMPDIR:-/tmp}/sus2sh-accum-skip-site-ders-XXXXXX")
trap 'rm -rf "$WORKDIR"' EXIT

MODEL="$WORKDIR/qtotal_l3k2.sus2"

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

run_train() {
  local mode=$1
  local dir=$2
  mkdir -p "$dir"
  (
    cd "$dir"
    SUS2_SH_ACCUM_SKIP_SITE_DERS="$mode" \
      "$BIN" train "$MODEL" "$REPO_ROOT/dev_test/NaCl_small.cfgs" \
        --max-iter=1 \
        --curr-pot-name=current.sus2 \
        --trained-pot-name=p.sus2 \
        --skip-preinit \
        --init-params=same >logout 2>&1
  )
}

run_train 0 "$WORKDIR/base"
run_train 1 "$WORKDIR/skip"

python3 - "$WORKDIR/base/current.sus2" "$WORKDIR/skip/current.sus2" <<'PY'
import re
import sys

float_re = re.compile(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?")

def values(path):
    with open(path, "r", encoding="utf-8") as fh:
        return [float(x) for x in float_re.findall(fh.read())]

base = values(sys.argv[1])
skip = values(sys.argv[2])
if len(base) != len(skip):
    raise SystemExit(f"numeric token count mismatch: {len(base)} vs {len(skip)}")

max_abs = 0.0
for i, (a, b) in enumerate(zip(base, skip)):
    diff = abs(a - b)
    max_abs = max(max_abs, diff)
    if diff > 1.0e-10:
        raise SystemExit(f"token {i} differs: {a} vs {b}, diff={diff}")

print(f"SH accum skip site derivatives train check passed, max_abs={max_abs:.3e}")
PY
