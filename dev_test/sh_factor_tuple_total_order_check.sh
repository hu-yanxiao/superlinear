#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 /path/to/mlp-sus2" >&2
  exit 2
fi

BIN=$1
WORKDIR=$(mktemp -d "${TMPDIR:-/tmp}/sus2sh-factor-order-XXXXXX")
trap 'rm -rf "$WORKDIR"' EXIT

MODEL="$WORKDIR/l4k4_b4.sus2"
"$BIN" init-sh "$MODEL" \
  --species-count=2 \
  --l-max=4 \
  --k-max=4 \
  --body-order=4 \
  --body-l-max=4,4,4,4 \
  --sh-factor-pruning=q-total \
  --write-sh-scalar-info \
  --cutoff=7.5 \
  --radial-basis-size=10 \
  --radial-basis-type=RBChebyshev_sss \
  --init-params=same >/dev/null

# q = k * (lmax + 1) + l. This scalar uses (k,l) factors
# (0,0), (0,1), (3,1). It is allowed by angular coupling:
# |0-1| <= 1 <= 0+1 and 0+1+1 is even. The old component-wise
# tuple pruning rejected it because l first increases and then k increases.
grep -q "{4, 0, 1, 16, -1, -1, 0}" "$MODEL"
