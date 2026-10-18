#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 /path/to/mlp-sus2" >&2
  exit 2
fi

BIN=$1
REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)
WORKDIR=$(mktemp -d "${TMPDIR:-/tmp}/sus2sh-rbasis-init-XXXXXX")
trap 'rm -rf "$WORKDIR"' EXIT

run_case() {
  local requested=$1
  local expected=$2
  local model="$WORKDIR/${requested}.sus2"
  local out="$WORKDIR/${requested}.cfg"

  "$BIN" init-sh "$model" \
    --species-count=2 \
    --l-max=3 \
    --k-max=3 \
    --body-order=5 \
    --body-l-max=3,3,2,2 \
    --cutoff=7.5 \
    --radial-basis-size=6 \
    --radial-basis-type="$requested" \
    --init-params=same >/dev/null

  grep -q "radial_basis_type = $expected" "$model"
  "$BIN" calc-efs "$model" "$REPO_ROOT/dev_test/NaCl_small.cfgs" "$out" >/dev/null
}

run_case RBLaguerre_log1p RBLaguerre_log1p
run_case laguerre_log1p RBLaguerre_log1p
run_case RBJacobi_sss RBJacobi_sss
run_case jacobi_sss RBJacobi_sss

echo "SH radial basis init smoke passed"
