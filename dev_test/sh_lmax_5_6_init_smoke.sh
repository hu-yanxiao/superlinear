#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 /path/to/mlp-sus2" >&2
  exit 2
fi

BIN=$1
WORKDIR=$(mktemp -d "${TMPDIR:-/tmp}/sus2sh-lmax-XXXXXX")
trap 'rm -rf "$WORKDIR"' EXIT

run_case() {
  local lmax=$1
  local body_lmax=$2
  local expected_radial=$3
  local model="$WORKDIR/l${lmax}.sus2"

  "$BIN" init-sh "$model" \
    --species-count=2 \
    --l-max="$lmax" \
    --k-max=3 \
    --body-order=6 \
    --body-l-max="$body_lmax" \
    --cutoff=6.0 \
    --radial-basis-size=4 \
    --radial-basis-type=RBChebyshev_sss_rational \
    --init-params=same >/dev/null

  grep -q "sh_l_max = $lmax" "$model"
  grep -q "sh_k_max = 3" "$model"
  grep -q "radial_funcs_count = $expected_radial" "$model"
  grep -Eq "alpha_moments_count = [1-9][0-9]*" "$model"
  grep -Eq "alpha_scalar_moments = [1-9][0-9]*" "$model"
}

run_case 5 "5,5,5,3,2" 18
run_case 6 "6,6,6,4,2" 21
