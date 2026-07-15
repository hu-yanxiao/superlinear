#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 /path/to/mlp-sus2" >&2
  exit 2
fi

BIN=$1
WORKDIR=$(mktemp -d "${TMPDIR:-/tmp}/sus2sh-format-XXXXXX")
trap 'rm -rf "$WORKDIR"' EXIT

COMMON_OPTS=(
  --species-count=2
  --l-max=4
  --k-max=4
  --body-order=6
  --body-l-max=4,4,4,2,1
  --cutoff=7.5
  --radial-basis-size=10
  --radial-basis-type=RBChebyshev_sss
  --init-params=same
)

LEGACY_MODEL="$WORKDIR/legacy_plain.mtp"
"$BIN" init-sh "$LEGACY_MODEL" "${COMMON_OPTS[@]}" \
  --sh-factor-pruning=legacy >/dev/null
grep -q "alpha_scalar_moments = 1004" "$LEGACY_MODEL"
grep -q "alpha_moments_count = 1960" "$LEGACY_MODEL"
if grep -q "sh_scalar_info" "$LEGACY_MODEL"; then
  echo "plain SH init-sh should not write sh_scalar_info metadata" >&2
  exit 1
fi

DEFAULT_MODEL="$WORKDIR/default_qtotal_plain.mtp"
EXPLICIT_TOTAL_MODEL="$WORKDIR/explicit_qtotal_plain.mtp"
"$BIN" init-sh "$DEFAULT_MODEL" "${COMMON_OPTS[@]}" >/dev/null
"$BIN" init-sh "$EXPLICIT_TOTAL_MODEL" "${COMMON_OPTS[@]}" \
  --sh-factor-pruning=q-total >/dev/null
grep -q "alpha_scalar_moments = 2059" "$DEFAULT_MODEL"
grep -q "alpha_moments_count = 3313" "$DEFAULT_MODEL"
cmp "$DEFAULT_MODEL" "$EXPLICIT_TOTAL_MODEL"

TOTAL_MODEL="$WORKDIR/qtotal_metadata.mtp"
"$BIN" init-sh "$TOTAL_MODEL" "${COMMON_OPTS[@]}" \
  --sh-factor-pruning=q-total \
  --write-sh-scalar-info >/dev/null
grep -q "alpha_scalar_moments = 2059" "$TOTAL_MODEL"
grep -q "alpha_moments_count = 3313" "$TOTAL_MODEL"
grep -q "sh_scalar_info_count = 2059" "$TOTAL_MODEL"
grep -q "{4, 0, 1, 16, -1, -1, 0}" "$TOTAL_MODEL"
