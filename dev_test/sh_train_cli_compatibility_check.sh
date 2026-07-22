#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 /path/to/mlp-sus2" >&2
  exit 2
fi

bin=$1
repo_root=$(cd "$(dirname "$0")/.." && pwd)
tmp_dir=$(mktemp -d "${TMPDIR:-/tmp}/sus2sh-train-cli-XXXXXX")
trap 'rm -rf "$tmp_dir"' EXIT

one_cfg="$tmp_dir/one.cfg"
awk '{print} /^END_CFG/{exit}' "$repo_root/example/train.cfg" >"$one_cfg"

# A single positional argument without any explicitly supplied SH option is
# the historical malformed two-argument train invocation.  Merely having
# known option keys in the parser map must not turn it into inline init-sh.
# The historical command dispatcher reports this usage error in stdout but
# returns process status 0, so assert the observable CLI behavior instead of
# relying on the legacy exit status.
"$bin" train "$one_cfg" >"$tmp_dir/missing-model.log" 2>&1 || true
grep -q "at least 2 arguments are required" "$tmp_dir/missing-model.log"
if grep -q "inline initial model written" "$tmp_dir/missing-model.log"; then
  echo "single-argument train created an implicit model without an SH option" >&2
  exit 1
fi

common_init=(
  --species-count=2
  --l-max=1
  --k-max=1
  --body-order=3
  --body-l-max=1,1,1,1
  --radial-basis-size=3
  --cutoff=5.0
  --init-params=same
)

# Explicit inline mode remains supported.
if ! "$bin" train "$one_cfg" --init-sh \
    "${common_init[@]}" \
    --inline-sh-model="$tmp_dir/explicit-init.mtp" \
    --trained-pot-name="$tmp_dir/explicit-trained.mtp" \
    --max-iter=0 >"$tmp_dir/explicit.log" 2>&1; then
  cat "$tmp_dir/explicit.log" >&2
  exit 1
fi
test -s "$tmp_dir/explicit-init.mtp"
grep -q "inline initial model written" "$tmp_dir/explicit.log"

# A genuinely supplied SH construction option also keeps the documented
# implicit one-argument mode.  outer-k-rank=0 is intentionally a non-empty,
# explicit value even though it selects the default mathematical model.
if ! "$bin" train "$one_cfg" \
    "${common_init[@]}" \
    --outer-k-rank=0 \
    --inline-sh-model="$tmp_dir/implicit-init.mtp" \
    --trained-pot-name="$tmp_dir/implicit-trained.mtp" \
    --max-iter=0 >"$tmp_dir/implicit.log" 2>&1; then
  cat "$tmp_dir/implicit.log" >&2
  exit 1
fi
test -s "$tmp_dir/implicit-init.mtp"
grep -q "inline initial model written" "$tmp_dir/implicit.log"

echo "SUS2-SH train CLI compatibility check passed"
