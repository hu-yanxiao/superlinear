#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
task_bin=${1:-./bin/mlp-sus2}
task_tmp=$(mktemp -d "${TMPDIR:-/tmp}/sus2sh-pure-w-contract-XXXXXX")
if [[ ${SUS2_TEST_KEEP_TMP:-0} == 1 ]]; then
  echo "keeping test directory: $task_tmp"
else
  trap 'rm -rf "$task_tmp"' EXIT
fi

task_common=(--species-count=2 --l-max=1 --k-max=2 --body-order=3
             --body-l-max=1,1,1,1 --radial-basis-size=3 --cutoff=5 --scaling-map=LK
             --two-layer-gate --two-layer-gate-mode=mu-scalar-full
             --two-layer-gate-body-order=3 --two-layer-gate-k-rank=2
             --two-layer-gate-radial-mixing=full --write-sh-scalar-info)
"$task_bin" init-sh "$task_tmp/pure.mtp" "${task_common[@]}" \
  --superlinear-mode=pure-w-channelwise --outer-k-rank=3 > "$task_tmp/pure.init.log"
"$task_bin" init-sh "$task_tmp/rank1.mtp" "${task_common[@]}" \
  --superlinear-mode=pure-w-channelwise --outer-k-rank=1 > "$task_tmp/rank1.init.log"
"$task_bin" init-sh "$task_tmp/legacy.mtp" "${task_common[@]}" \
  --outer-k-rank=2 > "$task_tmp/legacy.init.log"

python3 - "$task_tmp" <<'PY'
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
text = (root / "pure.mtp").read_text()
rank1 = (root / "rank1.mtp").read_text()
rank1 = re.sub(r"(?m)^outer_k_rank\s*=.*\n", "", rank1)
(root / "implicit-rank1.mtp").write_text(rank1)
cases = {
    "mismatched-modes": text.replace("outer_radial_mode = pure-matrix", "outer_radial_mode = legacy-plus-mixer"),
    "unknown-mode": text.replace("outer_channel_mode = same-channel", "outer_channel_mode = invalid"),
    "missing-mode": re.sub(r"(?m)^outer_radial_mode\s*=.*\n", "", text),
    "nonzero-p": text.replace("outer_k_rank1_weight_count = 0", "outer_k_rank1_weight_count = 1")
                     .replace("outer_k_rank1_weights = {}", "outer_k_rank1_weights = {1}"),
    "hidden-p": text.replace("outer_k_rank1_weights = {}", "outer_k_rank1_weights = {1}"),
    "huge-outer-rank": re.sub(r"(?m)^outer_k_rank\s*=.*$", "outer_k_rank = 2147483647", text),
    "overflow-outer-range": re.sub(r"(?m)^outer_k_rank1_basic_begin\s*=.*$",
                                    "outer_k_rank1_basic_begin = 2147483640", text),
}
def scalar(name):
    return int(re.search(rf"(?m)^\s*{name}\s*=\s*(\d+)\s*$", text)[1])
match = re.search(r"(?m)^(\s*sh_products\s*=\s*)\{(.*)\}\s*$", text)
products = [entry.split(",") for entry in re.findall(r"\{([^{}]+)\}", match[2])]
start = scalar("outer_k_rank1_product_begin")
begin = scalar("outer_k_rank1_basic_begin")
width, lmax = scalar("outer_k_rank"), scalar("sh_l_max")
labels, peers = {}, {}
node = begin
for l in range(lmax, -1, -1):
    for channel in range(width - 1, -1, -1):
        for m in range(-l, l + 1):
            labels[node] = (channel, l, m)
            peers[channel, l, m] = node
            node += 1
for product in products[start:]:
    left, right, target = map(int, product[:3])
    if left in labels and right in labels:
        channel, l, m = labels[right]
        product[1] = str(peers[(channel + 1) % width, l, m])
        assert int(product[1]) < target  # preserve valid topological ordering
        break
else:
    raise AssertionError("test fixture has no basic outer product")
replacement = match[1] + "{" + ", ".join("{" + ",".join(p) + "}" for p in products) + "}"
cases["cross-channel-product"] = text[:match.start()] + replacement + "\n" + text[match.end():]
for label, value in cases.items():
    assert value != text, f"fixture mutation failed: {label}"
    (root / f"{label}.mtp").write_text(value)
PY

"$task_bin" roundtrip-model-dev "$task_tmp/implicit-rank1.mtp" \
  "$task_tmp/implicit-rank1.roundtrip.mtp" > "$task_tmp/rank1.roundtrip.log"
for task_case in mismatched-modes unknown-mode missing-mode nonzero-p hidden-p \
                 huge-outer-rank overflow-outer-range cross-channel-product; do
  if "$task_bin" roundtrip-model-dev "$task_tmp/$task_case.mtp" \
      "$task_tmp/$task_case.roundtrip.mtp" > "$task_tmp/$task_case.log" 2>&1; then
    echo "invalid pure model accepted: $task_case" >&2
    exit 1
  fi
done
grep -q 'same-channel outer graph contains a cross-channel product' "$task_tmp/cross-channel-product.log"
grep -q 'weight count is inconsistent' "$task_tmp/nonzero-p.log"
grep -q 'outer weights must be an empty list' "$task_tmp/hidden-p.log"
grep -q 'outer primitive count exceeds supported index range' "$task_tmp/huge-outer-rank.log"
grep -q 'contracted primitive range is invalid\|primitive range is invalid' "$task_tmp/overflow-outer-range.log"

# Link a dedicated process against existing serial build objects; no make or
# source recompilation is performed, so other tests can use the binary safely.
task_objects=()
while IFS= read -r task_object; do
  if [[ $task_object != obj/ser/mlp/mlp.cpp.o ]]; then
    task_objects+=("$task_object")
  fi
done < <(rg --files -uuu obj/ser -g '*.o' | sort)
if [[ ${#task_objects[@]} == 0 ]]; then
  echo "serial build objects required under obj/ser; build the tested binary first" >&2
  exit 2
fi
task_libs=(-lblas -llapack)
if [[ $(uname -s) == Darwin ]]; then task_libs=(-framework Accelerate); fi
if [[ -n ${SUS2_TEST_LDLIBS:-} ]]; then read -r -a task_libs <<< "$SUS2_TEST_LDLIBS"; fi
"${CXX:-c++}" -std=c++11 -DMLIP_DEV -I./src -I./dev_src -I./cblas \
  dev_test/pure_w_model_contract_check.cpp "${task_objects[@]}" "${task_libs[@]}" \
  -o "$task_tmp/core-contract" > "$task_tmp/compile.log" 2>&1
"$task_tmp/core-contract" "$task_tmp/pure.mtp" "$task_tmp/legacy.mtp" \
  "$task_tmp/implicit-rank1.mtp" "$task_tmp/physical.roundtrip.mtp" \
  > "$task_tmp/core-contract.log" 2>&1
cat "$task_tmp/core-contract.log"
echo "pure_w_model_contract_check_ok"
