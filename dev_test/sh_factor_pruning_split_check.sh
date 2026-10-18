#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 /path/to/mlp-sus2" >&2
  exit 2
fi

BIN=$1
WORKDIR=$(mktemp -d "${TMPDIR:-/tmp}/sus2sh-pruning-split-XXXXXX")
trap 'rm -rf "$WORKDIR"' EXIT

COMMON_OPTS=(
  --species-count=2
  --l-max=4
  --k-max=4
  --body-order=4
  --body-l-max=4,4,4,4
  --cutoff=7.5
  --radial-basis-size=6
  --radial-basis-type=RBChebyshev_sss
  --init-params=same
  --write-sh-scalar-info
)

for mode in q-total legacy same-k; do
  "$BIN" init-sh "$WORKDIR/$mode.sus2" "${COMMON_OPTS[@]}" \
    --sh-factor-pruning="$mode" >/dev/null
done

"$BIN" init-sh "$WORKDIR/default.sus2" "${COMMON_OPTS[@]}" >/dev/null
cmp "$WORKDIR/default.sus2" "$WORKDIR/q-total.sus2"

"$BIN" init-sh "$WORKDIR/gate_q-total.sus2" "${COMMON_OPTS[@]}" \
  --sh-factor-pruning=q-total \
  --two-layer-gate >/dev/null
"$BIN" init-sh "$WORKDIR/gate_legacy.sus2" "${COMMON_OPTS[@]}" \
  --sh-factor-pruning=q-total \
  --two-layer-gate \
  --two-layer-gate-factor-pruning=legacy >/dev/null
"$BIN" init-sh "$WORKDIR/gate_same-k.sus2" "${COMMON_OPTS[@]}" \
  --sh-factor-pruning=q-total \
  --two-layer-gate \
  --two-layer-gate-factor-pruning=same-k >/dev/null
"$BIN" init-sh "$WORKDIR/multi.sus2" "${COMMON_OPTS[@]}" \
  --sh-factor-pruning=q-total \
  --two-layer-gate \
  --two-layer-gate-factor-pruning=legacy \
  --multi-layer-gate \
  --second-layer-gate-factor-pruning=same-k >/dev/null
"$BIN" init-sh "$WORKDIR/multi_inherit.sus2" "${COMMON_OPTS[@]}" \
  --sh-factor-pruning=q-total \
  --two-layer-gate \
  --two-layer-gate-factor-pruning=legacy \
  --multi-layer-gate >/dev/null
"$BIN" init-sh "$WORKDIR/multi_independent.sus2" "${COMMON_OPTS[@]}" \
  --sh-factor-pruning=q-total \
  --two-layer-gate \
  --two-layer-gate-factor-pruning=same-k \
  --multi-layer-gate \
  --second-layer-gate-factor-pruning=q-total >/dev/null

python3 - "$WORKDIR" <<'PY'
import pathlib
import re
import sys

root = pathlib.Path(sys.argv[1])
lmax = 4
kmax = 4


def field(text, name):
    match = re.search(rf"^{re.escape(name)}\s*=\s*(\d+)\s*$", text, re.M)
    if not match:
        raise SystemExit(f"missing {name}")
    return int(match.group(1))


def model(path):
    text = path.read_text()
    info_match = re.search(r"^sh_scalar_info\s*=\s*\{(.*?)\}\s*$", text, re.M | re.S)
    if not info_match:
        raise SystemExit(f"missing sh_scalar_info in {path}")
    specs = [
        tuple(int(value.strip()) for value in entry.split(","))
        for entry in re.findall(r"\{([^{}]+)\}", info_match.group(1))
    ]
    return text, specs


def q_lk(qid):
    return lmax - qid // kmax, kmax - 1 - qid % kmax


def qids(spec):
    return spec[1 : spec[0]]


def selected(path, field_name):
    text, specs = model(path)
    match = re.search(rf"^{field_name}\s*=\s*\{{([^}}]*)\}}", text, re.M)
    if not match:
        raise SystemExit(f"missing {field_name} in {path}")
    indices = [int(value.strip()) for value in match.group(1).split(",") if value.strip()]
    return {specs[index] for index in indices}


def gate_closure(path, field_name):
    text, _ = model(path)
    index_match = re.search(rf"^{field_name}\s*=\s*\{{([^}}]*)\}}", text, re.M)
    mapping_match = re.search(r"^alpha_moment_mapping\s*=\s*\{([^}]*)\}", text, re.M)
    products_match = re.search(r"^sh_products\s*=\s*\{(.*?)\}\s*$", text, re.M | re.S)
    if not index_match or not mapping_match or not products_match:
        raise SystemExit(f"missing graph data in {path}")
    indices = [int(value.strip()) for value in index_match.group(1).split(",") if value.strip()]
    mapping = [int(value.strip()) for value in mapping_match.group(1).split(",") if value.strip()]
    products = []
    for entry in re.findall(r"\{([^{}]+)\}", products_match.group(1)):
        values = [value.strip() for value in entry.split(",")]
        products.append(tuple(int(value) for value in values[:3]))
    required = {mapping[index] for index in indices}
    required_product_count = 0
    for left, right, target in reversed(products):
        if target not in required:
            continue
        required_product_count += 1
        required.add(left)
        required.add(right)
    return len(required), required_product_count


texts = {}
spec_lists = {}
spec_sets = {}
for mode in ("q-total", "legacy", "same-k"):
    texts[mode], spec_lists[mode] = model(root / f"{mode}.sus2")
    spec_sets[mode] = set(spec_lists[mode])

if not spec_sets["same-k"] < spec_sets["legacy"] < spec_sets["q-total"]:
    raise SystemExit("expected strict same-k subset legacy subset q-total scalar hierarchy")

expected_legacy = set()
expected_same_k = set()
for spec in spec_lists["q-total"]:
    channels = [q_lk(qid) for qid in qids(spec)]
    if not any(
        l1 < l2 or k1 < k2
        for (l1, k1), (l2, k2) in zip(channels, channels[1:])
    ):
        expected_legacy.add(spec)
        if len({k for _, k in channels}) == 1:
            expected_same_k.add(spec)
if spec_sets["legacy"] != expected_legacy:
    raise SystemExit("legacy graph is not the exact legacy subset of q-total")
if spec_sets["same-k"] != expected_same_k:
    raise SystemExit("same-k graph lost or added an angular-coupling path")

for spec in spec_lists["q-total"]:
    ids = qids(spec)
    if list(ids) != sorted(ids):
        raise SystemExit(f"q-total emitted a noncanonical tuple: {spec}")

for spec in spec_lists["legacy"]:
    channels = [q_lk(qid) for qid in qids(spec)]
    if any(l1 < l2 or k1 < k2 for (l1, k1), (l2, k2) in zip(channels, channels[1:])):
        raise SystemExit(f"legacy emitted a tuple outside separate l/k monotonic order: {spec}")

for spec in spec_lists["same-k"]:
    channels = [q_lk(qid) for qid in qids(spec)]
    if len({k for _, k in channels}) != 1:
        raise SystemExit(f"same-k emitted a cross-k scalar: {spec}")

for graph_field in ("alpha_moments_count", "sh_product_count", "alpha_scalar_moments"):
    counts = [field(texts[mode], graph_field) for mode in ("same-k", "legacy", "q-total")]
    if not counts[0] <= counts[1] <= counts[2]:
        raise SystemExit(f"non-monotonic {graph_field}: {counts}")

if selected(root / "gate_q-total.sus2", "two_layer_gate_scalar_indices") != spec_sets["q-total"]:
    raise SystemExit("gate1 default did not inherit outer q-total")
if selected(root / "gate_legacy.sus2", "two_layer_gate_scalar_indices") != spec_sets["legacy"]:
    raise SystemExit("gate1 legacy indices do not match the direct legacy graph")
if selected(root / "gate_same-k.sus2", "two_layer_gate_scalar_indices") != spec_sets["same-k"]:
    raise SystemExit("gate1 same-k indices do not match the direct same-k graph")
if selected(root / "multi.sus2", "two_layer_gate_scalar_indices") != spec_sets["legacy"]:
    raise SystemExit("multi-gate gate1 legacy indices are incorrect")
if selected(root / "multi.sus2", "second_layer_gate_scalar_indices") != spec_sets["same-k"]:
    raise SystemExit("multi-gate gate2 same-k indices are incorrect")
if selected(root / "multi_inherit.sus2", "second_layer_gate_scalar_indices") != spec_sets["legacy"]:
    raise SystemExit("gate2 default did not inherit gate1 legacy pruning")
if selected(root / "multi_independent.sus2", "two_layer_gate_scalar_indices") != spec_sets["same-k"]:
    raise SystemExit("independent multi-gate gate1 same-k indices are incorrect")
if selected(root / "multi_independent.sus2", "second_layer_gate_scalar_indices") != spec_sets["q-total"]:
    raise SystemExit("gate2 could not independently select the outer q-total graph")

closures = {
    mode: gate_closure(root / f"gate_{mode}.sus2", "two_layer_gate_scalar_indices")
    for mode in ("same-k", "legacy", "q-total")
}
moment_counts = [closures[mode][0] for mode in ("same-k", "legacy", "q-total")]
product_counts = [closures[mode][1] for mode in ("same-k", "legacy", "q-total")]
if not moment_counts[0] < moment_counts[1] < moment_counts[2]:
    raise SystemExit(f"gate required-moment closures are not strictly nested: {moment_counts}")
if not product_counts[0] < product_counts[1] < product_counts[2]:
    raise SystemExit(f"gate required-product closures are not strictly nested: {product_counts}")

print(
    "scalar_counts",
    "same-k", len(spec_sets["same-k"]),
    "legacy", len(spec_sets["legacy"]),
    "q-total", len(spec_sets["q-total"]),
)
print(
    "gate_closure_products",
    "same-k", closures["same-k"][1],
    "legacy", closures["legacy"][1],
    "q-total", closures["q-total"][1],
)
PY

if "$BIN" init-sh "$WORKDIR/reject_gate_richer.sus2" "${COMMON_OPTS[@]}" \
  --sh-factor-pruning=legacy \
  --two-layer-gate \
  --two-layer-gate-factor-pruning=q-total >/dev/null 2>&1; then
  echo "richer gate1 pruning should be rejected" >&2
  exit 1
fi

if "$BIN" init-sh "$WORKDIR/reject_gate_without_gate.sus2" "${COMMON_OPTS[@]}" \
  --two-layer-gate-factor-pruning=legacy >/dev/null 2>&1; then
  echo "gate1 pruning without --two-layer-gate should be rejected" >&2
  exit 1
fi

if "$BIN" init-sh "$WORKDIR/reject_gate2_without_multi.sus2" "${COMMON_OPTS[@]}" \
  --two-layer-gate \
  --second-layer-gate-factor-pruning=same-k >/dev/null 2>&1; then
  echo "gate2 pruning without --multi-layer-gate should be rejected" >&2
  exit 1
fi

echo "sh factor-pruning split check passed"
