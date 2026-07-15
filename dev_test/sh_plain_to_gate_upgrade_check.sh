#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

tmp_dir=".codex_tmp/sh_plain_to_gate_upgrade"
mkdir -p "$tmp_dir"

plain_init="$tmp_dir/plain_init.mtp"
train="$tmp_dir/one.cfg"
upgraded="$tmp_dir/upgraded_shared_gate.mtp"
upgrade_log="$tmp_dir/upgrade_train.log"
plain_metadata="$tmp_dir/plain_metadata.mtp"
upgraded_mixed="$tmp_dir/upgraded_mixed_gate.mtp"
mixed_upgrade_log="$tmp_dir/mixed_upgrade_train.log"
upgraded_inherit="$tmp_dir/upgraded_inherited_gate.mtp"
inherit_upgrade_log="$tmp_dir/inherit_upgrade_train.log"
rm -f "$plain_init" "$train" "$upgraded" "$upgrade_log" \
  "$plain_metadata" "$upgraded_mixed" "$mixed_upgrade_log" \
  "$upgraded_inherit" "$inherit_upgrade_log"
awk '{print} /^END_CFG/{exit}' example/train.cfg > "$train"

./bin/mlp-sus2 init-sh "$plain_init" \
  --species-count=2 \
  --l-max=2 \
  --k-max=2 \
  --body-order=4 \
  --body-l-max=2,2,2,2 \
  --radial-basis-size=4 \
  --cutoff=5.0

if grep -q "sh_scalar_info" "$plain_init"; then
  echo "plain init-sh unexpectedly wrote sh_scalar_info metadata" >&2
  exit 1
fi

./bin/mlp-sus2 train "$plain_init" "$train" \
  --trained-pot-name="$upgraded" \
  --max-iter=1 \
  --skip-preinit \
  --do-lin \
  --do-lin-rescale \
  --fine-tune \
  --two-layer-gate \
  --two-layer-gate-shared-radial \
  --energy-weight=1 \
  --force-weight=0 \
  --stress-weight=0 >"$upgrade_log"

grep -q "SUS2-SH plain-to-gate upgrade enabled" "$upgrade_log"
grep -q "plain-to-gate upgrade disabled do-lin/do-lin-rescale/fine-tune" "$upgrade_log"
if grep -q "fine-tune mode enabled" "$upgrade_log"; then
  echo "plain-to-gate upgrade should disable fine-tune rescale/linear warmup" >&2
  exit 1
fi
if grep -q "Fine-tune initial" "$upgrade_log"; then
  echo "plain-to-gate upgrade ran fine-tune initial rescale/linear warmup" >&2
  exit 1
fi

grep -q "sh_scalar_info_count" "$upgraded"
grep -q "two_layer_gate_enabled = true" "$upgraded"
grep -q "two_layer_gate_body_order_max = 4" "$upgraded"
grep -q "two_layer_gate_include_one_body = false" "$upgraded"
grep -q "two_layer_gate_radial_mode = shared-radial" "$upgraded"
grep -q "two_layer_gate_weights" "$upgraded"

./bin/mlp-sus2 init-sh "$plain_metadata" \
  --species-count=2 \
  --l-max=2 \
  --k-max=2 \
  --body-order=4 \
  --body-l-max=2,2,2,2 \
  --radial-basis-size=4 \
  --cutoff=5.0 \
  --sh-factor-pruning=q-total \
  --write-sh-scalar-info

./bin/mlp-sus2 train "$plain_metadata" "$train" \
  --trained-pot-name="$upgraded_mixed" \
  --max-iter=1 \
  --skip-preinit \
  --two-layer-gate \
  --two-layer-gate-body-order=4 \
  --two-layer-gate-factor-pruning=legacy \
  --multi-layer-gate \
  --second-layer-gate-body-order=4 \
  --second-layer-gate-factor-pruning=same-k \
  --energy-weight=1 \
  --force-weight=0 \
  --stress-weight=0 >"$mixed_upgrade_log"

python3 - "$upgraded_mixed" <<'PY'
import re
import sys

text = open(sys.argv[1]).read()
lmax = int(re.search(r"^sh_l_max\s*=\s*(\d+)", text, re.M).group(1))
kmax = int(re.search(r"^sh_k_max\s*=\s*(\d+)", text, re.M).group(1))
info_match = re.search(r"^sh_scalar_info\s*=\s*\{(.*?)\}\s*$", text, re.M | re.S)
if not info_match:
    raise SystemExit("mixed gate upgrade lost sh_scalar_info")
specs = [
    tuple(int(value.strip()) for value in entry.split(","))
    for entry in re.findall(r"\{([^{}]+)\}", info_match.group(1))
]


def indices(name):
    match = re.search(rf"^{name}\s*=\s*\{{([^}}]*)\}}", text, re.M)
    if not match:
        raise SystemExit(f"missing {name}")
    return [int(value.strip()) for value in match.group(1).split(",") if value.strip()]


def q_lk(qid):
    return lmax - qid // kmax, kmax - 1 - qid % kmax


gate1 = indices("two_layer_gate_scalar_indices")
gate2 = indices("second_layer_gate_scalar_indices")
if not gate1 or not gate2:
    raise SystemExit("mixed gate upgrade selected an empty scalar set")
for scalar in gate1:
    spec = specs[scalar]
    channels = [q_lk(qid) for qid in spec[1 : spec[0]]]
    if any(l1 < l2 or k1 < k2 for (l1, k1), (l2, k2) in zip(channels, channels[1:])):
        raise SystemExit("gate1 upgrade contains a scalar outside legacy pruning")
for scalar in gate2:
    spec = specs[scalar]
    channels = [q_lk(qid) for qid in spec[1 : spec[0]]]
    if len({k for _, k in channels}) != 1:
        raise SystemExit("gate2 upgrade contains a cross-k scalar")
if len(gate2) >= len(gate1):
    raise SystemExit("expected the same-k gate2 graph to be smaller than legacy gate1")
PY

./bin/mlp-sus2 train "$plain_metadata" "$train" \
  --trained-pot-name="$upgraded_inherit" \
  --max-iter=1 \
  --skip-preinit \
  --two-layer-gate \
  --two-layer-gate-body-order=2 \
  --two-layer-gate-factor-pruning=same-k \
  --multi-layer-gate \
  --second-layer-gate-body-order=4 \
  --energy-weight=1 \
  --force-weight=0 \
  --stress-weight=0 >"$inherit_upgrade_log"

python3 - "$upgraded_inherit" <<'PY'
import re
import sys

text = open(sys.argv[1]).read()
lmax = int(re.search(r"^sh_l_max\s*=\s*(\d+)", text, re.M).group(1))
kmax = int(re.search(r"^sh_k_max\s*=\s*(\d+)", text, re.M).group(1))
info_match = re.search(r"^sh_scalar_info\s*=\s*\{(.*?)\}\s*$", text, re.M | re.S)
specs = [
    tuple(int(value.strip()) for value in entry.split(","))
    for entry in re.findall(r"\{([^{}]+)\}", info_match.group(1))
]
index_match = re.search(r"^second_layer_gate_scalar_indices\s*=\s*\{([^}]*)\}", text, re.M)
indices = [int(value.strip()) for value in index_match.group(1).split(",") if value.strip()]


def q_lk(qid):
    return lmax - qid // kmax, kmax - 1 - qid % kmax


if not any(spec[0] == 4 for spec in (specs[index] for index in indices)):
    raise SystemExit("inherited gate2 test did not extend beyond gate1 body-order 2")
for index in indices:
    spec = specs[index]
    channels = [q_lk(qid) for qid in spec[1 : spec[0]]]
    if len({k for _, k in channels}) != 1:
        raise SystemExit("gate2 did not inherit gate1 same-k pruning during one-command upgrade")
PY
