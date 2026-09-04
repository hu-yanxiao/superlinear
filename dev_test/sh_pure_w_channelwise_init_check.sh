#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
bin=${1:-./bin/mlp-sus2}
task_tmp=$(mktemp -d "${TMPDIR:-/tmp}/sus2sh-pure-w-init-XXXXXX")
if [[ ${SUS2_TEST_KEEP_TMP:-0} == 1 ]]; then
  echo "keeping test directory: $task_tmp"
else
  trap 'rm -rf "$task_tmp"' EXIT
fi
awk '{print} /^END_CFG/{exit}' example/train.cfg > "$task_tmp/one.cfg"

common=(--species-count=2 --l-max=2 --k-max=2 --body-order=4
        --body-l-max=2,2,2,2 --radial-basis-size=3 --cutoff=5
        --superlinear-mode=pure-w-channelwise --two-layer-gate-mode=mu-scalar-full
        --two-layer-gate-body-order=5)

for scaling_map in K L LK; do
  for width in 1 4; do
    label="${scaling_map}-o${width}"
    "$bin" init-sh "$task_tmp/$label.mtp" "${common[@]}" \
      --scaling-map="$scaling_map" --outer-k-rank="$width" > "$task_tmp/$label.init.log"
    "$bin" roundtrip-model-dev "$task_tmp/$label.mtp" "$task_tmp/$label.roundtrip.mtp" \
      > "$task_tmp/$label.roundtrip.log"
    "$bin" calc-efs "$task_tmp/$label.roundtrip.mtp" "$task_tmp/one.cfg" \
      "$task_tmp/$label.pred.cfg" > "$task_tmp/$label.predict.log"
  done
done
"$bin" init-sh "$task_tmp/duplicate.mtp" "${common[@]}" --scaling-map=LK \
  --outer-k-rank=4 > "$task_tmp/duplicate.log"
cmp "$task_tmp/duplicate.mtp" "$task_tmp/LK-o4.mtp"
"$bin" init-sh "$task_tmp/angular.mtp" --species-count=2 --l-max=2 --k-max=1 \
  --body-order=4 --body-l-max=2,2,2,2 --radial-basis-size=3 --cutoff=5 \
  --sh-factor-pruning=same-k --write-sh-scalar-info > "$task_tmp/angular.log"

for invalid in --two-layer-gate-radial-mixing=none --two-layer-gate-k-rank=0 \
               --sh-factor-pruning=q-total --superlinear-mode=unknown \
               --outer-k-rank=2147483647; do
  if "$bin" init-sh "$task_tmp/invalid.mtp" "${common[@]}" --outer-k-rank=4 \
       "$invalid" > "$task_tmp/invalid.log" 2>&1; then
    echo "invalid pure-W option accepted: $invalid" >&2
    exit 1
  fi
done
if "$bin" init-sh "$task_tmp/invalid-legacy.mtp" --k-max=2 --outer-k-rank=4 \
     > "$task_tmp/invalid-legacy.log" 2>&1; then
  echo "legacy outer rank > K was unexpectedly accepted" >&2
  exit 1
fi

python3 - "$task_tmp" <<'PY'
import math
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
def scalar(text, name):
    return int(re.search(rf"(?m)^\s*{name}\s*=\s*(\d+)\s*$", text)[1])
def vector(text, name):
    line = re.search(rf"(?m)^\s*{name}\s*=\s*(\{{.*\}})\s*$", text)[1]
    return [float(x) for x in re.findall(r"[-+]?(?:\d*\.\d+|\d+\.?\d*)(?:[eE][-+]?\d+)?", line)]
def tuples(text, name, size):
    values = vector(text, name)
    assert len(values) % size == 0
    return [values[i:i+size] for i in range(0, len(values), size)]

angular_text = (root / "angular.mtp").read_text()
angular_count = scalar(angular_text, "alpha_scalar_moments")
def primitive(channel, l, m):
    return math.sin(0.37 * (1 + channel * 7 + l * 3 + m))
angular_values = []
for channel in range(4):
    values = [0.0] * scalar(angular_text, "alpha_moments_count")
    for node, (_, l, m) in enumerate(tuples(angular_text, "alpha_index_basic", 3)):
        values[node] = primitive(channel, int(l), int(m))
    for left, right, target, coeff in tuples(angular_text, "sh_products", 4):
        values[int(target)] += coeff * values[int(left)] * values[int(right)]
    angular_values.append([values[int(node)] for node in vector(angular_text, "alpha_moment_mapping")])
for mapping in ("K", "L", "LK"):
    for width in (1, 4):
        label = f"{mapping}-o{width}"
        text = (root / f"{label}.roundtrip.mtp").read_text()
        assert "outer_channel_mode = same-channel" in text
        assert "outer_radial_mode = pure-matrix" in text
        assert scalar(text, "outer_k_rank1_weight_count") == 0
        assert vector(text, "outer_k_rank1_weights") == []
        assert scalar(text, "two_layer_gate_k_rank") == 2
        assert scalar(text, "two_layer_gate_body_order_max") == 5
        assert scalar(text, "sh_body_order") == 4  # independent, non-subgraph gate
        assert scalar(text, "alpha_scalar_moments") == width * angular_count
        assert len(vector(text, "scal_coeffs")) == 2 * 2 * 2 * {"K": 2, "L": 3, "LK": 6}[mapping]
        coeff = vector(text, "two_layer_gate_radial_mixing_coeffs")
        assert len(coeff) == 3 * width * 2 * 3
        assert all(math.isfinite(x) and x != 0 for x in coeff)
        if width > 1:
            # Distinct directions, not merely rescalings erased by a chart.
            assert len({tuple(x / coeff[o*6] for x in coeff[o*6:o*6+6])
                        for o in range(width)}) == width
        # Propagate channel labels through the compiled outer DAG. This detects
        # incorrect remapping and any accidental cross-channel products.
        begin = scalar(text, "outer_k_rank1_basic_begin")
        labels = {}
        values = [0.0] * scalar(text, "alpha_moments_count")
        for l in range(2, -1, -1):
            for channel in range(width-1, -1, -1):
                for m in range(-l, l+1):
                    values[begin + len(labels)] = primitive(channel, l, m)
                    labels[begin + len(labels)] = channel
        products = tuples(text, "sh_products", 4)
        outer_products = products[scalar(text, "outer_k_rank1_product_begin"):]
        assert len(outer_products) == width * len(tuples(angular_text, "sh_products", 4))
        for left, right, target, cg_coeff in outer_products:
            left, right, target = map(int, (left, right, target))
            assert labels[left] == labels[right]
            assert target not in labels or labels[target] == labels[left]
            labels[target] = labels[left]
            values[target] += cg_coeff * values[left] * values[right]
        counts = [0] * width
        for moment, info in zip(vector(text, "alpha_moment_mapping"), tuples(text, "sh_scalar_info", 7)):
            channel = labels[int(moment)]
            counts[channel] += 1
            for q in info[1:6]:
                if q >= 0:
                    assert width - 1 - int(q) % width == channel
        assert counts == [angular_count] * width
        for index, moment in enumerate(vector(text, "alpha_moment_mapping")):
            channel = width - 1 - index % width
            assert abs(values[int(moment)] - angular_values[channel][index // width]) < 1e-12
        zero = re.sub(r"(?m)^(\s*two_layer_gate_radial_mixing_coeffs\s*=\s*)\{.*\}$",
                      lambda m: m[1] + "{" + ", ".join("0" for _ in coeff) + "}", text)
        (root / f"{label}.zero.mtp").write_text(zero)
print(f"pure_w_topology_ok angular_scalars={angular_count}; O=1,4; K/L/LK; gate_body>outer_body")
PY

for width in 1 4; do
  "$bin" roundtrip-model-dev "$task_tmp/LK-o${width}.zero.mtp" \
    "$task_tmp/zero-o${width}.roundtrip.mtp" > "$task_tmp/zero-o${width}.log"
  "$bin" calc-efs "$task_tmp/zero-o${width}.roundtrip.mtp" "$task_tmp/one.cfg" \
    "$task_tmp/zero-o${width}.cfg" > "$task_tmp/zero-o${width}.predict.log"
done
# With zero W, both widths have the same independent species constant; no
# legacy radial contribution may reappear even at O=1.
cmp "$task_tmp/zero-o1.cfg" "$task_tmp/zero-o4.cfg"
python3 - "$task_tmp/LK-o4.pred.cfg" "$task_tmp/zero-o4.cfg" <<'PY'
import re
import sys
from pathlib import Path
active, zero = [Path(p).read_text() for p in sys.argv[1:]]
def energy(text):
    return float(re.search(r"(?m)^\s*Energy\s*\n\s*(\S+)", text)[1])
assert abs(energy(active) - energy(zero)) > 1e-8, "zero-W comparison must change the physical energy"
print("pure_w_zero_kernel_no_legacy_fallback_ok")
PY

for initialization in default same; do
  args=(--superlinear-mode=pure-w-channelwise)
  if [[ $initialization == same ]]; then args+=(--init-params=same); fi
  "$bin" train "$task_tmp/LK-o4.mtp" "$task_tmp/one.cfg" \
    --max-iter=0 --skip-preinit --radial-smooth=0 --shift \
    "${args[@]}" \
    --curr-pot-name="$task_tmp/$initialization.current.mtp" \
    --best-pot-name="$task_tmp/$initialization.best.mtp" \
    --trained-pot-name="$task_tmp/$initialization.trained.mtp" \
    > "$task_tmp/$initialization.train.log" 2>&1
  if grep -q 'Random initialization of nonlinear coefficients' "$task_tmp/$initialization.train.log"; then
    echo "same/default unexpectedly randomized parameters" >&2
    exit 1
  fi
done
"$bin" train "$task_tmp/one.cfg" "${common[@]}" --scaling-map=LK --outer-k-rank=4 \
  --inline-sh-model="$task_tmp/inline.mtp" --max-iter=0 --skip-preinit --radial-smooth=0 \
  --trained-pot-name="$task_tmp/inline.trained.mtp" > "$task_tmp/inline.train.log" 2>&1
for label in random-a random-b random-other random-zero; do
  seed=2026090401
  args=(--init-params=random)
  if [[ $label == random-other ]]; then seed=2026090402; fi
  if [[ $label == random-zero ]]; then args+=(--two-layer-gate-radial-mixing-init-scale=0); fi
  "$bin" train "$task_tmp/LK-o4.mtp" "$task_tmp/one.cfg" "${args[@]}" \
    --init-random-seed="$seed" --max-iter=0 --skip-preinit --radial-smooth=0 \
    --trained-pot-name="$task_tmp/$label.trained.mtp" > "$task_tmp/$label.train.log" 2>&1
done
python3 - "$task_tmp" <<'PY'
import re
import sys
from pathlib import Path
root = Path(sys.argv[1])
def parameters(path):
    text = path.read_text()
    names = ("scal_coeffs", "two_layer_gate_radial_coeffs", "two_layer_gate_additive_coeffs",
             "two_layer_gate_type_coeffs", "two_layer_gate_weights", "two_layer_gate_k_rank_weights",
             "two_layer_gate_radial_mixing_coeffs", "species_coeffs", "moment_coeffs")
    return {name: tuple(float(x) for x in re.findall(r"[-+]?(?:\d*\.\d+|\d+\.?\d*)(?:[eE][-+]?\d+)?",
                re.search(rf"(?m)^\s*{name}\s*=\s*\{{(.*)\}}\s*$", text)[1])) for name in names}
expected = parameters(root / "LK-o4.roundtrip.mtp")
for name in ("default", "same", "inline"):
    assert parameters(root / f"{name}.trained.mtp") == expected
print("pure_w_same_default_retains_all_physical_parameters_ok")
random = {name: parameters(root / f"{name}.trained.mtp")
          for name in ("random-a", "random-b", "random-other", "random-zero")}
assert random["random-a"] == random["random-b"], "same random seed must reproduce all physical parameters"
w = "two_layer_gate_radial_mixing_coeffs"
g = "two_layer_gate_radial_coeffs"
assert random["random-a"][w] != expected[w]
assert random["random-a"][g] != expected[g]
assert random["random-other"][w] != random["random-a"][w]
assert random["random-other"][g] != random["random-a"][g]
assert any(value != 0 for value in random["random-a"][w])
assert max(abs(value) for value in random["random-a"][w]) <= 0.005
assert all(value == 0 for value in random["random-zero"][w])
assert random["random-zero"][g] == random["random-a"][g]
def legacy_radial(path):
    text = path.read_text()
    block = re.search(r"(?ms)^\s*radial_coeffs\s*$\s*(.*?)^\s*alpha_moments_count\s*=", text)[1]
    return tuple(float(x) for row in re.findall(r"\{([^}]*)\}", block)
                 for x in re.findall(r"[-+]?(?:\d*\.\d+|\d+\.?\d*)(?:[eE][-+]?\d+)?", row))
legacy = legacy_radial(root / "LK-o4.roundtrip.mtp")
assert legacy
for name in random:
    assert legacy_radial(root / f"{name}.trained.mtp") == legacy
print("pure_w_random_seed_reproducible_gate_and_W_changed_unused_R_retained_ok")
PY
for mismatch in --superlinear-mode=legacy --sh-factor-pruning=q-total --two-layer-gate-body-order=4; do
  if "$bin" train "$task_tmp/LK-o4.roundtrip.mtp" "$task_tmp/one.cfg" \
       "$mismatch" --max-iter=0 > "$task_tmp/mismatch.log" 2>&1; then
    echo "loaded pure model accepted an incompatible CLI architecture: $mismatch" >&2
    exit 1
  fi
done
echo "pure_w_channelwise_init_check_ok"
