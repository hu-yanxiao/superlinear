#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

bin=${1:-./bin/mlp-sus2}
bin=$(cd "$(dirname "$bin")" && pwd)/$(basename "$bin")
tmp_dir=$(mktemp -d "${TMPDIR:-/tmp}/sus2sh-k-rank-active-XXXXXX")
if [[ ${SUS2_TEST_KEEP_TMP:-0} == 1 ]]; then
  echo "keeping test directory: $tmp_dir"
else
  trap 'rm -rf "$tmp_dir"' EXIT
fi

one_cfg="$tmp_dir/one.cfg"
awk '{print} /^END_CFG/{exit}' example/train.cfg >"$one_cfg"
if [[ $(grep -c '^END_CFG' "$one_cfg") -ne 1 ]]; then
  echo "failed to prepare one-configuration test set" >&2
  exit 1
fi
python3 - "$one_cfg" <<'PY'
import sys

path = sys.argv[1]
lines = open(path, encoding="utf-8").readlines()
in_atoms = False
changed = 0
for index, raw in enumerate(lines):
    stripped = raw.strip()
    if stripped.startswith("AtomData:"):
        in_atoms = True
        continue
    if in_atoms and stripped:
        fields = raw.split()
        if len(fields) >= 8 and fields[0].lstrip("+-").isdigit():
            fields[2] = f"{float(fields[2])+0.017*(changed+1):.15e}"
            fields[3] = f"{float(fields[3])-0.011*(changed+1):.15e}"
            fields[4] = f"{float(fields[4])+0.007*(changed+1):.15e}"
            lines[index] = " ".join(fields) + "\n"
            changed += 1
            if changed == 3:
                break
if changed != 3:
    raise SystemExit("failed to break the symmetry of the rank test geometry")
open(path, "w", encoding="utf-8").writelines(lines)
PY

common=(
  --species-count=2
  --l-max=2
  --k-max=3
  --body-order=3
  --body-l-max=2,2,2,2
  --radial-basis-size=3
  --radial-basis-type=RBChebyshev_sss
  --cutoff=5.0
  --sh-factor-pruning=q-total
  --two-layer-gate
  --two-layer-gate-factor-pruning=q-total
  --two-layer-gate-mode=mu-scalar-full
  --two-layer-gate-body-order=3
  --two-layer-gate-shared-radial
  --init-params=same
)

compare_numeric_files() {
  python3 - "$1" "$2" "${3:-3.0e-11}" <<'PY'
import re
import sys

number = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")
left = [float(x) for x in number.findall(open(sys.argv[1], encoding="utf-8").read())]
right = [float(x) for x in number.findall(open(sys.argv[2], encoding="utf-8").read())]
if not left or len(left) != len(right):
    raise SystemExit(
        f"numeric output shape differs: {sys.argv[1]} ({len(left)}) "
        f"vs {sys.argv[2]} ({len(right)})")
worst = max(abs(a-b) for a, b in zip(left, right))
scale = max([1.0] + [abs(x) for x in left] + [abs(x) for x in right])
tol = float(sys.argv[3])*scale
if worst > tol:
    raise SystemExit(
        f"numeric parity failed: {sys.argv[1]} vs {sys.argv[2]} "
        f"worst={worst:.3e} tolerance={tol:.3e}")
print(f"numeric_parity worst={worst:.3e} scale={scale:.3e}")
PY
}

# R=K with identity W must be an exact change of representation, not merely a
# scalar-count match.  Activate both the gate and outer heads before comparing
# E/F/S and every public decomposition/descriptor path.
plain_init="$tmp_dir/plain-init.sus2"
identity_init="$tmp_dir/rank3-identity-init.sus2"
plain_model="$tmp_dir/plain-active.sus2"
identity_model="$tmp_dir/rank3-identity-active.sus2"
"$bin" init-sh "$plain_init" "${common[@]}" >/dev/null
"$bin" init-sh "$identity_init" "${common[@]}" \
  --outer-k-rank=3 --two-layer-gate-k-rank=3 >/dev/null
"$bin" roundtrip-model-dev "$plain_init" "$tmp_dir/plain-materialized.sus2" >/dev/null
"$bin" roundtrip-model-dev "$identity_init" "$tmp_dir/identity-materialized.sus2" >/dev/null

python3 - "$tmp_dir/plain-materialized.sus2" \
  "$tmp_dir/identity-materialized.sus2" "$plain_model" "$identity_model" <<'PY'
import re
import sys

plain_source, identity_source, plain_target, identity_target = sys.argv[1:]
number = r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?"

def vector(text, name):
    match = re.search(rf"(?m)^\s*{re.escape(name)}\s*=\s*\{{([^}}]*)\}}", text, re.S)
    if not match:
        raise SystemExit(f"missing {name}")
    return [float(x) for x in re.findall(number, match.group(1))]

def integer(text, name):
    match = re.search(rf"(?m)^\s*{re.escape(name)}\s*=\s*(\d+)\s*$", text)
    if not match:
        raise SystemExit(f"missing {name}")
    return int(match.group(1))

def replace_vector(text, name, values):
    match = re.search(rf"(?m)^\s*{re.escape(name)}\s*=\s*\{{([^}}]*)\}}", text, re.S)
    if not match:
        raise SystemExit(f"missing {name}")
    replacement = f"{name} = {{" + ", ".join(f"{x:.15e}" for x in values) + "}"
    return text[:match.start()] + replacement + text[match.end():]

plain = open(plain_source, encoding="utf-8").read()
identity = open(identity_source, encoding="utf-8").read()
if integer(identity, "outer_k_rank") != 3 or integer(identity, "two_layer_gate_k_rank") != 3:
    raise SystemExit("R=K identity fixture lost rank-3 metadata")
if integer(plain, "alpha_scalar_moments") != integer(identity, "alpha_scalar_moments"):
    raise SystemExit("R=K outer graph did not recover the full-K scalar count")
if len(vector(plain, "two_layer_gate_scalar_indices")) != len(
        vector(identity, "two_layer_gate_scalar_indices")):
    raise SystemExit("R=K gate graph did not recover the full-K scalar count")

kmax = integer(identity, "sh_k_max")
lmax = integer(identity, "sh_l_max")
for name in ("outer_k_rank1_weights", "two_layer_gate_k_rank_weights"):
    weights = vector(identity, name)
    expected_count = (lmax+1)*kmax*kmax
    if len(weights) != expected_count:
        raise SystemExit(f"{name} count is {len(weights)}, expected {expected_count}")
    for l in range(lmax+1):
        for channel in range(kmax):
            for k in range(kmax):
                value = weights[(l*kmax+channel)*kmax+k]
                expected = 1.0 if channel == k else 0.0
                if value != expected:
                    raise SystemExit(
                        f"{name} is not identity at l={l}, c={channel}, k={k}: {value}")

shared_names = (
    "species_coeffs",
    "moment_coeffs",
    "two_layer_gate_radial_coeffs",
    "two_layer_gate_weights",
    "two_layer_gate_additive_coeffs",
    "two_layer_gate_type_coeffs",
)
for name in shared_names:
    if len(vector(plain, name)) != len(vector(identity, name)):
        raise SystemExit(f"R=K changed shared vector length for {name}")

def active_values(name, count):
    if name == "species_coeffs":
        return [0.13*(i+1) for i in range(count)]
    if name == "moment_coeffs":
        return [
            (0.018 + 0.0004*(i % 11)) * (-1.0 if i % 3 == 1 else 1.0)
            for i in range(count)]
    if name == "two_layer_gate_weights":
        return [
            (0.021 + 0.0002*(i % 13)) * (-1.0 if i % 5 == 2 else 1.0)
            for i in range(count)]
    if name == "two_layer_gate_radial_coeffs":
        return [
            (0.09 + 0.004*(i % 7)) * (-1.0 if i % 6 == 4 else 1.0)
            for i in range(count)]
    return [
        (0.009 + 0.0003*(i % 7)) * (-1.0 if i % 4 == 1 else 1.0)
        for i in range(count)]

for name in shared_names:
    values = active_values(name, len(vector(plain, name)))
    plain = replace_vector(plain, name, values)
    identity = replace_vector(identity, name, values)

open(plain_target, "w", encoding="utf-8").write(plain)
open(identity_target, "w", encoding="utf-8").write(identity)
PY

for command in calc-efs calc-descriptors calc-partialE calc-eij; do
  plain_out="$tmp_dir/plain-${command}.out"
  identity_out="$tmp_dir/identity-${command}.out"
  "$bin" "$command" "$plain_model" "$one_cfg" "$plain_out" >/dev/null
  "$bin" "$command" "$identity_model" "$one_cfg" "$identity_out" >/dev/null
  compare_numeric_files "$plain_out" "$identity_out"
done

identity_roundtrip="$tmp_dir/rank3-identity-roundtrip.sus2"
"$bin" roundtrip-model-dev "$identity_model" "$identity_roundtrip" >/dev/null
"$bin" calc-efs "$identity_roundtrip" "$one_cfg" \
  "$tmp_dir/identity-roundtrip-efs.out" >/dev/null
compare_numeric_files "$tmp_dir/identity-calc-efs.out" \
  "$tmp_dir/identity-roundtrip-efs.out"

# Exercise an actually compressed, active gate R=2<K=3.  The asymmetric gate
# and outer matrices make channel/index transpositions observable.  The loss
# directional checks include E, F, and stress, while check-efs-fd-dev verifies
# the positional and cell chain rule independently of training targets.
rank2_init="$tmp_dir/rank2-init.sus2"
rank2_model="$tmp_dir/rank2-active.sus2"
"$bin" init-sh "$rank2_init" "${common[@]}" \
  --outer-k-rank=2 --two-layer-gate-k-rank=2 >/dev/null
"$bin" roundtrip-model-dev "$rank2_init" "$tmp_dir/rank2-materialized.sus2" >/dev/null

python3 - "$tmp_dir/rank2-materialized.sus2" "$rank2_model" <<'PY'
import re
import sys

source, target = sys.argv[1:]
text = open(source, encoding="utf-8").read()
number = r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?"

def integer(name):
    match = re.search(rf"(?m)^\s*{re.escape(name)}\s*=\s*(\d+)\s*$", text)
    if not match:
        raise SystemExit(f"missing {name}")
    return int(match.group(1))

def replace(name, make_values):
    global text
    match = re.search(rf"(?m)^\s*{re.escape(name)}\s*=\s*\{{([^}}]*)\}}", text, re.S)
    if not match:
        raise SystemExit(f"missing {name}")
    old = [float(x) for x in re.findall(number, match.group(1))]
    values = make_values(len(old))
    if len(values) != len(old):
        raise SystemExit(f"{name} replacement has {len(values)} values, expected {len(old)}")
    replacement = f"{name} = {{" + ", ".join(f"{x:.15e}" for x in values) + "}"
    text = text[:match.start()] + replacement + text[match.end():]

if integer("outer_k_rank") != 2 or integer("two_layer_gate_k_rank") != 2:
    raise SystemExit("active gate-rank2 fixture lost rank metadata")
kmax = integer("sh_k_max")
lmax = integer("sh_l_max")
expected_w = 2*(lmax+1)*kmax
if integer("outer_k_rank1_weight_count") != expected_w:
    raise SystemExit("unexpected outer rank2 W count")
if integer("two_layer_gate_k_rank_weight_count") != expected_w:
    raise SystemExit("unexpected gate rank2 W count")

replace("species_coeffs", lambda n: [0.11*(i+1) for i in range(n)])
replace("moment_coeffs", lambda n: [
    (0.024 + 0.0005*(i % 9)) * (-1.0 if i % 4 == 1 else 1.0)
    for i in range(n)])
replace("two_layer_gate_radial_coeffs", lambda n: [
    (0.12 + 0.005*(i % 7)) * (-1.0 if i % 6 == 4 else 1.0)
    for i in range(n)])
replace("two_layer_gate_weights", lambda n: [
    (0.028 + 0.0003*(i % 17)) * (-1.0 if i % 5 == 3 else 1.0)
    for i in range(n)])
replace("two_layer_gate_additive_coeffs", lambda n: [
    (0.18 + 0.006*(i % 5)) * (-1.0 if i % 3 == 1 else 1.0)
    for i in range(n)])
replace("two_layer_gate_type_coeffs", lambda n: [
    (0.62 + 0.011*(i % 7)) * (-1.0 if i % 4 == 2 else 1.0)
    for i in range(n)])

def outer_weights(_):
    values = []
    for l in range(lmax+1):
        values.extend((1.01+0.03*l, -0.27+0.01*l, 0.16-0.02*l))
        values.extend((-0.12+0.01*l, 0.74-0.02*l, 0.31+0.01*l))
    return values

def gate_weights(_):
    values = []
    for l in range(lmax+1):
        values.extend((0.88-0.02*l, -0.19+0.01*l, 0.07+0.02*l))
        values.extend((0.23+0.01*l, 0.61-0.03*l, -0.29+0.01*l))
    return values

replace("outer_k_rank1_weights", outer_weights)
replace("two_layer_gate_k_rank_weights", gate_weights)
open(target, "w", encoding="utf-8").write(text)
PY

python3 - "$rank2_model" "$tmp_dir/rank2-gate-w-perturbed.sus2" <<'PY'
import re
import sys

source, target = sys.argv[1:]
text = open(source, encoding="utf-8").read()
name = "two_layer_gate_k_rank_weights"
number = r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?"
match = re.search(rf"(?m)^\s*{name}\s*=\s*\{{([^}}]*)\}}", text, re.S)
if not match:
    raise SystemExit(f"missing {name}")
values = [float(x) for x in re.findall(number, match.group(1))]
values = [x+0.037*((i % 5)-2) for i, x in enumerate(values)]
replacement = f"{name} = {{" + ", ".join(f"{x:.15e}" for x in values) + "}"
open(target, "w", encoding="utf-8").write(
    text[:match.start()] + replacement + text[match.end():])
PY
"$bin" calc-efs "$rank2_model" "$one_cfg" "$tmp_dir/rank2-active-probe.cfg" >/dev/null
"$bin" calc-efs "$tmp_dir/rank2-gate-w-perturbed.sus2" "$one_cfg" \
  "$tmp_dir/rank2-gate-w-perturbed.cfg" >/dev/null
python3 - "$tmp_dir/rank2-active-probe.cfg" \
  "$tmp_dir/rank2-gate-w-perturbed.cfg" <<'PY'
import re
import sys

number = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")
a = [float(x) for x in number.findall(open(sys.argv[1], encoding="utf-8").read())]
b = [float(x) for x in number.findall(open(sys.argv[2], encoding="utf-8").read())]
if not a or len(a) != len(b):
    raise SystemExit("gate-rank2 activity probe output shape differs")
worst = max(abs(x-y) for x, y in zip(a, b))
scale = max([1.0] + [abs(x) for x in a] + [abs(x) for x in b])
if worst <= 1.0e-13*scale:
    raise SystemExit("active E/F/S is insensitive to an asymmetric gate-rank2 W perturbation")
print(f"gate_rank2_active_sensitivity worst={worst:.3e} scale={scale:.3e}")
PY

"$bin" calc-efs "$rank2_model" "$one_cfg" "$tmp_dir/rank2-target.cfg" >/dev/null
python3 - "$tmp_dir/rank2-target.cfg" <<'PY'
import sys

path = sys.argv[1]
lines = open(path, encoding="utf-8").readlines()
in_atoms = False
take_energy = False
take_stress = False
atom = 0
for index, raw in enumerate(lines):
    stripped = raw.strip()
    if stripped.startswith("AtomData:"):
        in_atoms = True
        continue
    if stripped == "Energy":
        in_atoms = False
        take_energy = True
        continue
    if stripped.startswith("PlusStress:"):
        take_stress = True
        continue
    if in_atoms and stripped:
        fields = raw.split()
        if len(fields) >= 8 and fields[0].lstrip("+-").isdigit():
            atom += 1
            for component in range(3):
                fields[5+component] = (
                    f"{float(fields[5+component]) + 0.006*((atom+component) % 5-2):.15e}")
            lines[index] = " ".join(fields) + "\n"
    elif take_energy and stripped:
        lines[index] = f"\t{float(stripped)+0.2:.15e}\n"
        take_energy = False
    elif take_stress and stripped:
        fields = raw.split()
        if len(fields) >= 6:
            lines[index] = "\t" + " \t".join(
                f"{float(value)+0.008*(component-2):.15e}"
                for component, value in enumerate(fields[:6])) + "\n"
            take_stress = False
if atom == 0 or take_energy or take_stress:
    raise SystemExit("failed to construct controlled E/F/S rank2 target")
open(path, "w", encoding="utf-8").writelines(lines)
PY

layout=$("$bin" check-loss-gradient-dev "$rank2_model" "$tmp_dir/rank2-target.cfg" \
  --max-configs=1 --energy-weight=1 --force-weight=0 --stress-weight=0 \
  --radial-smooth=0 --scalar-head-l2=0 --gate-scalar-l2=0 \
  --gate-mix-l2=0 --gate-full-l2=0 --gate-x-l2=0 \
  --coeff-start=0 --coeff-end=1 \
  --displacement=1.0e-5 --abs-tolerance=1 --rel-tolerance=1)
gate_begin=$(sed -n 's/.*gate_k_rank_begin=\([0-9][0-9]*\).*/\1/p' <<<"$layout")
gate_end=$(sed -n 's/.*gate_k_rank_end=\([0-9][0-9]*\).*/\1/p' <<<"$layout")
outer_begin=$(sed -n 's/.*outer_k_rank_begin=\([0-9][0-9]*\).*/\1/p' <<<"$layout")
outer_end=$(sed -n 's/.*outer_k_rank_end=\([0-9][0-9]*\).*/\1/p' <<<"$layout")
if [[ -z "$gate_begin" || -z "$gate_end" || -z "$outer_begin" || -z "$outer_end" \
      || $((gate_end-gate_begin)) -ne 18 || $((outer_end-outer_begin)) -ne 18 \
      || "$gate_end" -ne "$outer_begin" ]]; then
  echo "failed to discover contiguous active gate/outer rank2 blocks: $layout" >&2
  exit 1
fi

for range in "$gate_begin:$gate_end" "$outer_begin:$outer_end"; do
  begin=${range%%:*}
  end=${range##*:}
  "$bin" check-loss-gradient-dev "$rank2_model" "$tmp_dir/rank2-target.cfg" \
    --max-configs=1 --energy-weight=1 --force-weight=0 --stress-weight=0 \
    --radial-smooth=0 --scalar-head-l2=0 --gate-scalar-l2=0 \
    --gate-mix-l2=0 --gate-full-l2=0 --gate-x-l2=0 \
    --coeff-start="$begin" --coeff-end="$end" \
    --displacement=1.0e-5 --abs-tolerance=8.0e-6 --rel-tolerance=8.0e-4

  direction=$("$bin" check-loss-gradient-direction-dev \
    "$rank2_model" "$tmp_dir/rank2-target.cfg" \
    --max-configs=1 --energy-weight=1 --force-weight=0.1 --stress-weight=0.01 \
    --radial-smooth=0 --scalar-head-l2=0 --gate-scalar-l2=0 \
    --gate-mix-l2=0 --gate-full-l2=0 --gate-x-l2=0 \
    --coeff-start="$begin" --coeff-end="$end" \
    --displacement=1.0e-4 --abs-tolerance=2.0e-5 --rel-tolerance=5.0e-3)
  grep -q 'analytic_directional=' <<<"$direction"
done

"$bin" check-efs-fd-dev "$rank2_model" "$one_cfg" \
  --max-configs=1 --max-atoms=1 \
  --displacement=1.0e-5 --abs-tolerance=4.0e-5 --rel-tolerance=4.0e-3

rank2_roundtrip="$tmp_dir/rank2-roundtrip.sus2"
"$bin" roundtrip-model-dev "$rank2_model" "$rank2_roundtrip" >/dev/null
"$bin" calc-efs "$rank2_model" "$one_cfg" "$tmp_dir/rank2-efs.out" >/dev/null
"$bin" calc-efs "$rank2_roundtrip" "$one_cfg" \
  "$tmp_dir/rank2-roundtrip-efs.out" >/dev/null
compare_numeric_files "$tmp_dir/rank2-efs.out" "$tmp_dir/rank2-roundtrip-efs.out"

# The gate and outer ranks are independent topology parameters.  Exercise both
# asymmetric orderings with active coefficients, including the default
# mu-body-linear-combo gate path.  The rank-2/1 case has trainable W_g and W_o
# blocks; rank-1/2 has a trainable W_g block and the intentionally empty direct
# radial W_o block.
body_combo_common=()
for option in "${common[@]}"; do
  if [[ $option != --two-layer-gate-mode=* ]]; then
    body_combo_common+=("$option")
  fi
done

activate_mixed_rank_model() {
  local source=$1 target=$2 expected_outer=$3 expected_gate=$4 expected_mode=$5
  python3 - "$source" "$target" "$expected_outer" "$expected_gate" \
    "$expected_mode" <<'PY'
import re
import sys

source, target = sys.argv[1:3]
expected_outer, expected_gate = map(int, sys.argv[3:5])
expected_mode = sys.argv[5]
text = open(source, encoding="utf-8").read()
number = r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?"

def integer(name, default=None):
    match = re.search(rf"(?m)^\s*{re.escape(name)}\s*=\s*(\d+)\s*$", text)
    if not match:
        if default is not None:
            return default
        raise SystemExit(f"missing {name}")
    return int(match.group(1))

def string(name):
    match = re.search(rf"(?m)^\s*{re.escape(name)}\s*=\s*(\S+)\s*$", text)
    if not match:
        raise SystemExit(f"missing {name}")
    return match.group(1)

def replace(name, values_fn, required=True):
    global text
    match = re.search(
        rf"(?m)^\s*{re.escape(name)}\s*=\s*\{{([^}}]*)\}}", text, re.S)
    if not match:
        if required:
            raise SystemExit(f"missing {name}")
        return False
    old = [float(x) for x in re.findall(number, match.group(1))]
    values = values_fn(len(old))
    if len(values) != len(old):
        raise SystemExit(
            f"{name} replacement has {len(values)} values, expected {len(old)}")
    replacement = f"{name} = {{" + ", ".join(f"{x:.15e}" for x in values) + "}"
    text = text[:match.start()] + replacement + text[match.end():]
    return True

outer_rank = integer("outer_k_rank", 1)
gate_rank = integer("two_layer_gate_k_rank")
if outer_rank != expected_outer or gate_rank != expected_gate:
    raise SystemExit(
        f"mixed-rank metadata is {outer_rank}/{gate_rank}, "
        f"expected {expected_outer}/{expected_gate}")
if string("two_layer_gate_mode") != expected_mode:
    raise SystemExit(
        f"gate mode is {string('two_layer_gate_mode')}, expected {expected_mode}")

kmax = integer("sh_k_max")
lmax = integer("sh_l_max")
expected_outer_weights = expected_outer * (lmax + 1) * kmax
expected_gate_weights = expected_gate * (lmax + 1) * kmax
if integer("outer_k_rank1_weight_count") != expected_outer_weights:
    raise SystemExit("mixed-rank outer W serialization count is inconsistent")
if integer("two_layer_gate_k_rank_weight_count") != expected_gate_weights:
    raise SystemExit("mixed-rank gate W serialization count is inconsistent")

replace("species_coeffs", lambda n: [0.17 * (i + 1) for i in range(n)])
replace("moment_coeffs", lambda n: [
    (0.031 + 0.0007 * (i % 11)) * (-1.0 if i % 4 == 1 else 1.0)
    for i in range(n)])
replace("two_layer_gate_radial_coeffs", lambda n: [
    (0.14 + 0.006 * (i % 7)) * (-1.0 if i % 5 == 3 else 1.0)
    for i in range(n)])
replace("two_layer_gate_weights", lambda n: [
    (0.043 + 0.0009 * (i % 13)) * (-1.0 if i % 3 == 1 else 1.0)
    for i in range(n)])
has_body_mix = replace("two_layer_gate_body_mix_weights", lambda n: [
    (0.29 + 0.013 * (i % 9)) * (-1.0 if i % 4 == 2 else 1.0)
    for i in range(n)], required=False)
if (expected_mode == "mu-body-linear-combo") != has_body_mix:
    raise SystemExit("mixed-rank body-mix metadata does not match the gate mode")
replace("two_layer_gate_additive_coeffs", lambda n: [
    (0.23 + 0.008 * (i % 5)) * (-1.0 if i % 4 == 1 else 1.0)
    for i in range(n)])
replace("two_layer_gate_type_coeffs", lambda n: [
    (0.71 + 0.015 * (i % 7)) * (-1.0 if i % 5 == 2 else 1.0)
    for i in range(n)])

def rank_weights(rank, phase):
    values = []
    for l in range(lmax + 1):
        for channel in range(rank):
            for k in range(kmax):
                value = (0.81 + phase + 0.037 * l + 0.061 * channel
                         - 0.049 * k + 0.007 * channel * k)
                if (l + 2 * channel + k) % 4 == 1:
                    value = -value
                values.append(value)
    return values

# For outer rank one this deliberately writes a non-unit legacy compatibility
# shell.  The standalone loader must absorb it into the radial rows.
replace("outer_k_rank1_weights", lambda n: rank_weights(expected_outer, 0.0))
replace("two_layer_gate_k_rank_weights", lambda n: rank_weights(expected_gate, 0.19))
open(target, "w", encoding="utf-8").write(text)
PY
}

perturb_model_vector() {
  local source=$1 target=$2 name=$3
  python3 - "$source" "$target" "$name" <<'PY'
import re
import sys

source, target, name = sys.argv[1:]
text = open(source, encoding="utf-8").read()
number = r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?"
match = re.search(
    rf"(?m)^\s*{re.escape(name)}\s*=\s*\{{([^}}]*)\}}", text, re.S)
if not match:
    raise SystemExit(f"missing {name}")
values = [float(x) for x in re.findall(number, match.group(1))]
values = [value + 0.041 * ((i % 5) - 2) for i, value in enumerate(values)]
replacement = f"{name} = {{" + ", ".join(f"{x:.15e}" for x in values) + "}"
open(target, "w", encoding="utf-8").write(
    text[:match.start()] + replacement + text[match.end():])
PY
}

assert_numeric_difference() {
  python3 - "$1" "$2" "$3" <<'PY'
import re
import sys

left_path, right_path, label = sys.argv[1:]
number = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")
left = [float(x) for x in number.findall(open(left_path, encoding="utf-8").read())]
right = [float(x) for x in number.findall(open(right_path, encoding="utf-8").read())]
if not left or len(left) != len(right):
    raise SystemExit(f"{label}: sensitivity output shape differs")
worst = max(abs(a - b) for a, b in zip(left, right))
scale = max([1.0] + [abs(x) for x in left] + [abs(x) for x in right])
if worst <= 1.0e-13 * scale:
    raise SystemExit(f"{label}: active E/F/S is insensitive to the perturbation")
print(f"{label}_sensitivity worst={worst:.3e} scale={scale:.3e}")
PY
}

check_mixed_rank_gradients() {
  local label=$1 model=$2 expected_gate_count=$3 expected_outer_count=$4
  local layout gate_begin gate_end outer_begin outer_end begin end direction analytic
  layout=$("$bin" check-loss-gradient-dev "$model" "$tmp_dir/rank2-target.cfg" \
    --max-configs=1 --energy-weight=1 --force-weight=0 --stress-weight=0 \
    --radial-smooth=0 --scalar-head-l2=0 --gate-scalar-l2=0 \
    --gate-mix-l2=0 --gate-full-l2=0 --gate-x-l2=0 \
    --coeff-start=0 --coeff-end=1 \
    --displacement=1.0e-5 --abs-tolerance=1 --rel-tolerance=1)
  gate_begin=$(sed -n 's/.*gate_k_rank_begin=\([0-9][0-9]*\).*/\1/p' <<<"$layout")
  gate_end=$(sed -n 's/.*gate_k_rank_end=\([0-9][0-9]*\).*/\1/p' <<<"$layout")
  outer_begin=$(sed -n 's/.*outer_k_rank_begin=\([0-9][0-9]*\).*/\1/p' <<<"$layout")
  outer_end=$(sed -n 's/.*outer_k_rank_end=\([0-9][0-9]*\).*/\1/p' <<<"$layout")
  if [[ -z $gate_begin || -z $gate_end || -z $outer_begin || -z $outer_end \
        || $((gate_end-gate_begin)) -ne expected_gate_count \
        || $((outer_end-outer_begin)) -ne expected_outer_count \
        || $gate_end -ne $outer_begin ]]; then
    echo "$label: unexpected gate/outer coefficient layout: $layout" >&2
    exit 1
  fi

  for range in "$gate_begin:$gate_end" "$outer_begin:$outer_end"; do
    begin=${range%%:*}
    end=${range##*:}
    if [[ $begin -eq $end ]]; then
      continue
    fi
    "$bin" check-loss-gradient-dev "$model" "$tmp_dir/rank2-target.cfg" \
      --max-configs=1 --energy-weight=1 --force-weight=0 --stress-weight=0 \
      --radial-smooth=0 --scalar-head-l2=0 --gate-scalar-l2=0 \
      --gate-mix-l2=0 --gate-full-l2=0 --gate-x-l2=0 \
      --coeff-start="$begin" --coeff-end="$end" \
      --displacement=1.0e-5 --abs-tolerance=1.2e-5 --rel-tolerance=1.5e-3
    direction=$("$bin" check-loss-gradient-direction-dev \
      "$model" "$tmp_dir/rank2-target.cfg" \
      --max-configs=1 --energy-weight=1 --force-weight=0.1 --stress-weight=0.01 \
      --radial-smooth=0 --scalar-head-l2=0 --gate-scalar-l2=0 \
      --gate-mix-l2=0 --gate-full-l2=0 --gate-x-l2=0 \
      --coeff-start="$begin" --coeff-end="$end" \
      --displacement=1.0e-4 --abs-tolerance=3.0e-5 --rel-tolerance=7.0e-3)
    analytic=$(sed -n 's/.*analytic_directional=\([^ ]*\).*/\1/p' <<<"$direction")
    python3 - "$label" "$begin" "$end" "$analytic" <<'PY'
import math
import sys

label, begin, end, value = sys.argv[1:]
analytic = float(value)
if not math.isfinite(analytic) or abs(analytic) < 1.0e-14:
    raise SystemExit(
        f"{label}: coefficient range [{begin},{end}) has inactive directional gradient")
PY
  done

  "$bin" check-efs-fd-dev "$model" "$one_cfg" \
    --max-configs=1 --max-atoms=1 \
    --displacement=1.0e-5 --abs-tolerance=5.0e-5 --rel-tolerance=5.0e-3
}

outer2_gate1_init="$tmp_dir/outer2-gate1-body-init.sus2"
outer2_gate1_model="$tmp_dir/outer2-gate1-body-active.sus2"
"$bin" init-sh "$outer2_gate1_init" "${body_combo_common[@]}" \
  --outer-k-rank=2 --two-layer-gate-k-rank=1 >/dev/null
"$bin" roundtrip-model-dev "$outer2_gate1_init" \
  "$tmp_dir/outer2-gate1-body-materialized.sus2" >/dev/null
activate_mixed_rank_model "$tmp_dir/outer2-gate1-body-materialized.sus2" \
  "$outer2_gate1_model" 2 1 mu-body-linear-combo

outer1_gate2_init="$tmp_dir/outer1-gate2-full-init.sus2"
outer1_gate2_model="$tmp_dir/outer1-gate2-full-active-legacy-w.sus2"
"$bin" init-sh "$outer1_gate2_init" "${common[@]}" \
  --outer-k-rank=1 --two-layer-gate-k-rank=2 >/dev/null
"$bin" roundtrip-model-dev "$outer1_gate2_init" \
  "$tmp_dir/outer1-gate2-full-materialized.sus2" >/dev/null
activate_mixed_rank_model "$tmp_dir/outer1-gate2-full-materialized.sus2" \
  "$outer1_gate2_model" 1 2 mu-scalar-full

# The default body-linear-combo mixed model must depend actively on W_g.
perturb_model_vector "$outer2_gate1_model" \
  "$tmp_dir/outer2-gate1-body-gate-w-perturbed.sus2" \
  two_layer_gate_k_rank_weights
"$bin" calc-efs "$outer2_gate1_model" "$one_cfg" \
  "$tmp_dir/outer2-gate1-body-base.cfg" >/dev/null
"$bin" calc-efs "$tmp_dir/outer2-gate1-body-gate-w-perturbed.sus2" "$one_cfg" \
  "$tmp_dir/outer2-gate1-body-gate-w-perturbed.cfg" >/dev/null
assert_numeric_difference "$tmp_dir/outer2-gate1-body-base.cfg" \
  "$tmp_dir/outer2-gate1-body-gate-w-perturbed.cfg" \
  outer2_gate1_body_gate_w

check_mixed_rank_gradients outer2_gate1_body "$outer2_gate1_model" 9 18
check_mixed_rank_gradients outer1_gate2_full "$outer1_gate2_model" 18 0

# A direct outer-rank1 model may arrive in the old non-unit-W representation.
# Loading and saving must absorb W into radial rows, emit an all-one shell, and
# preserve every public E/F/stress value on the next reload.
legacy_rank1_canonical="$tmp_dir/outer1-gate2-canonical.sus2"
"$bin" calc-efs "$outer1_gate2_model" "$one_cfg" \
  "$tmp_dir/outer1-gate2-before-roundtrip.cfg" >/dev/null
"$bin" roundtrip-model-dev "$outer1_gate2_model" "$legacy_rank1_canonical" >/dev/null
"$bin" calc-efs "$legacy_rank1_canonical" "$one_cfg" \
  "$tmp_dir/outer1-gate2-after-roundtrip.cfg" >/dev/null
compare_numeric_files "$tmp_dir/outer1-gate2-before-roundtrip.cfg" \
  "$tmp_dir/outer1-gate2-after-roundtrip.cfg"
python3 - "$legacy_rank1_canonical" <<'PY'
import re
import sys

text = open(sys.argv[1], encoding="utf-8").read()
match = re.search(
    r"(?m)^\s*outer_k_rank1_weights\s*=\s*\{([^}]*)\}", text, re.S)
if not match:
    raise SystemExit("canonical direct-rank1 model lost its compatibility shell")
number = r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?"
weights = [float(x) for x in re.findall(number, match.group(1))]
if not weights or any(value != 1.0 for value in weights):
    raise SystemExit("canonical direct-rank1 compatibility shell is not all one")
print(f"legacy_rank1_canonical_weights count={len(weights)} all_one=true")
PY

echo "active K-rank math checks passed: identity, equal/mixed ranks, default body gate, and legacy rank1 roundtrip"
