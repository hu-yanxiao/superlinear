#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

bin=${1:-./bin/mlp-sus2}
tmp_dir=$(mktemp -d "${TMPDIR:-/tmp}/sus2sh-outer-k-rank1-XXXXXX")
if [[ ${SUS2_SH_TEST_KEEP_TMP:-0} == 1 ]]; then
  echo "keeping outer K-rank1 smoke artifacts in $tmp_dir" >&2
else
  trap 'rm -rf "$tmp_dir"' EXIT
fi

init_model="$tmp_dir/rank1-init.sus2"
bootstrap_model="$tmp_dir/rank1-bootstrap.sus2"
probe_model="$tmp_dir/rank1-probe.sus2"
base_radial_model="$tmp_dir/rank1-base-radial-unsupported.sus2"
roundtrip_model="$tmp_dir/rank1-roundtrip.sus2"
roundtrip_model2="$tmp_dir/rank1-roundtrip-2.sus2"
active_model="$tmp_dir/rank1-active.sus2"
one_cfg="$tmp_dir/one.cfg"
fd_cfg="$tmp_dir/fd-target.cfg"
efs_probe="$tmp_dir/probe-efs.cfg"
efs_roundtrip="$tmp_dir/roundtrip-efs.cfg"
efs_old="$tmp_dir/old-efs.cfg"
efs_old_canonical="$tmp_dir/old-canonical-efs.cfg"
descriptors="$tmp_dir/descriptors.txt"
partial="$tmp_dir/partial.txt"
eij="$tmp_dir/eij.txt"
old_bin=${OLD_KRANK_BIN:-}

compare_numeric_files() {
  python3 - "$1" "$2" "${3:-5e-12}" <<'PY'
import re
import sys

number = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")
left = [float(x) for x in number.findall(open(sys.argv[1], encoding="utf-8").read())]
right = [float(x) for x in number.findall(open(sys.argv[2], encoding="utf-8").read())]
if len(left) != len(right):
    raise SystemExit(f"numeric output length differs: {sys.argv[1]} {sys.argv[2]}")
worst = max((abs(a-b) for a, b in zip(left, right)), default=0.0)
scale = max([1.0] + [abs(x) for x in left] + [abs(x) for x in right])
tolerance = float(sys.argv[3]) * scale
if worst > tolerance:
    raise SystemExit(
        f"numeric parity failed: {sys.argv[1]} {sys.argv[2]} "
        f"worst={worst:.3e} tolerance={tolerance:.3e}")
print(f"numeric_parity worst={worst:.3e} scale={scale:.3e}")
PY
}

awk '{print} /^END_CFG/{exit}' example/train.cfg > "$one_cfg"
python3 - "$one_cfg" <<'PY'
import sys

path = sys.argv[1]
lines = open(path, encoding="utf-8").readlines()
in_atoms = False
changed = False
for index, raw in enumerate(lines):
    stripped = raw.strip()
    if stripped.startswith("AtomData:"):
        in_atoms = True
        continue
    if in_atoms and stripped:
        fields = raw.split()
        if len(fields) >= 8 and fields[0].lstrip("+-").isdigit():
            fields[2] = f"{float(fields[2]) + 0.037:.15e}"
            fields[3] = f"{float(fields[3]) - 0.021:.15e}"
            lines[index] = " ".join(fields) + "\n"
            changed = True
            break
if not changed:
    raise SystemExit("failed to perturb the finite-difference geometry")
open(path, "w", encoding="utf-8").writelines(lines)
PY

"$bin" init-sh "$init_model" \
  --species-count=2 \
  --l-max=4 \
  --k-max=4 \
  --body-order=4 \
  --body-l-max=4,4,4,4 \
  --radial-basis-size=4 \
  --radial-basis-type=RBChebyshev_sss \
  --cutoff=5.0 \
  --sh-factor-pruning=q-total \
  --two-layer-gate \
  --two-layer-gate-factor-pruning=q-total \
  --two-layer-gate-mode=mu-scalar-full \
  --two-layer-gate-body-order=4 \
  --two-layer-gate-shared-radial \
  --outer-k-rank1

# Materialize the implicit untrained radial/moment coefficients first.  The
# structural init-sh file intentionally omits those rows, while a legacy
# trained/checkpoint model contains them explicitly.
"$bin" roundtrip-model-dev "$init_model" "$bootstrap_model"

python3 - "$bootstrap_model" "$base_radial_model" <<'PY'
import sys

source, target = sys.argv[1:]
drop = {
    "two_layer_gate_radial_mode",
    "two_layer_gate_radial_coeff_count",
    "two_layer_gate_radial_coeffs",
}
lines = [line for line in open(source, encoding="utf-8")
         if line.split("=", 1)[0].strip() not in drop]
open(target, "w", encoding="utf-8").writelines(lines)
PY
if "$bin" calc-efs "$base_radial_model" "$one_cfg" \
     "$tmp_dir/base-radial-should-not-load.cfg" \
     >"$tmp_dir/base-radial.stdout" 2>"$tmp_dir/base-radial.stderr"; then
  echo "rank1 direct-radial silently changed a gate that reused base radial rows" >&2
  exit 1
fi
grep -q "requires an independent gate radial block" \
  "$tmp_dir/base-radial.stdout" "$tmp_dir/base-radial.stderr"

python3 - "$bootstrap_model" "$probe_model" <<'PY'
import re
import sys

source, target = sys.argv[1], sys.argv[2]
text = open(source, encoding="utf-8").read()

def integers(name):
    match = re.search(rf"{name}\s*=\s*\{{([^}}]*)\}}", text, re.S)
    if not match:
        raise SystemExit(f"missing {name}")
    return [int(x) for x in re.findall(r"-?\d+", match.group(1))]

def scalar(name):
    match = re.search(rf"{name}\s*=\s*(\d+)", text)
    if not match:
        raise SystemExit(f"missing {name}")
    return int(match.group(1))

if scalar("outer_k_rank1_weight_count") != 20:
    raise SystemExit("rank1 model should contain 20 l-major/k-minor weights")
basic_begin = scalar("outer_k_rank1_basic_begin")
if scalar("outer_k_rank1_basic_count") != 25:
    raise SystemExit("rank1 outer graph should contain 25 (l,m) primitives")
if any(moment < 0 or moment >= basic_begin
       for moment in integers("two_layer_gate_scalar_moment_indices")):
    raise SystemExit("gate terminal escaped the preserved K=4 graph")
mapping = integers("alpha_moment_mapping")
if not mapping or any(moment < basic_begin for moment in mapping):
    raise SystemExit("outer scalar mapping did not use the K=1 suffix graph")
gate_ordinals = integers("two_layer_gate_scalar_indices")
if gate_ordinals != list(range(len(gate_ordinals))):
    raise SystemExit("rank1 gate ordinals are not canonical q-total")

match = re.search(r"outer_k_rank1_weights\s*=\s*\{([^}]*)\}", text, re.S)
if not match:
    raise SystemExit("missing outer_k_rank1_weights")
weights = [float(x) for x in re.findall(
    r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?", match.group(1))]
if len(weights) != 20:
    raise SystemExit(f"expected 20 rank1 weights, got {len(weights)}")
for l in range(5):
    weights[4*l:4*l+4] = [1.0 + 0.03*l, -0.21 + 0.01*l, 0.14, 0.0]
replacement = "outer_k_rank1_weights = {" + ", ".join(
    f"{value:.15e}" for value in weights) + "}"
text = text[:match.start()] + replacement + text[match.end():]
open(target, "w", encoding="utf-8").write(text)
PY

"$bin" roundtrip-model-dev "$probe_model" "$roundtrip_model"
"$bin" calc-efs "$probe_model" "$one_cfg" "$efs_probe" >/dev/null
"$bin" calc-efs "$roundtrip_model" "$one_cfg" "$efs_roundtrip" >/dev/null
compare_numeric_files "$efs_probe" "$efs_roundtrip"
"$bin" roundtrip-model-dev "$roundtrip_model" "$roundtrip_model2"
cmp "$roundtrip_model" "$roundtrip_model2"

python3 - "$probe_model" "$roundtrip_model" <<'PY'
import re
import sys

legacy_text = open(sys.argv[1], encoding="utf-8").read()
direct_text = open(sys.argv[2], encoding="utf-8").read()
number = r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?"

def vector(text, name):
    match = re.search(rf"{name}\s*=\s*\{{([^}}]*)\}}", text, re.S)
    if not match:
        raise SystemExit(f"missing {name}")
    return [float(x) for x in re.findall(number, match.group(1))]

def scalar(text, name):
    match = re.search(rf"(?m)^\s*{name}\s*=\s*(\d+)\s*$", text)
    if not match:
        raise SystemExit(f"missing {name}")
    return int(match.group(1))

def radial_rows(text):
    try:
        section = text.split("radial_coeffs", 1)[1].split(
            "alpha_moments_count", 1)[0]
    except IndexError:
        raise SystemExit("missing radial coefficient section")
    return [[float(x) for x in re.findall(number, row)]
            for row in re.findall(r"\{([^{}]*)\}", section)]

weights = vector(legacy_text, "outer_k_rank1_weights")
shell = vector(direct_text, "outer_k_rank1_weights")
if len(shell) != len(weights) or any(x != 1.0 for x in shell):
    raise SystemExit("canonical rank1 file does not contain an all-one W shell")

lmax = scalar(legacy_text, "sh_l_max")
kmax = scalar(legacy_text, "sh_k_max")
rb = scalar(legacy_text, "radial_basis_size")
species = scalar(legacy_text, "species_count")
legacy_rows = radial_rows(legacy_text)
direct_rows = radial_rows(direct_text)
if len(legacy_rows) != kmax * (lmax + 1) or len(direct_rows) != len(legacy_rows):
    raise SystemExit("unexpected radial row count")
for k in range(kmax):
    for l in range(lmax + 1):
        mu = k * (lmax + 1) + l
        w = weights[l * kmax + k]
        if len(legacy_rows[mu]) != rb + species:
            raise SystemExit("unexpected radial row width")
        for xi in range(rb):
            expected = w * legacy_rows[mu][xi]
            actual = direct_rows[mu][xi]
            tol = 8.0e-15 * max(1.0, abs(expected), abs(actual))
            if abs(expected - actual) > tol:
                raise SystemExit(
                    f"radial absorption mismatch mu={mu} xi={xi}: "
                    f"expected={expected:.17e} actual={actual:.17e}")
        if direct_rows[mu][rb:] != legacy_rows[mu][rb:]:
            raise SystemExit(f"radial species tail changed for mu={mu}")
for name in (
        "two_layer_gate_radial_coeffs",
        "two_layer_gate_additive_coeffs",
        "two_layer_gate_type_coeffs",
        "two_layer_gate_weights",
        "species_coeffs",
        "moment_coeffs"):
    legacy_values = vector(legacy_text, name)
    direct_values = vector(direct_text, name)
    if len(legacy_values) != len(direct_values):
        raise SystemExit(f"{name} length changed during rank1 absorption")
    for index, (left, right) in enumerate(zip(legacy_values, direct_values)):
        tol = 8.0e-15 * max(1.0, abs(left), abs(right))
        if abs(left-right) > tol:
            raise SystemExit(f"{name}[{index}] changed during rank1 absorption")
print("rank1_direct_radial_absorption_ok rows=", len(direct_rows))
PY

probe_descriptors="$tmp_dir/probe-descriptors.txt"
canonical_descriptors="$tmp_dir/canonical-descriptors.txt"
probe_eij="$tmp_dir/probe-eij.txt"
canonical_eij="$tmp_dir/canonical-eij.txt"
"$bin" calc-descriptors "$probe_model" "$one_cfg" "$probe_descriptors" >/dev/null
"$bin" calc-descriptors "$roundtrip_model" "$one_cfg" "$canonical_descriptors" >/dev/null
"$bin" calc-eij "$probe_model" "$one_cfg" "$probe_eij" >/dev/null
"$bin" calc-eij "$roundtrip_model" "$one_cfg" "$canonical_eij" >/dev/null
compare_numeric_files "$probe_descriptors" "$canonical_descriptors"
compare_numeric_files "$probe_eij" "$canonical_eij"

if [[ -n "$old_bin" ]]; then
  "$old_bin" calc-efs "$probe_model" "$one_cfg" "$efs_old" >/dev/null
  "$old_bin" calc-efs "$roundtrip_model" "$one_cfg" "$efs_old_canonical" >/dev/null
  old_descriptors="$tmp_dir/old-descriptors.txt"
  old_eij="$tmp_dir/old-eij.txt"
  "$old_bin" calc-descriptors "$probe_model" "$one_cfg" "$old_descriptors" >/dev/null
  "$old_bin" calc-eij "$probe_model" "$one_cfg" "$old_eij" >/dev/null
  python3 - "$efs_old" "$efs_probe" "$efs_old_canonical" "$efs_roundtrip" \
               "$old_descriptors" "$probe_descriptors" "$old_eij" "$probe_eij" <<'PY'
import re
import sys

number = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")
for left_path, right_path in zip(sys.argv[1::2], sys.argv[2::2]):
    left = [float(x) for x in number.findall(open(left_path, encoding="utf-8").read())]
    right = [float(x) for x in number.findall(open(right_path, encoding="utf-8").read())]
    if len(left) != len(right):
        raise SystemExit(f"old/new output length differs: {left_path} {right_path}")
    worst = max((abs(a-b) for a, b in zip(left, right)), default=0.0)
    scale = max([1.0] + [abs(x) for x in left] + [abs(x) for x in right])
    if worst > 5.0e-11 * scale:
        raise SystemExit(
            f"old/new rank1 parity failed: {left_path} {right_path} "
            f"worst={worst:.3e} scale={scale:.3e}")
    print(f"old_new_rank1_parity worst={worst:.3e} scale={scale:.3e}")
PY
fi

python3 - "$roundtrip_model" "$active_model" <<'PY'
import re
import sys

source, target = sys.argv[1], sys.argv[2]
text = open(source, encoding="utf-8").read()
match = re.search(r"moment_coeffs\s*=\s*\{([^}]*)\}", text, re.S)
if not match:
    raise SystemExit("roundtrip model is missing moment_coeffs")
coeffs = [float(x) for x in re.findall(
    r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?", match.group(1))]
for q in range(len(coeffs)):
    coeffs[q] = (0.035 + 0.002 * (q % 7)) * (-1.0 if q % 3 == 1 else 1.0)
replacement = "moment_coeffs = {" + ", ".join(
    f"{value:.15e}" for value in coeffs) + "}"
text = text[:match.start()] + replacement + text[match.end():]
open(target, "w", encoding="utf-8").write(text)
PY

legacy_active="$tmp_dir/rank1-legacy-active.sus2"
python3 - "$probe_model" "$active_model" "$legacy_active" <<'PY'
import re
import sys

legacy_path, direct_path, target_path = sys.argv[1:]
legacy = open(legacy_path, encoding="utf-8").read()
direct = open(direct_path, encoding="utf-8").read()
pattern = re.compile(r"moment_coeffs\s*=\s*\{[^}]*\}", re.S)
match = pattern.search(direct)
if not match or not pattern.search(legacy):
    raise SystemExit("missing moment_coeffs while building legacy active fixture")
legacy = pattern.sub(match.group(0), legacy, count=1)
open(target_path, "w", encoding="utf-8").write(legacy)
PY

# Build a finite-difference target close to the active model prediction.  The
# untrained product basis can otherwise have an O(1e6) loss while some W
# derivatives are O(1e-4), so central differences of the loss become a test of
# floating-point cancellation instead of the chain rule.
"$bin" calc-efs "$active_model" "$one_cfg" "$fd_cfg" >/dev/null
python3 - "$fd_cfg" <<'PY'
import sys

path = sys.argv[1]
lines = open(path, encoding="utf-8").readlines()
in_atoms = False
expect_energy = False
expect_stress = False
atom_counter = 0
for index, raw in enumerate(lines):
    stripped = raw.strip()
    if stripped.startswith("AtomData:"):
        in_atoms = True
        continue
    if stripped == "Energy":
        in_atoms = False
        expect_energy = True
        continue
    if stripped.startswith("PlusStress:"):
        expect_stress = True
        continue
    if in_atoms and stripped:
        fields = raw.split()
        if len(fields) >= 8 and fields[0].lstrip("+-").isdigit():
            atom_counter += 1
            for component in range(3):
                fields[5 + component] = f"{float(fields[5 + component]) + 0.01 * ((atom_counter + component) % 5 - 2):.15e}"
            lines[index] = " ".join(fields) + "\n"
    elif expect_energy and stripped:
        lines[index] = f"\t{float(stripped) + 0.5:.15e}\n"
        expect_energy = False
    elif expect_stress and stripped:
        fields = raw.split()
        if len(fields) >= 6:
            fields = [f"{float(value) + 0.02 * (component - 2):.15e}"
                      for component, value in enumerate(fields[:6])]
            lines[index] = "\t" + " \t".join(fields) + "\n"
            expect_stress = False
if atom_counter == 0 or expect_energy or expect_stress:
    raise SystemExit("failed to construct controlled E/F/S finite-difference target")
open(path, "w", encoding="utf-8").writelines(lines)
PY

layout=$("$bin" check-loss-gradient-dev "$active_model" "$fd_cfg" \
  --max-configs=1 \
  --energy-weight=1 --force-weight=0 --stress-weight=0 \
  --radial-smooth=0 --scalar-head-l2=0 --gate-scalar-l2=0 \
  --gate-mix-l2=0 --gate-full-l2=0 --gate-x-l2=0 \
  --coeff-start=0 --coeff-end=1 \
  --displacement=1.0e-5 --abs-tolerance=1 --rel-tolerance=1)
rank1_begin=$(sed -n 's/.*outer_k_rank_begin=\([0-9][0-9]*\).*/\1/p' <<<"$layout")
rank1_end=$(sed -n 's/.*outer_k_rank_end=\([0-9][0-9]*\).*/\1/p' <<<"$layout")
linear_begin=$(sed -n 's/.*linear_begin=\([0-9][0-9]*\).*/\1/p' <<<"$layout")
radial_begin=$(tr ' ' '\n' <<<"$layout" | sed -n 's/^radial_begin=\([0-9][0-9]*\)$/\1/p')
if [[ -z "$rank1_begin" || -z "$rank1_end" || -z "$linear_begin" \
      || -z "$radial_begin" || "$rank1_begin" -ne "$rank1_end" \
      || "$rank1_end" -ne "$linear_begin" ]]; then
  echo "rank1 direct-radial layout still contains an optimizer W block: $layout" >&2
  exit 1
fi

if [[ -n "$old_bin" ]]; then
  old_layout=$("$old_bin" check-loss-gradient-dev "$legacy_active" "$fd_cfg" \
    --max-configs=1 --energy-weight=1 --force-weight=0 --stress-weight=0 \
    --radial-smooth=0 --scalar-head-l2=0 --gate-scalar-l2=0 \
    --gate-mix-l2=0 --gate-full-l2=0 --gate-x-l2=0 \
    --coeff-start=0 --coeff-end=1 \
    --displacement=1.0e-5 --abs-tolerance=1 --rel-tolerance=1)
  old_rank1_begin=$(sed -n 's/.*outer_k_rank_begin=\([0-9][0-9]*\).*/\1/p' <<<"$old_layout")
  if [[ -z "$old_rank1_begin" ]]; then
    echo "failed to discover legacy rank1 W offset: $old_layout" >&2
    exit 1
  fi
  analytic_at() {
    local exe=$1 model=$2 index=$3
    "$exe" check-loss-gradient-dev "$model" "$fd_cfg" \
      --max-configs=1 --energy-weight=1 --force-weight=0 --stress-weight=0 \
      --radial-smooth=0 --scalar-head-l2=0 --gate-scalar-l2=0 \
      --gate-mix-l2=0 --gate-full-l2=0 --gate-x-l2=0 \
      --coeff-start="$index" --coeff-end="$((index+1))" \
      --displacement=1.0e-5 --abs-tolerance=-1 --rel-tolerance=1.0e99 \
      | sed -n 's/.* analytic=\([^ ]*\).*/\1/p'
  }
  # l=0,k=1 -> mu=5 and W=-0.21 in the asymmetric fixture.
  legacy_row_begin=$((radial_begin + 5 * 6))
  direct_grads=()
  legacy_grads=()
  for xi in 0 1 2 3; do
    direct_grads+=("$(analytic_at "$bin" "$active_model" "$((legacy_row_begin+xi))")")
    legacy_grads+=("$(analytic_at "$old_bin" "$legacy_active" "$((legacy_row_begin+xi))")")
  done
  legacy_w_grad=$(analytic_at "$old_bin" "$legacy_active" "$((old_rank1_begin+1))")
  python3 - "$legacy_active" "$legacy_w_grad" \
    "${direct_grads[@]}" "${legacy_grads[@]}" <<'PY'
import re
import sys

path = sys.argv[1]
old_w_grad = float(sys.argv[2])
direct = [float(x) for x in sys.argv[3:7]]
legacy = [float(x) for x in sys.argv[7:11]]
text = open(path, encoding="utf-8").read()
number = r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?"
weights_match = re.search(r"outer_k_rank1_weights\s*=\s*\{([^}]*)\}", text, re.S)
section = text.split("radial_coeffs", 1)[1].split("alpha_moments_count", 1)[0]
rows = [[float(x) for x in re.findall(number, row)]
        for row in re.findall(r"\{([^{}]*)\}", section)]
weights = [float(x) for x in re.findall(number, weights_match.group(1))]
w = weights[1]
c = rows[5][:4]
for xi, (gc, gd) in enumerate(zip(legacy, direct)):
    expected = w*gd
    tol = 2.0e-8*max(1.0, abs(gc), abs(expected))
    if abs(gc-expected) > tol:
        raise SystemExit(
            f"radial chain rule failed xi={xi}: gc={gc:.17e} "
            f"W*gd={expected:.17e}")
expected_w = sum(ci*gdi for ci, gdi in zip(c, direct))
tol = 2.0e-8*max(1.0, abs(old_w_grad), abs(expected_w))
if abs(old_w_grad-expected_w) > tol:
    raise SystemExit(
        f"W chain rule failed: gW={old_w_grad:.17e} "
        f"sum(c*gd)={expected_w:.17e}")
print(f"rank1_chain_rule_ok W={w:.6g} gW={old_w_grad:.6e}")
PY
fi

# Check two signed effective radial rows coefficient-by-coefficient.  With
# lmax=4, k=1 maps to mu=5+l and carries a negative absorbed W in this fixture.
radial_stride=6
for mu in 0 5; do
  row_begin=$((radial_begin + mu * radial_stride))
  row_end=$((row_begin + 4))
  "$bin" check-loss-gradient-dev "$active_model" "$fd_cfg" \
    --max-configs=1 \
    --energy-weight=1 --force-weight=0 --stress-weight=0 \
    --radial-smooth=0 --scalar-head-l2=0 --gate-scalar-l2=0 \
    --gate-mix-l2=0 --gate-full-l2=0 --gate-x-l2=0 \
    --coeff-start="$row_begin" --coeff-end="$row_end" \
    --displacement=1.0e-5 --abs-tolerance=2.0e-6 --rel-tolerance=2.0e-4
done

# The regularizer is now defined on the effective radial coefficients d.  It is
# intentionally not compared with the old R(c), but its own chain rule must be
# self-consistent.
"$bin" check-loss-gradient-dev "$active_model" "$fd_cfg" \
  --max-configs=1 \
  --energy-weight=1 --force-weight=0.1 --stress-weight=0.01 \
  --radial-smooth=1.0e-6 --scalar-head-l2=0 --gate-scalar-l2=0 \
  --gate-mix-l2=0 --gate-full-l2=0 --gate-x-l2=0 \
  --coeff-start="$radial_begin" --coeff-end="$((radial_begin+4))" \
  --displacement=1.0e-5 --abs-tolerance=3.0e-6 --rel-tolerance=3.0e-4

"$bin" check-loss-gradient-direction-dev "$active_model" "$fd_cfg" \
  --max-configs=1 \
  --energy-weight=1 --force-weight=0.1 --stress-weight=0.01 \
  --radial-smooth=0 --scalar-head-l2=0 --gate-scalar-l2=0 \
  --gate-mix-l2=0 --gate-full-l2=0 --gate-x-l2=0 \
  --coeff-start="$radial_begin" --coeff-end="$((radial_begin+20*radial_stride))" \
  --displacement=1.0e-4 --abs-tolerance=2.0e-6 --rel-tolerance=2.0e-3

"$bin" check-efs-fd-dev "$active_model" "$one_cfg" \
  --max-configs=1 --max-atoms=1 \
  --displacement=1.0e-5 --abs-tolerance=2.0e-5 --rel-tolerance=2.0e-3

"$bin" check-linear-components-fd-dev "$active_model" "$one_cfg" \
  --max-configs=1 --max-atoms=1 \
  --displacement=1.0e-5 --abs-tolerance=2.0e-5 --rel-tolerance=2.0e-3

"$bin" calc-descriptors "$active_model" "$one_cfg" "$descriptors" >/dev/null
test -s "$descriptors"
"$bin" calc-partialE "$active_model" "$one_cfg" "$partial" >/dev/null
"$bin" calc-eij "$active_model" "$one_cfg" "$eij" >/dev/null

python3 - "$partial" "$eij" <<'PY'
import sys

partial_sum = 0.0
for raw in open(sys.argv[1], encoding="utf-8"):
    line = raw.strip()
    if line and not line.startswith("#"):
        partial_sum += sum(float(x) for x in line.split()[1:])

eij_sum = 0.0
rows = 0
for raw in open(sys.argv[2], encoding="utf-8"):
    fields = raw.split()
    if not fields:
        continue
    if len(fields) != 4:
        raise SystemExit(f"invalid calc-eij row: {raw.strip()}")
    eij_sum += float(fields[3])
    rows += 1
if rows == 0:
    raise SystemExit("calc-eij produced no rows")
abs_err = abs(eij_sum - partial_sum)
tol = 1.0e-11 * max(1.0, abs(partial_sum), abs(eij_sum))
if abs_err > tol:
    raise SystemExit(
        f"rank1 calc-eij sum mismatch: {eij_sum:.16e} vs {partial_sum:.16e}")
print(f"rank1_calc_eij_sum={eij_sum:.16e} "
      f"partial_sum={partial_sum:.16e} abs_err={abs_err:.3e}")
PY

physical_trace="$tmp_dir/direct-physical.csv"
weight_norm_trace="$tmp_dir/direct-weight-norm.csv"
physical_log="$tmp_dir/direct-physical.log"
weight_norm_log="$tmp_dir/direct-weight-norm.log"
"$bin" train "$active_model" "$fd_cfg" \
  --radial-parameterization=physical \
  --bfgs-coordinate-normalization=none \
  --optimizer-trace-file="$physical_trace" \
  --trained-pot-name="$tmp_dir/direct-physical-trained.sus2" \
  --max-iter=1 --skip-preinit \
  --energy-weight=1 --force-weight=0.1 --stress-weight=0 \
  --radial-smooth=0 >"$physical_log" 2>&1
"$bin" train "$active_model" "$fd_cfg" \
  --radial-parameterization=weight-norm \
  --bfgs-coordinate-normalization=none \
  --optimizer-trace-file="$weight_norm_trace" \
  --trained-pot-name="$tmp_dir/direct-weight-norm-trained.sus2" \
  --max-iter=1 --skip-preinit \
  --energy-weight=1 --force-weight=0.1 --stress-weight=0 \
  --radial-smooth=0 >"$weight_norm_log" 2>&1
grep -q "outer_zero_fallback_rows=5" "$weight_norm_log"
if grep -Eq "Repaired .*radial first|radial first-coeff sign repair" \
     "$physical_log" "$weight_norm_log"; then
  echo "direct rank1 training changed a signed/zero radial row before optimization" >&2
  exit 1
fi
python3 - "$physical_log" "$weight_norm_log" \
          "$physical_trace" "$weight_norm_trace" <<'PY'
import csv
import math
import re
import sys

def startup_loss(path):
    text = open(path, encoding="utf-8").read()
    match = re.search(
        r"optimizer physical startup objective total_loss=([^ ]+)", text)
    if not match:
        raise SystemExit(f"missing physical startup objective in {path}")
    return float(match.group(1))

def validate_trace(path):
    with open(path, newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise SystemExit(f"missing optimizer trace in {path}")
    for row in rows:
        for key, value in row.items():
            if value not in (None, "") and not math.isfinite(float(value)):
                raise SystemExit(f"non-finite optimizer trace value {key}={value}")

physical = startup_loss(sys.argv[1])
normalized = startup_loss(sys.argv[2])
scale = max(1.0, abs(physical), abs(normalized))
if abs(physical-normalized) > 1.0e-13*scale:
    raise SystemExit(
        f"zero-row fallback changed startup loss: {physical:.17e} "
        f"vs {normalized:.17e}")
validate_trace(sys.argv[3])
validate_trace(sys.argv[4])
print(f"direct_rank1_startup_loss={physical:.17e}")
PY

echo "outer K-rank1 direct-radial smoke passed: W=[${rank1_begin},${rank1_end}) radial_begin=${radial_begin}"
