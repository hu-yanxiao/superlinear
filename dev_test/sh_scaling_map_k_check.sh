#!/usr/bin/env bash
set -Eeuo pipefail
trap 'echo "scaling_map=K check failed at line ${LINENO}: ${BASH_COMMAND}" >&2' ERR

if [[ $# -ne 1 ]]; then
  echo "usage: $0 /path/to/mlp-sus2" >&2
  exit 2
fi

bin=$1
repo_root=$(cd "$(dirname "$0")/.." && pwd)
tmp_dir=$(mktemp -d "${TMPDIR:-/tmp}/sus2sh-scaling-map-k-XXXXXX")
trap 'rm -rf "$tmp_dir"' EXIT

one_cfg="$tmp_dir/one.cfg"
awk '{print} /^END_CFG/{exit}' "$repo_root/example/train.cfg" >"$one_cfg"
probe_cfg="$tmp_dir/probe.cfg"
awk 'BEGIN {n=0} /^BEGIN_CFG/{n++} n==2{print} n==2 && /^END_CFG/{exit}' \
  "$repo_root/example/train.cfg" >"$probe_cfg"

common=(
  --species-count=2
  --l-max=2
  --k-max=2
  --body-order=4
  --body-l-max=2,2,2,2
  --radial-basis-size=4
  --cutoff=5.0
  --sh-factor-pruning=q-total
  --init-params=same
  --write-sh-scalar-info
)

compare_cfg() {
  local lhs=$1
  local rhs=$2
  python3 - "$lhs" "$rhs" <<'PY'
import math
import re
import sys

number = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")

def physical_values(path):
    lines = open(path).read().splitlines()
    out = []
    atom_data = False
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("AtomData:"):
            atom_data = True
            continue
        if atom_data:
            if stripped == "Energy":
                atom_data = False
            elif stripped:
                values = [float(x) for x in number.findall(line)]
                if len(values) < 8:
                    raise SystemExit(f"malformed AtomData row in {path}: {line}")
                out.extend(values[5:8])
                continue
        if stripped == "Energy":
            out.append(float(number.search(lines[index + 1]).group()))
        elif stripped.startswith("PlusStress:") or stripped.startswith("Stress:"):
            out.extend(float(x) for x in number.findall(lines[index + 1]))
    return out

a = physical_values(sys.argv[1])
b = physical_values(sys.argv[2])
if len(a) != len(b):
    raise SystemExit(f"physical output length mismatch: {len(a)} != {len(b)}")
worst = 0.0
for x, y in zip(a, b):
    if not math.isfinite(x) or not math.isfinite(y):
        raise SystemExit("non-finite E/F/S value")
    worst = max(worst, abs(x - y))
    if abs(x - y) > 2.0e-11 + 2.0e-11 * max(abs(x), abs(y)):
        raise SystemExit(f"K/LK-expanded E/F/S mismatch: {x} vs {y}")
print(f"K/LK E/F/S max_abs={worst:.3e}")
PY
}

compare_numeric_file() {
  local lhs=$1
  local rhs=$2
  python3 - "$lhs" "$rhs" <<'PY'
import math
import re
import sys

number = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")

def values(path):
    return [float(value) for value in number.findall(open(path).read())]

a = values(sys.argv[1])
b = values(sys.argv[2])
if len(a) != len(b) or not a:
    raise SystemExit(f"numeric output mismatch: {len(a)} vs {len(b)}")
for x, y in zip(a, b):
    if not math.isfinite(x) or not math.isfinite(y):
        raise SystemExit("non-finite diagnostic output")
    if abs(x - y) > 2.0e-11 + 2.0e-11 * max(abs(x), abs(y)):
        raise SystemExit(f"numeric diagnostic mismatch: {x} vs {y}")
PY
}

check_counts() {
  local model=$1
  local expected_map=$2
  local expected_scal=$3
  local expected_gate_scal=${4:-0}
  python3 - "$model" "$expected_map" "$expected_scal" "$expected_gate_scal" <<'PY'
import re
import sys

path, expected_map, expected_scal, expected_gate_scal = sys.argv[1:]
text = open(path).read()

mapping = re.search(r"^scaling_map\s*=\s*(\S+)", text, re.M)
if not mapping or mapping.group(1) != expected_map:
    raise SystemExit(f"{path}: expected scaling_map={expected_map}")

def count_vector(name):
    match = re.search(rf"^{name}\s*=\s*\{{([^}}]*)\}}", text, re.M | re.S)
    if not match:
        raise SystemExit(f"{path}: missing {name}")
    return len(re.findall(
        r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?",
        match.group(1),
    ))

actual_scal = count_vector("scal_coeffs")
if actual_scal != int(expected_scal):
    raise SystemExit(f"{path}: scal_coeffs count {actual_scal} != {expected_scal}")

if int(expected_gate_scal):
    declared = re.search(
        r"^two_layer_gate_scaling_coeff_count\s*=\s*(\d+)", text, re.M
    )
    if not declared or int(declared.group(1)) != int(expected_gate_scal):
        raise SystemExit(f"{path}: wrong decoupled gate scaling count")
    actual_gate = count_vector("two_layer_gate_scaling_coeffs")
    if actual_gate != int(expected_gate_scal):
        raise SystemExit(
            f"{path}: gate scaling vector count {actual_gate} != {expected_gate_scal}"
        )
PY
}

make_k_lk_equivalent() {
  local k_model=$1
  local lk_model=$2
  local with_gate=${3:-0}
  python3 - "$k_model" "$lk_model" "$with_gate" <<'PY'
import re
import sys

k_path, lk_path, with_gate = sys.argv[1], sys.argv[2], int(sys.argv[3])
k_text = open(k_path).read()
species = 2
kmax = 2
lchannels = 3
pairs = species * species

def replace_vector(text, name, values):
    match = re.search(rf"^{name}\s*=\s*\{{([^}}]*)\}}", text, re.M | re.S)
    if not match:
        raise SystemExit(f"missing {name}")
    replacement = (
        name + " = {" + ", ".join(f"{value:.15e}" for value in values) + "}"
    )
    return text[:match.start()] + replacement + text[match.end():]

def distinct_k_blocks(slope0, shift0):
    blocks = []
    for k in range(kmax):
        slopes = [slope0 + 0.17 * k + 0.03 * pair for pair in range(pairs)]
        shifts = [shift0 + 0.11 * k + 0.02 * pair for pair in range(pairs)]
        blocks.append(slopes + shifts)
    return blocks

def flatten(blocks):
    return [value for block in blocks for value in block]

def expand(blocks):
    return flatten([blocks[mu // lchannels] for mu in range(kmax * lchannels)])

main_blocks = distinct_k_blocks(3.10, 1.55)
k_text = replace_vector(k_text, "scal_coeffs", flatten(main_blocks))

if with_gate:
    gate_blocks = distinct_k_blocks(2.85, 1.35)
    k_text = replace_vector(
        k_text, "two_layer_gate_scaling_coeffs", flatten(gate_blocks)
    )

    def vector_count(text, name):
        match = re.search(
            rf"^{name}\s*=\s*\{{([^}}]*)\}}", text, re.M | re.S
        )
        if not match:
            raise SystemExit(f"missing {name}")
        return len(re.findall(
            r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?",
            match.group(1),
        ))

    # Keep all gate signal paths active so the finite-difference check below
    # actually probes the shared-K gate scaling chain rule rather than a
    # trivially zero derivative.
    active_specs = {
        "two_layer_gate_radial_coeffs": (0.031, 0.006, 5),
        "two_layer_gate_additive_coeffs": (0.19, 0.013, 7),
        "two_layer_gate_type_coeffs": (0.73, 0.09, 4),
        "two_layer_gate_body_mix_weights": (0.17, 0.023, 9),
        "two_layer_gate_weights": (0.035, 0.0002, 17),
    }
    for name, (base, step, period) in active_specs.items():
        count = vector_count(k_text, name)
        values = [
            (base + step * (index % period))
            * (-1.0 if index % 5 == 2 else 1.0)
            for index in range(count)
        ]
        k_text = replace_vector(k_text, name, values)

if not re.search(r"^scaling_map\s*=\s*K\s*$", k_text, re.M):
    raise SystemExit("source model is not scaling_map=K")
lk_text = re.sub(
    r"^scaling_map\s*=\s*K\s*$",
    "scaling_map = LK",
    k_text,
    count=1,
    flags=re.M,
)
lk_text = replace_vector(lk_text, "scal_coeffs", expand(main_blocks))
if with_gate:
    expanded_gate = expand(gate_blocks)
    lk_text, replacements = re.subn(
        r"^(two_layer_gate_scaling_coeff_count\s*=\s*)\d+",
        rf"\g<1>{len(expanded_gate)}",
        lk_text,
        count=1,
        flags=re.M,
    )
    if replacements != 1:
        raise SystemExit("missing two_layer_gate_scaling_coeff_count")
    lk_text = replace_vector(
        lk_text, "two_layer_gate_scaling_coeffs", expanded_gate
    )

open(k_path, "w").write(k_text)
open(lk_path, "w").write(lk_text)
PY
}

for basis in RBChebyshev_sss RBLaguerre_log1p RBJacobi_sss; do
  k_init="$tmp_dir/${basis}-K-init.mtp"
  lk_init="$tmp_dir/${basis}-LK-init.mtp"
  k_model="$tmp_dir/${basis}-K.mtp"
  lk_model="$tmp_dir/${basis}-LK.mtp"
  k_pred="$tmp_dir/${basis}-K.cfg"
  lk_pred="$tmp_dir/${basis}-LK.cfg"

  "$bin" init-sh "$k_init" "${common[@]}" \
    --radial-basis-type="$basis" --scaling-map=K >/dev/null
  "$bin" init-sh "$lk_init" "${common[@]}" \
    --radial-basis-type="$basis" --scaling-map=LK >/dev/null
  grep -q "^scaling_map = K$" "$k_init"
  grep -q "^scaling_map = LK$" "$lk_init"
  "$bin" roundtrip-model-dev "$k_init" "$k_model" >/dev/null
  "$bin" roundtrip-model-dev "$lk_init" "$lk_model" >/dev/null
  make_k_lk_equivalent "$k_model" "$lk_model"

  # C=2, K=2, L=3: K has 2*C^2*K=16 scal/s coefficients, while
  # LK has 2*C^2*K*L=48.  Full mu radial rows remain present in both.
  check_counts "$k_model" K 16
  check_counts "$lk_model" LK 48

  "$bin" calc-efs "$k_model" "$one_cfg" "$k_pred" >/dev/null
  "$bin" calc-efs "$lk_model" "$one_cfg" "$lk_pred" >/dev/null
  compare_cfg "$k_pred" "$lk_pred"

  roundtrip="$tmp_dir/${basis}-K-roundtrip.mtp"
  "$bin" roundtrip-model-dev "$k_model" "$roundtrip" >/dev/null
  check_counts "$roundtrip" K 16
  "$bin" calc-efs "$roundtrip" "$one_cfg" "$tmp_dir/${basis}-roundtrip.cfg" >/dev/null
  compare_cfg "$k_pred" "$tmp_dir/${basis}-roundtrip.cfg"

  if [[ $basis == RBChebyshev_sss ]]; then
    "$bin" calc-descriptors "$k_model" "$one_cfg" "$tmp_dir/K.desc" >/dev/null
    "$bin" calc-descriptors "$lk_model" "$one_cfg" "$tmp_dir/LK.desc" >/dev/null
    compare_numeric_file "$tmp_dir/K.desc" "$tmp_dir/LK.desc"
    "$bin" calc-eij "$k_model" "$one_cfg" "$tmp_dir/K.eij" >/dev/null
    "$bin" calc-eij "$lk_model" "$one_cfg" "$tmp_dir/LK.eij" >/dev/null
    compare_numeric_file "$tmp_dir/K.eij" "$tmp_dir/LK.eij"
  fi
done

# Omitting the option must remain exactly the historical LK layout.
default_init="$tmp_dir/default-init.mtp"
default_model="$tmp_dir/default-roundtrip.mtp"
"$bin" init-sh "$default_init" "${common[@]}" \
  --radial-basis-type=RBChebyshev_sss >/dev/null
"$bin" roundtrip-model-dev "$default_init" "$default_model" >/dev/null
check_counts "$default_model" LK 48

# The gate radial map follows the same scal/s sharing rule.  Gate additive
# and full mu radial coefficients remain per mu.
gate_k_init="$tmp_dir/gate-K-init.mtp"
gate_lk_init="$tmp_dir/gate-LK-init.mtp"
gate_k_seed="$tmp_dir/gate-K-seed.mtp"
gate_lk_seed="$tmp_dir/gate-LK-seed.mtp"
gate_k="$tmp_dir/gate-K.mtp"
gate_lk="$tmp_dir/gate-LK.mtp"
"$bin" init-sh "$gate_k_init" "${common[@]}" \
  --radial-basis-type=RBChebyshev_sss \
  --scaling-map=K --two-layer-gate \
  --two-layer-gate-decoupled-scaling >/dev/null
"$bin" init-sh "$gate_lk_init" "${common[@]}" \
  --radial-basis-type=RBChebyshev_sss \
  --scaling-map=LK --two-layer-gate \
  --two-layer-gate-decoupled-scaling >/dev/null
"$bin" roundtrip-model-dev "$gate_k_init" "$gate_k_seed" >/dev/null
"$bin" roundtrip-model-dev "$gate_lk_init" "$gate_lk_seed" >/dev/null
check_counts "$gate_k_seed" K 16 16
check_counts "$gate_lk_seed" LK 48 48
"$bin" train "$gate_k_seed" "$one_cfg" \
  --trained-pot-name="$gate_k" \
  --max-iter=1 \
  --skip-preinit \
  --shift \
  --energy-weight=1 \
  --force-weight=0.2 \
  --stress-weight=0 \
  --radial-smooth=0 >/dev/null
make_k_lk_equivalent "$gate_k" "$gate_lk" 1
check_counts "$gate_k" K 16 16
check_counts "$gate_lk" LK 48 48
"$bin" calc-efs "$gate_k" "$probe_cfg" "$tmp_dir/gate-K.cfg" >/dev/null
"$bin" calc-efs "$gate_lk" "$probe_cfg" "$tmp_dir/gate-LK.cfg" >/dev/null
compare_cfg "$tmp_dir/gate-K.cfg" "$tmp_dir/gate-LK.cfg"

gate_perturbed="$tmp_dir/gate-K-perturbed.mtp"
python3 - "$gate_k" "$gate_perturbed" <<'PY'
import re
import sys

source, target = sys.argv[1:]
text = open(source).read()
match = re.search(
    r"^two_layer_gate_scaling_coeffs\s*=\s*\{([^}]*)\}",
    text,
    re.M | re.S,
)
if not match:
    raise SystemExit("missing two_layer_gate_scaling_coeffs")
number = r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?"
values = [float(value) for value in re.findall(number, match.group(1))]
values[0] += 0.1
replacement = (
    "two_layer_gate_scaling_coeffs = {"
    + ", ".join(f"{value:.15e}" for value in values)
    + "}"
)
text = text[:match.start()] + replacement + text[match.end():]
open(target, "w").write(text)
PY
"$bin" calc-efs "$gate_perturbed" "$probe_cfg" \
  "$tmp_dir/gate-K-perturbed.cfg" >/dev/null
python3 - "$tmp_dir/gate-K.cfg" "$tmp_dir/gate-K-perturbed.cfg" <<'PY'
import re
import sys

number = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")

def physical_values(path):
    lines = open(path).read().splitlines()
    out = []
    atom_data = False
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("AtomData:"):
            atom_data = True
            continue
        if atom_data:
            if stripped == "Energy":
                atom_data = False
            elif stripped:
                values = [float(value) for value in number.findall(line)]
                out.extend(values[5:8])
                continue
        if stripped == "Energy":
            out.append(float(number.search(lines[index + 1]).group()))
        elif stripped.startswith("PlusStress:") or stripped.startswith("Stress:"):
            out.extend(
                float(value) for value in number.findall(lines[index + 1])
            )
    return out

a = physical_values(sys.argv[1])
b = physical_values(sys.argv[2])
if len(a) != len(b):
    raise SystemExit("gate scaling activity output length mismatch")
worst = max((abs(x - y) for x, y in zip(a, b)), default=0.0)
if worst <= 1.0e-10:
    raise SystemExit("decoupled K gate scaling has no observable forward effect")
print(f"decoupled K gate scaling forward activity max_abs={worst:.3e}")
PY

gate_layout=$("$bin" check-loss-gradient-dev "$gate_k" "$probe_cfg" \
  --max-configs=1 \
  --energy-weight=1 \
  --force-weight=0.2 \
  --stress-weight=0 \
  --radial-smooth=0 \
  --gate-x-l2=0 \
  --coeff-start=0 \
  --coeff-end=1 \
  --displacement=1.0e-6 \
  --abs-tolerance=1.0e99 \
  --rel-tolerance=1.0e99 | grep 'checked_coeffs=')
gate_scaling_begin=$(awk '{for (i=1; i<=NF; ++i) if ($i ~ /^gate_scaling_begin=/) {sub(/^[^=]*=/, "", $i); print $i}}' <<<"$gate_layout")
gate_scaling_end=$(awk '{for (i=1; i<=NF; ++i) if ($i ~ /^gate_scaling_end=/) {sub(/^[^=]*=/, "", $i); print $i}}' <<<"$gate_layout")
if [[ ! $gate_scaling_begin =~ ^[0-9]+$ ||
      ! $gate_scaling_end =~ ^[0-9]+$ ||
      $gate_scaling_begin -ge $gate_scaling_end ]]; then
  echo "failed to resolve decoupled K gate scaling coefficient range" >&2
  exit 1
fi
gate_gradient_output=$("$bin" check-loss-gradient-direction-dev "$gate_k" "$probe_cfg" \
  --max-configs=1 \
  --energy-weight=1 \
  --force-weight=0.2 \
  --stress-weight=0 \
  --radial-smooth=0 \
  --gate-x-l2=0 \
  --coeff-start="$gate_scaling_begin" \
  --coeff-end="$gate_scaling_end" \
  --displacement=1.0e-3 \
  --abs-tolerance=3.0e-5 \
  --rel-tolerance=3.0e-4 | grep 'checked_coeffs=')
echo "$gate_gradient_output"
python3 - "$gate_gradient_output" <<'PY'
import re
import sys

line = sys.argv[1]
analytic = float(re.search(r"analytic_directional=([^ ]+)", line).group(1))
finite_difference = float(re.search(r"finite_difference=([^ ]+)", line).group(1))
if max(abs(analytic), abs(finite_difference)) <= 1.0e-8:
    raise SystemExit("decoupled K gate scaling gradient probe is inactive")
PY

# The main chain-rule risk is a shared K scal/s coefficient receiving the sum
# of all l-channel contributions.  An untrained init-params=same model has a
# zero scalar head, so its physical loss is independent of scal/s and a
# finite-difference check would pass vacuously (worst_index=-1, analytic=0).
# First activate the radial/scalar path on one configuration, then check it on
# a different configuration.  Keep an exactly expanded LK copy to verify that
# activation did not weaken the K/LK mathematical-equivalence check.
main_k_active="$tmp_dir/main-K-active.mtp"
main_lk_active="$tmp_dir/main-LK-active.mtp"
"$bin" train "$tmp_dir/RBChebyshev_sss-K.mtp" "$one_cfg" \
  --trained-pot-name="$main_k_active" \
  --max-iter=1 \
  --skip-preinit \
  --shift \
  --energy-weight=1 \
  --force-weight=0.2 \
  --stress-weight=0 \
  --radial-smooth=0 >/dev/null
make_k_lk_equivalent "$main_k_active" "$main_lk_active"
"$bin" calc-efs "$main_k_active" "$probe_cfg" \
  "$tmp_dir/main-K-active.cfg" >/dev/null
"$bin" calc-efs "$main_lk_active" "$probe_cfg" \
  "$tmp_dir/main-LK-active.cfg" >/dev/null
compare_cfg "$tmp_dir/main-K-active.cfg" "$tmp_dir/main-LK-active.cfg"

# Resolve the main scal/s block from the serialized coefficient layout instead
# of assuming that the two species shifts will always occupy indices [0,2).
main_layout=$("$bin" check-loss-gradient-dev "$main_k_active" "$probe_cfg" \
  --max-configs=1 \
  --energy-weight=1 \
  --force-weight=0.2 \
  --stress-weight=0 \
  --radial-smooth=0 \
  --gate-x-l2=0 \
  --coeff-start=0 \
  --coeff-end=1 \
  --displacement=1.0e-6 \
  --abs-tolerance=1.0e99 \
  --rel-tolerance=1.0e99 | grep 'checked_coeffs=')
main_radial_begin=$(awk '{for (i=1; i<=NF; ++i) if ($i ~ /^radial_begin=/) {sub(/^[^=]*=/, "", $i); print $i}}' <<<"$main_layout")
main_scaling_count=16
if [[ ! $main_radial_begin =~ ^[0-9]+$ ||
      $main_radial_begin -lt $main_scaling_count ]]; then
  echo "failed to resolve main K scal/s coefficient range" >&2
  exit 1
fi
main_scaling_end=$main_radial_begin
main_scaling_begin=$((main_scaling_end - main_scaling_count))

main_direction_output=$("$bin" check-loss-gradient-direction-dev \
  "$main_k_active" "$probe_cfg" \
  --max-configs=1 \
  --energy-weight=1 \
  --force-weight=0.2 \
  --stress-weight=0 \
  --radial-smooth=0 \
  --gate-x-l2=0 \
  --coeff-start="$main_scaling_begin" \
  --coeff-end="$main_scaling_end" \
  --displacement=1.0e-4 \
  --abs-tolerance=2.0e-5 \
  --rel-tolerance=2.0e-4 | grep 'checked_coeffs=')
echo "$main_direction_output"
python3 - "$main_direction_output" "$main_scaling_count" <<'PY'
import math
import re
import sys

line, expected_count = sys.argv[1], int(sys.argv[2])
checked = int(re.search(r"checked_coeffs=([^ ]+)", line).group(1))
analytic = float(re.search(r"analytic_directional=([^ ]+)", line).group(1))
finite_difference = float(re.search(r"finite_difference=([^ ]+)", line).group(1))
base_loss = float(re.search(r"base_loss=([^ ]+)", line).group(1))
if checked != expected_count:
    raise SystemExit(f"main K scal/s checked {checked} coefficients, expected {expected_count}")
if not all(math.isfinite(value) for value in (analytic, finite_difference, base_loss)):
    raise SystemExit("main K scal/s gradient probe produced a non-finite value")
if base_loss <= 0.0:
    raise SystemExit("main K scal/s gradient probe has no positive physical loss")
if min(abs(analytic), abs(finite_difference)) <= 1.0e-8:
    raise SystemExit("main K scal/s directional gradient probe is inactive")
PY

check_main_scaling_coordinate() {
  local offset=$1
  local label=$2
  local begin=$((main_scaling_begin + offset))
  local end=$((begin + 1))
  local output

  output=$("$bin" check-loss-gradient-direction-dev \
    "$main_k_active" "$probe_cfg" \
    --max-configs=1 \
    --energy-weight=1 \
    --force-weight=0.2 \
    --stress-weight=0 \
    --radial-smooth=0 \
    --gate-x-l2=0 \
    --coeff-start="$begin" \
    --coeff-end="$end" \
    --displacement=1.0e-4 \
    --abs-tolerance=2.0e-5 \
    --rel-tolerance=2.0e-4 | grep 'checked_coeffs=')
  echo "main K scal/s coordinate=$label $output"
  python3 - "$output" "$label" <<'PY'
import math
import re
import sys

line, label = sys.argv[1:]
checked = int(re.search(r"checked_coeffs=([^ ]+)", line).group(1))
analytic = float(re.search(r"analytic_directional=([^ ]+)", line).group(1))
finite_difference = float(re.search(r"finite_difference=([^ ]+)", line).group(1))
base_loss = float(re.search(r"base_loss=([^ ]+)", line).group(1))
if checked != 1:
    raise SystemExit(f"main K scal/s {label} checked {checked} coefficients, expected 1")
if not all(math.isfinite(value) for value in (analytic, finite_difference, base_loss)):
    raise SystemExit(f"main K scal/s {label} gradient probe is non-finite")
if base_loss <= 0.0:
    raise SystemExit(f"main K scal/s {label} probe has no positive physical loss")
if min(abs(analytic), abs(finite_difference)) <= 1.0e-8:
    raise SystemExit(f"main K scal/s {label} directional gradient probe is inactive")
PY
}

# The single-species probe activates pair (0,0).  In each K block the four
# slopes precede the four shifts, so these offsets independently exercise both
# coefficient kinds in both K channels.  Each coefficient is shared by all
# three l channels; therefore every finite difference checks their summed
# chain-rule contribution.
check_main_scaling_coordinate 0 slope-k0-pair00
check_main_scaling_coordinate 4 shift-k0-pair00
check_main_scaling_coordinate 8 slope-k1-pair00
check_main_scaling_coordinate 12 shift-k1-pair00

# Inline initialization must recognize --scaling-map as an SH construction
# option instead of falling back to the historical malformed train command.
"$bin" train "$one_cfg" "${common[@]}" \
  --radial-basis-type=RBChebyshev_sss \
  --scaling-map=K \
  --inline-sh-model="$tmp_dir/inline-K.mtp" \
  --trained-pot-name="$tmp_dir/inline-K-trained.mtp" \
  --max-iter=0 >"$tmp_dir/inline.log" 2>&1
grep -q "^scaling_map = K$" "$tmp_dir/inline-K.mtp"
grep -q "inline initial model written" "$tmp_dir/inline.log"

if "$bin" train "$tmp_dir/RBChebyshev_sss-K.mtp" "$one_cfg" \
    --scaling-map=LK --max-iter=0 >"$tmp_dir/map-mismatch.log" 2>&1; then
  echo "train unexpectedly converted an existing K model to LK" >&2
  exit 1
fi
grep -q "must match the loaded model" "$tmp_dir/map-mismatch.log"

if "$bin" init-sh "$tmp_dir/invalid-KL.mtp" "${common[@]}" \
    --radial-basis-type=RBChebyshev_sss --scaling-map=KL \
    >"$tmp_dir/invalid-KL.log" 2>&1; then
  echo "init-sh unexpectedly accepted scaling_map=KL for SUS2-SH" >&2
  exit 1
fi
grep -q "K, L, or LK" "$tmp_dir/invalid-KL.log"

echo "SUS2-SH scaling_map=K checks passed"
