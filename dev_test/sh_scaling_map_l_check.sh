#!/usr/bin/env bash
set -Eeuo pipefail
trap 'echo "scaling_map=L check failed at line ${LINENO}: ${BASH_COMMAND}" >&2' ERR

if [[ $# -ne 1 ]]; then
  echo "usage: $0 /path/to/mlp-sus2" >&2
  exit 2
fi

bin=$1
repo_root=$(cd "$(dirname "$0")/.." && pwd)
tmp_dir=$(mktemp -d "${TMPDIR:-/tmp}/sus2sh-scaling-map-l-XXXXXX")
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
  python3 - "$1" "$2" <<'PY'
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
                values = [float(value) for value in number.findall(line)]
                if len(values) < 8:
                    raise SystemExit(f"malformed AtomData row in {path}")
                out.extend(values[5:8])
                continue
        if stripped == "Energy":
            out.append(float(number.search(lines[index + 1]).group()))
        elif stripped.startswith("PlusStress:") or stripped.startswith("Stress:"):
            out.extend(float(value) for value in number.findall(lines[index + 1]))
    return out

a = physical_values(sys.argv[1])
b = physical_values(sys.argv[2])
if len(a) != len(b):
    raise SystemExit(f"physical output length mismatch: {len(a)} != {len(b)}")
worst = 0.0
for lhs, rhs in zip(a, b):
    if not math.isfinite(lhs) or not math.isfinite(rhs):
        raise SystemExit("non-finite E/F/S value")
    worst = max(worst, abs(lhs - rhs))
    if abs(lhs - rhs) > 2.0e-11 + 2.0e-11 * max(abs(lhs), abs(rhs)):
        raise SystemExit(f"L/LK-expanded E/F/S mismatch: {lhs} vs {rhs}")
print(f"L/LK E/F/S max_abs={worst:.3e}")
PY
}

compare_numeric_file() {
  python3 - "$1" "$2" <<'PY'
import math
import re
import sys

number = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")
a = [float(value) for value in number.findall(open(sys.argv[1]).read())]
b = [float(value) for value in number.findall(open(sys.argv[2]).read())]
if len(a) != len(b) or not a:
    raise SystemExit(f"numeric output mismatch: {len(a)} vs {len(b)}")
for lhs, rhs in zip(a, b):
    if not math.isfinite(lhs) or not math.isfinite(rhs):
        raise SystemExit("non-finite diagnostic output")
    if abs(lhs - rhs) > 2.0e-11 + 2.0e-11 * max(abs(lhs), abs(rhs)):
        raise SystemExit(f"numeric diagnostic mismatch: {lhs} vs {rhs}")
PY
}

check_counts() {
  python3 - "$1" "$2" "$3" <<'PY'
import re
import sys

path, expected_map, expected_count = sys.argv[1:]
text = open(path).read()
mapping = re.search(r"^scaling_map\s*=\s*(\S+)", text, re.M)
if not mapping or mapping.group(1) != expected_map:
    raise SystemExit(f"{path}: expected scaling_map={expected_map}")
match = re.search(r"^scal_coeffs\s*=\s*\{([^}]*)\}", text, re.M | re.S)
if not match:
    raise SystemExit(f"{path}: missing scal_coeffs")
count = len(re.findall(
    r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?",
    match.group(1),
))
if count != int(expected_count):
    raise SystemExit(f"{path}: scal_coeffs count {count} != {expected_count}")
PY
}

expand_l_to_lk() {
  local l_model=$1
  local lk_model=$2
  local seed_distinct=${3:-0}
  python3 - "$l_model" "$lk_model" "$seed_distinct" <<'PY'
import re
import sys

l_path, lk_path, seed_distinct = sys.argv[1], sys.argv[2], int(sys.argv[3])
text = open(l_path).read()
species = 2
kmax = 2
lchannels = 3
pairs = species * species
block_width = 2 * pairs

def vector(text, name):
    match = re.search(rf"^{name}\s*=\s*\{{([^}}]*)\}}", text, re.M | re.S)
    if not match:
        raise SystemExit(f"missing {name}")
    number = r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?"
    return match, [float(value) for value in re.findall(number, match.group(1))]

def replace_vector(text, name, values):
    match, _ = vector(text, name)
    replacement = name + " = {" + ", ".join(
        f"{value:.15e}" for value in values
    ) + "}"
    return text[:match.start()] + replacement + text[match.end():]

match, compact = vector(text, "scal_coeffs")
if len(compact) != lchannels * block_width:
    raise SystemExit(f"unexpected L scal_coeffs count: {len(compact)}")
if seed_distinct:
    compact = []
    for ell in range(lchannels):
        slopes = [3.10 + 0.19 * ell + 0.03 * pair for pair in range(pairs)]
        shifts = [1.55 + 0.13 * ell + 0.02 * pair for pair in range(pairs)]
        compact.extend(slopes + shifts)
    text = replace_vector(text, "scal_coeffs", compact)

blocks = [
    compact[ell * block_width:(ell + 1) * block_width]
    for ell in range(lchannels)
]
expanded = []
for mu in range(kmax * lchannels):
    expanded.extend(blocks[mu % lchannels])

if not re.search(r"^scaling_map\s*=\s*L\s*$", text, re.M):
    raise SystemExit("source model is not scaling_map=L")
lk_text = re.sub(
    r"^scaling_map\s*=\s*L\s*$", "scaling_map = LK", text,
    count=1, flags=re.M,
)
lk_text = replace_vector(lk_text, "scal_coeffs", expanded)
open(l_path, "w").write(text)
open(lk_path, "w").write(lk_text)
PY
}

for basis in RBChebyshev_sss RBLaguerre_log1p RBJacobi_sss; do
  l_init="$tmp_dir/${basis}-L-init.sus2"
  l_model="$tmp_dir/${basis}-L.sus2"
  lk_model="$tmp_dir/${basis}-LK-expanded.sus2"
  "$bin" init-sh "$l_init" "${common[@]}" \
    --radial-basis-type="$basis" --scaling-map=L >/dev/null
  "$bin" roundtrip-model-dev "$l_init" "$l_model" >/dev/null
  expand_l_to_lk "$l_model" "$lk_model" 1
  check_counts "$l_model" L 24
  check_counts "$lk_model" LK 48
  "$bin" calc-efs "$l_model" "$one_cfg" "$tmp_dir/${basis}-L.cfg" >/dev/null
  "$bin" calc-efs "$lk_model" "$one_cfg" "$tmp_dir/${basis}-LK.cfg" >/dev/null
  compare_cfg "$tmp_dir/${basis}-L.cfg" "$tmp_dir/${basis}-LK.cfg"

  roundtrip="$tmp_dir/${basis}-L-roundtrip.sus2"
  "$bin" roundtrip-model-dev "$l_model" "$roundtrip" >/dev/null
  check_counts "$roundtrip" L 24
  "$bin" calc-efs "$roundtrip" "$one_cfg" "$tmp_dir/${basis}-roundtrip.cfg" >/dev/null
  compare_cfg "$tmp_dir/${basis}-L.cfg" "$tmp_dir/${basis}-roundtrip.cfg"

  if [[ $basis == RBChebyshev_sss ]]; then
    "$bin" calc-descriptors "$l_model" "$one_cfg" "$tmp_dir/L.desc" >/dev/null
    "$bin" calc-descriptors "$lk_model" "$one_cfg" "$tmp_dir/LK.desc" >/dev/null
    compare_numeric_file "$tmp_dir/L.desc" "$tmp_dir/LK.desc"
    "$bin" calc-eij "$l_model" "$one_cfg" "$tmp_dir/L.eij" >/dev/null
    "$bin" calc-eij "$lk_model" "$one_cfg" "$tmp_dir/LK.eij" >/dev/null
    compare_numeric_file "$tmp_dir/L.eij" "$tmp_dir/LK.eij"
  fi
done

# Exercise L together with independent gate and outer K ranks.  The rank graph
# remains in its contracted domain; only scal/s sharing changes.
rank_l_init="$tmp_dir/rank-L-init.sus2"
rank_l="$tmp_dir/rank-L.sus2"
rank_lk="$tmp_dir/rank-LK-expanded.sus2"
"$bin" init-sh "$rank_l_init" "${common[@]}" \
  --radial-basis-type=RBChebyshev_sss --scaling-map=L \
  --two-layer-gate --outer-k-rank=2 --two-layer-gate-k-rank=1 >/dev/null
"$bin" roundtrip-model-dev "$rank_l_init" "$rank_l" >/dev/null
expand_l_to_lk "$rank_l" "$rank_lk" 1
"$bin" calc-efs "$rank_l" "$probe_cfg" "$tmp_dir/rank-L.cfg" >/dev/null
"$bin" calc-efs "$rank_lk" "$probe_cfg" "$tmp_dir/rank-LK.cfg" >/dev/null
compare_cfg "$tmp_dir/rank-L.cfg" "$tmp_dir/rank-LK.cfg"

# Activate the model, prove the L/LK expansion still agrees, and check the
# shared-L scal/s gradient so contributions from every k are accumulated.
active_l="$tmp_dir/active-L.sus2"
active_lk="$tmp_dir/active-LK-expanded.sus2"
"$bin" train "$tmp_dir/RBChebyshev_sss-L.sus2" "$one_cfg" \
  --trained-pot-name="$active_l" --scaling-map=L \
  --max-iter=1 --skip-preinit --shift \
  --energy-weight=1 --force-weight=0.2 --stress-weight=0 \
  --radial-smooth=0 >"$tmp_dir/active-train.log" 2>&1
[[ -s "$active_l" ]] || {
  cat "$tmp_dir/active-train.log" >&2
  echo "training did not save the requested active model (requires DEV binary)" >&2
  exit 1
}
python3 - "$tmp_dir/RBChebyshev_sss-L.sus2" "$active_l" <<'PY'
import re
import sys

def parameters(path):
    text = open(path).read()
    return [float(v) for field in re.findall(
        r"(?m)^\s*\w*coeffs\s*=\s*\{([^}]*)\}", text, re.S
    ) for v in re.findall(r"[-+]?(?:\d*\.\d+|\d+\.?\d*)(?:[eE][-+]?\d+)?", field)]

before, after = map(parameters, sys.argv[1:])
if not before or len(before) != len(after) or before == after:
    raise SystemExit("training must change model parameters before the active-model checks")
PY
"$bin" roundtrip-model-dev "$active_l" "$tmp_dir/active-L-roundtrip.sus2" >/dev/null
"$bin" calc-efs "$active_l" "$probe_cfg" "$tmp_dir/active-L-before-roundtrip.cfg" >/dev/null
"$bin" calc-efs "$tmp_dir/active-L-roundtrip.sus2" "$probe_cfg" \
  "$tmp_dir/active-L-after-roundtrip.cfg" >/dev/null
compare_cfg "$tmp_dir/active-L-before-roundtrip.cfg" "$tmp_dir/active-L-after-roundtrip.cfg"
expand_l_to_lk "$active_l" "$active_lk"
"$bin" calc-efs "$active_l" "$probe_cfg" "$tmp_dir/active-L.cfg" >/dev/null
"$bin" calc-efs "$active_lk" "$probe_cfg" "$tmp_dir/active-LK.cfg" >/dev/null
compare_cfg "$tmp_dir/active-L.cfg" "$tmp_dir/active-LK.cfg"

layout=$("$bin" check-loss-gradient-dev "$active_l" "$probe_cfg" \
  --max-configs=1 --energy-weight=1 --force-weight=0.2 --stress-weight=0 \
  --radial-smooth=0 --gate-x-l2=0 --coeff-start=0 --coeff-end=1 \
  --displacement=1.0e-6 --abs-tolerance=1.0e99 --rel-tolerance=1.0e99 |
  grep 'checked_coeffs=')
radial_begin=$(awk '{for (i=1; i<=NF; ++i) if ($i ~ /^radial_begin=/) {sub(/^[^=]*=/, "", $i); print $i}}' <<<"$layout")
scaling_count=24
if [[ ! $radial_begin =~ ^[0-9]+$ || $radial_begin -lt $scaling_count ]]; then
  echo "failed to resolve main L scal/s coefficient range" >&2
  exit 1
fi
scaling_end=$radial_begin
scaling_begin=$((scaling_end - scaling_count))
"$bin" check-loss-gradient-direction-dev "$active_l" "$probe_cfg" \
  --max-configs=1 --energy-weight=1 --force-weight=0.2 --stress-weight=0 \
  --radial-smooth=0 --gate-x-l2=0 \
  --coeff-start="$scaling_begin" --coeff-end="$scaling_end" \
  --displacement=1.0e-4 --abs-tolerance=2.0e-5 --rel-tolerance=2.0e-4 |
  grep 'checked_coeffs='

if "$bin" init-sh "$tmp_dir/bad.sus2" "${common[@]}" \
  --radial-basis-type=RBChebyshev_sss --scaling-map=KL \
  >"$tmp_dir/bad.out" 2>"$tmp_dir/bad.err"; then
  echo "invalid scaling_map=KL unexpectedly succeeded" >&2
  exit 1
fi
grep -q "scaling-map should be K, L, LK, or pair" "$tmp_dir/bad.err"

echo "scaling_map=L trainer checks passed"
