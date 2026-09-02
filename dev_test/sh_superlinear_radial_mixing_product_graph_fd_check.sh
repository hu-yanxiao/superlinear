#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

patched_bin=${PATCHED_BIN:-./bin/mlp-sus2}
old_bin=${OLD_BIN:-}
tmp_dir=".codex_tmp/sh_superlinear_radial_mixing_product_graph_fd"

if [[ ! -x "$patched_bin" ]]; then
  echo "missing patched binary: $patched_bin" >&2
  exit 2
fi
if [[ -n "$old_bin" && ! -x "$old_bin" ]]; then
  echo "missing old binary: $old_bin" >&2
  exit 2
fi

rm -rf "$tmp_dir"
mkdir -p "$tmp_dir"

cfg="$tmp_dir/asym8.cfg"
zero_model="$tmp_dir/full-radial-zero.mtp"
active_model="$tmp_dir/full-radial-active.mtp"
zero_out="$tmp_dir/full-radial-zero.efs.cfg"
active_out="$tmp_dir/full-radial-active.efs.cfg"

cat > "$cfg" <<'CFG'
BEGIN_CFG
 Size
       8
 Supercell
   8.5000000000    0.0000000000    0.0000000000
   0.0000000000    8.5000000000    0.0000000000
   0.0000000000    0.0000000000    8.5000000000
 AtomData:  id type       cartes_x      cartes_y      cartes_z     fx          fy          fz
     1      0       1.130000      1.720000      1.410000      0.030000    -0.020000     0.010000
     2      1       2.470000      1.190000      3.060000     -0.010000     0.040000    -0.030000
     3      0       3.810000      2.680000      1.970000      0.020000     0.010000    -0.040000
     4      1       5.080000      3.330000      4.440000     -0.030000    -0.010000     0.020000
     5      0       1.860000      5.210000      5.640000      0.010000    -0.050000     0.030000
     6      1       4.620000      5.870000      2.580000     -0.020000     0.030000     0.040000
     7      0       6.190000      1.930000      5.160000      0.050000    -0.010000    -0.020000
     8      1       2.940000      6.360000      1.240000     -0.040000     0.020000     0.010000
 Energy
    -12.345678
 PlusStress:  xx          yy          zz          yz          xz          xy
    0.030000   -0.020000    0.010000    0.004000   -0.003000    0.002000
END_CFG
CFG

"$patched_bin" init-sh "$zero_model" \
  --species-count=2 \
  --l-max=2 \
  --k-max=3 \
  --body-order=4 \
  --body-l-max=2,2,2,2 \
  --radial-basis-size=4 \
  --cutoff=5.0 \
  --write-sh-scalar-info \
  --init-params=same \
  --two-layer-gate \
  --two-layer-gate-mode=mu-scalar-full \
  --two-layer-gate-body-order=3 \
  --two-layer-gate-k-rank=2 \
  --outer-k-rank=2 \
  --two-layer-gate-radial-mixing=full >/dev/null

python3 - "$zero_model" "$active_model" <<'PY'
import re
import sys
from pathlib import Path

zero_path, active_path = map(Path, sys.argv[1:3])
text = zero_path.read_text(encoding="utf-8")
number = r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?"

def rewrite_vector(src, name, make_values):
    match = re.search(
        rf"(?ms)^\s*{re.escape(name)}\s*=\s*\{{([^}}]*)\}}",
        src,
    )
    if not match:
        if name == "species_coeffs":
            old_count = coeff_count("species_count")
        elif name == "moment_coeffs":
            old_count = coeff_count("alpha_scalar_moments")
        else:
            raise SystemExit(f"missing vector {name}")
        values = make_values(old_count)
        return (
            src.rstrip()
            + "\n"
            + f"{name} = {{"
            + ", ".join(f"{x:.15e}" for x in values)
            + "}\n"
        )
    old = [float(x) for x in re.findall(number, match.group(1))]
    values = make_values(len(old))
    replacement = f"{name} = {{" + ", ".join(f"{x:.15e}" for x in values) + "}"
    return src[:match.start()] + replacement + src[match.end():]

def coeff_count(name):
    match = re.search(rf"(?m)^\s*{re.escape(name)}\s*=\s*(\d+)\s*$", text)
    if not match:
        raise SystemExit(f"missing {name}")
    return int(match.group(1))

mix_count = coeff_count("two_layer_gate_radial_mixing_coeff_count")
if mix_count != 48:
    raise SystemExit(f"unexpected radial mixing coeff count: {mix_count}")

shared = text
shared_vectors = {
    "species_coeffs": lambda n: [0.11 * (i + 1) for i in range(n)],
    "moment_coeffs": lambda n: [
        2.0 * (0.075 + 0.0017 * (i % 13)) * (-1.0 if i % 4 == 1 else 1.0)
        for i in range(n)
    ],
    "two_layer_gate_radial_coeffs": lambda n: [
        (0.13 + 0.006 * (i % 7)) * (-1.0 if i % 6 == 4 else 1.0)
        for i in range(n)
    ],
    "two_layer_gate_additive_coeffs": lambda n: [
        (0.21 + 0.009 * (i % 5)) * (-1.0 if i % 3 == 1 else 1.0)
        for i in range(n)
    ],
    "two_layer_gate_type_coeffs": lambda n: [
        (0.68 + 0.013 * (i % 7)) * (-1.0 if i % 4 == 2 else 1.0)
        for i in range(n)
    ],
    "two_layer_gate_weights": lambda n: [
        (0.047 + 0.0008 * (i % 17)) * (-1.0 if i % 5 == 3 else 1.0)
        for i in range(n)
    ],
    "two_layer_gate_k_rank_weights": lambda n: [
        (0.72 + 0.031 * (i % 9)) * (-1.0 if i % 4 == 1 else 1.0)
        for i in range(n)
    ],
    "outer_k_rank1_weights": lambda n: [
        (0.83 + 0.027 * (i % 11)) * (-1.0 if i % 5 == 2 else 1.0)
        for i in range(n)
    ],
}
for name, make_values in shared_vectors.items():
    shared = rewrite_vector(shared, name, make_values)

zero = rewrite_vector(shared, "two_layer_gate_radial_mixing_coeffs", lambda n: [0.0] * n)
active = rewrite_vector(
    shared,
    "two_layer_gate_radial_mixing_coeffs",
    lambda n: [8.5 * ((i % 7) - 3) / 3.0 for i in range(n)],
)

zero_path.write_text(zero, encoding="utf-8")
active_path.write_text(active, encoding="utf-8")
PY

"$patched_bin" calc-efs "$zero_model" "$cfg" "$zero_out" >/dev/null
"$patched_bin" calc-efs "$active_model" "$cfg" "$active_out" >/dev/null

python3 - "$zero_out" "$active_out" <<'PY'
import sys

def parse(path):
    energy = None
    forces = []
    stress = None
    cols = None
    read_energy = False
    read_stress = False
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            s = line.strip()
            if s.startswith("AtomData:"):
                headers = s.split()[1:]
                cols = {name: headers.index(name) for name in ("fx", "fy", "fz")}
            elif s.startswith("Energy"):
                read_energy = True
            elif read_energy and s:
                energy = float(s.split()[0])
                read_energy = False
            elif s.startswith("PlusStress:"):
                read_stress = True
            elif read_stress and s:
                vals = [float(v) for v in s.split()[:6]]
                stress = [vals[0], vals[1], vals[2], vals[3], vals[4], vals[5]]
                read_stress = False
            elif cols is not None and s and s[0].isdigit():
                f = s.split()
                forces.extend(float(f[cols[name]]) for name in ("fx", "fy", "fz"))
    return energy, forces, stress

e0, f0, s0 = parse(sys.argv[1])
e1, f1, s1 = parse(sys.argv[2])
de = abs(e1 - e0)
df = max(abs(a - b) for a, b in zip(f0, f1))
ds = max(abs(a - b) for a, b in zip(s0, s1))
if de <= 1.0e-8 or df <= 1.0e-8 or ds <= 1.0e-8:
    raise SystemExit(
        f"radial mixing not strongly active: dE={de:.3e} dF={df:.3e} dS={ds:.3e}"
    )
print(f"nonzero_radial_mixing_effect dE={de:.3e} dF={df:.3e} dS={ds:.3e}")
PY

layout=$("$patched_bin" check-loss-gradient-dev "$active_model" "$cfg" \
  --max-configs=1 \
  --energy-weight=0 \
  --force-weight=1 \
  --stress-weight=0 \
  --radial-smooth=0 \
  --scalar-head-l2=0 \
  --gate-scalar-l2=0 \
  --gate-mix-l2=0 \
  --gate-full-l2=0 \
  --gate-x-l2=0 \
  --displacement=1.0e-5 \
  --abs-tolerance=1e99 \
  --rel-tolerance=1e99 | tail -n 1)

layout_field() {
  local key=$1
  tr ' ' '\n' <<<"$layout" | sed -n "s/^${key}=//p" | head -n 1
}

radial_begin=$(layout_field radial_begin)
radial_end=$(layout_field radial_end)
mixing_begin=$(layout_field gate_radial_mixing_begin)
mixing_end=$(layout_field gate_radial_mixing_end)
linear_begin=$(layout_field linear_begin)

if [[ -z "$radial_begin" || -z "$radial_end" || -z "$mixing_begin" || -z "$mixing_end" || -z "$linear_begin" ]]; then
  echo "failed to parse coefficient layout: $layout" >&2
  exit 1
fi
if (( linear_begin <= mixing_end )); then
  echo "unexpected layout without linear/product coefficients: $layout" >&2
  exit 1
fi

run_gradient_case() {
  local label=$1
  local bin=$2
  local expected=$3
  local range_name=$4
  local begin=$5
  local end=$6
  local hvt=$7
  local weights=$8
  local model=${9:-$active_model}
  local ew=0
  local fw=0
  local sw=0
  case "$weights" in
    E) ew=1 ;;
    F) fw=1 ;;
    S) sw=1 ;;
    EFS) ew=1; fw=0.3; sw=0.2 ;;
    *) echo "bad weight mode: $weights" >&2; exit 2 ;;
  esac

  local out="$tmp_dir/${label}_${range_name}_hvt${hvt}_${weights}.out"
  set +e
  SUS2_SH_PRODUCT_ROWS=0 SUS2_SH_PRODUCT_HVT_REVERSE="$hvt" \
    "$bin" check-loss-gradient-dev "$model" "$cfg" \
      --max-configs=1 \
      --energy-weight="$ew" \
      --force-weight="$fw" \
      --stress-weight="$sw" \
      --radial-smooth=0 \
      --scalar-head-l2=0 \
      --gate-scalar-l2=0 \
      --gate-mix-l2=0 \
      --gate-full-l2=0 \
      --gate-x-l2=0 \
      --displacement=1.0e-5 \
      --abs-tolerance=1.0e-7 \
      --rel-tolerance=1.0e-3 \
      --coeff-start="$begin" \
      --coeff-end="$end" >"$out" 2>&1
  local rc=$?
  set -e

  local line
  line=$(tail -n 1 "$out")
  if [[ "$expected" == "pass" ]]; then
    if (( rc != 0 )); then
      echo "$label $range_name HVT=$hvt $weights expected pass but failed" >&2
      tail -n 5 "$out" >&2
      exit 1
    fi
  else
    if (( rc == 0 )); then
      echo "$label $range_name HVT=$hvt $weights expected failure but passed" >&2
      tail -n 5 "$out" >&2
      exit 1
    fi
  fi

  python3 - "$label" "$range_name" "$hvt" "$weights" "$expected" "$line" <<'PY'
import re
import sys

label, range_name, hvt, weights, expected, line = sys.argv[1:7]

def field(name, default=None):
    match = re.search(rf"(?:^| ){re.escape(name)}=([^ ]+)", line)
    if not match:
        if default is not None:
            return default
        raise SystemExit(f"missing {name} in: {line}")
    return match.group(1)

checked = int(field("checked_coeffs"))
abs_err = float(field("abs_err", "nan"))
rel_err = float(field("rel_err", "nan"))
if checked <= 0:
    raise SystemExit(f"{label} {range_name} HVT={hvt} {weights}: no coefficients checked")
if expected == "fail" and not (abs_err > 1.0e-7 or rel_err > 1.0e-3):
    raise SystemExit(
        f"{label} {range_name} HVT={hvt} {weights}: failure was not the expected gradient mismatch"
    )
print(
    f"{label} {range_name} HVT={hvt} {weights} {expected}: "
    f"checked={checked} abs_err={abs_err:.3e} rel_err={rel_err:.3e}"
)
PY
}

"$patched_bin" check-efs-fd-dev "$active_model" "$cfg" \
  --max-configs=1 \
  --max-atoms=8 \
  --displacement=1.0e-4 \
  --abs-tolerance=1.0e-4 \
  --rel-tolerance=2.0e-3 >/dev/null

for hvt in 0 1; do
  for weights in E F S EFS; do
    run_gradient_case patched "$patched_bin" pass base_radial "$radial_begin" "$radial_end" "$hvt" "$weights"
    run_gradient_case patched "$patched_bin" pass radial_mixing "$mixing_begin" "$mixing_end" "$hvt" "$weights"
    run_gradient_case patched_zero_c "$patched_bin" pass radial_mixing "$mixing_begin" "$mixing_end" "$hvt" "$weights" "$zero_model"
  done
done

if [[ -n "$old_bin" ]]; then
  for hvt in 0 1; do
    run_gradient_case old "$old_bin" pass base_radial "$radial_begin" "$radial_end" "$hvt" E
    for weights in F S EFS; do
      run_gradient_case old "$old_bin" fail base_radial "$radial_begin" "$radial_end" "$hvt" "$weights"
    done
  done
else
  echo "OLD_BIN not set; skipping old f4bc77a failure assertions"
fi

rows_log="$tmp_dir/product_rows_trace.log"
SUS2_SH_PRODUCT_ROWS=1 SUS2_SH_PRODUCT_ROWS_TRACE=1 \
  "$patched_bin" calc-efs "$active_model" "$cfg" "$tmp_dir/product_rows_probe.cfg" \
    >"$rows_log" 2>&1
if grep -q "SUS2-SH product rows enabled" "$rows_log"; then
  run_gradient_case patched_rows "$patched_bin" pass base_radial "$radial_begin" "$radial_end" 0 EFS
else
  echo "product_rows_unavailable_for_radial_mixing_outer_rank_path"
fi

echo "superlinear radial-mixing product-graph FD check passed"
