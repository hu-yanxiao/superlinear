#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

bin=${1:-./bin/mlp-sus2}
old_bin=${OLD_BIN:-}
tmp_dir=".local_tmp/sh_superlinear_radial_mixing_gate_tangent_efs_fd"

if [[ ! -x "$bin" ]]; then
  echo "missing mlp-sus2 binary: $bin" >&2
  exit 2
fi
if [[ -n "$old_bin" && ! -x "$old_bin" ]]; then
  echo "missing old binary: $old_bin" >&2
  exit 2
fi

rm -rf "$tmp_dir"
mkdir -p "$tmp_dir"

cfg="$tmp_dir/tiny_asym.cfg"
seed_model="$tmp_dir/seed-random.sus2"
inline_model="$tmp_dir/seed-topology.sus2"
active_model="$tmp_dir/active-c.sus2"
zero_model="$tmp_dir/zero-c.sus2"

cat > "$cfg" <<'CFG'
BEGIN_CFG
 Size
       5
 Supercell
   7.0000000000    0.1000000000    0.0500000000
   0.0000000000    6.8000000000    0.2000000000
   0.0500000000    0.0000000000    7.2000000000
 AtomData:  id type       cartes_x      cartes_y      cartes_z     fx          fy          fz
     1      0       2.000000      2.000000      2.000000      0.030000    -0.020000     0.010000
     2      0       3.300000      2.150000      2.250000     -0.010000     0.040000    -0.030000
     3      0       2.300000      3.350000      2.420000      0.020000     0.010000    -0.040000
     4      0       2.450000      2.550000      3.300000     -0.030000    -0.010000     0.020000
     5      0       3.520000      3.170000      3.050000      0.010000    -0.050000     0.030000
 Energy
    -3.456789
 PlusStress:  xx          yy          zz          yz          xz          xy
    0.030000   -0.020000    0.010000    0.004000   -0.003000    0.002000
END_CFG
CFG

"$bin" train "$cfg" \
  --init-sh \
  --species-count=1 \
  --l-max=0 \
  --k-max=1 \
  --body-order=4 \
  --body-l-max=0,0,0,0 \
  --radial-basis-size=3 \
  --cutoff=5.0 \
  --write-sh-scalar-info \
  --two-layer-gate \
  --two-layer-gate-mode=mu-scalar-full \
  --two-layer-gate-body-order=3 \
  --two-layer-gate-k-rank=1 \
  --outer-k-rank=1 \
  --two-layer-gate-radial-mixing=full \
  --init-params=random \
  --init-random-seed=2026090203 \
  --skip-preinit \
  --max-iter=0 \
  --inline-sh-model="$inline_model" \
  --trained-pot-name="$seed_model" \
  >"$tmp_dir/init.log" 2>&1

python3 - "$seed_model" "$active_model" "$zero_model" <<'PY'
import re
import sys
from pathlib import Path

seed_path, active_path, zero_path = map(Path, sys.argv[1:4])
text = seed_path.read_text(encoding="utf-8")
number = r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?"

def find_int(src, name):
    match = re.search(rf"(?m)^\s*{re.escape(name)}\s*=\s*(\d+)\s*$", src)
    if not match:
        raise SystemExit(f"missing {name}")
    return int(match.group(1))

def rewrite_vector(src, name, values):
    match = re.search(
        rf"(?ms)^\s*{re.escape(name)}\s*=\s*\{{([^}}]*)\}}",
        src,
    )
    if not match:
        raise SystemExit(f"missing vector {name}")
    old = [float(x) for x in re.findall(number, match.group(1))]
    if len(old) != len(values):
        raise SystemExit(f"{name}: expected {len(old)} values, got {len(values)}")
    replacement = (
        f"{name} = {{"
        + ", ".join(f"{x:.15e}" for x in values)
        + "}"
    )
    return src[:match.start()] + replacement + src[match.end():]

def rewrite_radial_coeffs(src, values):
    match = re.search(
        r"(?ms)^(\s*radial_coeffs\s*\n)(.*?)(?=^\s*alpha_moments_count\s*=)",
        src,
    )
    if not match:
        raise SystemExit("missing radial_coeffs block")
    rows = list(re.finditer(r"\{([^}]*)\}", match.group(2)))
    if len(rows) != 1:
        raise SystemExit(f"expected one radial row, found {len(rows)}")
    old = [float(x) for x in re.findall(number, rows[0].group(1))]
    if len(old) != len(values):
        raise SystemExit(
            f"radial_coeffs: expected {len(old)} values, got {len(values)}"
        )
    row = rows[0]
    block = (
        match.group(2)[:row.start()]
        + "{"
        + ", ".join(f"{x:.15e}" for x in values)
        + "}"
        + match.group(2)[row.end():]
    )
    replacement = match.group(1) + block
    return src[:match.start()] + replacement + src[match.end():]

if find_int(text, "species_count") != 1:
    raise SystemExit("fixture must stay C=1")
if find_int(text, "sh_l_max") != 0 or find_int(text, "sh_k_max") != 1:
    raise SystemExit("fixture must stay l=0,k=1")
if find_int(text, "two_layer_gate_radial_mixing_coeff_count") != 3:
    raise SystemExit("unexpected radial-mixing C count")

active = text
active = rewrite_vector(active, "scal_coeffs", [2.90, 2.05])
active = rewrite_radial_coeffs(active, [0.48, -0.37, 0.29, 1.16])
active = rewrite_vector(active, "species_coeffs", [0.19])
active = rewrite_vector(active, "moment_coeffs", [0.42, -0.31, 0.23])
active = rewrite_vector(
    active, "two_layer_gate_radial_coeffs", [0.55, -0.43, 0.34]
)
active = rewrite_vector(active, "two_layer_gate_additive_coeffs", [0.82])
active = rewrite_vector(active, "two_layer_gate_type_coeffs", [1.37])
active = rewrite_vector(active, "two_layer_gate_weights", [0.74, -0.52])
active = rewrite_vector(active, "two_layer_gate_k_rank_weights", [1.63])
active = rewrite_vector(active, "outer_k_rank1_weights", [1.0])
active = rewrite_vector(
    active, "two_layer_gate_radial_mixing_coeffs", [7.0, -5.5, 4.25]
)
zero = rewrite_vector(
    active, "two_layer_gate_radial_mixing_coeffs", [0.0, 0.0, 0.0]
)

active_path.write_text(active, encoding="utf-8")
zero_path.write_text(zero, encoding="utf-8")
PY

layout=$("$bin" check-loss-gradient-dev "$active_model" "$cfg" \
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
gate_radial_begin=$(layout_field gate_radial_begin)
gate_radial_end=$(layout_field gate_radial_end)
gate_additive_begin=$(layout_field gate_additive_begin)
gate_additive_end=$(layout_field gate_additive_end)
gate_type_begin=$(layout_field gate_type_begin)
gate_type_end=$(layout_field gate_type_end)
gate_weight_begin=$(layout_field gate_weight_begin)
gate_weight_end=$(layout_field gate_weight_end)
mixing_begin=$(layout_field gate_radial_mixing_begin)
mixing_end=$(layout_field gate_radial_mixing_end)
gate_k_rank_begin=$(layout_field gate_k_rank_begin)
gate_k_rank_end=$(layout_field gate_k_rank_end)

if [[ "$radial_begin" != "3" || "$radial_end" != "7" \
      || "$mixing_begin" != "14" || "$mixing_end" != "17" ]]; then
  echo "unexpected tiny coefficient layout: $layout" >&2
  exit 1
fi

"$bin" calc-efs "$zero_model" "$cfg" "$tmp_dir/zero.efs.cfg" >/dev/null
"$bin" calc-efs "$active_model" "$cfg" "$tmp_dir/active.efs.cfg" >/dev/null

python3 - "$tmp_dir/zero.efs.cfg" "$tmp_dir/active.efs.cfg" <<'PY'
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
                stress = [float(v) for v in s.split()[:6]]
                read_stress = False
            elif cols is not None and s and s[0].isdigit():
                fields = s.split()
                forces.extend(float(fields[cols[name]]) for name in ("fx", "fy", "fz"))
    return energy, forces, stress

e0, f0, s0 = parse(sys.argv[1])
e1, f1, s1 = parse(sys.argv[2])
de = abs(e1 - e0)
df = max(abs(a - b) for a, b in zip(f0, f1))
ds = max(abs(a - b) for a, b in zip(s0, s1))
if de <= 1.0e-8 or df <= 1.0e-8 or ds <= 1.0e-8:
    raise SystemExit(
        f"radial mixing not active enough: dE={de:.3e} dF={df:.3e} dS={ds:.3e}"
    )
print(f"active-C signal dE={de:.3e} dF={df:.3e} dS={ds:.3e}")
PY

common_grad_args=(
  --max-configs=1
  --radial-smooth=0
  --scalar-head-l2=0
  --gate-scalar-l2=0
  --gate-mix-l2=0
  --gate-full-l2=0
  --gate-x-l2=0
  --displacement=1.0e-5
  --abs-tolerance=3.0e-7
  --rel-tolerance=3.0e-3
)

weights_to_args() {
  case "$1" in
    E) echo "--energy-weight=1 --force-weight=0 --stress-weight=0" ;;
    F) echo "--energy-weight=0 --force-weight=1 --stress-weight=0" ;;
    S) echo "--energy-weight=0 --force-weight=0 --stress-weight=1" ;;
    EFS) echo "--energy-weight=1 --force-weight=0.35 --stress-weight=0.25" ;;
    *) echo "bad weights: $1" >&2; exit 2 ;;
  esac
}

run_gradient_case() {
  local label=$1
  local model=$2
  local range_name=$3
  local begin=$4
  local end=$5
  local hvt=$6
  local weights=$7
  local rows=${8:-0}
  local out="$tmp_dir/${label}_${range_name}_hvt${hvt}_${weights}_rows${rows}.out"

  read -r -a weight_args <<<"$(weights_to_args "$weights")"
  if [[ "$rows" == "1" ]]; then
    SUS2_SH_PRODUCT_ROWS=1 \
      "$bin" check-loss-gradient-dev "$model" "$cfg" \
        "${weight_args[@]}" \
        "${common_grad_args[@]}" \
        --coeff-start="$begin" \
        --coeff-end="$end" >"$out" 2>&1
  else
    SUS2_SH_PRODUCT_ROWS=0 SUS2_SH_PRODUCT_HVT_REVERSE="$hvt" \
      "$bin" check-loss-gradient-dev "$model" "$cfg" \
        "${weight_args[@]}" \
        "${common_grad_args[@]}" \
        --coeff-start="$begin" \
        --coeff-end="$end" >"$out" 2>&1
  fi

  python3 - "$label" "$range_name" "$hvt" "$weights" "$rows" "$(tail -n 1 "$out")" <<'PY'
import math
import re
import sys

label, range_name, hvt, weights, rows, line = sys.argv[1:7]

def field(name):
    match = re.search(rf"(?:^| ){re.escape(name)}=([^ ]+)", line)
    if not match:
        raise SystemExit(f"missing {name} in: {line}")
    return match.group(1)

checked = int(field("checked_coeffs"))
abs_err = float(field("abs_err"))
rel_err = float(field("rel_err"))
if checked <= 0:
    raise SystemExit(f"{label} {range_name}: no coefficients checked")
if not (math.isfinite(abs_err) and math.isfinite(rel_err)):
    raise SystemExit(f"{label} {range_name}: non-finite error in {line}")
print(
    f"{label} {range_name} HVT={hvt} rows={rows} {weights}: "
    f"checked={checked} abs_err={abs_err:.3e} rel_err={rel_err:.3e}"
)
PY
}

run_direction_case() {
  local label=$1
  local model=$2
  local range_name=$3
  local begin=$4
  local end=$5
  local hvt=$6
  local weights=$7
  local out="$tmp_dir/${label}_${range_name}_direction_hvt${hvt}_${weights}.out"

  read -r -a weight_args <<<"$(weights_to_args "$weights")"
  SUS2_SH_PRODUCT_ROWS=0 SUS2_SH_PRODUCT_HVT_REVERSE="$hvt" \
    "$bin" check-loss-gradient-direction-dev "$model" "$cfg" \
      "${weight_args[@]}" \
      --max-configs=1 \
      --radial-smooth=0 \
      --scalar-head-l2=0 \
      --gate-scalar-l2=0 \
      --gate-mix-l2=0 \
      --gate-full-l2=0 \
      --gate-x-l2=0 \
      --displacement=1.0e-5 \
      --abs-tolerance=3.0e-7 \
      --rel-tolerance=3.0e-3 \
      --coeff-start="$begin" \
      --coeff-end="$end" >"$out" 2>&1

  python3 - "$label" "$range_name" "$hvt" "$weights" "$(tail -n 1 "$out")" <<'PY'
import math
import re
import sys

label, range_name, hvt, weights, line = sys.argv[1:6]

def field(name):
    match = re.search(rf"(?:^| ){re.escape(name)}=([^ ]+)", line)
    if not match:
        raise SystemExit(f"missing {name} in: {line}")
    return match.group(1)

checked = int(field("checked_coeffs"))
analytic = float(field("analytic_directional"))
fd = float(field("finite_difference"))
abs_err = float(field("abs_err"))
rel_err = float(field("rel_err"))
if checked <= 0:
    raise SystemExit(f"{label} {range_name}: no coefficients checked")
if abs(analytic) <= 1.0e-10 or abs(fd) <= 1.0e-10:
    raise SystemExit(
        f"{label} {range_name} HVT={hvt} {weights}: zero-C direct gradient vanished "
        f"analytic={analytic:.3e} fd={fd:.3e}"
    )
if not all(math.isfinite(x) for x in (analytic, fd, abs_err, rel_err)):
    raise SystemExit(f"{label} {range_name}: non-finite direction result in {line}")
print(
    f"{label} {range_name} direction HVT={hvt} {weights}: "
    f"checked={checked} analytic={analytic:.3e} fd={fd:.3e} "
    f"abs_err={abs_err:.3e} rel_err={rel_err:.3e}"
)
PY
}

"$bin" check-efs-fd-dev "$active_model" "$cfg" \
  --max-configs=1 \
  --max-atoms=5 \
  --displacement=1.0e-4 \
  --abs-tolerance=2.0e-4 \
  --rel-tolerance=5.0e-3 >/dev/null

ranges=(
  "radial_scaling_shift:1:3"
  "base_radial:${radial_begin}:${radial_end}"
  "gate_radial:${gate_radial_begin}:${gate_radial_end}"
  "gate_additive:${gate_additive_begin}:${gate_additive_end}"
  "gate_type:${gate_type_begin}:${gate_type_end}"
  "gate_weight:${gate_weight_begin}:${gate_weight_end}"
  "gate_k_rank:${gate_k_rank_begin}:${gate_k_rank_end}"
  "radial_mixing:${mixing_begin}:${mixing_end}"
)

for hvt in 0 1; do
  for weights in E F S EFS; do
    for spec in "${ranges[@]}"; do
      IFS=: read -r name begin end <<<"$spec"
      run_gradient_case active "$active_model" "$name" "$begin" "$end" "$hvt" "$weights"
    done
    run_direction_case zero_c_direct "$zero_model" radial_mixing "$mixing_begin" "$mixing_end" "$hvt" "$weights"
  done
done

for spec in "radial_scaling_shift:1:3" \
            "base_radial:${radial_begin}:${radial_end}" \
            "gate_weight:${gate_weight_begin}:${gate_weight_end}" \
            "gate_k_rank:${gate_k_rank_begin}:${gate_k_rank_end}" \
            "radial_mixing:${mixing_begin}:${mixing_end}"; do
  IFS=: read -r name begin end <<<"$spec"
  run_gradient_case product_rows "$active_model" "$name" "$begin" "$end" 0 EFS 1
done

if [[ -n "$old_bin" ]]; then
  for hvt in 0 1; do
    for weights in F S EFS; do
      out="$tmp_dir/old_scaling_shift_hvt${hvt}_${weights}.out"
      read -r -a weight_args <<<"$(weights_to_args "$weights")"
      set +e
      SUS2_SH_PRODUCT_ROWS=0 SUS2_SH_PRODUCT_HVT_REVERSE="$hvt" \
        "$old_bin" check-loss-gradient-dev "$active_model" "$cfg" \
          "${weight_args[@]}" \
          "${common_grad_args[@]}" \
          --coeff-start=1 \
          --coeff-end=3 >"$out" 2>&1
      rc=$?
      set -e
      if (( rc == 0 )); then
        echo "old binary unexpectedly passed active-C scaling/shift $weights HVT=$hvt" >&2
        tail -n 5 "$out" >&2
        exit 1
      fi
      echo "old active-C scaling/shift HVT=$hvt $weights failed as expected: $(tail -n 1 "$out")"
    done
  done
fi

echo "superlinear radial-mixing gate-tangent EFS FD check passed"
