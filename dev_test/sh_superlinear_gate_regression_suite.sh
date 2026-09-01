#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

bin=./bin/mlp-sus2
if [[ ! -x "$bin" ]]; then
  echo "missing $bin; build it first with ./configure && make mlp" >&2
  exit 2
fi

run_check() {
  local name=$1
  shift
  echo "[superlinear-suite] BEGIN $name"
  "$@"
  echo "[superlinear-suite] PASS  $name"
}

run_two_layer_radial_mixing_suite() {
  local tmp_dir=.codex_tmp/sh_two_layer_gate_radial_mixing_suite
  mkdir -p "$tmp_dir"

  local cfg="$tmp_dir/asym8.cfg"
  local cfg_rot="$tmp_dir/asym8_rot.cfg"
  local base_raw_model="$tmp_dir/base-no-mixing.raw.mtp"
  local full_zero_raw_model="$tmp_dir/full-radial-zero.raw.mtp"
  local base_model="$tmp_dir/base-no-mixing.mtp"
  local full_zero_model="$tmp_dir/full-radial-zero.mtp"
  local full_active_model="$tmp_dir/full-radial-active.mtp"
  local full_active_roundtrip="$tmp_dir/full-radial-active.roundtrip.mtp"

  local base_out="$tmp_dir/base.efs.cfg"
  local zero_out="$tmp_dir/full_zero.efs.cfg"
  local active_out="$tmp_dir/full_active.efs.cfg"
  local active_roundtrip_out="$tmp_dir/full_active_roundtrip.efs.cfg"

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

python3 - "$cfg" "$cfg_rot" <<'PY'
import re
import sys

src, dst = sys.argv[1], sys.argv[2]
L = 8.5
c = L / 2.0
lines = open(src, encoding="utf-8").read().splitlines()

atom_re = re.compile(
    r"\s*\d+\s+\d+(?:\s+[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?){6}\s*"
)
in_atoms = False
rotate_next_stress = False
out_lines = []
for line in lines:
    marker = line.lstrip()
    if marker.startswith("AtomData:"):
        in_atoms = True
        out_lines.append(line)
        continue
    if marker.startswith("Energy"):
        in_atoms = False
    if in_atoms and atom_re.fullmatch(line):
        aid, typ, x, y, z, fx, fy, fz = line.split()[:8]
        x, y, z = float(x), float(y), float(z)
        fx, fy, fz = float(fx), float(fy), float(fz)
        xr, yr = c - (y - c), c + (x - c)
        out_lines.append(
            f"{int(aid):6d} {int(typ):6d} {xr:14.8f} {yr:14.8f} {z:14.8f}"
            f" {-fy:14.8f} {fx:14.8f} {fz:14.8f}"
        )
        continue
    if marker.startswith("PlusStress:"):
        rotate_next_stress = True
        out_lines.append(line)
        continue
    if rotate_next_stress and marker:
        xx, yy, zz, yz, xz, xy = map(float, marker.split()[:6])
        # S' = R S R^T for R = [[0,-1,0],[1,0,0],[0,0,1]].
        out_lines.append(
            f"   {yy:.10f}   {xx:.10f}   {zz:.10f}"
            f"   {xz:.10f}   {-yz:.10f}   {-xy:.10f}"
        )
        rotate_next_stress = False
        continue
    out_lines.append(line)

open(dst, "w", encoding="utf-8").write("\n".join(out_lines) + "\n")
PY

  local common_args=(
    --species-count=2
    --l-max=2
    --k-max=3
    --body-order=4
    --body-l-max=2,2,2,2
    --radial-basis-size=4
    --cutoff=5.0
    --write-sh-scalar-info
    --init-params=same
    --two-layer-gate
    --two-layer-gate-mode=mu-scalar-full
    --two-layer-gate-body-order=3
    --two-layer-gate-k-rank=2
    --outer-k-rank=2
  )

  "$bin" init-sh "$base_raw_model" "${common_args[@]}" >/dev/null
  "$bin" init-sh "$full_zero_raw_model" "${common_args[@]}" \
    --two-layer-gate-radial-mixing=full >/dev/null
  "$bin" roundtrip-model-dev "$base_raw_model" "$base_model" >/dev/null
  "$bin" roundtrip-model-dev "$full_zero_raw_model" "$full_zero_model" >/dev/null

python3 - "$full_zero_model" "$full_active_model" <<'PY'
import re
import sys
from pathlib import Path

full_zero_model, full_active_model = sys.argv[1:3]

full_zero_text = Path(full_zero_model).read_text(encoding="utf-8")
def find_int(name, text):
    m = re.search(rf"^\s*{re.escape(name)}\s*=\s*(\d+)\s*$", text, re.M)
    if not m:
        return None
    return int(m.group(1))

if "two_layer_gate_radial_mixing = full" not in full_zero_text:
    raise SystemExit("radial mixing mode marker missing from full-zero model")

count = find_int("two_layer_gate_radial_mixing_coeff_count", full_zero_text)
if count is None:
    raise SystemExit("two_layer_gate_radial_mixing_coeff_count missing")

zero_coeffs = [0.0 for _ in range(count)]
active_coeffs = [0.17 * ((i % 7) - 3) / 3.0 for i in range(count)]

pattern = r"^\s*two_layer_gate_radial_mixing_coeffs\s*=\s*\{[^\n]*\}\s*$"
if not re.search(pattern, full_zero_text, re.M):
    raise SystemExit("two_layer_gate_radial_mixing_coeffs missing")

def format_coeffs(coeffs):
    return "two_layer_gate_radial_mixing_coeffs = {" + ", ".join(f"{x:.15e}" for x in coeffs) + "}"

def replace_coeffs(text, values):
    return re.sub(
        pattern,
        format_coeffs(values),
        text,
        count=1,
        flags=re.M,
    )

full_zero_text = replace_coeffs(full_zero_text, zero_coeffs)
full_active_text = replace_coeffs(full_zero_text, active_coeffs)

number = r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?"

def rewrite_vector(text, name, make_values):
    match = re.search(
        rf"(?m)^\s*{re.escape(name)}\s*=\s*\{{([^}}]*)\}}",
        text,
        re.S,
    )
    if not match:
        raise SystemExit(f"missing active fixture vector {name}")
    old = [float(x) for x in re.findall(number, match.group(1))]
    values = make_values(len(old))
    if len(values) != len(old):
        raise SystemExit(f"{name}: replacement size {len(values)} != {len(old)}")
    replacement = f"{name} = {{" + ", ".join(f"{x:.15e}" for x in values) + "}"
    return text[:match.start()] + replacement + text[match.end():]

active_vectors = {
    "species_coeffs": lambda n: [0.11 * (i + 1) for i in range(n)],
    "moment_coeffs": lambda n: [
        (0.075 + 0.0017 * (i % 13)) * (-1.0 if i % 4 == 1 else 1.0)
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
for name, make_values in active_vectors.items():
    full_active_text = rewrite_vector(full_active_text, name, make_values)

Path(full_zero_model).write_text(full_zero_text, encoding="utf-8")
Path(full_active_model).write_text(full_active_text, encoding="utf-8")

if find_int("two_layer_gate_radial_mixing_coeff_count", full_zero_text) != count:
    raise SystemExit("radial mixing coeff_count changed unexpectedly")
PY

  local layout
  layout=$("$bin" check-loss-gradient-dev "$full_zero_model" "$cfg" \
    --max-configs=1 \
    --energy-weight=1 \
    --force-weight=1 \
    --stress-weight=1 \
    --gate-x-l2=0 \
    --displacement=1.0e-3 \
    --abs-tolerance=1e99 \
    --rel-tolerance=1e99 | tail -n 1)

  local radial_begin radial_end gate_radial_begin gate_radial_end \
        gate_scaling_begin gate_scaling_end \
        gate_additive_begin gate_additive_end \
        gate_type_begin gate_type_end \
        gate_weight_begin gate_weight_end \
        gate_radial_mixing_begin gate_radial_mixing_end \
        gate_k_rank_begin gate_k_rank_end \
        outer_k_rank_begin outer_k_rank_end

  layout_field() {
    local key=$1
    tr ' ' '\n' <<<"$layout" | sed -n "s/^${key}=//p" | head -n 1
  }
  radial_begin=$(layout_field radial_begin)
  radial_end=$(layout_field radial_end)
  gate_radial_begin=$(layout_field gate_radial_begin)
  gate_radial_end=$(layout_field gate_radial_end)
  gate_scaling_begin=$(layout_field gate_scaling_begin)
  gate_scaling_end=$(layout_field gate_scaling_end)
  gate_additive_begin=$(layout_field gate_additive_begin)
  gate_additive_end=$(layout_field gate_additive_end)
  gate_type_begin=$(layout_field gate_type_begin)
  gate_type_end=$(layout_field gate_type_end)
  gate_weight_begin=$(layout_field gate_weight_begin)
  gate_weight_end=$(layout_field gate_weight_end)
  gate_radial_mixing_begin=$(layout_field gate_radial_mixing_begin)
  gate_radial_mixing_end=$(layout_field gate_radial_mixing_end)
  gate_k_rank_begin=$(layout_field gate_k_rank_begin)
  gate_k_rank_end=$(layout_field gate_k_rank_end)
  outer_k_rank_begin=$(layout_field outer_k_rank_begin)
  outer_k_rank_end=$(layout_field outer_k_rank_end)

  "$bin" calc-efs "$base_model" "$cfg" "$base_out" >/dev/null
  "$bin" calc-efs "$full_zero_model" "$cfg" "$zero_out" >/dev/null
  python3 - "$base_out" "$zero_out" <<'PY'
import re
import sys

def parse_vals(path):
    text = open(path, encoding="utf-8").read()
    return [float(x) for x in re.findall(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?", text)]

a = parse_vals(sys.argv[1])
b = parse_vals(sys.argv[2])
if len(a) != len(b):
    raise SystemExit(f"length mismatch: {len(a)} vs {len(b)}")
for x, y in zip(a, b):
    if abs(x - y) > 1.0e-12 * max(1.0, abs(x), abs(y)):
        raise SystemExit("zero radial-mixing output mismatch")
print("zero_mixing_parity ok")
PY

  "$bin" calc-efs "$full_active_model" "$cfg" "$active_out" >/dev/null
  python3 - "$zero_out" "$active_out" <<'PY'
import re
import math
import sys

def parse(path):
    en = None
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
                cols = {
                    "fx": headers.index("fx"),
                    "fy": headers.index("fy"),
                    "fz": headers.index("fz"),
                    "site_en": headers.index("site_en"),
                }
            elif s.startswith("Energy"):
                read_energy = True
            elif read_energy and s:
                en = float(s.split()[0])
                read_energy = False
            elif s.startswith("PlusStress:"):
                read_stress = True
            elif read_stress and s:
                vals = [float(v) for v in s.split()[:6]]
                stress = [
                    [vals[0], vals[5], vals[4]],
                    [vals[5], vals[1], vals[3]],
                    [vals[4], vals[3], vals[2]],
                ]
                read_stress = False
            elif cols is not None and s and s[0].isdigit():
                f = s.split()
                forces.append((float(f[cols["fx"]]), float(f[cols["fy"]]), float(f[cols["fz"]])))
    return en, forces, stress

en0, f0, s0 = parse(sys.argv[1])
en1, f1, s1 = parse(sys.argv[2])
de = abs(en0 - en1)
df = 0.0
if len(f0) != len(f1):
    raise SystemExit("force count mismatch")
for a, b in zip(f0, f1):
    for x, y in zip(a, b):
        d = abs(x - y)
        df = max(df, d)
ds = 0.0
for i in range(3):
    for j in range(3):
        ds = max(ds, abs(s0[i][j] - s1[i][j]))
if de <= 1.0e-10 and df <= 1.0e-10 and ds <= 1.0e-10:
    raise SystemExit("nonzero radial mixing did not change E/F/S")
print(f"nonzero_mixing_effect e={de:.3e} f={df:.3e} s={ds:.3e}")
PY

  "$bin" roundtrip-model-dev "$full_active_model" "$full_active_roundtrip" >/dev/null
  "$bin" calc-efs "$full_active_roundtrip" "$cfg" "$active_roundtrip_out" >/dev/null
  python3 - "$active_out" "$active_roundtrip_out" <<'PY'
import re
import sys

def parse_vals(path):
    txt = open(path, encoding="utf-8").read()
    return [float(x) for x in re.findall(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?", txt)]

a = parse_vals(sys.argv[1])
b = parse_vals(sys.argv[2])
if len(a) != len(b):
    raise SystemExit(f"roundtrip length mismatch: {len(a)} vs {len(b)}")
for x, y in zip(a, b):
    if abs(x - y) > 1.0e-12 * max(1.0, abs(x), abs(y)):
        raise SystemExit("roundtrip output changed")
print("roundtrip_parity ok")
PY
  grep -q "two_layer_gate_radial_mixing = full" "$full_active_model"
  grep -q "two_layer_gate_radial_mixing = full" "$full_active_roundtrip"
  grep -q "two_layer_gate_radial_mixing_coeff_count = 48" "$full_active_roundtrip"

  "$bin" check-efs-fd-dev "$full_active_model" "$cfg" \
    --max-configs=1 \
    --max-atoms=8 \
    --displacement=1.0e-4 \
    --abs-tolerance=1.0e-4 \
    --rel-tolerance=2.0e-3 >/dev/null

  run_direction_block() {
    local block_name=$1
    local coeff_start=$2
    local coeff_end=$3
    local modes=${4:-"E F S EFS"}
    if (( coeff_end <= coeff_start )); then
      echo "[superlinear-suite] SKIP $block_name (empty)"
      return 0
    fi
    for weights in $modes; do
      case "$weights" in
        E) ew=1; fw=0; sw=0 ;;
        F) ew=0; fw=1; sw=0 ;;
        S) ew=0; fw=0; sw=1 ;;
        EFS) ew=1; fw=0.3; sw=0.2 ;;
      esac
      local line
      line=$("$bin" check-loss-gradient-direction-dev "$full_active_model" "$cfg" \
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
        --displacement=1.0e-4 \
        --abs-tolerance=2e-6 \
        --rel-tolerance=4e-3 \
        --coeff-start="$coeff_start" \
        --coeff-end="$coeff_end" | tail -n 1)
      python3 - "$block_name" "$weights" "$coeff_start" "$coeff_end" "$line" <<'PY'
import re
import sys
import math

name, mode, b, e, line = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]

def field(tok):
    m = re.search(rf"(?:^| ){re.escape(tok)}=([^ ]+)", line)
    if not m:
        raise SystemExit(f"{name} {mode} {b}:{e}: missing {tok}")
    return m.group(1)

checked = int(field("checked_coeffs"))
analytic = float(field("analytic_directional"))
fd = float(field("finite_difference"))
if checked <= 0:
    raise SystemExit(f"{name} {mode} {b}:{e} checked_coeffs={checked}")
if not (math.isfinite(analytic) and math.isfinite(fd)):
    raise SystemExit(f"{name} {mode} {b}:{e} nonfinite directional result")
if abs(analytic) <= 1.0e-13 and abs(fd) <= 1.0e-13:
    raise SystemExit(f"{name} {mode} {b}:{e} directional derivative numerically zero")
print(f"{name} {mode} checked={checked} an={analytic:.3e} fd={fd:.3e}")
PY
    done
  }

  run_direction_block "base_species" 0 2 "E EFS"
  run_direction_block "base_scaling" 2 "$radial_begin"
  run_direction_block "base_radial" "$radial_begin" "$radial_end"
  run_direction_block "gate_radial" "$gate_radial_begin" "$gate_radial_end"
  run_direction_block "shared_scaling" "$gate_scaling_begin" "$gate_scaling_end"
  run_direction_block "gate_additive" "$gate_additive_begin" "$gate_additive_end"
  run_direction_block "gate_type" "$gate_type_begin" "$gate_type_end"
  run_direction_block "gate_weight" "$gate_weight_begin" "$gate_weight_end"
  run_direction_block "gate_k_rank" "$gate_k_rank_begin" "$gate_k_rank_end"
  run_direction_block "outer_k_rank" "$outer_k_rank_begin" "$outer_k_rank_end"
  run_direction_block "radial_mixing" "$gate_radial_mixing_begin" "$gate_radial_mixing_end"

  "$bin" calc-efs "$full_active_model" "$cfg" "$active_out" >/dev/null
  "$bin" calc-efs "$full_active_model" "$cfg_rot" "$tmp_dir/full_active_rotated.efs.cfg" >/dev/null
  python3 - "$active_out" "$tmp_dir/full_active_rotated.efs.cfg" <<'PY'
import math
import sys

def parse(path):
    forces = []
    sites = []
    stress = None
    cols = None
    capture = False
    in_atoms = False
    for raw in open(path, encoding="utf-8"):
        s = raw.strip()
        if not s:
            continue
        if s.startswith("AtomData:"):
            in_atoms = True
            header = s.split()[1:]
            cols = {
                "fx": header.index("fx"),
                "fy": header.index("fy"),
                "fz": header.index("fz"),
                "site_en": header.index("site_en"),
            }
        elif s.startswith("Energy"):
            in_atoms = False
        elif s.startswith("PlusStress:"):
            capture = True
        elif capture:
            vals = [float(x) for x in s.split()[:6]]
            stress = [[vals[0], vals[5], vals[4]], [vals[5], vals[1], vals[3]], [vals[4], vals[3], vals[2]]]
            capture = False
        elif in_atoms and cols is not None and s[0].isdigit():
            parts = s.split()
            forces.append((float(parts[cols["fx"]]), float(parts[cols["fy"]]), float(parts[cols["fz"]])))
            sites.append(float(parts[cols["site_en"]]))
    return sum(sites), forces, stress

E0, F0, S0 = parse(sys.argv[1])
E1, F1, S1 = parse(sys.argv[2])
if len(F0) != len(F1):
    raise SystemExit("rotated force count mismatch")

de = abs(E0 - E1)
df = 0.0
for a, b in zip(F0, F1):
    b_back = (b[1], -b[0], b[2])
    df = max(df, max(abs(a[0] - b_back[0]), abs(a[1] - b_back[1]), abs(a[2] - b_back[2]))
)
R = [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
S1_back = [
    [
        sum(R[k][i] * S1[k][l] * R[l][j]
            for k in range(3) for l in range(3))
        for j in range(3)
    ] for i in range(3)
]
ds = max(abs(S0[i][j] - S1_back[i][j]) for i in range(3) for j in range(3))

if de > 1.0e-10 or df > 1.0e-8 or ds > 1.0e-8:
    raise SystemExit(f"rotation check failed: dE={de:.3e}, dF={df:.3e}, dS={ds:.3e}")
print(f"rotation_equivariance_ok dE={de:.3e} dF={df:.3e} dS={ds:.3e}")
PY

  if [[ ${SUS2_TEST_KEEP_TMP:-0} != 1 ]]; then
    rm -rf "$tmp_dir"
  fi
}

run_check edge_l1_init_legacy_load \
  bash dev_test/sh_two_layer_gate_edge_l1_init_check.sh
run_check edge_l1_zero_compat \
  bash dev_test/sh_two_layer_gate_edge_l1_zero_compat_check.sh
run_check edge_l1_forward_nonvacuous \
  bash dev_test/sh_two_layer_gate_edge_l1_forward_check.sh
run_check edge_l1_parameter_fd \
  bash dev_test/sh_two_layer_gate_edge_l1_loss_gradient_check.sh
run_check edge_l1_efs_coordinate_fd \
  bash dev_test/sh_two_layer_gate_edge_l1_efs_fd_check.sh
run_check edge_l1_linear_components \
  bash dev_test/sh_two_layer_gate_edge_l1_linear_components_check.sh
run_check edge_l1_rotation_equivariance \
  bash dev_test/sh_two_layer_gate_edge_l1_rotation_check.sh
run_check two_layer_gate_radial_mixing \
  run_two_layer_radial_mixing_suite

if [[ "${SUS2_SH_SUPERLINEAR_INCLUDE_READOUT:-1}" != "0" ]]; then
  run_check gate_readout_diagnostics \
    bash dev_test/sh_two_layer_gate_readout_diagnostics_check.sh "$bin"
  run_check gate_readout_zero_gate_efs_fd \
    bash dev_test/sh_two_layer_gate_readout_zero_gate_efs_fd_check.sh
fi

if [[ "${SUS2_SH_SUPERLINEAR_INCLUDE_PROFILE:-1}" != "0" ]]; then
  run_check edge_l1_pair_cache_profile \
    bash dev_test/sh_two_layer_gate_edge_l1_pair_cache_profile_check.sh
  run_check edge_l1_weighted_cache_profile \
    bash dev_test/sh_two_layer_gate_edge_l1_weighted_cache_profile_check.sh
fi

echo "[superlinear-suite] all requested superlinear gate checks passed"
