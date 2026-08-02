#!/usr/bin/env bash
# Exact release3 regression for the combined scaling_map=K and rank path.
#
# The fixture is deliberately L=4/K=6 with independent outer/gate rank R=3.
# It checks that K scaling is exactly equivalent to an LK model formed by
# replicating K's scal/s pair across l, and then exercises all active K,
# radial, gate-rank and outer-rank parameter derivatives plus E/F/stress FD.
set -Eeuo pipefail
trap 'echo "scaling_map=K rank3 check failed at line ${LINENO}: ${BASH_COMMAND}" >&2' ERR

repo_root=$(cd "$(dirname "$0")/.." && pwd)
bin=${1:-$repo_root/bin/mlp-sus2}
bin=$(cd "$(dirname "$bin")" && pwd)/$(basename "$bin")
[[ -x "$bin" ]] || { echo "missing executable: $bin" >&2; exit 2; }

tmp_dir=$(mktemp -d "${TMPDIR:-/tmp}/sus2sh-scaling-k-rank3-XXXXXX")
if [[ ${SUS2_TEST_KEEP_TMP:-0} == 1 ]]; then
  echo "keeping test directory: $tmp_dir"
else
  trap 'rm -rf "$tmp_dir"' EXIT
fi

one_cfg="$tmp_dir/one.cfg"
awk '{print} /^END_CFG/{exit}' "$repo_root/example/train.cfg" >"$one_cfg"
[[ $(grep -c '^END_CFG' "$one_cfg") == 1 ]]

# Break symmetry so that the positional FD check covers non-zero force paths.
python3 - "$one_cfg" <<'PY'
import sys
path = sys.argv[1]
lines = open(path, encoding="utf-8").readlines()
in_atoms = False
changed = 0
for i, raw in enumerate(lines):
    stripped = raw.strip()
    if stripped.startswith("AtomData:"):
        in_atoms = True
        continue
    if in_atoms and stripped:
        fields = raw.split()
        if len(fields) >= 8 and fields[0].lstrip("+-").isdigit():
            fields[2] = f"{float(fields[2]) + .017*(changed+1):.16e}"
            fields[3] = f"{float(fields[3]) - .011*(changed+1):.16e}"
            fields[4] = f"{float(fields[4]) + .007*(changed+1):.16e}"
            lines[i] = " ".join(fields) + "\n"
            changed += 1
            if changed == 3:
                break
if changed != 3:
    raise SystemExit("failed to perturb three atoms")
open(path, "w", encoding="utf-8").writelines(lines)
PY

common=(
  --species-count=2 --l-max=4 --k-max=6 --body-order=3
  --body-l-max=4,4,4,4 --radial-basis-size=3
  --radial-basis-type=RBChebyshev_sss --cutoff=5.0
  --sh-factor-pruning=q-total --scaling-map=K
  --two-layer-gate --two-layer-gate-factor-pruning=q-total
  --two-layer-gate-mode=mu-scalar-full --two-layer-gate-body-order=3
  --two-layer-gate-shared-radial --outer-k-rank=3
  --two-layer-gate-k-rank=3 --init-params=same
)
"$bin" init-sh "$tmp_dir/k.init.mtp" "${common[@]}" >/dev/null
"$bin" roundtrip-model-dev "$tmp_dir/k.init.mtp" "$tmp_dir/k.active.mtp" >/dev/null

# Set nonzero, asymmetric model parameters and make the exact LK expansion.
# Radial functions retain their native mu=(l,k) layout in both representations.
python3 - "$tmp_dir/k.active.mtp" "$tmp_dir/k.lkeq.mtp" <<'PY'
import re, sys
kpath, lkpath = sys.argv[1:]
text = open(kpath, encoding="utf-8").read()
number = r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?"

def vector(name, content=None):
    content = text if content is None else content
    match = re.search(rf"(?m)^\s*{re.escape(name)}\s*=\s*\{{([^}}]*)\}}", content, re.S)
    if not match: raise SystemExit(f"missing {name}")
    return [float(x) for x in re.findall(number, match.group(1))]

def replace(content, name, values):
    match = re.search(rf"(?m)^\s*{re.escape(name)}\s*=\s*\{{([^}}]*)\}}", content, re.S)
    if not match: raise SystemExit(f"missing {name}")
    value = name + " = {" + ", ".join(f"{x:.16e}" for x in values) + "}"
    return content[:match.start()] + value + content[match.end():]

def integer(name):
    match = re.search(rf"(?m)^\s*{re.escape(name)}\s*=\s*(\d+)\s*$", text)
    if not match: raise SystemExit(f"missing {name}")
    return int(match.group(1))

kmax, lmax, species = integer("sh_k_max"), integer("sh_l_max"), integer("species_count")
if (kmax, lmax) != (6, 4): raise SystemExit("fixture did not retain L=4/K=6")
if integer("outer_k_rank") != 3 or integer("two_layer_gate_k_rank") != 3:
    raise SystemExit("fixture did not retain rank=3")

for name, base, step, period in (
    ("species_coeffs", .17, .031, 5),
    ("moment_coeffs", .021, .0007, 11),
    ("two_layer_gate_radial_coeffs", .11, .004, 7),
    ("two_layer_gate_weights", .028, .0005, 13),
    ("two_layer_gate_additive_coeffs", .19, .007, 5),
    ("two_layer_gate_type_coeffs", .71, .019, 7),
):
    old = vector(name)
    values = [(base + step*(i % period)) * (-1. if i % 5 == 2 else 1.) for i in range(len(old))]
    text = replace(text, name, values)

expected_w = 3*(lmax+1)*kmax
for name, phase in (("two_layer_gate_k_rank_weights", .19), ("outer_k_rank1_weights", .0)):
    if len(vector(name)) != expected_w:
        raise SystemExit(f"{name}: expected {expected_w} rank weights")
    values = []
    for l in range(lmax+1):
        for channel in range(3):
            for k in range(kmax):
                x = .83 + phase + .041*l + .067*channel - .053*k + .009*channel*k
                values.append(-x if (l + channel + 2*k) % 4 == 1 else x)
    text = replace(text, name, values)

# K scal/s: [k][species-pair slopes][species-pair shifts].
pairs = species*species
kblocks = []
for k in range(kmax):
    kblocks.append([2.73 + .23*k + .031*p for p in range(pairs)] +
                   [1.19 + .17*k + .027*p for p in range(pairs)])
text = replace(text, "scal_coeffs", [x for block in kblocks for x in block])
open(kpath, "w", encoding="utf-8").write(text)

lk = re.sub(r"(?m)^scaling_map\s*=\s*K\s*$", "scaling_map = LK", text, count=1)
lk = replace(lk, "scal_coeffs", [x for k in range(kmax) for _l in range(lmax+1) for x in kblocks[k]])
open(lkpath, "w", encoding="utf-8").write(lk)
if len(vector("scal_coeffs", text)) != 48 or len(vector("scal_coeffs", lk)) != 240:
    raise SystemExit("unexpected K/LK scal coefficient counts")
print(f"fixture L=4 K=6 rank=3: K_scal=48 LK_scal=240 Wg=Wo={expected_w}")
PY

"$bin" calc-efs "$tmp_dir/k.active.mtp" "$one_cfg" "$tmp_dir/k.cfg" >/dev/null
"$bin" calc-efs "$tmp_dir/k.lkeq.mtp" "$one_cfg" "$tmp_dir/lk.cfg" >/dev/null
python3 - "$tmp_dir/k.cfg" "$tmp_dir/lk.cfg" <<'PY'
import math, re, sys
number = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")
left = [float(x) for x in number.findall(open(sys.argv[1]).read())]
right = [float(x) for x in number.findall(open(sys.argv[2]).read())]
if not left or len(left) != len(right): raise SystemExit("K/LK EFS shape mismatch")
worst = max(abs(x-y) for x, y in zip(left, right))
scale = max(1., *(abs(x) for x in left), *(abs(x) for x in right))
if not all(math.isfinite(x) for x in left + right) or worst > 2e-11*scale:
    raise SystemExit(f"K/LK E/F/stress mismatch: {worst:.3e}, scale {scale:.3e}")
print(f"K_LK_rank3_EFS_exact max_abs={worst:.3e} scale={scale:.3e}")
PY

# Offset E, F and stress to make all loss-gradient paths non-vacuous.
python3 - "$tmp_dir/k.cfg" "$tmp_dir/target.cfg" <<'PY'
import sys
src, dst = sys.argv[1:]
lines = open(src).readlines(); atoms = energy = stress = False; n = 0
for i, raw in enumerate(lines):
    stripped = raw.strip()
    if stripped.startswith("AtomData:"): atoms = True; continue
    if stripped == "Energy": atoms = False; energy = True; continue
    if stripped.startswith("PlusStress:") or stripped.startswith("Stress:"): stress = True; continue
    if atoms and stripped:
        fields = raw.split()
        if len(fields) >= 8 and fields[0].lstrip("+-").isdigit():
            n += 1
            for c in range(3): fields[5+c] = f"{float(fields[5+c]) + .006*((n+c)%5-2):.16e}"
            lines[i] = " ".join(fields) + "\n"
    elif energy and stripped:
        lines[i] = f"\t{float(stripped)+.2:.16e}\n"; energy = False
    elif stress and stripped:
        fields = raw.split()
        if len(fields) >= 6:
            lines[i] = "\t" + " \t".join(f"{float(x)+.008*(c-2):.16e}" for c, x in enumerate(fields[:6])) + "\n"
            stress = False
if n == 0 or energy or stress: raise SystemExit("failed to construct E/F/stress target")
open(dst, "w").writelines(lines)
PY

layout=$("$bin" check-loss-gradient-dev "$tmp_dir/k.active.mtp" "$tmp_dir/target.cfg" \
  --max-configs=1 --energy-weight=1 --force-weight=0 --stress-weight=0 \
  --radial-smooth=0 --scalar-head-l2=0 --gate-scalar-l2=0 --gate-mix-l2=0 --gate-full-l2=0 --gate-x-l2=0 \
  --coeff-start=0 --coeff-end=1 --displacement=1e-5 --abs-tolerance=1 --rel-tolerance=1)
echo "$layout" | grep 'checked_coeffs='
field() { awk -v key="$1" '{for (i=1;i<=NF;i++) if ($i ~ ("^" key "=[0-9]+$")) {sub(/^[^=]*=/,"",$i); print $i}}' <<<"$layout"; }
radial_begin=$(field radial_begin); radial_end=$(field radial_end)
gate_radial_begin=$(field gate_radial_begin); gate_radial_end=$(field gate_radial_end)
gate_begin=$(field gate_k_rank_begin); gate_end=$(field gate_k_rank_end)
outer_begin=$(field outer_k_rank_begin); outer_end=$(field outer_k_rank_end)
scal_begin=$((radial_begin - 48)); scal_end=$radial_begin
[[ $scal_begin -ge 0 && $radial_begin -lt $radial_end && $gate_radial_begin -lt $gate_radial_end ]]
[[ $((gate_end-gate_begin)) == 90 && $((outer_end-outer_begin)) == 90 && $gate_end == $outer_begin ]]

check_block() {
  local name=$1 begin=$2 end=$3
  "$bin" check-loss-gradient-dev "$tmp_dir/k.active.mtp" "$tmp_dir/target.cfg" \
    --max-configs=1 --energy-weight=1 --force-weight=.1 --stress-weight=.01 \
    --radial-smooth=0 --scalar-head-l2=0 --gate-scalar-l2=0 --gate-mix-l2=0 --gate-full-l2=0 --gate-x-l2=0 \
    --coeff-start="$begin" --coeff-end="$end" --displacement=1e-5 --abs-tolerance=1.5e-5 --rel-tolerance=2e-3 | grep 'checked_coeffs='
  "$bin" check-loss-gradient-direction-dev "$tmp_dir/k.active.mtp" "$tmp_dir/target.cfg" \
    --max-configs=1 --energy-weight=1 --force-weight=.1 --stress-weight=.01 \
    --radial-smooth=0 --scalar-head-l2=0 --gate-scalar-l2=0 --gate-mix-l2=0 --gate-full-l2=0 --gate-x-l2=0 \
    --coeff-start="$begin" --coeff-end="$end" --displacement=1e-4 --abs-tolerance=3e-5 --rel-tolerance=7e-3 \
    | tee "$tmp_dir/${name}.direction.txt" | grep 'checked_coeffs='
  python3 - "$name" "$tmp_dir/${name}.direction.txt" <<'PY'
import math, re, sys
name, text = sys.argv[1], open(sys.argv[2]).read()
analytic = float(re.search(r"analytic_directional=([^ ]+)", text).group(1))
finite_difference = float(re.search(r"finite_difference=([^ ]+)", text).group(1))
if not (math.isfinite(analytic) and math.isfinite(finite_difference)) or min(abs(analytic), abs(finite_difference)) < 1e-10:
    raise SystemExit(f"{name}: inactive/nonfinite directional gradient")
print(f"{name}_gradient_active analytic={analytic:.6e} fd={finite_difference:.6e}")
PY
}
check_block K_scaling "$scal_begin" "$scal_end"
check_block main_radial "$radial_begin" "$radial_end"
check_block gate_radial "$gate_radial_begin" "$gate_radial_end"
check_block gate_rank3_W "$gate_begin" "$gate_end"
check_block outer_rank3_W "$outer_begin" "$outer_end"

"$bin" check-efs-fd-dev "$tmp_dir/k.active.mtp" "$one_cfg" --max-configs=1 --max-atoms=2 \
  --displacement=1e-5 --abs-tolerance=5e-5 --rel-tolerance=5e-3 | tee "$tmp_dir/efs_fd.txt"
grep -q 'force_components=' "$tmp_dir/efs_fd.txt"
grep -q 'stress_components=' "$tmp_dir/efs_fd.txt"

# Optional MPI addendum: four configurations force two ranks to own data,
# then compare the trained serial/MPI model predictions componentwise.
if [[ ${RUN_MPI:-0} == 1 ]]; then
  command -v mpirun >/dev/null
  "$bin" help train 2>&1 | grep -q 'Mode: MPI'
  four_cfg="$tmp_dir/four.cfg"
  awk 'BEGIN{n=0} {print} /^END_CFG/{n++; if(n==4) exit}' "$repo_root/example/train.cfg" >"$four_cfg"
  [[ $(grep -c '^END_CFG' "$four_cfg") == 4 ]]
  train_opts=(--max-iter=2 --skip-preinit --do-samp --init-params=same --energy-weight=1 --force-weight=.01 --stress-weight=.001
              --radial-smooth=0 --scalar-head-l2=0 --gate-scalar-l2=0 --gate-mix-l2=0 --gate-full-l2=0 --gate-x-l2=0)
  "$bin" train "$tmp_dir/k.active.mtp" "$four_cfg" "${train_opts[@]}" --trained-pot-name="$tmp_dir/serial.mtp" >"$tmp_dir/serial.train.log" 2>&1
  timeout 300 mpirun -n 2 "$bin" train "$tmp_dir/k.active.mtp" "$four_cfg" "${train_opts[@]}" --trained-pot-name="$tmp_dir/mpi2.mtp" >"$tmp_dir/mpi2.train.log" 2>&1
  "$bin" calc-efs "$tmp_dir/serial.mtp" "$four_cfg" "$tmp_dir/serial.cfg" >/dev/null
  "$bin" calc-efs "$tmp_dir/mpi2.mtp" "$four_cfg" "$tmp_dir/mpi2.cfg" >/dev/null
  python3 - "$tmp_dir/serial.cfg" "$tmp_dir/mpi2.cfg" <<'PY'
import math, re, sys
number = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")
def values(path):
    output = {"energy": [], "force": [], "stress": []}; atoms = energy = stress = False
    for line in open(path):
        stripped = line.strip()
        if stripped.startswith("AtomData:"): atoms = True; continue
        if stripped == "Energy": atoms = False; energy = True; continue
        if stripped.startswith("PlusStress:") or stripped.startswith("Stress:"): stress = True; continue
        if atoms and stripped:
            fields = line.split()
            if len(fields) >= 8 and fields[0].lstrip("+-").isdigit(): output["force"].extend(float(x) for x in fields[5:8])
        elif energy and stripped: output["energy"].append(float(number.search(stripped).group())); energy = False
        elif stress and stripped: output["stress"].extend(float(x) for x in number.findall(stripped)[:6]); stress = False
    return output
left, right = values(sys.argv[1]), values(sys.argv[2]); tolerance = {"energy": 1e-7, "force": 1e-8, "stress": 1e-5}
parts = []
for key in ("energy", "force", "stress"):
    if not left[key] or len(left[key]) != len(right[key]): raise SystemExit(f"MPI {key} shape mismatch")
    worst = max(abs(a-b) for a,b in zip(left[key], right[key]))
    if not all(math.isfinite(x) for x in left[key]+right[key]) or worst > tolerance[key]: raise SystemExit(f"MPI {key} mismatch {worst:.3e}")
    parts.append(f"{key}={worst:.3e}")
print("K_rank3_MPI_serial_training_parity " + " ".join(parts))
PY
fi

echo "PASS scaling_map=K L4/K6 outer/gate-rank3: exact LK, parameter and E/F/stress FD"
