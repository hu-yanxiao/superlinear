#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

bin=${1:-./bin/mlp-sus2}
tmp_dir=$(mktemp -d "${TMPDIR:-/tmp}/sus2sh-k-rank-cli-XXXXXX")
if [[ ${SUS2_TEST_KEEP_TMP:-0} == 1 ]]; then
  echo "keeping test directory: $tmp_dir"
else
  trap 'rm -rf "$tmp_dir"' EXIT
fi

one_cfg="$tmp_dir/one.cfg"
awk '{print} /^END_CFG/{exit}' example/train.cfg > "$one_cfg"

common_args=(
  --species-count=2
  --l-max=4
  --k-max=4
  --body-order=6
  --body-l-max=4,4,4,2,1
  --radial-basis-size=4
  --radial-basis-type=RBChebyshev_sss
  --cutoff=5.0
  --sh-factor-pruning=q-total
  --two-layer-gate
  --two-layer-gate-factor-pruning=q-total
  --two-layer-gate-mode=mu-scalar-full
  --two-layer-gate-body-order=5
  --two-layer-gate-shared-radial
)

no_rank="$tmp_dir/no-rank.sus2"
explicit_zero="$tmp_dir/explicit-zero.sus2"
rank2="$tmp_dir/outer-rank2.sus2"
gate1_outer1="$tmp_dir/gate1-outer1.sus2"
rank4_identity="$tmp_dir/gate4-outer4.sus2"
native_k1="$tmp_dir/native-k1.sus2"

"$bin" init-sh "$no_rank" "${common_args[@]}" >/dev/null
"$bin" init-sh "$explicit_zero" "${common_args[@]}" \
  --outer-k-rank=0 --two-layer-gate-k-rank=0 >/dev/null
cmp "$no_rank" "$explicit_zero"

if "$bin" init-sh "$tmp_dir/invalid.sus2" "${common_args[@]}" \
     --outer-k-rank=1.5 >/dev/null 2>&1; then
  echo "fractional --outer-k-rank was accepted" >&2
  exit 1
fi
if "$bin" init-sh "$tmp_dir/invalid.sus2" "${common_args[@]}" \
     --outer-k-rank1=false >/dev/null 2>&1; then
  echo "valued --outer-k-rank1 compatibility flag was accepted" >&2
  exit 1
fi
if "$bin" init-sh "$tmp_dir/invalid.sus2" "${common_args[@]}" \
     --two-layer-gate-k-rank=1 >/dev/null 2>&1; then
  echo "gate rank without outer rank was accepted" >&2
  exit 1
fi

"$bin" init-sh "$rank2" "${common_args[@]}" --outer-k-rank=2 >/dev/null
"$bin" init-sh "$gate1_outer1" "${common_args[@]}" \
  --outer-k-rank=1 --two-layer-gate-k-rank=1 >/dev/null
"$bin" init-sh "$rank4_identity" "${common_args[@]}" \
  --outer-k-rank=4 --two-layer-gate-k-rank=4 >/dev/null

native_args=("${common_args[@]}")
for index in "${!native_args[@]}"; do
  if [[ ${native_args[$index]} == --k-max=4 ]]; then
    native_args[$index]=--k-max=1
  fi
done
"$bin" init-sh "$native_k1" "${native_args[@]}" >/dev/null

python3 - "$no_rank" "$rank2" "$gate1_outer1" "$rank4_identity" "$native_k1" <<'PY'
import re
import sys

no_rank, rank2, both1, both4, native1 = sys.argv[1:]

def text(path):
    return open(path, encoding="utf-8").read()

def scalar(data, name):
    match = re.search(rf"(?m)^\s*{re.escape(name)}\s*=\s*(\d+)\s*$", data)
    if not match:
        raise SystemExit(f"missing {name}")
    return int(match.group(1))

def vector_count(data, name):
    match = re.search(rf"{re.escape(name)}\s*=\s*\{{([^}}]*)\}}", data, re.S)
    if not match:
        raise SystemExit(f"missing {name}")
    return len(re.findall(r"[-+]?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?", match.group(1)))

def vector(data, name):
    match = re.search(rf"{re.escape(name)}\s*=\s*\{{([^}}]*)\}}", data, re.S)
    if not match:
        raise SystemExit(f"missing {name}")
    return [float(x) for x in re.findall(
        r"[-+]?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?", match.group(1))]

n = text(no_rank)
r2 = text(rank2)
b1 = text(both1)
b4 = text(both4)
n1 = text(native1)

if "outer_k_rank1_enabled" in n or "two_layer_gate_k_rank" in n:
    raise SystemExit("default rank=0 model contains rank metadata")
if scalar(n, "alpha_scalar_moments") != 2059:
    raise SystemExit("l4/k4/body6 no-rank outer scalar count should be 2059")
if scalar(n, "two_layer_gate_weight_count") != 30060:
    raise SystemExit("l4/k4/body5 no-rank full gate head should contain 30060 weights")

if scalar(r2, "outer_k_rank") != 2:
    raise SystemExit("outer rank2 metadata missing")
if scalar(r2, "outer_k_rank1_basic_count") != 50:
    raise SystemExit("outer rank2 should contain 50 contracted primitives")
if scalar(r2, "outer_k_rank1_weight_count") != 40:
    raise SystemExit("outer rank2 should contain 40 contraction weights")
if scalar(r2, "alpha_scalar_moments") != 224:
    raise SystemExit("l4/rank2/body6 outer scalar count should be 224")

if "outer_k_rank =" in b1:
    raise SystemExit("rank1 file should omit generic rank metadata for old-reader compatibility")
if scalar(b1, "outer_k_rank1_basic_count") != 25:
    raise SystemExit("outer rank1 should contain 25 contracted primitives")
if scalar(b1, "outer_k_rank1_weight_count") != 20:
    raise SystemExit("outer rank1 should contain a 20-value compatibility shell")
if any(value != 1.0 for value in vector(b1, "outer_k_rank1_weights")):
    raise SystemExit("outer rank1 compatibility shell should be all ones")
if scalar(b1, "two_layer_gate_k_rank") != 1:
    raise SystemExit("gate rank1 metadata missing")
if scalar(b1, "two_layer_gate_k_rank_basic_count") != 25:
    raise SystemExit("gate rank1 should contain 25 contracted primitives")
if scalar(b1, "two_layer_gate_k_rank_weight_count") != 20:
    raise SystemExit("gate rank1 should contain 20 contraction weights")
if scalar(b1, "alpha_scalar_moments") != 36:
    raise SystemExit("l4/rank1/body6 outer scalar count should be 36")
if vector_count(b1, "two_layer_gate_scalar_indices") != 32:
    raise SystemExit("l4/rank1/body5 gate scalar count should be 32")
if scalar(b1, "two_layer_gate_weight_count") != 640:
    raise SystemExit("mu-scalar-full gate rank1 head should contain 20*32=640 weights")

if scalar(b4, "alpha_scalar_moments") != 2059:
    raise SystemExit("rank4 identity outer graph should recover 2059 scalars")
if vector_count(b4, "two_layer_gate_scalar_indices") != 1503:
    raise SystemExit("rank4 identity gate graph should recover 1503 scalars")
if scalar(n1, "alpha_scalar_moments") != 36:
    raise SystemExit("native l4/k1/body6 model should contain 36 outer scalars")
if vector_count(n1, "two_layer_gate_scalar_indices") != 32:
    raise SystemExit("native l4/k1/body5 model should contain 32 gate scalars")
if "outer_k_rank1_enabled" in n1 or "two_layer_gate_k_rank" in n1:
    raise SystemExit("native K=1 control unexpectedly contains rank metadata")

print("metadata_counts_ok no_rank=2059 rank2=224 rank1=36 gate_rank1=32")
PY

rank2_layout=$("$bin" check-loss-gradient-dev "$rank2" "$one_cfg" \
  --max-configs=1 --energy-weight=1 --force-weight=0 --stress-weight=0 \
  --radial-smooth=0 --scalar-head-l2=0 --gate-scalar-l2=0 \
  --gate-mix-l2=0 --gate-full-l2=0 --gate-x-l2=0 \
  --coeff-start=0 --coeff-end=1 \
  --displacement=1.0e-5 --abs-tolerance=1 --rel-tolerance=1)
rank2_begin=$(sed -n 's/.*outer_k_rank_begin=\([0-9][0-9]*\).*/\1/p' <<<"$rank2_layout")
rank2_end=$(sed -n 's/.*outer_k_rank_end=\([0-9][0-9]*\).*/\1/p' <<<"$rank2_layout")
if [[ -z "$rank2_begin" || -z "$rank2_end" \
      || $((rank2_end-rank2_begin)) -ne 40 ]]; then
  echo "rank2 optimizer W block is not 40 coefficients: $rank2_layout" >&2
  exit 1
fi
"$bin" check-loss-gradient-direction-dev "$rank2" "$one_cfg" \
  --max-configs=1 --energy-weight=1 --force-weight=0.1 --stress-weight=0.01 \
  --radial-smooth=0 --scalar-head-l2=0 --gate-scalar-l2=0 \
  --gate-mix-l2=0 --gate-full-l2=0 --gate-x-l2=0 \
  --coeff-start="$rank2_begin" --coeff-end="$rank2_end" \
  --displacement=1.0e-5 --abs-tolerance=2.0e-5 --rel-tolerance=3.0e-3

rank2_roundtrip="$tmp_dir/outer-rank2-roundtrip.sus2"
rank2_roundtrip_efs="$tmp_dir/outer-rank2-roundtrip.cfg"
"$bin" roundtrip-model-dev "$rank2" "$rank2_roundtrip" >/dev/null
"$bin" calc-efs "$rank2_roundtrip" "$one_cfg" "$rank2_roundtrip_efs" >/dev/null

no_rank_efs="$tmp_dir/no-rank-efs.cfg"
rank2_efs="$tmp_dir/rank2-efs.cfg"
rank4_efs="$tmp_dir/rank4-efs.cfg"
"$bin" calc-efs "$no_rank" "$one_cfg" "$no_rank_efs" >/dev/null
"$bin" calc-efs "$rank2" "$one_cfg" "$rank2_efs" >/dev/null
"$bin" calc-efs "$rank4_identity" "$one_cfg" "$rank4_efs" >/dev/null
cmp "$rank2_efs" "$rank2_roundtrip_efs"
python3 - "$no_rank_efs" "$rank4_efs" <<'PY'
import re
import sys

number = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")
def values(path):
    return [float(x) for x in number.findall(open(path, encoding="utf-8").read())]
a, b = values(sys.argv[1]), values(sys.argv[2])
if len(a) != len(b):
    raise SystemExit("rank4 identity E/F/S output shape differs from no-rank")
worst = max((abs(x-y) for x, y in zip(a, b)), default=0.0)
scale = max([1.0] + [abs(x) for x in a] + [abs(x) for x in b])
if worst > 2.0e-11 * scale:
    raise SystemExit(f"rank4 identity parity failed: worst={worst:.3e}, scale={scale:.3e}")
print(f"rank4_identity_efs_worst={worst:.3e}")
PY

roundtrip="$tmp_dir/gate1-outer1-roundtrip.sus2"
roundtrip_efs="$tmp_dir/gate1-outer1-roundtrip.cfg"
base_efs="$tmp_dir/gate1-outer1-base.cfg"
"$bin" roundtrip-model-dev "$gate1_outer1" "$roundtrip" >/dev/null
"$bin" calc-efs "$gate1_outer1" "$one_cfg" "$base_efs" >/dev/null
"$bin" calc-efs "$roundtrip" "$one_cfg" "$roundtrip_efs" >/dev/null
cmp "$base_efs" "$roundtrip_efs"

active="$tmp_dir/gate1-outer1-active.sus2"
python3 - "$roundtrip" "$active" <<'PY'
import re
import sys

source, target = sys.argv[1:]
data = open(source, encoding="utf-8").read()
number = r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?"

def replace_vector(name, update):
    global data
    match = re.search(rf"{name}\s*=\s*\{{([^}}]*)\}}", data, re.S)
    if not match:
        raise SystemExit(f"missing {name}")
    values = [float(x) for x in re.findall(number, match.group(1))]
    update(values)
    replacement = f"{name} = {{" + ", ".join(f"{x:.15e}" for x in values) + "}"
    data = data[:match.start()] + replacement + data[match.end():]

replace_vector("moment_coeffs", lambda v: v.__setitem__(slice(None), [
    (0.025 + 0.001 * (i % 9)) * (-1.0 if i % 3 == 1 else 1.0)
    for i in range(len(v))]))
replace_vector("two_layer_gate_weights", lambda v: v.__setitem__(slice(None), [
    (0.035 + 0.0002 * (i % 17)) * (-1.0 if i % 5 == 2 else 1.0)
    for i in range(len(v))]))
replace_vector("two_layer_gate_k_rank_weights", lambda v: v.__setitem__(slice(None), [
    0.91, -0.17, 0.13, -0.07] * (len(v) // 4)))
replace_vector("outer_k_rank1_weights", lambda v: v.__setitem__(slice(None), [
    1.03, -0.19, 0.11, -0.05] * (len(v) // 4)))
open(target, "w", encoding="utf-8").write(data)
PY

target_cfg="$tmp_dir/target.cfg"
"$bin" calc-efs "$active" "$one_cfg" "$target_cfg" >/dev/null
python3 - "$target_cfg" <<'PY'
import sys

path = sys.argv[1]
lines = open(path, encoding="utf-8").readlines()
in_atoms = False
energy_next = False
stress_next = False
atom = 0
for i, raw in enumerate(lines):
    stripped = raw.strip()
    if stripped.startswith("AtomData:"):
        in_atoms = True
        continue
    if stripped == "Energy":
        in_atoms = False
        energy_next = True
        continue
    if stripped.startswith("PlusStress:"):
        stress_next = True
        continue
    if in_atoms and stripped:
        fields = raw.split()
        if len(fields) >= 8 and fields[0].lstrip("+-").isdigit():
            atom += 1
            for c in range(3):
                fields[5+c] = f"{float(fields[5+c]) + 0.006*((atom+c)%5-2):.15e}"
            lines[i] = " ".join(fields) + "\n"
    elif energy_next and stripped:
        lines[i] = f"\t{float(stripped) + 0.2:.15e}\n"
        energy_next = False
    elif stress_next and stripped:
        fields = raw.split()
        if len(fields) >= 6:
            lines[i] = "\t" + " \t".join(
                f"{float(x) + 0.008*(c-2):.15e}" for c, x in enumerate(fields[:6])) + "\n"
            stress_next = False
if atom == 0 or energy_next or stress_next:
    raise SystemExit("failed to build controlled E/F/S target")
open(path, "w", encoding="utf-8").writelines(lines)
PY

layout=$("$bin" check-loss-gradient-dev "$active" "$target_cfg" \
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
      || $((gate_end-gate_begin)) -ne 20 || "$outer_begin" -ne "$outer_end" \
      || "$gate_end" -ne "$outer_begin" ]]; then
  echo "failed to discover gate W and empty direct outer-W blocks: $layout" >&2
  exit 1
fi

for range in "$gate_begin:$gate_end"; do
  begin=${range%%:*}
  end=${range##*:}
  "$bin" check-loss-gradient-dev "$active" "$target_cfg" \
    --max-configs=1 --energy-weight=1 --force-weight=0 --stress-weight=0 \
    --radial-smooth=0 --scalar-head-l2=0 --gate-scalar-l2=0 \
    --gate-mix-l2=0 --gate-full-l2=0 --gate-x-l2=0 \
    --coeff-start="$begin" --coeff-end="$end" \
    --displacement=1.0e-5 --abs-tolerance=5.0e-6 --rel-tolerance=5.0e-4
  energy_direction=$("$bin" check-loss-gradient-direction-dev "$active" "$target_cfg" \
    --max-configs=1 --energy-weight=1 --force-weight=0 --stress-weight=0 \
    --radial-smooth=0 --scalar-head-l2=0 --gate-scalar-l2=0 \
    --gate-mix-l2=0 --gate-full-l2=0 --gate-x-l2=0 \
    --coeff-start="$begin" --coeff-end="$end" \
    --displacement=1.0e-5 --abs-tolerance=5.0e-6 --rel-tolerance=5.0e-4)
  analytic=$(sed -n 's/.*analytic_directional=\([^ ]*\).*/\1/p' <<<"$energy_direction")
  python3 - "$analytic" <<'PY'
import sys
value = float(sys.argv[1])
if abs(value) < 1.0e-14:
    raise SystemExit("rank contraction block has a numerically zero energy directional gradient")
PY
done

"$bin" check-loss-gradient-direction-dev "$active" "$target_cfg" \
  --max-configs=1 --energy-weight=1 --force-weight=0.1 --stress-weight=0.01 \
  --radial-smooth=0 --scalar-head-l2=0 --gate-scalar-l2=0 \
  --gate-mix-l2=0 --gate-full-l2=0 --gate-x-l2=0 \
  --coeff-start="$gate_begin" --coeff-end="$outer_end" \
  --displacement=1.0e-4 --abs-tolerance=1.0e-5 --rel-tolerance=3.0e-3

"$bin" check-efs-fd-dev "$active" "$one_cfg" \
  --max-configs=1 --max-atoms=1 \
  --displacement=1.0e-5 --abs-tolerance=3.0e-5 --rel-tolerance=3.0e-3

for model in "$rank2" "$active" "$native_k1"; do
  stem=$(basename "$model" .sus2)
  descriptors="$tmp_dir/$stem-descriptors.txt"
  partial="$tmp_dir/$stem-partial.txt"
  eij="$tmp_dir/$stem-eij.txt"
  "${bin}" calc-descriptors "$model" "$one_cfg" "$descriptors" >/dev/null
  "${bin}" calc-partialE "$model" "$one_cfg" "$partial" >/dev/null
  "${bin}" calc-eij "$model" "$one_cfg" "$eij" >/dev/null
  test -s "$descriptors"
  python3 - "$partial" "$eij" "$stem" <<'PY'
import sys

partial_sum = 0.0
for raw in open(sys.argv[1], encoding="utf-8"):
    fields = raw.split()
    if fields and not fields[0].startswith("#"):
        partial_sum += sum(float(x) for x in fields[1:])
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
err = abs(partial_sum-eij_sum)
tol = 2.0e-11*max(1.0, abs(partial_sum), abs(eij_sum))
if err > tol:
    raise SystemExit(
        f"{sys.argv[3]} calc-eij sum mismatch: partial={partial_sum:.16e} "
        f"eij={eij_sum:.16e} err={err:.3e} tol={tol:.3e}")
print(f"{sys.argv[3]}_calc_eij_abs_err={err:.3e} "
      f"scale={max(abs(partial_sum), abs(eij_sum)):.3e}")
PY
done

echo "generic K-rank CLI smoke passed: Wg=[${gate_begin},${gate_end}) direct-Wo=[${outer_begin},${outer_end})"
