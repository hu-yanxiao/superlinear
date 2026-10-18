#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

bin=${1:-./bin/mlp-sus2}
bin=$(cd "$(dirname "$bin")" && pwd)/$(basename "$bin")
tmp_dir=$(mktemp -d "${TMPDIR:-/tmp}/sus2sh-outer-rank-no-gate-XXXXXX")
if [[ ${SUS2_TEST_KEEP_TMP:-0} == 1 ]]; then
  echo "keeping test directory: $tmp_dir"
else
  trap 'rm -rf "$tmp_dir"' EXIT
fi

one_cfg="$tmp_dir/one.cfg"
awk '{print} /^END_CFG/{exit}' example/train.cfg >"$one_cfg"
linear_cfg="$tmp_dir/linear-train.cfg"
python3 - "$linear_cfg" <<'PY'
import math
import sys

path = sys.argv[1]
base = [
    (1, 0, 1.13, 1.72, 1.41),
    (2, 1, 2.47, 1.19, 3.06),
    (3, 0, 3.81, 2.68, 1.97),
    (4, 1, 5.08, 3.33, 4.44),
    (5, 0, 1.86, 5.21, 5.64),
    (6, 1, 4.62, 5.87, 2.58),
    (7, 0, 6.19, 1.93, 5.16),
    (8, 1, 2.94, 6.36, 1.24),
]
with open(path, "w", encoding="utf-8") as out:
    for sample in range(24):
        scale = 1.0 + 0.002 * (sample - 11.5)
        cell = 8.5 * scale
        out.write("BEGIN_CFG\n Size\n      8\n Supercell\n")
        out.write(f"  {cell:.10f} 0 0\n  0 {cell:.10f} 0\n  0 0 {cell:.10f}\n")
        out.write(" AtomData:  id type       cartes_x      cartes_y      cartes_z     fx          fy          fz\n")
        for atom_id, atom_type, x, y, z in base:
            phase = 0.37 * (sample + 1) * (atom_id + 1)
            px = x * scale + 0.035 * math.sin(phase)
            py = y * scale + 0.030 * math.cos(0.71 * phase)
            pz = z * scale + 0.025 * math.sin(1.13 * phase)
            fx = 0.02 * math.cos(phase)
            fy = -0.018 * math.sin(0.83 * phase)
            fz = 0.016 * math.cos(1.21 * phase)
            out.write(
                f"{atom_id:7d} {atom_type:6d} {px:12.7f} {py:12.7f} {pz:12.7f} "
                f"{fx:11.7f} {fy:11.7f} {fz:11.7f}\n")
        out.write(" Energy\n")
        out.write(f"  {-12.5 + 0.075 * sample + 0.03 * math.sin(sample):.12f}\n")
        out.write(" PlusStress:  xx yy zz yz xz xy\n")
        out.write(
            f"  {0.001 * sample:.8f} {-0.0007 * sample:.8f} "
            f"{0.0004 * sample:.8f} 0.0003 -0.0002 0.0001\n")
        out.write("END_CFG\n")
PY

common=(
  --species-count=2
  --l-max=2
  --k-max=3
  --body-order=4
  --body-l-max=2,2,2,2
  --radial-basis-size=3
  --radial-basis-type=RBChebyshev_sss
  --cutoff=5.0
  --sh-factor-pruning=q-total
  --init-params=same
)

plain="$tmp_dir/plain.sus2"
rank1="$tmp_dir/rank1.sus2"
rank2="$tmp_dir/rank2.sus2"
rank3="$tmp_dir/rank3.sus2"
"$bin" init-sh "$plain" "${common[@]}" >/dev/null
"$bin" init-sh "$rank1" "${common[@]}" --outer-k-rank=1 >/dev/null
"$bin" init-sh "$rank2" "${common[@]}" --outer-k-rank=2 >/dev/null
"$bin" init-sh "$rank3" "${common[@]}" --outer-k-rank=3 >/dev/null

python3 - "$plain" "$rank1" "$rank2" "$rank3" <<'PY'
import re
import sys

plain_path, rank1_path, rank2_path, rank3_path = sys.argv[1:]

def load(path):
    return open(path, encoding="utf-8").read()

def integer(text, name):
    match = re.search(rf"(?m)^\s*{re.escape(name)}\s*=\s*(\d+)\s*$", text)
    if not match:
        raise SystemExit(f"missing {name}")
    return int(match.group(1))

plain = load(plain_path)
models = [(1, load(rank1_path)), (2, load(rank2_path)), (3, load(rank3_path))]
for rank, text in models:
    if "two_layer_gate_" in text:
        raise SystemExit(f"gate-free outer rank {rank} contains gate metadata")
    if integer(text, "outer_k_rank1_product_begin") != 0:
        raise SystemExit(f"gate-free outer rank {rank} preserved an unused raw product graph")
    if integer(text, "outer_k_rank1_basic_begin") != integer(text, "alpha_index_basic_count"):
        raise SystemExit(f"gate-free outer rank {rank} has a gap before contracted basics")
    if integer(text, "sh_scalar_info_count") != integer(text, "alpha_scalar_moments"):
        raise SystemExit(f"gate-free outer rank {rank} omitted outer scalar metadata")
if "outer_k_rank =" in models[0][1]:
    raise SystemExit("rank-one compatibility format unexpectedly contains generic rank metadata")
if integer(models[1][1], "outer_k_rank") != 2:
    raise SystemExit("rank-two metadata is missing")
if integer(models[2][1], "outer_k_rank") != 3:
    raise SystemExit("rank-three metadata is missing")
if integer(plain, "alpha_scalar_moments") != integer(models[2][1], "alpha_scalar_moments"):
    raise SystemExit("R=K gate-free graph did not recover the full-K scalar count")
if integer(models[1][1], "alpha_scalar_moments") >= integer(plain, "alpha_scalar_moments"):
    raise SystemExit("R<K gate-free graph did not reduce the outer scalar count")
print("gate_free_outer_rank_metadata_ok")
PY

for model in "$rank1" "$rank2" "$rank3"; do
  stem=$(basename "$model" .sus2)
  first="$tmp_dir/$stem-roundtrip1.sus2"
  second="$tmp_dir/$stem-roundtrip2.sus2"
  "$bin" roundtrip-model-dev "$model" "$first" >/dev/null
  "$bin" roundtrip-model-dev "$first" "$second" >/dev/null
  "$bin" calc-efs "$model" "$one_cfg" "$tmp_dir/$stem-base.cfg" >/dev/null
  "$bin" calc-efs "$second" "$one_cfg" "$tmp_dir/$stem-roundtrip.cfg" >/dev/null
  cmp "$tmp_dir/$stem-base.cfg" "$tmp_dir/$stem-roundtrip.cfg"
  if grep -q '^two_layer_gate_' "$second"; then
    echo "$stem roundtrip introduced gate metadata" >&2
    exit 1
  fi
done

"$bin" calc-efs "$plain" "$one_cfg" "$tmp_dir/plain.cfg" >/dev/null
"$bin" calc-efs "$rank3" "$one_cfg" "$tmp_dir/rank3.cfg" >/dev/null
python3 - "$tmp_dir/plain.cfg" "$tmp_dir/rank3.cfg" <<'PY'
import re
import sys

number = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")
def values(path):
    return [float(x) for x in number.findall(open(path, encoding="utf-8").read())]
a, b = values(sys.argv[1]), values(sys.argv[2])
if not a or len(a) != len(b):
    raise SystemExit("R=K gate-free E/F/S output shape differs from full K")
worst = max(abs(x-y) for x, y in zip(a, b))
scale = max([1.0] + [abs(x) for x in a] + [abs(x) for x in b])
if worst > 2.0e-11 * scale:
    raise SystemExit(f"R=K gate-free E/F/S parity failed: {worst:.3e}")
print(f"gate_free_rank3_identity_efs_worst={worst:.3e}")
PY

layout=$("$bin" check-loss-gradient-dev "$rank2" "$one_cfg" \
  --max-configs=1 --energy-weight=1 --force-weight=0 --stress-weight=0 \
  --radial-smooth=0 --scalar-head-l2=0 --gate-scalar-l2=0 \
  --gate-mix-l2=0 --gate-full-l2=0 --gate-x-l2=0 \
  --coeff-start=0 --coeff-end=1 \
  --displacement=1e-5 --abs-tolerance=1 --rel-tolerance=1)
outer_begin=$(sed -n 's/.*outer_k_rank_begin=\([0-9][0-9]*\).*/\1/p' <<<"$layout")
outer_end=$(sed -n 's/.*outer_k_rank_end=\([0-9][0-9]*\).*/\1/p' <<<"$layout")
if [[ -z "$outer_begin" || -z "$outer_end" || $((outer_end-outer_begin)) -ne 18 ]]; then
  echo "failed to discover the gate-free outer-rank weight block: $layout" >&2
  exit 1
fi
"$bin" check-loss-gradient-direction-dev "$rank2" "$one_cfg" \
  --max-configs=1 --energy-weight=1 --force-weight=0.1 --stress-weight=0.01 \
  --radial-smooth=0 --scalar-head-l2=0 --gate-scalar-l2=0 \
  --gate-mix-l2=0 --gate-full-l2=0 --gate-x-l2=0 \
  --coeff-start="$outer_begin" --coeff-end="$outer_end" \
  --displacement=1e-5 --abs-tolerance=2e-5 --rel-tolerance=3e-3

trained="$tmp_dir/rank2-trained.sus2"
"$bin" train "$rank2" "$linear_cfg" \
  --trained-pot-name="$trained" \
  --max-iter=1 --skip-preinit --do-lin --do-lin-steps=1 \
  --do-lin-freq=1 --optimizer=lbfgs --lbfgs-initial-step=1e-8 \
  --init-params=same \
  --energy-weight=1 --force-weight=0 --stress-weight=0 \
  --radial-parameterization=weight-norm \
  --radial-smooth=0 --scalar-head-l2=1e-5 --gate-scalar-l2=0 \
  --gate-mix-l2=0 --gate-full-l2=0 --gate-x-l2=0 >/dev/null
test -s "$trained"
"$bin" calc-efs "$trained" "$one_cfg" "$tmp_dir/rank2-trained.cfg" >/dev/null

if "$bin" train "$rank2" "$one_cfg" --two-layer-gate \
     --max-iter=0 --skip-preinit --trained-pot-name="$tmp_dir/invalid.sus2" \
     >"$tmp_dir/upgrade.err" 2>&1; then
  echo "an existing gate-free ranked model was incorrectly upgraded to a gate model" >&2
  exit 1
fi
grep -q 'initialize a new model with --two-layer-gate and --outer-k-rank together' \
  "$tmp_dir/upgrade.err"

echo "gate-free outer K-rank checks passed"
