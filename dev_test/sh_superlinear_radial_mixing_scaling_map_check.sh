#!/usr/bin/env bash
set -Eeuo pipefail
trap 'echo "direct radial-mixer scaling-map check failed at line ${LINENO}: ${BASH_COMMAND}" >&2' ERR

repo_root=$(cd "$(dirname "$0")/.." && pwd)
bin=${1:-$repo_root/bin/mlp-sus2}
bin=$(cd "$(dirname "$bin")" && pwd)/$(basename "$bin")
[[ -x "$bin" ]] || { echo "missing executable: $bin" >&2; exit 2; }

tmp_dir=$(mktemp -d "${TMPDIR:-/tmp}/sus2sh-direct-mixer-scaling-XXXXXX")
if [[ ${SUS2_TEST_KEEP_TMP:-0} == 1 ]]; then
  echo "keeping test directory: $tmp_dir"
else
  trap 'rm -rf "$tmp_dir"' EXIT
fi

cfg="$tmp_dir/one.cfg"
awk '{print} /^END_CFG/{exit}' "$repo_root/example/train.cfg" >"$cfg"

common=(
  --species-count=2 --l-max=2 --k-max=3 --body-order=4
  --body-l-max=2,2,2,2 --radial-basis-size=3
  --radial-basis-type=RBChebyshev_sss --cutoff=5.0
  --sh-factor-pruning=q-total --init-params=same
  --two-layer-gate --two-layer-gate-mode=mu-scalar-full
  --two-layer-gate-body-order=3 --two-layer-gate-k-rank=2
  --outer-k-rank=2 --two-layer-gate-radial-mixing=full
)

for map in K L pair; do
  "$bin" init-sh "$tmp_dir/${map}.init.sus2" "${common[@]}" \
    --scaling-map="$map" >/dev/null
  "$bin" roundtrip-model-dev \
    "$tmp_dir/${map}.init.sus2" "$tmp_dir/${map}.sus2" >/dev/null
done

# Seed distinct compact scaling blocks, activate the direct mixer, and create
# mathematically identical LK expansions. Mixer C stays (l,outer,k,b) in both
# files; only the radial slope/shift sharing map is expanded.
python3 - "$tmp_dir/K.sus2" "$tmp_dir/K.LK.sus2" K \
            "$tmp_dir/L.sus2" "$tmp_dir/L.LK.sus2" L \
            "$tmp_dir/pair.sus2" "$tmp_dir/pair.LK.sus2" pair <<'PY'
import re
import sys

number = r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?"
species, kmax, lcount, rb_size, outer = 2, 3, 3, 3, 2
pair_count = species * species
block_width = 2 * pair_count
expected_mixer = lcount * outer * kmax * rb_size


def vector(text, name):
    match = re.search(
        rf"(?m)^\s*{re.escape(name)}\s*=\s*\{{([^}}]*)\}}", text, re.S
    )
    if not match:
        raise SystemExit(f"missing {name}")
    return match, [float(x) for x in re.findall(number, match.group(1))]


def replace(text, name, values):
    match, _ = vector(text, name)
    value = name + " = {" + ", ".join(f"{x:.16e}" for x in values) + "}"
    return text[:match.start()] + value + text[match.end():]


for source, target, compact_map in zip(sys.argv[1::3], sys.argv[2::3], sys.argv[3::3]):
    text = open(source, encoding="utf-8").read()
    if "two_layer_gate_radial_mixing_layout = direct-k" not in text:
        raise SystemExit("missing direct-k layout marker")
    count = re.search(
        r"two_layer_gate_radial_mixing_coeff_count\s*=\s*(\d+)", text
    )
    if not count or int(count.group(1)) != expected_mixer:
        raise SystemExit("wrong direct mixer coefficient count")

    compact_count = 1 if compact_map == "pair" else kmax if compact_map == "K" else lcount
    blocks = []
    for channel in range(compact_count):
        slopes = [2.4 + 0.31 * channel + 0.037 * pair for pair in range(pair_count)]
        shifts = [1.1 + 0.23 * channel + 0.029 * pair for pair in range(pair_count)]
        blocks.append(slopes + shifts)
    text = replace(text, "scal_coeffs", [x for block in blocks for x in block])

    _, mixer = vector(text, "two_layer_gate_radial_mixing_coeffs")
    if len(mixer) != expected_mixer:
        raise SystemExit("wrong direct mixer vector length")
    mixer = [0.08 * ((i % 11) - 5) / 5.0 for i in range(expected_mixer)]
    text = replace(text, "two_layer_gate_radial_mixing_coeffs", mixer)
    open(source, "w", encoding="utf-8").write(text)

    expanded = []
    for k in range(kmax):
        for ell in range(lcount):
            block = blocks[0] if compact_map == "pair" else blocks[k] if compact_map == "K" else blocks[ell]
            expanded.extend(block)
    lk = re.sub(
        rf"(?m)^scaling_map\s*=\s*{compact_map}\s*$",
        "scaling_map = LK",
        text,
        count=1,
    )
    lk = replace(lk, "scal_coeffs", expanded)
    open(target, "w", encoding="utf-8").write(lk)
PY

compare_efs() {
  local compact_map=$1
  "$bin" calc-efs "$tmp_dir/${compact_map}.sus2" "$cfg" \
    "$tmp_dir/${compact_map}.cfg" >/dev/null
  "$bin" calc-efs "$tmp_dir/${compact_map}.LK.sus2" "$cfg" \
    "$tmp_dir/${compact_map}.LK.cfg" >/dev/null
  python3 - "$compact_map" "$tmp_dir/${compact_map}.cfg" \
    "$tmp_dir/${compact_map}.LK.cfg" <<'PY'
import math
import re
import sys

label, left_path, right_path = sys.argv[1:]
number = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")
left = [float(x) for x in number.findall(open(left_path).read())]
right = [float(x) for x in number.findall(open(right_path).read())]
if not left or len(left) != len(right):
    raise SystemExit(f"{label}/LK output shape mismatch")
worst = max(abs(a - b) for a, b in zip(left, right))
scale = max(1.0, *(abs(x) for x in left), *(abs(x) for x in right))
if not all(math.isfinite(x) for x in left + right) or worst > 2.0e-11 * scale:
    raise SystemExit(f"{label}/LK EFS mismatch: {worst:.3e}, scale={scale:.3e}")
print(f"direct_mixer_{label}_LK_EFS max_abs={worst:.3e} scale={scale:.3e}")
PY
}

compare_efs K
compare_efs L
compare_efs pair

for model in K.sus2 L.sus2 pair.sus2 K.LK.sus2; do
  "$bin" check-efs-fd-dev "$tmp_dir/$model" "$cfg" \
    --max-configs=1 --max-atoms=2 --displacement=1.0e-5 \
    --abs-tolerance=7.0e-5 --rel-tolerance=7.0e-3 >/dev/null
done

layout_field() {
  local model=$1 key=$2
  "$bin" check-loss-gradient-dev "$tmp_dir/$model" "$cfg" \
    --max-configs=1 --energy-weight=1 --force-weight=.1 --stress-weight=.01 \
    --gate-x-l2=0 --coeff-start=0 --coeff-end=1 --displacement=1.0e-5 \
    --abs-tolerance=1 --rel-tolerance=1 2>/dev/null |
    tail -1 | awk -v key="$key" '{for(i=1;i<=NF;i++) if($i ~ ("^" key "=[0-9]+$")){sub(/^[^=]*=/,"",$i); print $i}}'
}

check_direction() {
  local label=$1 model=$2 begin=$3 end=$4
  local output
  output=$("$bin" check-loss-gradient-direction-dev "$tmp_dir/$model" "$cfg" \
    --max-configs=1 --energy-weight=1 --force-weight=.1 --stress-weight=.01 \
    --gate-x-l2=0 --coeff-start="$begin" --coeff-end="$end" \
    --displacement=1.0e-4 --abs-tolerance=4.0e-5 --rel-tolerance=8.0e-3)
  echo "$label $output" | tail -1
  python3 - "$label" "$output" <<'PY'
import math
import re
import sys

label, text = sys.argv[1:]
analytic = float(re.search(r"analytic_directional=([^ ]+)", text).group(1))
fd = float(re.search(r"finite_difference=([^ ]+)", text).group(1))
if not (math.isfinite(analytic) and math.isfinite(fd)):
    raise SystemExit(f"{label}: non-finite directional derivative")
if max(abs(analytic), abs(fd)) < 1.0e-11:
    raise SystemExit(f"{label}: vacuous directional derivative")
PY
}

mix_begin=$(layout_field K.sus2 gate_radial_mixing_begin)
mix_end=$(layout_field K.sus2 gate_radial_mixing_end)
[[ $((mix_end - mix_begin)) == 54 ]]
check_direction direct_mixer_C_K K.sus2 "$mix_begin" "$mix_end"

for spec in "K.sus2:24:K" "L.sus2:24:L" "pair.sus2:8:pair" "K.LK.sus2:72:LK"; do
  IFS=: read -r model scaling_count label <<<"$spec"
  radial_begin=$(layout_field "$model" radial_begin)
  check_direction "direct_mixer_scaling_${label}" "$model" \
    "$((radial_begin - scaling_count))" "$radial_begin"
done

echo "direct K-to-outer radial-mixer scaling-map checks passed"
