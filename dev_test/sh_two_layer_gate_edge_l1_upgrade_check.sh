#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

tmp_dir=".codex_tmp/sh_two_layer_gate_edge_l1_upgrade"
mkdir -p "$tmp_dir"

init_base="$tmp_dir/init_gate.mtp"
base="$tmp_dir/base_gate.mtp"
upgraded="$tmp_dir/upgraded_edge_l1_gate.mtp"
edge_ref="$tmp_dir/edge_l1_reference.mtp"
train="$tmp_dir/one.cfg"
base_pred="$tmp_dir/base_pred.cfg"
upgraded_pred="$tmp_dir/upgraded_pred.cfg"
base_train_log="$tmp_dir/base_train.log"
upgrade_log="$tmp_dir/upgrade_train.log"
rm -f "$init_base" "$base" "$upgraded" "$edge_ref" "$train" "$base_pred" "$upgraded_pred" "$base_train_log" "$upgrade_log"
awk '{print} /^END_CFG/{exit}' example/train.cfg > "$train"

common_opts=(
  --species-count=2
  --l-max=2
  --k-max=2
  --body-order=4
  --body-l-max=2,2,2,2
  --radial-basis-size=4
  --cutoff=5.0
  --write-sh-scalar-info
  --two-layer-gate
  --two-layer-gate-mode=mu-scalar-full
)

./bin/mlp-sus2 init-sh "$init_base" "${common_opts[@]}" >/dev/null
./bin/mlp-sus2 init-sh "$edge_ref" "${common_opts[@]}" --two-layer-gate-edge-l1 >/dev/null
./bin/mlp-sus2 train "$init_base" "$train" \
  --trained-pot-name="$base" \
  --max-iter=1 \
  --skip-preinit \
  --shift \
  --energy-weight=0 \
  --force-weight=0 \
  --stress-weight=0 \
  --radial-smooth=0 >"$base_train_log"

if grep -q "two_layer_gate_edge_l1_enabled = true" "$base"; then
  echo "base gate model unexpectedly has edge-L1 metadata" >&2
  exit 1
fi

./bin/mlp-sus2 calc-efs "$base" "$train" "$base_pred" >/dev/null

./bin/mlp-sus2 train "$base" "$train" \
  --trained-pot-name="$upgraded" \
  --max-iter=1 \
  --skip-preinit \
  --shift \
  --two-layer-gate \
  --two-layer-gate-mode=mu-scalar-full \
  --two-layer-gate-edge-l1 \
  --energy-weight=0 \
  --force-weight=0 \
  --stress-weight=0 \
  --radial-smooth=0 >"$upgrade_log"

grep -q "SUS2-SH two-layer gate edge-L1 upgrade enabled" "$upgrade_log"
grep -q "two_layer_gate_edge_l1_enabled = true" "$upgraded"
grep -q "two_layer_gate_edge_l1_source_moment_count" "$upgraded"
grep -q "two_layer_gate_edge_l1_source_moment_indices" "$upgraded"
if grep -q "two_layer_gate_edge_l1_source_mu_count" "$upgraded"; then
  echo "edge-L1 upgrade should write full L=1 moment sources, not raw mu sources" >&2
  exit 1
fi

source_count=$(awk '/two_layer_gate_edge_l1_source_moment_count/ {print $3}' "$upgraded")
if [[ "$source_count" -le 2 ]]; then
  echo "edge-L1 upgrade should include multi-body L=1 source moments; got $source_count" >&2
  exit 1
fi
weight_count=$(awk '/two_layer_gate_edge_l1_weight_count/ {print $3}' "$upgraded")
if [[ "$weight_count" -ne $((6 * source_count)) ]]; then
  echo "edge-L1 upgrade weight count should be radial_func_count * source_count; got $weight_count for $source_count sources" >&2
  exit 1
fi

python3 - "$upgraded" "$edge_ref" <<'PY'
import re
import sys

def source_indices(path):
    text = open(path).read()
    match = re.search(r"two_layer_gate_edge_l1_source_moment_indices\s*=\s*\{([^}]*)\}", text, re.S)
    if not match:
        raise SystemExit(f"missing edge-L1 source list in {path}")
    return [int(x) for x in re.split(r"[,\s]+", match.group(1).strip()) if x]

upgraded_sources = source_indices(sys.argv[1])
ref_sources = source_indices(sys.argv[2])
if upgraded_sources != ref_sources:
    raise SystemExit(
        f"upgraded edge-L1 sources differ from init-sh reference: {upgraded_sources} vs {ref_sources}"
    )

text = open(sys.argv[1]).read()
match = re.search(r"two_layer_gate_edge_l1_weights\s*=\s*\{([^}]*)\}", text, re.S)
if not match:
    raise SystemExit("missing edge-L1 weights")
weights = [float(x) for x in re.split(r"[,\s]+", match.group(1).strip()) if x]
if not weights:
    raise SystemExit("empty edge-L1 weights")
max_abs = max(abs(x) for x in weights)
if max_abs > 1.0e-14:
    raise SystemExit(f"edge-L1 upgrade weights should initialize to zero; max_abs={max_abs:.3e}")
print(f"edge-L1 zero initialization OK: count={len(weights)} max_abs={max_abs:.3e}")
PY

./bin/mlp-sus2 calc-efs "$upgraded" "$train" "$upgraded_pred" >/dev/null

python3 - "$base_pred" "$upgraded_pred" <<'PY'
import math
import sys

def values(path):
    out = []
    fcols = None
    need_energy_value = False
    need_stress_value = False
    with open(path) as handle:
        for raw in handle:
            line = raw.strip()
            parts = line.split()
            if not parts:
                continue
            if need_energy_value:
                out.append(float(parts[0]))
                need_energy_value = False
            elif need_stress_value:
                out.extend(float(x) for x in parts)
                need_stress_value = False
            elif parts[0] == "Energy":
                if len(parts) > 1:
                    out.append(float(parts[1]))
                else:
                    need_energy_value = True
            elif parts[0] == "PlusStress:":
                need_stress_value = True
            elif line.startswith("AtomData:"):
                header = line.split()[1:]
                fcols = [header.index(name) for name in ("fx", "fy", "fz")]
            elif fcols is not None and line[:1].isdigit() and len(parts) > max(fcols):
                out.extend(float(parts[idx]) for idx in fcols)
    return out

a = values(sys.argv[1])
b = values(sys.argv[2])
if len(a) != len(b):
    raise SystemExit(f"value count mismatch: {len(a)} vs {len(b)}")
err = max(abs(x - y) for x, y in zip(a, b))
if not math.isfinite(err) or err > 1.0e-10:
    raise SystemExit(f"zero-initialized edge-L1 upgrade changed predictions: {err}")
print(f"edge-L1 upgrade compatibility OK: max_err={err:.3e}")
PY
