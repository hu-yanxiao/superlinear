#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

tmp_dir=".codex_tmp/sh_two_layer_gate_edge_l1_zero"
mkdir -p "$tmp_dir"

base="$tmp_dir/base_gate.mtp"
edge="$tmp_dir/edge_gate_zero.mtp"
base_pred="$tmp_dir/base_pred.cfg"
edge_pred="$tmp_dir/edge_pred.cfg"

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
)

./bin/mlp-sus2 init-sh "$base" "${common_opts[@]}" >/dev/null
./bin/mlp-sus2 init-sh "$edge" "${common_opts[@]}" --two-layer-gate-edge-l1 >/dev/null
./bin/mlp-sus2 calc-efs "$base" example/train.cfg "$base_pred" >/dev/null
./bin/mlp-sus2 calc-efs "$edge" example/train.cfg "$edge_pred" >/dev/null

python3 - "$base_pred" "$edge_pred" <<'PY'
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
                parts = line.split()
                out.extend(float(parts[idx]) for idx in fcols)
    return out

a = values(sys.argv[1])
b = values(sys.argv[2])
if len(a) != len(b):
    raise SystemExit(f"value count mismatch: {len(a)} vs {len(b)}")
err = max(abs(x - y) for x, y in zip(a, b))
if not math.isfinite(err) or err > 1.0e-10:
    raise SystemExit(f"zero edge-L1 gate predictions differ: {err}")
print(f"zero edge-L1 compatibility OK: max_err={err:.3e}")
PY
