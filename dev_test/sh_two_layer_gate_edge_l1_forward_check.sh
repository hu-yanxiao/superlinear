#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

tmp_dir=".codex_tmp/sh_two_layer_gate_edge_l1_forward"
mkdir -p "$tmp_dir"

zero_model="$tmp_dir/edge_l1_zero.mtp"
nonzero_model="$tmp_dir/edge_l1_nonzero.mtp"
zero_pred="$tmp_dir/zero_pred.cfg"
nonzero_pred="$tmp_dir/nonzero_pred.cfg"

./bin/mlp-sus2 init-sh "$zero_model" \
  --species-count=2 \
  --l-max=2 \
  --k-max=2 \
  --body-order=4 \
  --body-l-max=2,2,2,2 \
  --radial-basis-size=4 \
  --cutoff=5.0 \
  --write-sh-scalar-info \
  --two-layer-gate \
  --two-layer-gate-edge-l1 >/dev/null

python3 - "$zero_model" "$nonzero_model" <<'PY'
import re
import sys

src, dst = sys.argv[1:3]
base_text = open(src).read()

def with_linear_coeffs(text):
    species = int(re.search(r"species_count = (\d+)", text).group(1))
    alpha = int(re.search(r"alpha_scalar_moments = (\d+)", text).group(1))
    species_line = "species_coeffs = {" + ", ".join("1.000000000000000e+00" for _ in range(species)) + "}"
    moment_line = "moment_coeffs = {" + ", ".join("1.000000000000000e+00" for _ in range(alpha)) + "}"
    if "species_coeffs =" in text:
        text = re.sub(r"species_coeffs = \{[^}]*\}", species_line, text, flags=re.S)
    else:
        text += "\n" + species_line
    if "moment_coeffs =" in text:
        text = re.sub(r"moment_coeffs = \{[^}]*\}", moment_line, text, flags=re.S)
    else:
        text += "\n" + moment_line + "\n"
    return text

zero_text = with_linear_coeffs(base_text)
match = re.search(r"two_layer_gate_edge_l1_weights = \{([^}]*)\}", zero_text, re.S)
if not match:
    raise SystemExit("missing two_layer_gate_edge_l1_weights")
alpha_basic = int(re.search(r"alpha_index_basic_count = (\d+)", zero_text).group(1))
source_count = int(re.search(r"two_layer_gate_edge_l1_source_moment_count = (\d+)", zero_text).group(1))
source_match = re.search(r"two_layer_gate_edge_l1_source_moment_indices = \{([^}]*)\}", zero_text, re.S)
if not source_match:
    raise SystemExit("missing two_layer_gate_edge_l1_source_moment_indices")
sources = [int(x) for x in re.findall(r"[-+0-9]+", source_match.group(1))]
if len(sources) != source_count:
    raise SystemExit("edge-L1 source count mismatch")
try:
    nonraw_source = next(i for i, base in enumerate(sources) if base >= alpha_basic)
except StopIteration:
    raise SystemExit("edge-L1 full source list does not contain a multi-body L=1 source")
weights = [float(x) for x in re.findall(r"[-+0-9.eE]+", match.group(1))]
if len(weights) % source_count != 0:
    raise SystemExit("edge-L1 weight count is not divisible by source count")
target_count = len(weights) // source_count
if target_count == 0:
    raise SystemExit("empty two_layer_gate_edge_l1_weights")
for i in range(len(weights)):
    weights[i] = 0.0
for mu in range(target_count):
    weights[mu * source_count + nonraw_source] = 1000.0 * ((mu % 5) - 2)
replacement = (
    "two_layer_gate_edge_l1_weights = {"
    + ", ".join(f"{x:.15e}" for x in weights)
    + "}"
)
nonzero_text = zero_text[:match.start()] + replacement + zero_text[match.end():]
open(src, "w").write(zero_text)
open(dst, "w").write(nonzero_text)
PY

./bin/mlp-sus2 calc-efs "$zero_model" example/train.cfg "$zero_pred" >/dev/null
./bin/mlp-sus2 calc-efs "$nonzero_model" example/train.cfg "$nonzero_pred" >/dev/null

python3 - "$zero_pred" "$nonzero_pred" <<'PY'
import math
import sys

def site_energies(path):
    out = []
    with open(path) as handle:
        for raw in handle:
            line = raw.strip()
            parts = raw.strip().split()
            if not parts:
                continue
            if line.startswith("AtomData:"):
                header = line.split()[1:]
                site_col = header.index("site_en")
                continue
            if "site_col" in locals() and line[:1].isdigit() and len(parts) > site_col:
                out.append(float(parts[site_col]))
    return out

a = site_energies(sys.argv[1])
b = site_energies(sys.argv[2])
if len(a) != len(b):
    raise SystemExit(f"site energy count mismatch: {len(a)} vs {len(b)}")
delta = max(abs(x - y) for x, y in zip(a, b))
if not math.isfinite(delta) or delta < 1.0e-12:
    raise SystemExit(f"nonzero edge-L1 gate did not change site energies: delta={delta}")
print(f"nonzero edge-L1 forward sensitivity OK: max_site_energy_delta={delta:.3e}")
PY
