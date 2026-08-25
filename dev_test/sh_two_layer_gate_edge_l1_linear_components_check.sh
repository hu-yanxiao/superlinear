#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

bin="${MLP_BIN:-./bin/mlp-sus2}"
tmp_dir=".codex_tmp/sh_two_layer_gate_edge_l1_linear_components"
mkdir -p "$tmp_dir"

init_model="$tmp_dir/edge_l1_linear_components_init.mtp"
trained_model="$tmp_dir/edge_l1_linear_components_trained.mtp"
model="$tmp_dir/edge_l1_linear_components.mtp"
cfg="$tmp_dir/non_symmetric.cfg"
train_log="$tmp_dir/train.log"
trained_readout_log="$tmp_dir/trained_readout.log"
components_log="$tmp_dir/components.log"
readout_log="$tmp_dir/readout.log"
rm -f "$init_model" "$trained_model" "$model" "$cfg" \
  "$train_log" "$trained_readout_log" "$components_log" "$readout_log"

"$bin" init-sh "$init_model" \
  --species-count=2 \
  --l-max=2 \
  --k-max=2 \
  --body-order=4 \
  --body-l-max=2,2,2,2 \
  --radial-basis-size=4 \
  --init-params=same \
  --cutoff=5.0 \
  --write-sh-scalar-info \
  --two-layer-gate \
  --two-layer-gate-mode=mu-scalar-full \
  --two-layer-gate-shared-radial \
  --two-layer-gate-edge-l1 >/dev/null

python3 - "$cfg" <<'PY'
import sys

cfg_path = sys.argv[1]
atoms = [
    (1, 0, 1.13, 1.72, 1.41, 0.03, -0.02, 0.01),
    (2, 1, 2.47, 1.19, 3.06, -0.01, 0.04, -0.03),
    (3, 0, 3.81, 2.68, 1.97, 0.02, 0.01, -0.04),
    (4, 1, 5.08, 3.33, 4.44, -0.03, -0.01, 0.02),
    (5, 0, 1.86, 5.21, 5.64, 0.01, -0.05, 0.03),
    (6, 1, 4.62, 5.87, 2.58, -0.02, 0.03, 0.04),
    (7, 0, 6.19, 1.93, 5.16, 0.05, -0.01, -0.02),
    (8, 1, 2.94, 6.36, 1.24, -0.04, 0.02, 0.01),
]
with open(cfg_path, "w") as out:
    out.write("BEGIN_CFG\n")
    out.write(" Size\n")
    out.write(f"      {len(atoms)}\n")
    out.write(" Supercell\n")
    out.write("   8.5000000000    0.0000000000    0.0000000000\n")
    out.write("   0.0000000000    8.5000000000    0.0000000000\n")
    out.write("   0.0000000000    0.0000000000    8.5000000000\n")
    out.write(" AtomData:  id type       cartes_x      cartes_y      cartes_z     fx          fy          fz\n")
    for row in atoms:
        out.write("%7d %6d %12.6f %12.6f %12.6f %11.6f %11.6f %11.6f\n" % row)
    out.write(" Energy\n")
    out.write("    -12.345678\n")
    out.write(" PlusStress:  xx          yy          zz          yz          xz          xy\n")
    out.write("    0.030000   -0.020000    0.010000    0.004000   -0.003000    0.002000\n")
    out.write("END_CFG\n")
PY

python3 - "$init_model" <<'PY'
import re
import sys

model_path = sys.argv[1]
text = open(model_path).read()


def replace(name, value_fn):
    global text
    match = re.search(rf"{name} = \{{([^}}]*)\}}", text, re.S)
    if not match:
        raise SystemExit(f"missing {name}")
    old = [float(item.strip()) for item in match.group(1).split(",") if item.strip()]
    new = [value_fn(index, value) for index, value in enumerate(old)]
    block = name + " = {" + ", ".join(f"{value:.15e}" for value in new) + "}"
    text = text[:match.start()] + block + text[match.end():]


replace(
    "two_layer_gate_weights",
    lambda index, _: (1.0 if index % 2 == 0 else -1.0) * (0.15 + 0.003 * (index % 13)),
)
replace(
    "two_layer_gate_additive_coeffs",
    lambda index, _: (1.0 if index % 3 else -1.0) * (0.7 + 0.02 * (index % 5)),
)
replace("two_layer_gate_type_coeffs", lambda index, _: 0.85 + 0.03 * (index % 7))
replace(
    "two_layer_gate_edge_l1_weights",
    lambda index, _: (1.0 if index % 2 == 0 else -1.0) * (0.2 + 0.01 * (index % 9)),
)
open(model_path, "w").write(text)
PY

"$bin" train "$init_model" "$cfg" \
  --trained-pot-name="$trained_model" \
  --max-iter=1 \
  --skip-preinit \
  --do-lin \
  --do-lin-steps=1 \
  --do-lin-freq=1 \
  --energy-weight=1 \
  --force-weight=1 \
  --stress-weight=0 >"$train_log" 2>&1

"$bin" check-linear-readout-dev "$trained_model" "$cfg" \
  --max-configs=1 \
  --max-atoms=2 \
  --abs-tolerance=2e-8 \
  --rel-tolerance=2e-7 >"$trained_readout_log" 2>&1

cp "$trained_model" "$model"

python3 - "$model" <<'PY'
import re
import sys

model_path = sys.argv[1]
text = open(model_path).read()


def values(name):
    match = re.search(rf"{name} = \{{([^}}]*)\}}", text, re.S)
    if not match:
        raise SystemExit(f"missing {name}")
    return match, [float(item.strip()) for item in match.group(1).split(",") if item.strip()]


def replace(name, value_fn):
    global text
    match, old = values(name)
    new = [value_fn(index, value) for index, value in enumerate(old)]
    block = name + " = {" + ", ".join(f"{value:.15e}" for value in new) + "}"
    text = text[:match.start()] + block + text[match.end():]


# Full TrainLinear canonicalizes the center-species multiplier to one and moves
# each fitted atomic offset into regression_coeffs.  Exercise that post-solve
# gauge so the component reconstruction is required to equal direct E/F.
replace("species_coeffs", lambda index, _: 1.0)
replace(
    "moment_coeffs",
    lambda index, _: (1.0 if index % 2 == 0 else -1.0) * (0.03 + 0.001 * (index % 11)),
)
replace(
    "two_layer_gate_weights",
    lambda index, _: (1.0 if index % 2 == 0 else -1.0) * (0.15 + 0.003 * (index % 13)),
)
replace(
    "two_layer_gate_additive_coeffs",
    lambda index, _: (1.0 if index % 3 else -1.0) * (0.7 + 0.02 * (index % 5)),
)
replace(
    "two_layer_gate_type_coeffs",
    lambda index, _: 0.85 + 0.03 * (index % 7),
)
replace(
    "two_layer_gate_edge_l1_weights",
    lambda index, _: (1.0 if index % 2 == 0 else -1.0) * (0.2 + 0.01 * (index % 9)),
)
open(model_path, "w").write(text)
PY

"$bin" check-linear-components-fd-dev "$model" "$cfg" \
  --max-configs=1 \
  --max-atoms=2 \
  --displacement=1e-5 \
  --abs-tolerance=3e-6 \
  --rel-tolerance=3e-4 >"$components_log" 2>&1

"$bin" check-linear-readout-dev "$model" "$cfg" \
  --max-configs=1 \
  --max-atoms=2 \
  --abs-tolerance=2e-9 \
  --rel-tolerance=2e-8 >"$readout_log" 2>&1

grep -q "checked_components=" "$components_log"
grep -q "force_components=" "$trained_readout_log"
grep -q "force_components=" "$readout_log"
cat "$trained_readout_log"
cat "$components_log"
cat "$readout_log"
