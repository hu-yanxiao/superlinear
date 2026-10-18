#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 /path/to/mlp-sus2" >&2
  exit 2
fi

BIN=$1
REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)
WORKDIR=$(mktemp -d "${TMPDIR:-/tmp}/sus2sh-readout-diagnostics-XXXXXX")
trap 'rm -rf "$WORKDIR"' EXIT

MODEL_ZERO="$WORKDIR/readout_zero.sus2"
MODEL_ON="$WORKDIR/readout_on.sus2"
MODEL_DISABLED="$WORKDIR/readout_disabled.sus2"
MODEL_GATE_ZERO="$WORKDIR/readout_gate_zero.sus2"
MODEL_LMP_ZERO="$WORKDIR/readout_lmp_zero.sus2"
MODEL_LMP_ON="$WORKDIR/readout_lmp_on.sus2"
ONE_CFG="$WORKDIR/one.cfg"

awk '{print} /^END_CFG/{exit}' "$REPO_ROOT/example/train.cfg" > "$ONE_CFG"

"$BIN" init-sh "$MODEL_ZERO" \
  --species-count=2 \
  --l-max=2 \
  --k-max=2 \
  --body-order=4 \
  --body-l-max=2,2,2,2 \
  --sh-factor-pruning=q-total \
  --write-sh-scalar-info \
  --two-layer-gate \
  --two-layer-gate-shared-radial \
  --two-layer-gate-mode=mu-scalar-full \
  --two-layer-gate-readout \
  --cutoff=5.0 \
  --radial-basis-size=4 \
  --radial-basis-type=RBChebyshev_sss \
  --init-params=same >/dev/null

python3 - \
  "$MODEL_ZERO" "$MODEL_ON" "$MODEL_DISABLED" "$MODEL_GATE_ZERO" \
  "$MODEL_LMP_ZERO" "$MODEL_LMP_ON" <<'PY'
import re
import sys

zero_path, on_path, disabled_path, gate_zero_path, lmp_zero_path, lmp_on_path = sys.argv[1:]
text = open(zero_path, encoding="utf-8").read()

number = r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?"

def replace_list(source, name, transform):
    match = re.search(rf"{name}\s*=\s*\{{([^}}]*)\}}", source, re.S)
    if not match:
        raise SystemExit(f"missing {name}")
    values = [float(x) for x in re.findall(number, match.group(1))]
    replacement = name + " = {" + ", ".join(
        f"{transform(i, value):.15e}" for i, value in enumerate(values)
    ) + "}"
    return source[:match.start()] + replacement + source[match.end():]

with_gate = replace_list(
    text,
    "two_layer_gate_weights",
    lambda i, value: 0.17 if i == 0 else value,
)
zero = replace_list(
    with_gate,
    "two_layer_gate_readout_coeffs",
    lambda i, value: 0.0,
)
on = replace_list(
    with_gate,
    "two_layer_gate_readout_coeffs",
    lambda i, value: (1.0 if i % 2 == 0 else -1.0) * (0.025 + 0.0005 * i),
)
gate_zero = replace_list(
    on,
    "two_layer_gate_weights",
    lambda i, value: 0.0,
)
disabled, disabled_count = re.subn(
    r"\ntwo_layer_gate_readout_enabled\s*=\s*true\s*\n"
    r"two_layer_gate_readout_coeff_count\s*=\s*\d+\s*\n"
    r"two_layer_gate_readout_coeffs\s*=\s*\{[^}]*\}\s*\n",
    "\n",
    zero,
    count=1,
    flags=re.S,
)
if disabled_count != 1:
    raise SystemExit("failed to remove readout metadata")

open(zero_path, "w", encoding="utf-8").write(zero)
open(on_path, "w", encoding="utf-8").write(on)
open(disabled_path, "w", encoding="utf-8").write(disabled)
open(gate_zero_path, "w", encoding="utf-8").write(gate_zero)
open(lmp_zero_path, "w", encoding="utf-8").write(
    zero.replace("radial_basis_type = RBChebyshev_sss",
                 "radial_basis_type = RBChebyshev_sss_lmp")
)
open(lmp_on_path, "w", encoding="utf-8").write(
    on.replace("radial_basis_type = RBChebyshev_sss",
               "radial_basis_type = RBChebyshev_sss_lmp")
)
PY

"$BIN" calc-descriptors "$MODEL_ZERO" "$ONE_CFG" "$WORKDIR/desc_zero.txt" >/dev/null
"$BIN" calc-descriptors "$MODEL_ON" "$ONE_CFG" "$WORKDIR/desc_on.txt" >/dev/null
"$BIN" calc-descriptors "$MODEL_GATE_ZERO" "$ONE_CFG" "$WORKDIR/desc_gate_zero.txt" >/dev/null
"$BIN" calc-descriptors "$MODEL_LMP_ON" "$ONE_CFG" "$WORKDIR/desc_lmp_on.txt" >/dev/null

"$BIN" calc-eij "$MODEL_ZERO" "$ONE_CFG" "$WORKDIR/eij_zero.txt" >/dev/null
"$BIN" calc-eij "$MODEL_DISABLED" "$ONE_CFG" "$WORKDIR/eij_disabled.txt" >/dev/null
"$BIN" calc-eij "$MODEL_ON" "$ONE_CFG" "$WORKDIR/eij_on.txt" >/dev/null
SUS2_SH_CALC_EIJ_FAST=0 \
  "$BIN" calc-eij "$MODEL_ON" "$ONE_CFG" "$WORKDIR/eij_on_slow.txt" >/dev/null
SUS2_SH_DISABLE_TWO_LAYER_EDGE_CACHE=1 \
  "$BIN" calc-eij "$MODEL_ON" "$ONE_CFG" "$WORKDIR/eij_on_no_cache.txt" >/dev/null
"$BIN" calc-eij "$MODEL_LMP_ZERO" "$ONE_CFG" "$WORKDIR/eij_lmp_zero.txt" >/dev/null
"$BIN" calc-eij "$MODEL_LMP_ON" "$ONE_CFG" "$WORKDIR/eij_lmp_on.txt" >/dev/null

"$BIN" calc-partialE "$MODEL_ZERO" "$ONE_CFG" "$WORKDIR/partial_zero.txt" >/dev/null
"$BIN" calc-partialE "$MODEL_ON" "$ONE_CFG" "$WORKDIR/partial_on.txt" >/dev/null
"$BIN" calc-partialE "$MODEL_LMP_ZERO" "$ONE_CFG" "$WORKDIR/partial_lmp_zero.txt" >/dev/null
"$BIN" calc-partialE "$MODEL_LMP_ON" "$ONE_CFG" "$WORKDIR/partial_lmp_on.txt" >/dev/null

"$BIN" calc-efs "$MODEL_ZERO" "$ONE_CFG" "$WORKDIR/efs_zero.cfg" >/dev/null
"$BIN" calc-efs "$MODEL_DISABLED" "$ONE_CFG" "$WORKDIR/efs_disabled.cfg" >/dev/null
"$BIN" calc-efs "$MODEL_ON" "$ONE_CFG" "$WORKDIR/efs_on.cfg" >/dev/null
"$BIN" calc-efs "$MODEL_LMP_ZERO" "$ONE_CFG" "$WORKDIR/efs_lmp_zero.cfg" >/dev/null
"$BIN" calc-efs "$MODEL_LMP_ON" "$ONE_CFG" "$WORKDIR/efs_lmp_on.cfg" >/dev/null

python3 - \
  "$MODEL_ON" \
  "$WORKDIR/desc_zero.txt" \
  "$WORKDIR/desc_on.txt" \
  "$WORKDIR/desc_gate_zero.txt" \
  "$WORKDIR/desc_lmp_on.txt" \
  "$WORKDIR/eij_zero.txt" \
  "$WORKDIR/eij_disabled.txt" \
  "$WORKDIR/eij_on.txt" \
  "$WORKDIR/eij_on_slow.txt" \
  "$WORKDIR/eij_on_no_cache.txt" \
  "$WORKDIR/eij_lmp_zero.txt" \
  "$WORKDIR/eij_lmp_on.txt" \
  "$WORKDIR/partial_zero.txt" \
  "$WORKDIR/partial_on.txt" \
  "$WORKDIR/partial_lmp_zero.txt" \
  "$WORKDIR/partial_lmp_on.txt" \
  "$WORKDIR/efs_zero.cfg" \
  "$WORKDIR/efs_disabled.cfg" \
  "$WORKDIR/efs_on.cfg" \
  "$WORKDIR/efs_lmp_zero.cfg" \
  "$WORKDIR/efs_lmp_on.cfg" <<'PY'
import math
import re
import sys

(
    model_path,
    desc_zero_path,
    desc_on_path,
    desc_gate_zero_path,
    desc_lmp_on_path,
    eij_zero_path,
    eij_disabled_path,
    eij_on_path,
    eij_slow_path,
    eij_no_cache_path,
    eij_lmp_zero_path,
    eij_lmp_on_path,
    partial_zero_path,
    partial_on_path,
    partial_lmp_zero_path,
    partial_lmp_on_path,
    efs_zero_path,
    efs_disabled_path,
    efs_on_path,
    efs_lmp_zero_path,
    efs_lmp_on_path,
) = sys.argv[1:]

model = open(model_path, encoding="utf-8").read()

def scalar_int(name):
    match = re.search(rf"^\s*{name}\s*=\s*(\d+)\s*$", model, re.M)
    if not match:
        raise SystemExit(f"missing {name}")
    return int(match.group(1))

outer_count = scalar_int("alpha_scalar_moments")
readout_count = scalar_int("two_layer_gate_readout_coeff_count")
expected_columns = 1 + outer_count + readout_count

def descriptor_rows(path):
    rows = []
    for raw in open(path, encoding="utf-8"):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        row = [float(x) for x in line.split()]
        if len(row) != expected_columns:
            raise SystemExit(
                f"descriptor column mismatch in {path}: got {len(row)}, "
                f"expected {expected_columns}"
            )
        rows.append(row)
    if not rows:
        raise SystemExit(f"no descriptor rows in {path}")
    return rows

desc_zero = descriptor_rows(desc_zero_path)
desc_on = descriptor_rows(desc_on_path)
desc_gate_zero = descriptor_rows(desc_gate_zero_path)
descriptor_rows(desc_lmp_on_path)
if desc_zero != desc_on:
    raise SystemExit("descriptors changed when only readout coefficients changed")

outer_changed = False
for active, inactive in zip(desc_on, desc_gate_zero):
    if any(abs(a - b) > 1.0e-12
           for a, b in zip(active[1:1 + outer_count],
                           inactive[1:1 + outer_count])):
        outer_changed = True
    for a, b in zip(active[1 + outer_count:], inactive[1 + outer_count:]):
        if abs(a - b) > 1.0e-12:
            raise SystemExit("gate scalar descriptor changed with outer gate weights")
if not outer_changed:
    raise SystemExit("outer descriptors did not respond to nonzero gate weights")

def eij_rows(path):
    rows = []
    for raw in open(path, encoding="utf-8"):
        parts = raw.split()
        if not parts:
            continue
        if len(parts) != 4:
            raise SystemExit(f"invalid calc-eij row in {path}: {raw.strip()}")
        rows.append((int(parts[0]), int(parts[1]), float(parts[2]), float(parts[3])))
    if not rows:
        raise SystemExit(f"no calc-eij rows in {path}")
    return rows

eij_zero = eij_rows(eij_zero_path)
eij_disabled = eij_rows(eij_disabled_path)
eij_on = eij_rows(eij_on_path)
eij_slow = eij_rows(eij_slow_path)
eij_no_cache = eij_rows(eij_no_cache_path)
eij_lmp_zero = eij_rows(eij_lmp_zero_path)
eij_lmp_on = eij_rows(eij_lmp_on_path)
if len(eij_zero) != len(eij_disabled):
    raise SystemExit("zero-readout and disabled-readout calc-eij row counts differ")
for zero, disabled in zip(eij_zero, eij_disabled):
    if zero[:2] != disabled[:2] or abs(zero[2] - disabled[2]) > 1.0e-12:
        raise SystemExit("zero-readout and disabled-readout pair metadata differ")
    tol = 2.0e-12 * max(1.0, abs(zero[3]), abs(disabled[3]))
    if abs(zero[3] - disabled[3]) > tol:
        raise SystemExit(
            "zero-readout and disabled-readout calc-eij mismatch: "
            f"{zero[3]:.16e} vs {disabled[3]:.16e}"
        )
if len(eij_on) != len(eij_slow):
    raise SystemExit("fast and slow calc-eij row counts differ")
for fast, slow in zip(eij_on, eij_slow):
    if fast[:2] != slow[:2] or abs(fast[2] - slow[2]) > 1.0e-12:
        raise SystemExit("fast and slow calc-eij pair metadata differ")
    tol = 2.0e-10 * max(1.0, abs(fast[3]), abs(slow[3]))
    if abs(fast[3] - slow[3]) > tol:
        raise SystemExit(
            f"fast/slow calc-eij mismatch: {fast[3]:.16e} vs {slow[3]:.16e}"
        )
if len(eij_on) != len(eij_no_cache):
    raise SystemExit("cached and no-cache calc-eij row counts differ")
for cached, uncached in zip(eij_on, eij_no_cache):
    if cached[:2] != uncached[:2] or abs(cached[2] - uncached[2]) > 1.0e-12:
        raise SystemExit("cached and no-cache calc-eij pair metadata differ")
    tol = 2.0e-10 * max(1.0, abs(cached[3]), abs(uncached[3]))
    if abs(cached[3] - uncached[3]) > tol:
        raise SystemExit(
            "cached/no-cache calc-eij mismatch: "
            f"{cached[3]:.16e} vs {uncached[3]:.16e}"
        )

def cfg_energy(path):
    lines = open(path, encoding="utf-8").read().splitlines()
    for i, line in enumerate(lines):
        if line.strip() == "Energy":
            return float(lines[i + 1].strip())
    raise SystemExit(f"missing Energy in {path}")

def partial_interaction_sum(path):
    total = 0.0
    rows = 0
    for raw in open(path, encoding="utf-8"):
        parts = raw.split()
        if not parts or parts[0].startswith("#"):
            continue
        # Column zero is the one-body site constant.  calc-eij and the
        # remaining partialE columns both decompose only the interaction
        # energy, including the optional readout columns.
        total += sum(float(value) for value in parts[1:])
        rows += 1
    if rows == 0:
        raise SystemExit(f"no calc-partialE rows in {path}")
    return total

delta_eij = sum(row[3] for row in eij_on) - sum(row[3] for row in eij_zero)
if abs(cfg_energy(efs_zero_path) - cfg_energy(efs_disabled_path)) > 2.0e-12:
    raise SystemExit("zero-readout and disabled-readout calc-efs mismatch")
delta_efs = cfg_energy(efs_on_path) - cfg_energy(efs_zero_path)
delta_partial = (
    partial_interaction_sum(partial_on_path)
    - partial_interaction_sum(partial_zero_path)
)
tol = 2.0e-8 * max(1.0, abs(delta_efs))
if abs(delta_eij - delta_efs) > tol:
    raise SystemExit(
        "readout effective-pair delta mismatch: "
        f"calc-eij={delta_eij:.16e}, calc-efs={delta_efs:.16e}, tol={tol:.3e}"
    )
if abs(delta_partial - delta_efs) > tol:
    raise SystemExit(
        "readout partial-energy delta mismatch: "
        f"calc-partialE={delta_partial:.16e}, calc-efs={delta_efs:.16e}, "
        f"tol={tol:.3e}"
    )

delta_lmp_eij = (
    sum(row[3] for row in eij_lmp_on)
    - sum(row[3] for row in eij_lmp_zero)
)
delta_lmp_efs = cfg_energy(efs_lmp_on_path) - cfg_energy(efs_lmp_zero_path)
delta_lmp_partial = (
    partial_interaction_sum(partial_lmp_on_path)
    - partial_interaction_sum(partial_lmp_zero_path)
)
lmp_tol = 2.0e-8 * max(1.0, abs(delta_lmp_efs))
if abs(delta_lmp_eij - delta_lmp_efs) > lmp_tol:
    raise SystemExit(
        "_lmp readout effective-pair delta mismatch: "
        f"calc-eij={delta_lmp_eij:.16e}, calc-efs={delta_lmp_efs:.16e}, "
        f"tol={lmp_tol:.3e}"
    )
if abs(delta_lmp_partial - delta_lmp_efs) > lmp_tol:
    raise SystemExit(
        "_lmp readout partial-energy delta mismatch: "
        f"calc-partialE={delta_lmp_partial:.16e}, "
        f"calc-efs={delta_lmp_efs:.16e}, tol={lmp_tol:.3e}"
    )

print(
    "readout diagnostics passed: "
    f"descriptor_columns={expected_columns}, delta={delta_efs:.12e}"
)
PY
