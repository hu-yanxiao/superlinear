#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 /path/to/mlp-sus2" >&2
  exit 2
fi

BIN=$1
REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)
WORKDIR=$(mktemp -d "${TMPDIR:-/tmp}/sus2sh-calc-eij-XXXXXX")
trap 'rm -rf "$WORKDIR"' EXIT

MODEL="$WORKDIR/l2k2_b4.sus2"
LMP_MODEL="$WORKDIR/l2k2_b4_lmp.sus2"
GATE_MODEL="$WORKDIR/l2k2_b4_gate.sus2"
GATE_LMP_MODEL="$WORKDIR/l2k2_b4_gate_lmp.sus2"
ONE_CFG="$WORKDIR/one.cfg"
PARTIAL="$WORKDIR/partial.txt"
EIJ="$WORKDIR/eij.txt"
LMP_EIJ="$WORKDIR/lmp_eij.txt"
GATE_LMP_EIJ="$WORKDIR/gate_lmp_eij.txt"
SMALL_CFG="$WORKDIR/small_periodic.cfg"
SMALL_EIJ="$WORKDIR/small_eij.txt"

awk '{print} /^END_CFG/{exit}' "$REPO_ROOT/example/train.cfg" > "$ONE_CFG"

"$BIN" init-sh "$MODEL" \
  --species-count=2 \
  --l-max=2 \
  --k-max=2 \
  --body-order=4 \
  --body-l-max=2,2,2,2 \
  --sh-factor-pruning=q-total \
  --write-sh-scalar-info \
  --cutoff=5.0 \
  --radial-basis-size=4 \
  --radial-basis-type=RBChebyshev_sss \
  --init-params=same >/dev/null

"$BIN" calc-partialE "$MODEL" "$ONE_CFG" "$PARTIAL" >/dev/null
"$BIN" calc-eij "$MODEL" "$ONE_CFG" "$EIJ" >/dev/null
if [[ ! -s "$EIJ" ]]; then
  echo "calc-eij did not create a non-empty output file" >&2
  exit 1
fi

python3 - "$MODEL" "$LMP_MODEL" <<'PY'
import sys

model, lmp_model = sys.argv[1], sys.argv[2]
text = open(model, encoding="utf-8").read()
text = text.replace("radial_basis_type = RBChebyshev_sss",
                    "radial_basis_type = RBChebyshev_sss_lmp")
open(lmp_model, "w", encoding="utf-8").write(text)
PY

"$BIN" calc-eij "$LMP_MODEL" "$ONE_CFG" "$LMP_EIJ" >/dev/null
if [[ ! -s "$LMP_EIJ" ]]; then
  echo "_lmp calc-eij did not create a non-empty output file" >&2
  exit 1
fi

"$BIN" init-sh "$GATE_MODEL" \
  --species-count=2 \
  --l-max=2 \
  --k-max=2 \
  --body-order=4 \
  --body-l-max=2,2,2,2 \
  --sh-factor-pruning=q-total \
  --two-layer-gate \
  --two-layer-gate-shared-radial \
  --cutoff=5.0 \
  --radial-basis-size=4 \
  --radial-basis-type=RBChebyshev_sss \
  --init-params=same >/dev/null

python3 - "$GATE_MODEL" "$GATE_LMP_MODEL" <<'PY'
import re
import sys

model, lmp_model = sys.argv[1], sys.argv[2]
text = open(model, encoding="utf-8").read()
match = re.search(r"two_layer_gate_weights\s*=\s*\{([^}]*)\}", text, re.S)
if not match:
    raise SystemExit("missing two_layer_gate_weights")
weights = [float(x) for x in re.findall(
    r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?", match.group(1))]
if not weights:
    raise SystemExit("empty two_layer_gate_weights")
weights[0] = 0.25
replacement = "two_layer_gate_weights = {" + ", ".join(f"{x:.15e}" for x in weights) + "}"
text = text[:match.start()] + replacement + text[match.end():]
text = text.replace("radial_basis_type = RBChebyshev_sss",
                    "radial_basis_type = RBChebyshev_sss_lmp")
open(lmp_model, "w", encoding="utf-8").write(text)
PY

"$BIN" calc-eij "$GATE_LMP_MODEL" "$ONE_CFG" "$GATE_LMP_EIJ" >/dev/null
if [[ ! -s "$GATE_LMP_EIJ" ]]; then
  echo "gate _lmp calc-eij did not create a non-empty output file" >&2
  exit 1
fi

python3 - "$PARTIAL" "$EIJ" <<'PY'
import math
import sys

partial_path, eij_path = sys.argv[1], sys.argv[2]

partial_sum = 0.0
with open(partial_path, "r", encoding="utf-8") as fh:
    for raw in fh:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        values = [float(x) for x in line.split()]
        partial_sum += sum(values[1:])

eij_sum = 0.0
line_count = 0
with open(eij_path, "r", encoding="utf-8") as fh:
    for raw in fh:
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 4:
            raise SystemExit(f"calc-eij row should have 4 columns, got {len(parts)}: {line}")
        t0, t1 = int(parts[0]), int(parts[1])
        if t0 > t1:
            raise SystemExit(f"pair type columns are not sorted: {line}")
        dist = float(parts[2])
        if dist <= 0.0:
            raise SystemExit(f"non-positive pair distance: {line}")
        eij_sum += float(parts[3])
        line_count += 1

if line_count == 0:
    raise SystemExit("calc-eij produced no pair rows")

tol = 1.0e-8 * max(1.0, abs(partial_sum))
if abs(eij_sum - partial_sum) > tol:
    raise SystemExit(
        f"EP sum mismatch: calc-eij={eij_sum:.16e}, partialE_scalar_sum={partial_sum:.16e}, tol={tol:.3e}"
    )
PY

cat > "$SMALL_CFG" <<'CFG'
BEGIN_CFG
 Size
      1
 Supercell
   1.0000000000    0.0000000000    0.0000000000
   0.0000000000    1.0000000000    0.0000000000
   0.0000000000    0.0000000000    1.0000000000
 AtomData:  id type       cartes_x      cartes_y      cartes_z
      1      0     0.000000     0.000000     0.000000
 Energy
      0.0
END_CFG
CFG

"$BIN" calc-eij "$MODEL" "$SMALL_CFG" "$SMALL_EIJ" >/dev/null
if [[ ! -s "$SMALL_EIJ" ]]; then
  echo "calc-eij did not create a non-empty periodic-image output file" >&2
  exit 1
fi

python3 - "$SMALL_EIJ" <<'PY'
import sys

rows = [line.split() for line in open(sys.argv[1], encoding="utf-8") if line.strip()]
if len(rows) <= 1:
    raise SystemExit(f"expected multiple periodic-image pair rows, got {len(rows)}")
for row in rows:
    if len(row) != 4:
        raise SystemExit(f"calc-eij row should have 4 columns: {' '.join(row)}")
    if row[0] != "0" or row[1] != "0":
        raise SystemExit(f"unexpected type pair in one-atom test: {' '.join(row)}")
print(f"calc-eij periodic-image row count: {len(rows)}")
PY
