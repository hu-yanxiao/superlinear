#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

bin=${1:-./bin/mlp-sus2}
bin=$(cd "$(dirname "$bin")" && pwd)/$(basename "$bin")
tmp_dir=$(mktemp -d "${TMPDIR:-/tmp}/sus2sh-radial-cutoff-XXXXXX")
if [[ ${SUS2_TEST_KEEP_TMP:-0} == 1 ]]; then
  echo "keeping test directory: $tmp_dir"
else
  trap 'rm -rf "$tmp_dir"' EXIT
fi

one_cfg="$tmp_dir/one.cfg"
base_cfg="$tmp_dir/base.cfg"
plus_cfg="$tmp_dir/plus.cfg"
minus_cfg="$tmp_dir/minus.cfg"
awk '{print} /^END_CFG/{exit}' example/train.cfg >"$one_cfg"

python3 - "$one_cfg" "$base_cfg" "$plus_cfg" "$minus_cfg" <<'PY'
import sys

source, base, plus, minus = sys.argv[1:]
lines = open(source, encoding="utf-8").read().splitlines(True)
for index, line in enumerate(lines):
    if line.strip().startswith("AtomData:"):
        names = line.split()[1:]
        atom_line = index + 1
        x_column = names.index("cartes_x")
        break
else:
    raise SystemExit("missing AtomData section")

def write(path, displacement):
    output = lines.copy()
    fields = output[atom_line].split()
    fields[x_column] = f"{float(fields[x_column]) + 0.019 + displacement:.16e}"
    output[atom_line] = "\t".join(fields) + "\n"
    open(path, "w", encoding="utf-8").writelines(output)

write(base, 0.0)
write(plus, 1.0e-4)
write(minus, -1.0e-4)
PY

common=(
  --species-count=2
  --l-max=2
  --k-max=2
  --body-order=4
  --body-l-max=2,2,2,2
  --radial-basis-size=4
  --cutoff=5.0
  --scaling-map=K
  --sh-factor-pruning=q-total
  --write-sh-scalar-info
)

check_force_fd() {
  local base_path=$1
  local plus_path=$2
  local minus_path=$3
  local label=$4
  python3 - "$base_path" "$plus_path" "$minus_path" "$label" <<'PY'
import math
import sys

def read(path):
    energy = None
    forces = []
    force_columns = None
    expect_energy = False
    for raw in open(path, encoding="utf-8"):
        line = raw.strip()
        if not line:
            continue
        if expect_energy:
            energy = float(line.split()[0])
            expect_energy = False
        elif line == "Energy":
            expect_energy = True
        elif line.startswith("AtomData:"):
            names = line.split()[1:]
            force_columns = [names.index(name) for name in ("fx", "fy", "fz")]
        elif force_columns is not None and line[0].isdigit():
            fields = line.split()
            if len(fields) > max(force_columns):
                forces.append([float(fields[i]) for i in force_columns])
    if energy is None or not forces:
        raise SystemExit(f"failed to parse {path}")
    values = [energy] + [component for row in forces for component in row]
    if not all(math.isfinite(value) for value in values):
        raise SystemExit(f"non-finite polynomial-cutoff prediction in {path}")
    return energy, forces

base_energy, base_forces = read(sys.argv[1])
plus_energy, _ = read(sys.argv[2])
minus_energy, _ = read(sys.argv[3])
dedx = (plus_energy - minus_energy) / 2.0e-4
force = base_forces[0][0]
error = abs(force + dedx)
scale = max(1.0, abs(force), abs(dedx))
if error > 3.0e-5 + 2.0e-3 * scale:
    raise SystemExit(
        f"{sys.argv[4]} polynomial-cutoff force FD mismatch: "
        f"force={force:.12e} dE/dx={dedx:.12e} error={error:.3e}"
    )
print(f"{sys.argv[4]}_polynomial_cutoff_force_fd={error:.3e}")
PY
}

legacy_default="$tmp_dir/legacy-default.sus2"
legacy_explicit="$tmp_dir/legacy-explicit.sus2"
"$bin" init-sh "$legacy_default" "${common[@]}" \
  --radial-basis-type=RBChebyshev_sss >/dev/null
"$bin" init-sh "$legacy_explicit" "${common[@]}" \
  --radial-basis-type=RBChebyshev_sss --radial-cutoff-type=legacy >/dev/null
cmp "$legacy_default" "$legacy_explicit"
if grep -q 'radial_cutoff_' "$legacy_default"; then
  echo "legacy/default model unexpectedly contains cutoff metadata" >&2
  exit 1
fi

legacy_roundtrip="$tmp_dir/legacy-roundtrip.sus2"
"$bin" roundtrip-model-dev "$legacy_default" "$legacy_roundtrip" >/dev/null
if grep -q 'radial_cutoff_' "$legacy_roundtrip"; then
  echo "old-format roundtrip introduced cutoff metadata" >&2
  exit 1
fi
"$bin" calc-efs "$legacy_default" "$base_cfg" "$tmp_dir/legacy-before.cfg" >/dev/null
"$bin" calc-efs "$legacy_roundtrip" "$base_cfg" "$tmp_dir/legacy-after.cfg" >/dev/null
cmp "$tmp_dir/legacy-before.cfg" "$tmp_dir/legacy-after.cfg"

if "$bin" init-sh "$tmp_dir/invalid-p-only.sus2" "${common[@]}" \
     --radial-basis-type=RBChebyshev_sss --radial-cutoff-p=6 \
     >"$tmp_dir/invalid-p-only.log" 2>&1; then
  echo "radial-cutoff-p without polynomial type was incorrectly accepted" >&2
  exit 1
fi
grep -q 'only applies to --radial-cutoff-type=polynomial' "$tmp_dir/invalid-p-only.log"

if "$bin" init-sh "$tmp_dir/invalid-p-zero.sus2" "${common[@]}" \
     --radial-basis-type=RBChebyshev_sss --radial-cutoff-type=polynomial \
     --radial-cutoff-p=0 >"$tmp_dir/invalid-p-zero.log" 2>&1; then
  echo "radial-cutoff-p=0 was incorrectly accepted" >&2
  exit 1
fi
grep -q 'radial-cutoff-p should be in' "$tmp_dir/invalid-p-zero.log"

bases=(RBChebyshev_sss RBChebyshev_sss_rational RBLaguerre_log1p RBJacobi_sss)
for basis in "${bases[@]}"; do
  stem=${basis#RB}
  model="$tmp_dir/$stem-p5.sus2"
  explicit="$tmp_dir/$stem-p5-explicit.sus2"
  type_only="$tmp_dir/$stem-p5-type-only.sus2"
  roundtrip="$tmp_dir/$stem-roundtrip.sus2"
  p7="$tmp_dir/$stem-p7.sus2"

  "$bin" init-sh "$model" "${common[@]}" \
    --radial-basis-type="$basis" --radial-cutoff-type=polynomial >/dev/null
  "$bin" init-sh "$explicit" "${common[@]}" \
    --radial-basis-type="$basis" --radial-cutoff-type=polynomial \
    --radial-cutoff-p=5 >/dev/null
  cmp "$model" "$explicit"
  grep -q $'^\tradial_cutoff_type = polynomial$' "$model"
  grep -q $'^\tradial_cutoff_p = 5$' "$model"

  grep -v $'^\tradial_cutoff_p = ' "$model" >"$type_only"
  "$bin" calc-efs "$type_only" "$base_cfg" "$tmp_dir/$stem-type-only.cfg" >/dev/null

  "$bin" roundtrip-model-dev "$model" "$roundtrip" >/dev/null
  grep -q $'^\tradial_cutoff_type = polynomial$' "$roundtrip"
  grep -q $'^\tradial_cutoff_p = 5$' "$roundtrip"

  "$bin" calc-efs "$model" "$base_cfg" "$tmp_dir/$stem-base.cfg" >/dev/null
  "$bin" calc-efs "$roundtrip" "$base_cfg" "$tmp_dir/$stem-roundtrip.cfg" >/dev/null
  cmp "$tmp_dir/$stem-base.cfg" "$tmp_dir/$stem-roundtrip.cfg"
  cmp "$tmp_dir/$stem-base.cfg" "$tmp_dir/$stem-type-only.cfg"

  "$bin" calc-efs "$model" "$plus_cfg" "$tmp_dir/$stem-plus.cfg" >/dev/null
  "$bin" calc-efs "$model" "$minus_cfg" "$tmp_dir/$stem-minus.cfg" >/dev/null
  check_force_fd "$tmp_dir/$stem-base.cfg" "$tmp_dir/$stem-plus.cfg" \
    "$tmp_dir/$stem-minus.cfg" "$basis"

  lmp_basis="${basis}_lmp"
  lmp_model="$tmp_dir/$stem-lmp-p5.sus2"
  lmp_roundtrip="$tmp_dir/$stem-lmp-roundtrip.sus2"
  sed "s/radial_basis_type = $basis/radial_basis_type = $lmp_basis/" \
    "$model" >"$lmp_model"
  "$bin" roundtrip-model-dev "$lmp_model" "$lmp_roundtrip" >/dev/null
  grep -q $'^\tradial_cutoff_type = polynomial$' "$lmp_roundtrip"
  grep -q $'^\tradial_cutoff_p = 5$' "$lmp_roundtrip"
  "$bin" calc-efs "$lmp_model" "$base_cfg" "$tmp_dir/$stem-lmp-base.cfg" >/dev/null
  "$bin" calc-efs "$lmp_roundtrip" "$base_cfg" \
    "$tmp_dir/$stem-lmp-roundtrip.cfg" >/dev/null
  cmp "$tmp_dir/$stem-lmp-base.cfg" "$tmp_dir/$stem-lmp-roundtrip.cfg"
  "$bin" calc-efs "$lmp_model" "$plus_cfg" "$tmp_dir/$stem-lmp-plus.cfg" >/dev/null
  "$bin" calc-efs "$lmp_model" "$minus_cfg" "$tmp_dir/$stem-lmp-minus.cfg" >/dev/null
  check_force_fd "$tmp_dir/$stem-lmp-base.cfg" "$tmp_dir/$stem-lmp-plus.cfg" \
    "$tmp_dir/$stem-lmp-minus.cfg" "$lmp_basis"

  "$bin" init-sh "$p7" "${common[@]}" \
    --radial-basis-type="$basis" --radial-cutoff-type=polynomial \
    --radial-cutoff-p=7 >/dev/null
  grep -q $'^\tradial_cutoff_p = 7$' "$p7"
  "$bin" calc-efs "$p7" "$base_cfg" "$tmp_dir/$stem-p7.cfg" >/dev/null
  if cmp -s "$tmp_dir/$stem-base.cfg" "$tmp_dir/$stem-p7.cfg"; then
    echo "$basis p=5 and p=7 predictions unexpectedly match" >&2
    exit 1
  fi
done

echo "selectable radial cutoff checks passed"
