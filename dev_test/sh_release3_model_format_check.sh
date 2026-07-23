#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 /path/to/mlp-sus2" >&2
  exit 2
fi

bin=$1
repo_root=$(cd "$(dirname "$0")/.." && pwd)
tmp_dir=$(mktemp -d "${TMPDIR:-/tmp}/sus2sh-model-format-XXXXXX")
trap 'rm -rf "$tmp_dir"' EXIT

common_args=(
  --species-count=2
  --l-max=2
  --k-max=2
  --body-order=4
  --body-l-max=2,2,2,2
  --radial-basis-size=4
  --radial-basis-type=RBChebyshev_sss
  --cutoff=5.0
  --sh-factor-pruning=q-total
  --write-sh-scalar-info
  --two-layer-gate
  --two-layer-gate-factor-pruning=q-total
  --two-layer-gate-mode=mu-scalar-full
  --two-layer-gate-body-order=4
  --two-layer-gate-shared-radial
  --init-params=same
)

no_extension=$tmp_dir/no-extension.mtp
rank2=$tmp_dir/rank2.mtp
readout=$tmp_dir/readout.mtp

"$bin" init-sh "$no_extension" "${common_args[@]}" >/dev/null
"$bin" init-sh "$rank2" "${common_args[@]}" --outer-k-rank=2 >/dev/null
"$bin" init-sh "$readout" "${common_args[@]}" --two-layer-gate-readout >/dev/null

assert_version() {
  local model=$1
  local expected=$2
  local actual
  actual=$(sed -n '2s/^version = //p' "$model")
  if [[ $actual != "$expected" ]]; then
    echo "$model has version '$actual', expected '$expected'" >&2
    exit 1
  fi
}

assert_version "$no_extension" 1.1.0
assert_version "$rank2" 1.2.0
assert_version "$readout" 1.2.0

for model in "$no_extension" "$rank2" "$readout"; do
  roundtrip=$tmp_dir/$(basename "${model%.mtp}")-roundtrip.mtp
  "$bin" roundtrip-model-dev "$model" "$roundtrip" >/dev/null
done
assert_version "$tmp_dir/no-extension-roundtrip.mtp" 1.1.0
assert_version "$tmp_dir/rank2-roundtrip.mtp" 1.2.0
assert_version "$tmp_dir/readout-roundtrip.mtp" 1.2.0

# Both supported headers remain readable. Saving canonicalizes the header from
# the actual feature set, so a legacy rank file is upgraded and a feature-free
# 1.2 file is downgraded.
legacy_rank=$tmp_dir/rank2-header-1.1.mtp
extended_plain=$tmp_dir/no-extension-header-1.2.mtp
sed '2s/1\.2\.0/1.1.0/' "$rank2" > "$legacy_rank"
sed '2s/1\.1\.0/1.2.0/' "$no_extension" > "$extended_plain"
"$bin" roundtrip-model-dev "$legacy_rank" "$tmp_dir/legacy-rank-roundtrip.mtp" >/dev/null
"$bin" roundtrip-model-dev "$extended_plain" "$tmp_dir/extended-plain-roundtrip.mtp" >/dev/null
assert_version "$tmp_dir/legacy-rank-roundtrip.mtp" 1.2.0
assert_version "$tmp_dir/extended-plain-roundtrip.mtp" 1.1.0

unsupported=$tmp_dir/unsupported-version.mtp
sed '2s/1\.2\.0/9.9.9/' "$rank2" > "$unsupported"
if "$bin" roundtrip-model-dev "$unsupported" "$tmp_dir/should-not-exist.mtp" \
     >"$tmp_dir/unsupported.log" 2>&1; then
  echo "standalone reader accepted an unsupported SUS2-SH model version" >&2
  exit 1
fi
grep -F 'SUS2-SH model version must be "1.1.0" or "1.2.0"' \
  "$tmp_dir/unsupported.log" >/dev/null

# Kokkos deliberately reuses PairSUS2MTP::settings/read_file, so one host-side
# format guard covers both CPU and Kokkos pair styles.
python3 - "$repo_root" <<'PY'
from pathlib import Path
import sys

root = Path(sys.argv[1])
cpu = (root / "interfaces/lammps/ML-SUS2/pair_sus2_mtp.cpp").read_text()
kk_h = (root / "interfaces/lammps/ML-SUS2-KK/pair_sus2_mtp_kokkos.h").read_text()
kk_cpp = (root / "interfaces/lammps/ML-SUS2-KK/pair_sus2_mtp_kokkos.cpp").read_text()

if 'model_file_version == "1.1.0" || model_file_version == "1.2.0"' not in cpu:
    raise SystemExit("LAMMPS host reader does not explicitly accept 1.1.0 and 1.2.0")
if 'keyword == "two_layer_gate_readout_enabled"' not in cpu:
    raise SystemExit("LAMMPS host reader lacks gate-readout metadata support")
if 'keyword == "multi_layer_gate_enabled"' not in cpu or \
        '"do not support "' not in cpu or \
        '"multi_layer_gate_enabled models; use standalone mlp-sus2."' not in cpu:
    raise SystemExit("LAMMPS host reader lacks explicit multi-layer rejection")
for token in (
    "two_layer_gate_readout_coeff_count",
    "two_layer_gate_readout_coeffs",
    "readout_has_nonzero_coeffs",
):
    if token not in cpu:
        raise SystemExit(f"LAMMPS host reader/compute lacks {token}")
if "d_two_layer_gate_readout_coeffs" not in kk_h or \
        "ComputeGateReadoutEnergy" not in kk_cpp or \
        "ComputeGateReadoutAdjointSeeds" not in kk_cpp:
    raise SystemExit("LAMMPS Kokkos gate-readout path is incomplete")
seed = kk_cpp.find('"ComputeGateReadoutAdjointSeeds"')
full = kk_cpp.find('"ComputeGateFullScalarAdjointsTeam"', seed)
body = kk_cpp.find('"ComputeGateAdjointDers', seed)
if seed < 0 or (full < 0 and body < 0):
    raise SystemExit("LAMMPS Kokkos readout seeds are not ordered before backprop")
if "public PairSUS2MTP" not in kk_h or "PairSUS2MTP::settings(" not in kk_cpp:
    raise SystemExit("Kokkos no longer demonstrably reuses the guarded host reader")
PY

echo "release3 model format checks passed"
