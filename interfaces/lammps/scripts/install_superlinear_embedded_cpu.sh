#!/usr/bin/env bash
set -euo pipefail
if [[ $# != 1 ]]; then
  echo "Usage: $0 /path/to/isolated/lammps-source" >&2
  exit 2
fi
pkg_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
lammps_root=$(cd "$1" && pwd)
[[ -f "$lammps_root/src/pair.h" ]] || {
  echo "Need a LAMMPS source tree" >&2; exit 2;
}
# This target intentionally has no native model library. Refuse an installed
# native-wrapper source rather than silently linking that library or deleting
# the user's existing installation. Use an isolated source tree for candidates.
if [[ -f "$lammps_root/src/pair_sus2_superlinear.cpp" ]]; then
  echo "Native wrapper already installed; use a separate source tree for the standalone target" >&2
  exit 2
fi
sources=(pair_sus2_embedded)
for source in "$pkg_root"/ML-SUS2/sus2_sl_*.cpp; do
  sources+=("$(basename "$source" .cpp)")
done
for source in "${sources[@]}"; do
  for suffix in cpp h; do
    [[ -f "$pkg_root/ML-SUS2/$source.$suffix" ]] || {
      echo "Incomplete embedded package: $source.$suffix" >&2; exit 2;
    }
  done
done
mkdir -p "$lammps_root/src/ML-SUS2" "$lammps_root/src/MAKE/OPTIONS"
for source in "${sources[@]}"; do
  cp "$pkg_root/ML-SUS2/$source.cpp" "$pkg_root/ML-SUS2/$source.h" "$lammps_root/src/ML-SUS2/"
  cp "$pkg_root/ML-SUS2/$source.cpp" "$pkg_root/ML-SUS2/$source.h" "$lammps_root/src/"
done
cp "$pkg_root/ML-SUS2/sus2_sl_zbl.h" "$lammps_root/src/ML-SUS2/"
cp "$pkg_root/ML-SUS2/sus2_sl_zbl.h" "$lammps_root/src/"
cp "$pkg_root/MAKE/OPTIONS/Makefile.ml_sus2_embedded_base_avx2" \
   "$pkg_root/MAKE/OPTIONS/Makefile.ml_sus2_embedded_avx2" "$lammps_root/src/MAKE/OPTIONS/"
echo "Installed self-contained sus2/embedded sources in $lammps_root"
echo "Build: make -C $lammps_root/src ml_sus2_embedded_avx2 -j 4"
echo "No SUS2 inference/training archive or private include tree is required."
