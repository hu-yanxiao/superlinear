#!/usr/bin/env bash
# Install only the standalone superlinear engine and its Kokkos pair into an
# isolated clean LAMMPS source tree. Never build, submit, clean or publish here.
set -euo pipefail
if [[ $# != 1 ]]; then
  echo "Usage: $0 /absolute/path/to/isolated-clean-lammps-source" >&2
  exit 2
fi
pkg_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
lammps_root=$(cd "$1" && pwd -P)
if [[ "$lammps_root" == /work/phy-weigw/apps/lammps-10Dec2025 ]]; then
  echo "Refusing canonical CPU source tree; install into an isolated git archive" >&2
  exit 2
fi
for file in src/pair.h src/version.h src/KOKKOS/kokkos_base.h cmake/CMakeLists.txt lib/kokkos/CMakeLists.txt; do
  [[ -f "$lammps_root/$file" ]] || { echo "Incomplete LAMMPS source tree: $file" >&2; exit 2; }
done
if ! grep -q 'LAMMPS_VERSION "10 Dec 2025"' "$lammps_root/src/version.h"; then
  echo "Qualified build route requires LAMMPS 10 Dec 2025" >&2
  exit 2
fi
# CMake must see package sources in their original subdirectories. Old make
# installs root-level files, style headers and Obj_* directories. Refuse them
# instead of deleting or cleaning anything in a potentially active checkout.
shopt -s nullglob
legacy=("$lammps_root"/src/Obj_* "$lammps_root"/src/lmp_*
        "$lammps_root"/src/pair_sus2_mtp.* "$lammps_root"/src/sus2_mtp_*
        "$lammps_root"/src/pair_sus2_superlinear.*
        "$lammps_root"/src/KOKKOS/pair_sus2_mtp_kokkos.*)
if (( ${#legacy[@]} )); then
  echo "Existing old-make objects/binaries or historical SUS2 package detected; use a clean isolated source tree" >&2
  exit 2
fi
sources=(pair_sus2_embedded)
for source in "$pkg_root"/ML-SUS2/sus2_sl_*.cpp; do
  sources+=("$(basename "$source" .cpp)")
done
for source in "${sources[@]}"; do
  for suffix in cpp h; do
    [[ -f "$pkg_root/ML-SUS2/$source.$suffix" ]] || {
      echo "Incomplete standalone superlinear package: $source.$suffix" >&2; exit 2;
    }
  done
done
for suffix in cpp h; do
  [[ -f "$pkg_root/ML-SUS2-KK/pair_sus2_embedded_kokkos.$suffix" ]] || {
    echo "Missing Kokkos pair: pair_sus2_embedded_kokkos.$suffix" >&2; exit 2;
  }
done
headers=("$pkg_root"/ML-SUS2-KK/sus2_sl_kokkos_*.h)
(( ${#headers[@]} >= 3 )) || { echo "Missing Kokkos engine/math/radial headers" >&2; exit 2; }
[[ -f "$pkg_root/ML-SUS2/sus2_sl_zbl.h" ]] || { echo "Missing standalone ZBL header" >&2; exit 2; }

# Root-level standalone CPU units enter LAMMPS core's source glob. Only the new
# pair and its device headers enter KOKKOS; no historical sus2mtp file is copied.
installed=()
copy_if_changed() {
  if ! cmp -s "$1" "$2"; then cp "$1" "$2"; fi
}
for source in "${sources[@]}"; do
  for suffix in cpp h; do
    copy_if_changed "$pkg_root/ML-SUS2/$source.$suffix" "$lammps_root/src/$source.$suffix"
    installed+=("src/$source.$suffix")
  done
done
copy_if_changed "$pkg_root/ML-SUS2/sus2_sl_zbl.h" "$lammps_root/src/sus2_sl_zbl.h"
installed+=(src/sus2_sl_zbl.h)
for suffix in cpp h; do
  copy_if_changed "$pkg_root/ML-SUS2-KK/pair_sus2_embedded_kokkos.$suffix" "$lammps_root/src/KOKKOS/pair_sus2_embedded_kokkos.$suffix"
  installed+=("src/KOKKOS/pair_sus2_embedded_kokkos.$suffix")
done
for header in "${headers[@]}"; do
  copy_if_changed "$header" "$lammps_root/src/KOKKOS/$(basename "$header")"
  installed+=("src/KOKKOS/$(basename "$header")")
done
manifest="$lammps_root/SUPERLINEAR_KOKKOS_INSTALLED.sha256"
(
  cd "$lammps_root"
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "${installed[@]}"
  else
    shasum -a 256 "${installed[@]}"
  fi
) > "$manifest"
echo "Installed standalone sus2/embedded and sus2/embedded/kk into $lammps_root"
echo "Source hashes: $manifest"
echo "Configure into a NEW out-of-source build directory; never reuse a historical sus2mtp cache."
echo "Serial host control (not GPU qualification):"
printf 'cmake -S %q -B /absolute/path/to/new-serial-build -D CMAKE_BUILD_TYPE=Release -D CMAKE_CXX_STANDARD=17 -D BUILD_MPI=OFF -D BUILD_OMP=OFF -D PKG_KOKKOS=ON -D KOKKOS_PREC=double -D Kokkos_ENABLE_SERIAL=ON -D Kokkos_ENABLE_OPENMP=OFF -D Kokkos_ENABLE_CUDA=OFF\n' "$lammps_root/cmake"
echo "cmake --build /absolute/path/to/new-serial-build --target lmp --parallel 4"
echo "No build, scheduler submission or production replacement has been performed."
