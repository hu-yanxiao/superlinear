#!/usr/bin/env bash
set -euo pipefail
if [[ $# != 2 ]]; then
  echo "Usage: $0 /path/to/lammps /path/to/built-native-library" >&2
  exit 2
fi
pkg_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
lammps_root=$(cd "$1" && pwd)
native_build=$(cd "$2" && pwd)
[[ -f "$lammps_root/src/pair.h" && -f "$native_build/lib/libsus2_superlinear_native.a" && -f "$native_build/Makefile.native-config" ]] || {
  echo "Need a LAMMPS source tree and a successfully built native library" >&2; exit 2;
}
native_install="$lammps_root/lib/sus2-superlinear"
mkdir -p "$lammps_root/src/ML-SUS2" "$native_install" "$lammps_root/src/MAKE/OPTIONS"
cp "$pkg_root"/ML-SUS2/pair_sus2_superlinear.{cpp,h} "$lammps_root/src/ML-SUS2/"
cp "$pkg_root"/ML-SUS2/pair_sus2_superlinear.{cpp,h} "$lammps_root/src/"
cp -R "$native_build/include" "$native_install/"
cp "$native_build/lib/libsus2_superlinear_native.a" "$native_install/"
cp "$native_build/Makefile.native-config" "$native_install/"
cp "$pkg_root/MAKE/OPTIONS/Makefile.ml_sus2_avx2" \
  "$pkg_root/MAKE/OPTIONS/Makefile.ml_sus2_superlinear_avx2" "$lammps_root/src/MAKE/OPTIONS/"
echo "Installed only sus2/superlinear and its isolated native library into $lammps_root"
echo "Build: make -C $lammps_root/src ml_sus2_superlinear_avx2 -j 16"
echo "Existing sus2mtp CPU/Kokkos sources are preserved. No CMake GPU integration is installed."
