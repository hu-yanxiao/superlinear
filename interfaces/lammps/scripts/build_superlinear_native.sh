#!/usr/bin/env bash
set -euo pipefail
if [[ $# != 1 ]]; then
  echo "Usage: $0 /absolute/output-directory" >&2
  echo "Optional: CXX, CXXFLAGS, CPPFLAGS, AR, JOBS, SUS2_NATIVE_BLAS_LIBS" >&2
  exit 2
fi
pkg_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
sus2_root=$(cd "$pkg_root/../.." && pwd)
mkdir -p "$1"
build_dir=$(cd "$1" && pwd)
if [[ "$build_dir" == *' '* || "$sus2_root" == *' '* ]]; then
  echo "Native make build currently requires paths without spaces" >&2
  exit 2
fi
native_cxx=${CXX:-c++}
native_flags=${CXXFLAGS:--O3 -fPIC -std=c++17}
native_cppflags=${CPPFLAGS:-}
native_blas=${SUS2_NATIVE_BLAS_LIBS:-}
if [[ -n ${MKLROOT:-} ]]; then
  native_cppflags+=" -DMLIP_INTEL_MKL -DMKL_ILP64 -I$MKLROOT/include"
  native_blas=${native_blas:--L$MKLROOT/lib/intel64 -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl}
elif [[ -z "$native_blas" ]]; then
  if [[ $(uname -s) == Darwin ]]; then native_blas='-framework Accelerate'; else native_blas='-lblas'; fi
fi
# No MLIP_MPI: the adapter owns communication and the library must never issue
# collectives on MPI_COMM_WORLD, especially in LAMMPS partition runs.
if [[ "$native_cppflags $native_flags" == *-DMLIP_MPI* ]]; then
  echo "Do not define MLIP_MPI in the isolated native inference library" >&2
  exit 2
fi
flags_tmp=$(mktemp "$build_dir/.native-flags.XXXXXX")
printf '%s\n' "$sus2_root" "$native_cxx" "$native_flags" "$native_cppflags" > "$flags_tmp"
if [[ -f "$build_dir/native-build-flags" ]] && cmp -s "$flags_tmp" "$build_dir/native-build-flags"; then
  rm "$flags_tmp"
else
  mv "$flags_tmp" "$build_dir/native-build-flags"
fi
make -f "$pkg_root/native/Makefile" -j "${JOBS:-4}" \
  SUS2_ROOT="$sus2_root" BUILD_DIR="$build_dir" CXX="$native_cxx" \
  CXXFLAGS="$native_flags" CPPFLAGS="$native_cppflags -DMLIP_DEV -I$sus2_root/cblas" AR="${AR:-ar}"
mkdir -p "$build_dir/include/sus2-native/src/common" "$build_dir/include/sus2-native/dev_src"
cp "$sus2_root"/src/*.h "$build_dir/include/sus2-native/src/"
cp "$sus2_root"/src/common/*.h "$build_dir/include/sus2-native/src/common/"
cp "$sus2_root"/dev_src/*.h "$build_dir/include/sus2-native/dev_src/"
# Package a link smoke test, including native class vtables and the full reader.
"$native_cxx" -std=c++17 ${native_cppflags} -I"$build_dir/include" "$pkg_root/native/link_smoke.cpp" \
  "$build_dir/lib/libsus2_superlinear_native.a" ${native_blas} -o "$build_dir/native-link-smoke"
"$build_dir/native-link-smoke"
# Record the selected ABI flags: the installed pair's header view must agree.
{
  echo "SUS2_SUPERLINEAR_CPPFLAGS = $native_cppflags"
  echo "SUS2_SUPERLINEAR_BLAS_LIBS = $native_blas"
} > "$build_dir/Makefile.native-config"
echo "Native library ready: $build_dir/lib/libsus2_superlinear_native.a"
