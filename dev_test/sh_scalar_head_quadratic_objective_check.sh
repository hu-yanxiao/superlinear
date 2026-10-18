#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "usage: $0 PURE_W_MODEL CFG_FILE [OBJECT_DIR]" >&2
  echo "optional env: CXX, SUS2_TEST_CPPFLAGS, SUS2_TEST_LDFLAGS," >&2
  echo "              SUS2_TEST_RUNNER, SUS2_TEST_BUILD_DIR" >&2
  exit 2
fi

model_path="$1"
cfg_path="$2"
object_dir="${3:-$root_dir/obj/ser}"
build_dir="${SUS2_TEST_BUILD_DIR:-$root_dir/.local_tmp/scalar-head-objective-check}"
cxx="${CXX:-c++}"
mkdir -p "$build_dir"

if [[ ! -f "$object_dir/mtpr.cpp.o" || ! -f "$object_dir/mtpr_trainer.cpp.o" ]]; then
  echo "object directory does not contain current mtpr and trainer objects: $object_dir" >&2
  exit 2
fi
if [[ ! -f "$object_dir/mlp/mtpr_train.cpp.o" ]]; then
  echo "object directory lacks the Rescale/helper object: $object_dir/mlp/mtpr_train.cpp.o" >&2
  exit 2
fi

extra_cppflags=()
extra_ldflags=()
runner=()
if [[ -n "${SUS2_TEST_CPPFLAGS:-}" ]]; then
  read -r -a extra_cppflags <<<"$SUS2_TEST_CPPFLAGS"
fi
if [[ -n "${SUS2_TEST_LDFLAGS:-}" ]]; then
  read -r -a extra_ldflags <<<"$SUS2_TEST_LDFLAGS"
elif [[ "$(uname -s)" == "Darwin" ]]; then
  extra_cppflags+=(
    -I/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/System/Library/Frameworks/Accelerate.framework/Frameworks/vecLib.framework/Headers
  )
  extra_ldflags+=( -framework Accelerate )
else
  echo "set SUS2_TEST_LDFLAGS to the BLAS/LAPACK/runtime flags used for OBJECT_DIR" >&2
  exit 2
fi
if [[ -n "${SUS2_TEST_RUNNER:-}" ]]; then
  read -r -a runner <<<"$SUS2_TEST_RUNNER"
fi

shopt -s nullglob
objects=(
  "$object_dir"/*.o
  "$object_dir"/common/*.o
  "$object_dir"/drivers/*.o
  "$object_dir"/mlp/mtpr_train.cpp.o
)
shopt -u nullglob
if (( ${#objects[@]} == 0 )); then
  echo "no link objects found below $object_dir" >&2
  exit 2
fi

common_compile=( -std=c++11 -O2 -I"$root_dir" "${extra_cppflags[@]}" )
"$cxx" "${common_compile[@]}" \
  -c "$root_dir/native_tests/scalar_head_site_components_check.cpp" \
  -o "$build_dir/scalar_head_site_components_check.o"
"$cxx" "${common_compile[@]}" \
  -c "$root_dir/native_tests/scalar_head_quadratic_objective_check.cpp" \
  -o "$build_dir/scalar_head_quadratic_objective_check.o"

"$cxx" -O2 \
  "$build_dir/scalar_head_site_components_check.o" \
  "${objects[@]}" "${extra_ldflags[@]}" \
  -o "$build_dir/scalar_head_site_components_check"
"$cxx" -O2 \
  "$build_dir/scalar_head_quadratic_objective_check.o" \
  "${objects[@]}" "${extra_ldflags[@]}" \
  -o "$build_dir/scalar_head_quadratic_objective_check"

if [[ -n "${SUS2_TEST_RUNNER:-}" ]]; then
  "${runner[@]}" "$build_dir/scalar_head_site_components_check" \
    "$model_path" "$cfg_path"
  "${runner[@]}" "$build_dir/scalar_head_quadratic_objective_check" \
    "$model_path" "$cfg_path"
else
  "$build_dir/scalar_head_site_components_check" "$model_path" "$cfg_path"
  "$build_dir/scalar_head_quadratic_objective_check" "$model_path" "$cfg_path"
fi

echo "scalar-head site components and physical full-quadratic objective checks passed"
