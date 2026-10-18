#!/usr/bin/env bash
set -eo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
[[ $# == 3 ]] || { echo 'usage: MODEL CFG OBJECT_DIR' >&2; exit 2; }
OBJECTS=$3
BUILD=${SUS2_TEST_BUILD_DIR:-$ROOT/.local_tmp/pure-w-state-check}
mkdir -p "$BUILD"
CPP=()
LD=()
RUN=()
[[ -z ${SUS2_TEST_CPPFLAGS:-} ]] || read -r -a CPP <<< "$SUS2_TEST_CPPFLAGS"
if [[ -n ${SUS2_TEST_LDFLAGS:-} ]]; then
  read -r -a LD <<< "$SUS2_TEST_LDFLAGS"
elif [[ $(uname -s) == Darwin ]]; then
  LD=(-framework Accelerate)
else
  echo 'set SUS2_TEST_LDFLAGS for the object build' >&2; exit 2
fi
[[ -z ${SUS2_TEST_RUNNER:-} ]] || read -r -a RUN <<< "$SUS2_TEST_RUNNER"
LINK=("$OBJECTS"/*.o "$OBJECTS"/common/*.o "$OBJECTS"/drivers/*.o "$OBJECTS"/mlp/mtpr_train.cpp.o)
"${CXX:-c++}" -std=c++11 -O2 -I"$ROOT" "${CPP[@]}" \
 "$ROOT/native_tests/pure_w_execution_state_check.cpp" "${LINK[@]}" "${LD[@]}" -o "$BUILD/check"
SUS2_SH_PURE_W_KERNEL=basis SUS2_SH_PURE_W_GRAPH=batched SUS2_SH_PURE_W_PROFILE=1 \
 "${RUN[@]}" "$BUILD/check" "$1" "$2"
