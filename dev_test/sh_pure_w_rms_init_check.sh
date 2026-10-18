#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
task_bin=${1:-./bin/mlp-sus2}
mkdir -p .local_tmp
task_tmp=$(mktemp -d "$PWD/.local_tmp/pure-w-rms-api-XXXXXX")
echo "RMS API evidence: $task_tmp"

# Existing objects are used only for linking this focused harness. This script
# never rebuilds or replaces the model executable or production object files.
task_mode=${SUS2_TEST_OBJECT_MODE:-ser}
task_objects=()
while IFS= read -r task_object; do
  [[ $task_object == "obj/$task_mode/mlp/mlp.cpp.o" ]] || task_objects+=("$task_object")
done < <(rg --files -uuu "obj/$task_mode" -g '*.o' | sort)
[[ ${#task_objects[@]} -gt 0 ]] || { echo "Build the tested object files first" >&2; exit 2; }
task_defines=(-DMLIP_DEV)
task_cxx=${CXX:-c++}
task_libs=(-lblas -llapack)
if [[ $(uname -s) == Darwin ]]; then task_libs=(-framework Accelerate); fi
if [[ $task_mode == mpi ]]; then
  task_defines+=(-DMLIP_MPI)
  task_cxx=${CXX:-mpicxx}
fi
if [[ -n ${SUS2_TEST_LDLIBS:-} ]]; then read -r -a task_libs <<< "$SUS2_TEST_LDLIBS"; fi
"$task_cxx" -std=c++11 "${task_defines[@]}" -I./src -I./dev_src -I./cblas \
  dev_test/pure_w_rms_init_check.cpp "${task_objects[@]}" "${task_libs[@]}" \
  -o "$task_tmp/rms-api" > "$task_tmp/compile.log" 2>&1

task_common=(--species-count=2 --l-max=2 --body-order=3 --body-l-max=2,2,2,2
             --radial-basis-size=4 --cutoff=5 --scaling=1
             --two-layer-gate --two-layer-gate-mode=mu-scalar-full --two-layer-gate-body-order=3
             --write-sh-scalar-info)
"$task_bin" init-sh "$task_tmp/legacy.sus2" "${task_common[@]}" --scaling-map=K \
  --k-max=2 --outer-k-rank=2 --two-layer-gate-k-rank=2 > "$task_tmp/legacy.init.log"
for task_map in K L LK; do
  for task_rank in 1 3; do
    "$task_bin" init-sh "$task_tmp/$task_map.o$task_rank.sus2" "${task_common[@]}" \
      --superlinear-mode=pure-w-channelwise --scaling-map="$task_map" \
      --k-max=2 --outer-k-rank="$task_rank" --two-layer-gate-k-rank=2 \
      > "$task_tmp/$task_map.o$task_rank.init.log"
  done
  "$task_bin" init-sh "$task_tmp/$task_map.k1.sus2" "${task_common[@]}" \
    --superlinear-mode=pure-w-channelwise --scaling-map="$task_map" \
    --k-max=1 --outer-k-rank=3 --two-layer-gate-k-rank=1 \
    --two-layer-gate-decoupled-scaling > "$task_tmp/$task_map.k1.init.log"
  for task_cache_off in 0 1; do
    SUS2_SH_DISABLE_TWO_LAYER_EDGE_CACHE="$task_cache_off" \
      OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
      "$task_tmp/rms-api" "$task_tmp/$task_map.o1.sus2" "$task_tmp/$task_map.o3.sus2" \
      "$task_tmp/$task_map.k1.sus2" "$task_tmp/legacy.sus2" \
      native_tests/pure_w_regularization.cfg "$task_tmp/$task_map.off$task_cache_off.roundtrip.sus2" \
      > "$task_tmp/$task_map.off$task_cache_off.log" 2>&1
    tail -1 "$task_tmp/$task_map.off$task_cache_off.log"
  done
done
echo pure_w_rms_init_check_ok
