#!/bin/bash
# Server-only preparation. Does not submit jobs or change any source checkpoint.
set -euo pipefail
task_repo=/work/phy-weigw/20260321_Test/superlinear-radial-coefficient-gate-work-codex
task_source=/work/phy-weigw/hyx/vo2/9.1-adaptive-radial-pytorch-20260906-be41881
task_campaign=/work/phy-weigw/hyx/vo2/9.1-adaptive-pytorch-to-cpp-5000-33-200c-20260907
task_data=/work/phy-weigw/hyx/vo2/9.1/train.cfg
task_check=/work/phy-weigw/hyx/vo2/9.1-adaptive-efs-optimization-20260906/inputs/check3.cfg
task_binary=$task_repo/bin/mlp-sus2
task_oracle=$task_source/O2/runtime/mlp-sus2
task_code_commit=${1:?full committed campaign script identity}
[[ "$task_code_commit" =~ ^[0-9a-f]{40}$ ]]
[[ $(git -C "$task_repo" rev-parse HEAD) == "$task_code_commit" ]]
[[ -z $(git -C "$task_repo" status --porcelain --untracked-files=no) ]]
test ! -e "$task_campaign"
check_sha() { [[ $(sha256sum "$1" | cut -d ' ' -f1) == "$2" ]]; }
check_sha "$task_binary" 37657b894abb7321926cf5ef37f66f19761ac6a55bdd0cd39814981d9a65d888
check_sha "$task_oracle" 1134bffec5f2905f53b55c81f36df3e5ab71d10c510e1cfe9bc2fdada1189688
check_sha "$task_data" 0020c54b9a460ea31e8253c67970ed4434ea602e73e3870914233a4ca399c0c1
check_sha "$task_check" a714581ef641de514e148cc033e6324a9b55d10d1e609ed9a213d41a2b8fcb53
declare -A task_hash=(
  [2]=8c9fb148c4be58d978280a8045ef8f96487f7646830d63a0c837b217d837f126
  [4]=239ff63ee445bb60dab3276384c739841f0fe552d8c77e91186fa583384a1056
  [8]=d727f3f1914d83a2ca35d8a8c45f23d2eefc363aa491a7b82a81cecb006a2ac1
)
for task_outer in 2 4 8; do
  task_training=$task_source/O$task_outer
  grep -qx 'exit_code=0' "$task_training/exit_status.txt"
  grep -q 'Epoch 500:' "$task_training/train.log"
  check_sha "$task_training/current.best.mtp" "${task_hash[$task_outer]}"
  check_sha "$task_training/current.mtp" "${task_hash[$task_outer]}"
  check_sha "$task_training/inputs/train.cfg" 0020c54b9a460ea31e8253c67970ed4434ea602e73e3870914233a4ca399c0c1
done
mkdir "$task_campaign"
for task_outer in 2 4 8; do
  task_run=$task_campaign/O$task_outer
  task_training=$task_source/O$task_outer
  mkdir -p "$task_run/inputs" "$task_run/runtime" "$task_run/logs" "$task_run/preflight"
  cp "$task_training/current.best.mtp" "$task_run/inputs/resume.mtp"
  cp "$task_data" "$task_run/inputs/train.cfg"
  cp "$task_check" "$task_run/inputs/check3.cfg"
  cp "$task_binary" "$task_run/runtime/mlp-sus2"
  cp "$task_oracle" "$task_run/runtime/mlp-sus2-pytorch-oracle"
  cp "$task_repo/dev_test/vo2_superlinear_ab/compare_cfg_numeric.py" "$task_run/runtime/compare_cfg_numeric.py"
  cp "$task_repo/dev_test/vo2_adaptive_pytorch_cpp_continue/train_33_200c.lsf" "$task_run/train.lsf"
  cp "$task_training/exit_status.txt" "$task_run/inputs/pytorch_exit_status.txt"
  grep 'Epoch 500:' "$task_training/train.log" > "$task_run/inputs/pytorch_epoch500.txt"
  check_sha "$task_run/inputs/resume.mtp" "${task_hash[$task_outer]}"
  check_sha "$task_training/current.best.mtp" "${task_hash[$task_outer]}"
  check_sha "$task_run/inputs/train.cfg" 0020c54b9a460ea31e8253c67970ed4434ea602e73e3870914233a4ca399c0c1
  check_sha "$task_run/runtime/mlp-sus2" 37657b894abb7321926cf5ef37f66f19761ac6a55bdd0cd39814981d9a65d888
  {
    date -Is
    printf 'source_model=%s/current.best.mtp\nsource_pytorch_commit=be41881063a9ae881cb56e2dfd19a8df066341c9\n' "$task_training"
    printf 'binary_source_commit=cb685aaacaaa9ee1939a751021ba9d205aa95011\nscript_commit=%s\n' "$task_code_commit"
    printf 'outer_rank=%s\nqueue=33\nmpi_ranks=200\nallocation=5x40\nmax_optimizer_steps=5000\n' "$task_outer"
    printf 'source_epochs_completed=500\noptimizer_history=fresh-BFGS\nmodel_reinitialization=false\nrescale=false\n'
    printf 'objective=PyTorch-compatible-EFS-and-scalar-radial-regularization\nstd_weight=0\nstdd_weight=0\n'
  } > "$task_run/provenance.txt"
  chmod 444 "$task_run/inputs/resume.mtp" "$task_run/inputs/train.cfg" "$task_run/inputs/check3.cfg"
  (cd "$task_run" && sha256sum inputs/* runtime/* train.lsf provenance.txt > frozen.sha256)
  bash -n "$task_run/train.lsf"
  printf 'PREPARED %s\n' "$task_run"
done
date -Is > "$task_campaign/PREPARED"
